/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file lower_buffer_arguments.cc
 * \brief Lower CallNode::buffer_map to an underlying type
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <variant>

namespace tvm {
namespace tir {

class BufferArgUsageCollector : public StmtExprVisitor {
 public:
  struct BufferUsage {
    Buffer parameter;
    std::vector<BufferRegion> arguments;

    BufferUsage(Buffer parameter) : parameter(parameter) {}

    void AddArgument(BufferRegion arg) {
      bool contiguous_parameter = parameter->strides.empty();
      if (contiguous_parameter) {
        bool contiguous_argument = [&]() {
          if (arg->buffer->strides.size()) {
            return false;
          }

          CHECK_EQ(arg->buffer->shape.size(), arg->region.size());

          // Skip past any leading extent==1 portions
          size_t i = 0;
          for (; i < arg->region.size(); i++) {
            const auto& extent = arg->region[i]->extent;
            if (!is_const_int(extent, 1)) {
              break;
            }
          }

          // One dimension may partially cover the buffer dimension.
          // (e.g. The region `Buf[0:1, 4:8, 0:16]` in a buffer of shape
          // `[16, 16, 16]`.)
          i++;

          arith::Analyzer analyzer;
          // All remaining dimensions must cover entire buffer dimension
          for (; i < arg->region.size(); i++) {
            const auto& range = arg->region[i];
            const auto& dim = arg->buffer->shape[i];
            if (!is_const_int(range->min, 0)) {
              return false;
            }
            if (!analyzer.CanProveEqual(range->extent, dim)) {
              return false;
            }
          }
          return true;
        }();
        CHECK(contiguous_argument)
            << "Buffer parameter " << parameter << " requires a contiguous argument, "
            << "but the provided region " << arg << " is not contiguous.";
      }

      arguments.push_back(arg);
    }
  };
  struct FuncUsage {
    bool exposed_externally{false};
    std::vector<std::variant<Var, BufferUsage>> arg_usage;

    FuncUsage(bool exposed_externally, std::vector<std::variant<Var, BufferUsage>> arg_usage)
        : exposed_externally(exposed_externally), arg_usage(std::move(arg_usage)) {}

    BufferUsage* GetBufferArg(size_t i) {
      ICHECK_LE(i, arg_usage.size());
      auto* buffer_usage = std::get_if<BufferUsage>(&arg_usage[i]);
      ICHECK(buffer_usage) << "Parameter " << i << " was defined as accepting a variable, "
                           << "but was called with a Buffer";
      return buffer_usage;
    }

    Array<ObjectRef> GetParameterArray() const {
      Array<ObjectRef> out;
      for (const auto& param : arg_usage) {
        if (auto* ptr = std::get_if<Var>(&param)) {
          out.push_back(*ptr);
        } else if (auto* ptr = std::get_if<BufferUsage>(&param)) {
          out.push_back(ptr->parameter);
        } else {
          LOG(FATAL) << "Internal error visiting variant";
        }
      }
      return out;
    }

    struct Definition {
      Var var;
      PrimExpr value;
      ObjectPath path;
    };

    std::tuple<std::vector<Definition>, std::vector<AssertStmt>> CollectImpliedArguments(
        const Array<PrimExpr>& args, const Map<Var, BufferRegion>& buffer_map) const {
      ICHECK_EQ(args.size(), arg_usage.size());

      std::unordered_map<const VarNode*, Definition> defined_params;
      for (size_t i = 0; i < args.size(); i++) {
        const auto& arg = args[i];
        const auto& usage = arg_usage[i];
        if (auto* ptr = std::get_if<Var>(&usage)) {
          tir::Var var_param = *ptr;
          if (auto* var_arg = arg.as<VarNode>()) {
            auto opt = buffer_map.Get(GetRef<Var>(var_arg));
            CHECK(!opt.defined()) << "Parameter " << (*ptr)->name_hint
                                  << " expected a value argument, "
                                  << "but received buffer argument " << opt.value();
          }
          defined_params.emplace(var_param.get(),
                                 Definition{var_param, arg, ObjectPath::Root((*ptr)->name_hint)});
        }
      }

      std::vector<AssertStmt> implied_conditions;
      std::vector<Definition> implied_params;
      for (size_t i = 0; i < args.size(); i++) {
        const PrimExpr& arg = args[i];
        if (auto* usage = std::get_if<BufferUsage>(&arg_usage[i])) {
          BufferRegion arg_buf = [&]() {
            auto var_ptr = arg.as<VarNode>();
            CHECK(var_ptr) << "Parameter " << i << " was expected to be a buffer, "
                           << "but instead received argument " << arg;
            auto opt = buffer_map.Get(GetRef<Var>(var_ptr));
            CHECK(opt) << "Parameter " << i << " was expected to be a buffer, "
                       << "but instead received argument " << arg;
            return opt.value();
          }();

          Buffer argument_slice = [&]() {
            auto region_mins = arg_buf->region.Map([](const auto& region) { return region->min; });
            auto region_extents =
                arg_buf->region.Map([](const auto& region) { return region->extent; });
            return arg_buf->buffer.MakeSlice(region_mins, region_extents);
          }();

          Buffer parameter_buf = usage->parameter;
          ObjectPath root_path = ObjectPath::Root(parameter_buf->name);

          auto check_expr = [&](const PrimExpr& param_expr, const ObjectPath& param_path,
                                const PrimExpr& arg_expr) {
            if (auto* var_ptr = param_expr.as<VarNode>()) {
              Var var = GetRef<Var>(var_ptr);
              if (auto it = defined_params.find(var_ptr); it != defined_params.end()) {
                std::stringstream ss;
                ss << "Callee requires that " << it->second.path << " == " << param_path;
                implied_conditions.push_back(
                    AssertStmt(it->second.value == arg_expr, StringImm(ss.str()), Stmt()));
              } else {
                Definition definition{var, arg_expr, param_path};
                defined_params.emplace(var_ptr, definition);
                implied_params.push_back(definition);
              }
            } else {
              std::stringstream ss;
              ss << "Callee requires that " << param_path << " == " << param_expr;
              implied_conditions.push_back(
                  AssertStmt(param_expr == arg_expr, StringImm(ss.str()), Stmt()));
            }
          };

          check_expr(parameter_buf->elem_offset, root_path->Attr("elem_offset"),
                     argument_slice->elem_offset);

          CHECK_EQ(parameter_buf->shape.size(), argument_slice->shape.size())
              << "Buffer parameter " << parameter_buf << " was expected to be passed a "
              << parameter_buf->shape.size()
              << "-dimensional region, but was instead passed the region " << arg_buf;
          for (size_t i = 0; i < parameter_buf->shape.size(); i++) {
            check_expr(parameter_buf->shape[i], root_path->Attr("shape")->ArrayIndex(i),
                       argument_slice->shape[i]);
          }

          if (parameter_buf->strides.size()) {
            CHECK_GE(arg_buf->buffer->shape.size(), parameter_buf->shape.size());
            size_t offset = arg_buf->buffer->shape.size() - parameter_buf->shape.size();
            for (size_t i = 0; i < offset; i++) {
              CHECK(is_const_int(arg_buf->region[i]->extent, 1))
                  << "Region " << arg_buf << " must have extent==1 for the first " << offset
                  << " dimensions, in order to be compatible with the parameter " << parameter_buf
                  << " with shape " << parameter_buf->shape;
            }
            Array<PrimExpr> argument_strides = [&]() {
              if (argument_slice->strides.size()) {
                return argument_slice->strides;
              } else {
                return argument_slice.MakeStrideView()->strides;
              }
            }();

            CHECK_EQ(argument_strides.size(), parameter_buf->strides.size())
                << "Buffer parameter requires " << parameter_buf->strides.size() << " strides ("
                << parameter_buf->strides << "), but the argument buffer only defines "
                << argument_strides.size() << "strides (" << argument_strides << ")";
            for (size_t i = 0; i < argument_strides.size(); i++) {
              check_expr(parameter_buf->strides[i], root_path->Attr("strides")->ArrayIndex(i),
                         argument_strides[i]);
            }
          }
        }
      }

      // Filter out trivially passed/failed asserts.  Additional
      // simplifications will be done later during tir.Simplify, but
      // this cleans up the resulting PrimFunc for constant-foldable
      // conditions.
      Array<String> failed_conditions;
      for (const auto& cond : implied_conditions) {
        if (auto* as_int = cond->condition.as<IntImmNode>()) {
          if (!as_int->value) {
            failed_conditions.push_back(Downcast<StringImm>(cond->message)->value);
          }
        }
      }
      if (failed_conditions.size()) {
        auto arg_or_buf = args.Map([&](const PrimExpr& arg) -> ObjectRef {
          if (auto* ptr = arg.as<VarNode>()) {
            if (auto opt = buffer_map.Get(GetRef<Var>(ptr))) {
              return opt.value();
            }
          }
          return arg;
        });
        ICHECK_EQ(failed_conditions.size(), 0)
            << "Arguments " << arg_or_buf << " are not compatible with parameters "
            << GetParameterArray() << ": " << failed_conditions;
      }

      implied_conditions.erase(std::remove_if(implied_conditions.begin(), implied_conditions.end(),
                                              [&](const AssertStmt& cond) -> bool {
                                                return cond->condition.as<IntImmNode>();
                                              }),
                               implied_conditions.end());

      return {implied_params, implied_conditions};
    }
  };

  static auto Collect(const IRModule& mod) {
    BufferArgUsageCollector collector;

    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto* ptr = base_func.as<PrimFuncNode>()) {
        auto func = GetRef<PrimFunc>(ptr);

        bool exposed_externally = func->GetAttr<String>(tvm::attr::kGlobalSymbol).defined();

        std::vector<std::variant<Var, BufferUsage>> arg_usage;
        for (const auto& param : func->params) {
          if (auto opt = func->buffer_map.Get(param)) {
            arg_usage.push_back(BufferUsage(opt.value()));
          } else {
            arg_usage.push_back(param);
          }
        }
        collector.usage_.emplace(gvar, FuncUsage(exposed_externally, std::move(arg_usage)));
      }
    }

    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto* ptr = base_func.as<PrimFuncNode>()) {
        collector.VisitStmt(ptr->body);
      }
    }

    return std::move(collector.usage_);
  }

 private:
  void VisitExpr_(const CallNode* op) override {
    StmtExprVisitor::VisitExpr_(op);

    auto* gvar_ptr = op->op.as<GlobalVarNode>();
    if (!gvar_ptr) {
      return;
    }
    auto gvar = GetRef<GlobalVar>(gvar_ptr);

    auto usage_it = usage_.find(gvar);
    ICHECK(usage_it != usage_.end()) << "CallNode attempts to make internal call to " << gvar
                                     << " exists, but no such PrimFunc exists in the IRModule";
    auto& func_usage = usage_it->second;

    ICHECK_EQ(op->args.size(), func_usage.arg_usage.size())
        << "Function " << gvar << " defined with " << func_usage.arg_usage.size()
        << " parameters, but was called with " << op->args.size() << " arguments.";

    for (size_t i = 0; i < op->args.size(); i++) {
      if (auto* var_ptr = op->args[i].as<VarNode>()) {
        auto var = GetRef<Var>(var_ptr);
        if (auto region = op->buffer_map.Get(var)) {
          auto buf = region.value()->buffer;
          func_usage.GetBufferArg(i)->AddArgument(region.value());
        }
      }
    }
  }

  static bool IsStatic(const Buffer& buf) {
    for (const auto& dim : buf->shape) {
      if (!dim->IsInstance<IntImmNode>()) {
        return false;
      }
    }
    for (const auto& stride : buf->strides) {
      if (!stride->IsInstance<IntImmNode>()) {
        return false;
      }
    }

    return true;
  }

  std::unordered_map<GlobalVar, FuncUsage, ObjectPtrHash, ObjectPtrEqual> usage_;
};

class BufferArgumentRewriter : public StmtExprMutator {
 public:
  using Parent = StmtExprMutator;

  static IRModule Apply(IRModule mod,
                        const std::unordered_map<GlobalVar, BufferArgUsageCollector::FuncUsage,
                                                 ObjectPtrHash, ObjectPtrEqual>& usage) {
    BufferArgumentRewriter rewriter(usage);

    bool made_change = false;
    Map<GlobalVar, BaseFunc> functions;

    for (const auto& kv : mod->functions) {
      const auto& gvar = kv.first;
      const auto& base_func = kv.second;

      auto func = [&]() -> BaseFunc {
        auto* func_ptr = base_func.as<PrimFuncNode>();
        if (!func_ptr) return base_func;
        auto func = GetRef<PrimFunc>(func_ptr);

        auto buffer_map = func->buffer_map;
        auto params = rewriter.UpdateParams(gvar, func->params, &buffer_map);
        auto body = rewriter.VisitStmt(func->body);

        if (params.same_as(func->params) && body.same_as(func->params)) {
          return base_func;
        } else {
          auto* write_ptr = func.CopyOnWrite();
          write_ptr->params = params;
          write_ptr->buffer_map = buffer_map;
          write_ptr->body = body;
          return func;
        }
      }();

      if (!func.same_as(base_func)) {
        made_change = true;
      }
      functions.Set(gvar, func);
    }

    if (made_change) {
      mod.CopyOnWrite()->functions = functions;
    }
    return mod;
  }

 private:
  BufferArgumentRewriter(const std::unordered_map<GlobalVar, BufferArgUsageCollector::FuncUsage,
                                                  ObjectPtrHash, ObjectPtrEqual>& usage)
      : usage_(usage) {}

  Stmt VisitStmt(const Stmt& stmt) override {
    Stmt node = Parent::VisitStmt(stmt);

    for (auto it = deferred_asserts_.rbegin(); it != deferred_asserts_.rend(); it++) {
      node = AssertStmt((*it)->condition, (*it)->message, node);
    }
    deferred_asserts_.clear();

    return node;
  }

  PrimExpr VisitExpr_(const LetNode* op) override {
    // TODO(Lunderberg): Handle this case.  Will need to replace the
    // bound variable in any deferred_asserts_ that are generated
    // within the body of the let binding.
    bool cache = inside_let_expr_;
    auto ret = Parent::VisitExpr_(op);
    inside_let_expr_ = cache;
    return ret;
  }

  PrimExpr VisitExpr_(const CallNode* op) override {
    if (op->op.same_as(builtin::if_then_else())) {
      // TODO(Lunderberg): Handle this case.  Will need to apply the
      // conditional to any deferred_asserts_ that are generated
      // inside the then/else of the conditional.
      bool cache = inside_if_then_expr_;
      auto ret = Parent::VisitExpr_(op);
      inside_if_then_expr_ = cache;
      return ret;
    }

    auto node = Downcast<Call>(Parent::VisitExpr_(op));

    auto* gvar_ptr = node->op.as<GlobalVarNode>();
    if (!gvar_ptr) {
      return std::move(node);
    }

    auto gvar = GetRef<GlobalVar>(gvar_ptr);

    auto usage_it = usage_.find(gvar);
    if (usage_it == usage_.end()) {
      return std::move(node);
    }
    const auto& func_usage = usage_it->second;

    if (func_usage.exposed_externally) {
      return std::move(node);
    }

    auto buffer_map = node->buffer_map;
    auto args = node->args.Map([&](const PrimExpr& arg) -> PrimExpr {
      if (auto* var_ptr = arg.as<VarNode>()) {
        Var var = GetRef<Var>(var_ptr);
        if (auto opt_buf = node->buffer_map.Get(var)) {
          buffer_map.erase(var);
          return opt_buf.value()->buffer->data;
        }
      }
      return arg;
    });
    if (args.same_as(node->args) && buffer_map.same_as(node->buffer_map)) {
      return std::move(node);
    }

    auto [implied_arguments, implied_conditions] =
        func_usage.CollectImpliedArguments(node->args, node->buffer_map);
    for (const auto& implied_arg : implied_arguments) {
      args.push_back(implied_arg.value);
    }

    // Because `tir::AssertStmt` has no equivalent that derives from
    // PrimExpr, the asserts need to be placed prior to the Stmt
    // that contains the Call.
    for (const auto& implied_cond : implied_conditions) {
      ICHECK(!inside_let_expr_)
          << "tir.LowerBufferArguments does not yet support rewrite of CallNode " << node
          << ", because it requires pre-condition checks that must be hoisted "
          << "outside of the tir::Let expression that contains the CallNode";
      ICHECK(!inside_if_then_expr_)
          << "tir.LowerBufferArguments does not yet support rewrite of CallNode " << node
          << ", because it requires pre-condition checks that must be hoisted "
          << "outside of a tir::if_then_else() expression containing the CallNode";
      deferred_asserts_.push_back(implied_cond);
    }

    return Call(node->dtype, node->op, args, buffer_map);
  }

  Array<Var> UpdateParams(const GlobalVar& gvar, const Array<Var>& params,
                          Map<Var, Buffer>* buffer_map) {
    auto usage_it = usage_.find(gvar);
    if (usage_it == usage_.end()) {
      return params;
    }
    const auto& func_usage = usage_it->second;

    if (func_usage.exposed_externally) {
      return params;
    }

    Map<Var, BufferRegion> dummy_buffer_region_map;
    Array<Var> new_params = params.Map([&](const auto& var) {
      if (auto opt_buf = buffer_map->Get(var)) {
        dummy_buffer_region_map.Set(var, BufferRegion::FullRegion(opt_buf.value()));
        buffer_map->erase(var);
        return opt_buf.value()->data;
      } else {
        return var;
      }
    });

    if (new_params.same_as(params)) {
      return params;
    }

    // Using CollectImpliedArguments() for the parameter updates, even
    // though we need to construct the dummy region map, removes the
    // potential of producing out-of-order changes at the call site
    // and the callee's signature.
    auto [implied_arguments, implied_conditions] = func_usage.CollectImpliedArguments(
        params.Map([](Var var) -> PrimExpr { return var; }), dummy_buffer_region_map);
    for (const auto& implied : implied_arguments) {
      new_params.push_back(implied.var);
    }
    return new_params;
  }

  const std::unordered_map<GlobalVar, BufferArgUsageCollector::FuncUsage, ObjectPtrHash,
                           ObjectPtrEqual>& usage_;
  std::vector<AssertStmt> deferred_asserts_;
  bool inside_let_expr_{false};
  bool inside_if_then_expr_{false};
};

namespace transform {
Pass LowerBufferArguments() {
  auto pass_func = [](IRModule mod, PassContext ctx) -> IRModule {
    auto usage = BufferArgUsageCollector::Collect(mod);

    return BufferArgumentRewriter::Apply(std::move(mod), usage);
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tir.LowerBufferArguments", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerBufferArguments").set_body_typed(LowerBufferArguments);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
