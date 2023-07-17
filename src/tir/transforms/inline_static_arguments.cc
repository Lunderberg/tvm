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
 *  Lower intrinsic calls and ops to device specific ir when possible.
 * \file inline_static_arguments.cc
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ir/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

namespace tvm {
namespace tir {

namespace {

using ArgSet = std::unordered_set<PrimExpr, StructuralHash, StructuralEqual>;
using ParamArgMap = std::unordered_map<Var, ArgSet, ObjectPtrHash, ObjectPtrEqual>;

}  // namespace

class ArgumentCollector : public StmtExprVisitor {
 public:
  ArgumentCollector(const IRModule& mod, ParamArgMap* arg_map) : mod_(mod), arg_map_(arg_map) {}

 private:
  void VisitExpr_(const CallNode* op) override {
    StmtExprVisitor::VisitExpr_(op);

    if (auto opt = op->op.as<GlobalVar>()) {
      auto gvar = opt.value();
      if (auto opt_base_func = mod_->functions.Get(gvar)) {
        if (auto opt_prim_func = opt_base_func.value().as<PrimFunc>()) {
          auto prim_func = opt_prim_func.value();
          auto global_symbol = prim_func->GetAttr<String>(tvm::attr::kGlobalSymbol);
          if (!global_symbol) {
            CollectArgs(op, prim_func);
          }
        }
      }
    }
  }

  void CollectArgs(const CallNode* caller, const PrimFunc& callee) {
    CHECK_EQ(caller->args.size(), callee->params.size())
        << "Call to subroutine " << caller->op << " provided " << caller->args.size()
        << " arguments (args = " << caller->args << "), "
        << "but callee only accepts " << callee->params.size()
        << " parameters (params = " << callee->params << ").";

    for (size_t i = 0; i < caller->args.size(); i++) {
      Var param = callee->params[i];
      PrimExpr arg = caller->args[i];
      if (auto opt = callee->buffer_map.Get(param)) {
        auto param_buf = opt.value();

        if (auto dltensor_arg = arg.as<CallNode>();
            dltensor_arg && dltensor_arg->op.same_as(builtin::tvm_stack_make_array())) {
          auto arg_shape = Downcast<Call>(dltensor_arg->args[1])->args;
          CHECK_EQ(param_buf->shape.size(), arg_shape.size())
              << "Callee " << caller->op << " accepts buffer arg " << param_buf << " of shape "
              << param_buf->shape << ", but argument had incompatible shape " << arg_shape;
          CollectParamArg(param_buf->shape, arg_shape);

          if (param_buf->strides.size()) {
            if (auto arg_strides = dltensor_arg->args[2].as<CallNode>()) {
              CHECK_EQ(param_buf->shape.size(), arg_shape.size())
                  << "Callee " << caller->op << "accepts buffer arg " << param_buf
                  << " with strides " << param_buf->strides
                  << ", but argument had incompatible strides " << arg_strides;
              CollectParamArg(param_buf->strides, arg_strides->args);
            }
          }

          auto arg_elem_offset = dltensor_arg->args[5];
          CollectParamArg(param_buf->elem_offset, arg_elem_offset);
        }

      } else {
        CollectParamArg(param, arg);
      }
    }
  }

  void CollectParamArg(const Array<PrimExpr>& param, const Array<PrimExpr>& arg) {
    ICHECK_EQ(param.size(), arg.size())
        << "Internal error: "
        << "Size mismatch should be caught earlier with more user-friendly error message";
    for (size_t i = 0; i < param.size(); i++) {
      CollectParamArg(param[i], arg[i]);
    }
  }

  void CollectParamArg(const PrimExpr& param, const PrimExpr& arg) {
    // A var may be an implicitly-defined parameter from a buffer.
    if (auto opt = param.as<Var>()) {
      (*arg_map_)[opt.value()].insert(arg);
    }
  }

  const IRModule& mod_;
  ParamArgMap* arg_map_;
};

class StaticArgumentRemover : public StmtExprMutator {
 public:
  StaticArgumentRemover(const IRModule& mod, const Map<Var, PrimExpr>& var_remap) {
    // Collect the gvar/index pairs that should be removed.
    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto opt = base_func.as<PrimFunc>()) {
        auto prim_func = opt.value();
        for (size_t i = 0; i < prim_func->params.size(); i++) {
          if (var_remap.count(prim_func->params[i])) {
            to_remove_[gvar.get()].insert(i);
          }
        }
      }
    }
  }

 private:
  PrimExpr VisitExpr_(const CallNode* op) override {
    auto node = Downcast<Call>(StmtExprMutator::VisitExpr_(op));

    if (auto* gvar = node->op.as<GlobalVarNode>()) {
      if (auto it = to_remove_.find(gvar); it != to_remove_.end()) {
        const auto& removed_indices = it->second;

        Array<PrimExpr> new_args;
        for (size_t i = 0; i < node->args.size(); i++) {
          if (!removed_indices.count(i)) {
            new_args.push_back(node->args[i]);
          }
        }
        node.CopyOnWrite()->args = new_args;
      }
    }

    return std::move(node);
  }

  std::unordered_map<const GlobalVarNode*, std::unordered_set<size_t>> to_remove_;
};

namespace transform {

Pass InlineStaticArguments() {
  auto pass_func = [](IRModule mod, PassContext ctx) -> IRModule {
    int num_internal_methods = [&mod]() {
      int count = 0;
      for (const auto& [gvar, base_func] : mod->functions) {
        if (auto prim_func = base_func.as<PrimFunc>()) {
          auto global_symbol = prim_func.value()->GetAttr<String>(tvm::attr::kGlobalSymbol);
          if (!global_symbol) {
            count++;
          }
        }
      }
      return count;
    }();

    if (num_internal_methods == 0) {
      // Early bail-out for most common case in which the IRModule has
      // no internal methods.
      return mod;
    }

    ParamArgMap arg_map;
    ArgumentCollector collector(mod, &arg_map);
    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto* prim_func = base_func.as<PrimFuncNode>()) {
        collector(prim_func->body);
      }
    }

    auto make_replacement_map = [](const ParamArgMap& arg_map) -> Map<Var, PrimExpr> {
      Map<Var, PrimExpr> var_remap;
      for (const auto& [var, args] : arg_map) {
        if (args.size() == 1) {
          if (auto unique_int = args.begin()->as<IntImm>()) {
            var_remap.Set(var, unique_int.value());
          }
        }
      }
      return var_remap;
    };

    Map<Var, PrimExpr> var_remap = make_replacement_map(arg_map);

    if (var_remap.size() == 0) {
      // Internal subroutines exist, but they are called with either
      // dynamic arguments, or with varying static arguments, so no
      // inlining is possible.
      return mod;
    }

    // While inlining a static argument, it may result in another
    // argument becoming static.  (e.g. A static buffer shape that is
    // passed to SubroutineA, which then passes the buffer to
    // SubroutineB.)  Because variables are SSA across the entire
    // module, we can iteratively apply the replacements to known
    // arguments, rather than iteratively mutating the PrimFunc.
    //
    // Because any recursive calls would require the propagated
    // parameter to be dynamic, the maximum number of iterations is
    // the maximum depth of the call graph.
    arith::Analyzer analyzer;
    for (int i = 0; i < num_internal_methods; i++) {
      bool converged = true;

      for (auto& [var, args] : arg_map) {
        ArgSet new_args;
        for (const auto& arg : args) {
          auto new_arg = Substitute(arg, var_remap);
          if (!new_arg.same_as(arg)) {
            converged = false;
            new_arg = analyzer.Simplify(new_arg);
          }
          new_args.insert(new_arg);
        }
        args = std::move(new_args);
      }

      if (converged) {
        break;
      } else {
        var_remap = make_replacement_map(arg_map);
      }
    }

    StaticArgumentRemover mutator(mod, var_remap);
    IRModule updated_callsite;
    for (auto [gvar, func] : mod->functions) {
      if (auto opt = func.as<PrimFunc>()) {
        auto prim_func = opt.value();
        auto new_body = mutator(prim_func->body);
        if (!new_body.same_as(prim_func->body)) {
          prim_func.CopyOnWrite()->body = new_body;
          updated_callsite->Add(gvar, prim_func);
        }
      }
    }
    mod.CopyOnWrite()->Update(updated_callsite);

    IRModule updated_callee;
    for (auto [gvar, base_func] : mod->functions) {
      if (auto opt = base_func.as<PrimFunc>()) {
        auto prim_func = opt.value();

        // The specialization map may only contain variables for this
        // specific PrimFunc.
        Map<Var, ObjectRef> specialize_map;
        auto collect_specialize = [&](const PrimExpr& expr) {
          if (auto opt = expr.as<Var>()) {
            auto var = opt.value();
            if (auto remap = var_remap.Get(var)) {
              specialize_map.Set(var, remap.value());
            }
          }
        };

        for (const auto& param : prim_func->params) {
          if (auto opt = prim_func->buffer_map.Get(param)) {
            auto buf = opt.value();
            for (const auto& dim : buf->shape) {
              collect_specialize(dim);
            }
            for (const auto& stride : buf->strides) {
              collect_specialize(stride);
            }
            collect_specialize(buf->elem_offset);
          } else {
            collect_specialize(param);
          }
        }

        if (specialize_map.size()) {
          updated_callee->Add(gvar, Specialize(prim_func, specialize_map));
        }
      }
    }

    mod.CopyOnWrite()->Update(updated_callee);

    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tir.InlineStaticArguments", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InlineStaticArguments").set_body_typed(InlineStaticArguments);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
