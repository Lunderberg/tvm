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

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>

#include <optional>
#include <variant>

#include "../../../tir/analysis/stmt_to_primfunc.h"
#include "../../../tir/schedule/utils.h"
#include "../../../tir/transforms/ir_utils.h"
#include "../primitive.h"

namespace tvm {
namespace relax {

namespace {

Optional<String> UniqueBlockName(const tir::Stmt& stmt) {
  struct Visitor : tir::StmtVisitor {
    void VisitStmt_(const tir::BlockNode* op) override {
      if (op->name_hint.size()) {
        names.push_back(op->name_hint);
      }
      tir::StmtVisitor::VisitStmt_(op);
    }
    Array<String> names;
  };
  Visitor visitor;
  visitor(stmt);
  if (visitor.names.size() == 1) {
    return visitor.names[0];
  } else {
    return NullOpt;
  }
};

struct ArgumentMapping {
  struct ForwardOriginalArgument {
    size_t index;
  };
  struct HoistedArgument {
    tir::Buffer result;
    size_t producer_index;
    size_t primfunc_arg_index;
  };

  struct Argument {
    using ParamType = std::variant<tir::Var, tir::Buffer>;
    ParamType tir_param;
    std::variant<ForwardOriginalArgument, HoistedArgument> source;

    Argument(ParamType tir_param, std::variant<ForwardOriginalArgument, HoistedArgument> source)
        : tir_param(std::move(tir_param)), source(std::move(source)) {}

    std::string name() const {
      if (auto* ptr = std::get_if<tir::Var>(&tir_param)) {
        return (*ptr)->name_hint;
      } else if (auto* ptr = std::get_if<tir::Buffer>(&tir_param)) {
        return (*ptr)->name;
      } else {
        LOG(FATAL) << "Internal error, unhandled variant";
      }
    }
  };

  struct Function {
    GlobalVar id;
    std::vector<Argument> args;
  };

  Function original;
  std::vector<Function> mapping;

  ArgumentMapping(const tir::PrimFunc& original_func, GlobalVar original_id,
                  const Array<tir::PrimFunc>& fragments, const Array<GlobalVar>& fragment_ids) {
    ICHECK_EQ(fragments.size(), fragment_ids.size());

    auto get_param_obj = [](const tir::PrimFunc& func,
                            size_t i) -> std::variant<tir::Var, tir::Buffer> {
      tir::Var var = func->params[i];
      if (auto opt = func->buffer_map.Get(var)) {
        return opt.value();
      } else {
        return var;
      }
    };

    std::unordered_map<const Object*, Argument> arg_lookup;
    std::vector<Argument> original_args;
    for (size_t i = 0; i < original_func->params.size(); i++) {
      tir::Var var = original_func->params[i];
      if (auto opt = original_func->buffer_map.Get(var)) {
        tir::Buffer buf = opt.value();
        ShapeExpr shape(buf->shape);
      } else {
      }
      auto tir_param = get_param_obj(original_func, i);
      auto key = std::visit([](const ObjectRef& obj) { return obj.get(); }, tir_param);
      Argument arg(tir_param, ForwardOriginalArgument{i});
      original_args.push_back(arg);
      arg_lookup.insert_or_assign(key, arg);
    }
    original = Function{original_id, original_args};

    for (size_t func_i = 0; func_i < fragments.size(); func_i++) {
      const GlobalVar& fragment_id = fragment_ids[func_i];
      const tir::PrimFunc& fragment = fragments[func_i];
      std::vector<Argument> args;

      for (size_t arg_i = 0; arg_i < fragment->params.size(); arg_i++) {
        auto tir_param = get_param_obj(fragment, arg_i);
        auto key = std::visit([](const ObjectRef& obj) { return obj.get(); }, tir_param);
        if (auto res = arg_lookup.find(key); res != arg_lookup.end()) {
          args.push_back(res->second);
        } else if (auto* as_buf = std::get_if<tir::Buffer>(&tir_param)) {
          Argument arg(tir_param, HoistedArgument{*as_buf, func_i, arg_i});
          args.push_back(arg);
          arg_lookup.insert_or_assign(key, arg);
        } else {
          LOG(FATAL) << "Expected argument to either be forwarded from original, or a buffer";
        }
      }
      mapping.push_back({fragment_id, std::move(args)});
    }
  }

  Optional<relax::Expr> UpdateArguments(const relax::Call& call) const {
    static const Op& call_tir_op = Op::Get("relax.call_tir");

    if (call->op != call_tir_op) {
      return NullOpt;
    }

    ICHECK_GE(call->args.size(), 2);
    if (!call->args[0].same_as(original.id)) {
      return NullOpt;
    }
    const Array<relax::Expr>& original_inputs = Downcast<Tuple>(call->args[1])->fields;

    CHECK_EQ(original_inputs.size() + call->sinfo_args.size(), mapping.size())
        << "The original PrimFunc accepts " << mapping.size()
        << " parameters, but the call into it has " << call->args.size() << " input arguments and "
        << call->sinfo_args.size() << " output arguments";

    Array<relax::BindingBlock> binding_blocks;
    Array<relax::Call> generated_tir_calls;
    Array<Array<relax::Var>> generated_tir_call_outputs;

    size_t num_inputs = original_inputs.size();
    auto original_outputs = [&]() -> Array<StructInfo> {
      ICHECK_EQ(call->sinfo_args.size(), 1);
      if (auto* as_tuple = call->sinfo_args[0].as<TupleStructInfoNode>()) {
        return as_tuple->fields;
      } else {
        return call->sinfo_args;
      }
    }();
    size_t num_outputs = original_outputs.size();

    Map<tir::Var, PrimExpr> shape_remap;
    for (size_t i = 0; i < original.args.size(); i++) {
      auto sinfo = [&]() {
        if (i < num_inputs) {
          return GetStructInfo(original_inputs[i]);
        } else if (i - num_inputs < num_outputs) {
          return original_outputs[i - num_inputs];
        } else {
          LOG(FATAL) << "Internal error, argument size mismatch should be caught by now";
        }
      }();
      auto* buf_ptr = std::get_if<tir::Buffer>(&original.args[i].tir_param);
      auto* tensor_info = sinfo.as<TensorStructInfoNode>();
      if (buf_ptr && tensor_info) {
        if (auto* shape = tensor_info->shape.as<ShapeExprNode>()) {
          const tir::Buffer& buf = *buf_ptr;
          CHECK_EQ(shape->values.size(), buf->shape.size())
              << "Attempting to rewrite PrimFunc " << original.id << ", but the "
              << shape->values.size() << "-dimensional relax::Expr " << original_inputs[i]
              << " cannot bind to the " << buf->shape.size() << "-dimensional buffer argument "
              << buf->name;
          for (size_t dim_i = 0; dim_i < buf->shape.size(); dim_i++) {
            const PrimExpr& buffer_dim = buf->shape[i];
            const PrimExpr& arg_dim = shape->values[i];
            if (auto* as_var = buffer_dim.as<tir::VarNode>()) {
              shape_remap.Set(GetRef<tir::Var>(as_var), arg_dim);
            }
          }
        }
      }
    }

    std::vector<std::string> names = [&]() {
      std::vector<std::string> names(num_inputs + num_outputs, "");
      for (const auto& func : mapping) {
        for (const auto& arg : func.args) {
          if (auto* forward = std::get_if<ForwardOriginalArgument>(&arg.source)) {
            names[forward->index] = arg.name();
          }
        }
      }
      for (size_t i = 0; i < names.size(); i++) {
        if (names[i].size() == 0) {
          std::stringstream ss;
          ss << "arg_" << i;
          names[i] = ss.str();
        }
      }
      return names;
    }();

    Array<relax::Expr> input_vars;
    {
      Array<Binding> bindings;
      for (size_t i = 0; i < num_inputs; i++) {
        const auto& orig = original_inputs[i];
        if (auto ptr = orig.as<relax::VarNode>()) {
          input_vars.push_back(GetRef<relax::Expr>(ptr));
        } else {
          Var var(names[i], GetStructInfo(orig));
          input_vars.push_back(var);
          bindings.push_back(VarBinding(var, orig));
        }
      }
      if (bindings.size()) {
        binding_blocks.push_back(BindingBlock(bindings));
      }
    }
    Array<relax::Var> output_vars;
    for (size_t i = 0; i < num_outputs; i++) {
      output_vars.push_back(Var(names[num_inputs + i], original_outputs[i]));
    }

    for (const auto& func : mapping) {
      Array<relax::Expr> args = {};
      Array<relax::Var> call_output_vars;

      for (const auto& arg : func.args) {
        if (auto* forward = std::get_if<ForwardOriginalArgument>(&arg.source)) {
          if (forward->index < num_inputs) {
            args.push_back(input_vars[forward->index]);
          } else if (forward->index - num_inputs < num_outputs) {
            call_output_vars.push_back(output_vars[forward->index - num_inputs]);
          } else {
            LOG(FATAL) << "Internal error, call requires argument " << forward->index
                       << " of a function with " << num_inputs << " input args and " << num_outputs
                       << " output args";
          }

        } else if (auto* hoisted = std::get_if<HoistedArgument>(&arg.source)) {
          if (hoisted->producer_index < generated_tir_calls.size()) {
            size_t num_producer_inputs =
                Downcast<Tuple>(generated_tir_calls[hoisted->producer_index]->args[1])
                    ->fields.size();
            ICHECK_GE(hoisted->primfunc_arg_index, num_producer_inputs)
                << "Call to " << func.id << " needs the output from earlier call to "
                << mapping[hoisted->producer_index].id << ", which should be found at index "
                << hoisted->primfunc_arg_index << ".  However, the earlier call takes "
                << num_producer_inputs << " inputs, so index " << hoisted->primfunc_arg_index
                << " is an input";

            size_t out_i = hoisted->primfunc_arg_index - num_producer_inputs;
            ICHECK_LT(out_i, generated_tir_call_outputs[hoisted->producer_index].size())
                << "Call to " << func.id << " needs the output from earlier call to "
                << mapping[hoisted->producer_index].id << ", which should be found at index "
                << hoisted->primfunc_arg_index << ".  However, the earlier call takes "
                << num_producer_inputs << " inputs and "
                << generated_tir_call_outputs[hoisted->producer_index].size()
                << "outputs, so index " << hoisted->primfunc_arg_index << " doesn't exist";

            Expr arg = generated_tir_call_outputs[hoisted->producer_index][out_i];
            args.push_back(arg);
          } else if (hoisted->producer_index == generated_tir_calls.size()) {
            auto parameter_shape = hoisted->result->shape;
            auto argument_shape = parameter_shape.Map(
                [&shape_remap](const PrimExpr& expr) { return Substitute(expr, shape_remap); });
            TensorStructInfo sinfo_arg(ShapeExpr(argument_shape), hoisted->result->dtype);
            call_output_vars.push_back(relax::Var(hoisted->result->name, sinfo_arg));
          } else {
            LOG(FATAL) << "Internal error, cannot access output of " << hoisted->producer_index
                       << "-th in call to " << generated_tir_calls.size() << "-th PrimFunc";
          }
        } else {
          LOG(FATAL) << "Internal error, no variant matched";
        }
      }

      auto [new_call, bindings] = [&]() {
        if (call_output_vars.size() == 1) {
          relax::Call new_call(call_tir_op, {func.id, Tuple(args)}, Attrs(),
                               {GetStructInfo(call_output_vars[0])});
          Array<Binding> bindings = {VarBinding(call_output_vars[0], new_call)};
          return std::tuple{new_call, bindings};
        } else {
          relax::Call new_call(call_tir_op, {func.id, Tuple(args)}, Attrs(),
                               {TupleStructInfo(call_output_vars.Map(GetStructInfo))});
          Array<Binding> bindings;
          for (size_t i = 0; i < call_output_vars.size(); i++) {
            Expr item = TupleGetItem(new_call, i);
            Var var = call_output_vars[i];
            bindings.push_back(VarBinding(var, item));
          };
          return std::tuple{new_call, bindings};
        }
      }();

      relax::BindingBlock block_binding(bindings);

      binding_blocks.push_back(block_binding);
      generated_tir_calls.push_back(new_call);
      generated_tir_call_outputs.push_back(call_output_vars);
    }

    ICHECK(binding_blocks.size());

    auto output = [&]() -> Expr {
      // Special case, but a common one: The final call_tir produces
      // all the outputs that were produced by the original PrimFunc.
      const auto& last_call_tir_output = generated_tir_call_outputs.back();
      if (output_vars.size() == last_call_tir_output.size()) {
        bool all_same = true;
        for (size_t i = 0; i < output_vars.size(); i++) {
          if (!output_vars[i].same_as(last_call_tir_output[i])) {
            all_same = false;
            break;
          }
        }
        if (all_same) {
          Expr output = generated_tir_calls.back();
          generated_tir_calls.pop_back();
          binding_blocks.pop_back();
          generated_tir_call_outputs.pop_back();
          return output;
        }
      }

      if (output_vars.size() == 1) {
        return output_vars[0];
      } else {
        return Tuple(output_vars.Map([](const relax::Var& var) -> Expr { return var; }));
      }
    }();

    return SeqExpr(binding_blocks, output);
  }

  friend std::ostream& operator<<(std::ostream& os, const ArgumentMapping& map) {
    if (map.mapping.empty()) {
      return os << "ArgumentMapping()";
    }

    os << "ArgumentMapping(";
    for (size_t func_i = 0; func_i < map.mapping.size(); func_i++) {
      const auto& mapping = map.mapping[func_i];
      const auto& arguments = mapping.args;

      for (size_t arg_i = 0; arg_i < arguments.size(); arg_i++) {
        const auto& arg = arguments[arg_i];
        if (auto* hoisted = std::get_if<HoistedArgument>(&arg.source);
            hoisted && hoisted->producer_index == func_i) {
          os << "\n\t" << arg.name();
          os << " = alloc("
             << "shape = " << hoisted->result->shape << "),";
        }
      }

      os << "\n\t" << mapping.id << "(";
      for (size_t arg_i = 0; arg_i < arguments.size(); arg_i++) {
        const auto& arg = arguments[arg_i];

        if (arg_i) {
          os << ", ";
        }

        os << arg.name();
      }
      os << "),";
    }
    os << "\n)";
    return os;
  }
};

class Mutator : public ExprMutator {
 public:
  static Optional<Function> Apply(GlobalVar to_split, const ArgumentMapping& arg_mapping,
                                  Function expr) {
    Mutator mutator(to_split, arg_mapping);
    Function output = Downcast<Function>(mutator(expr));
    if (mutator.updated_) {
      return output;
    } else {
      return NullOpt;
    }
  }

  using ExprMutator::VisitExpr_;
  Expr VisitExpr_(const CallNode* call) override {
    Expr expr = VisitExprPostOrder_(call);
    call = expr.as<CallNode>();

    if (auto opt = arg_mapping_.UpdateArguments(Downcast<Call>(expr))) {
      updated_ = true;
      return opt.value();
    } else {
      return expr;
    }
  }

 private:
  Mutator(GlobalVar to_split, const ArgumentMapping& arg_mapping)
      : to_split_(to_split), arg_mapping_(arg_mapping) {}

  GlobalVar to_split_;
  const ArgumentMapping& arg_mapping_;
  bool updated_{false};
};

}  // namespace

Array<GlobalVar> SplitTIR(tir::ScheduleState self, const tir::StmtSRef& block_sref,
                          GlobalVar tir_primfunc, Array<String> new_primfunc_names) {
  tir::PrimFunc original = Downcast<tir::PrimFunc>(self->mod->functions.Get(tir_primfunc));

  tir::Block block_to_extract = GetRef<tir::Block>(TVM_SREF_TO_BLOCK(block_sref));

  Array<tir::StmtSRef> loops = GetLoops(block_sref);
  tir::StmtSRef scope_sref = loops.size() ? loops[0] : block_sref;

  tir::Stmt stmt_to_extract = GetRef<tir::Stmt>(scope_sref->stmt);

  tir::StmtSRef root_sref = GetScopeRoot(self, block_sref, /* require_stage_pipline = */ true);
  tir::Block root_block = Downcast<tir::Block>(GetRef<tir::Stmt>(root_sref->stmt));

  if (root_block->body.same_as(stmt_to_extract)) {
    // Extracting the entirety of a function is a no-op, because no
    // remainder function is left behind.
    //
    // TODO: Should the "extracted" function still be renamed?
    return {tir_primfunc};
  }

  auto root_body = root_block->body.as<tir::SeqStmtNode>();
  ICHECK(root_body) << "SplitTIR expects an independent stage to be extracted";
  const auto& stages = root_body->seq;

  size_t stmt_index = [&]() {
    for (size_t i = 0; i < stages.size(); i++) {
      if (stages[i].same_as(stmt_to_extract)) {
        return i;
      }
    }
    LOG(FATAL) << "Could not find stage to extract in the body of the root block";
  }();

  auto [fragments, fragment_names] = [&]() {
    size_t user_name_i = 0;
    auto next_id = [&](const tir::Stmt& stmt, const String& suffix) -> String {
      if (user_name_i < new_primfunc_names.size()) {
        std::string out = new_primfunc_names[user_name_i];
        user_name_i++;
        return out;
      }
      if (auto opt = UniqueBlockName(stmt)) {
        return opt.value();
      }
      return tir_primfunc->name_hint + suffix;
    };
    auto extracted_id = next_id(stmt_to_extract, "_stage");

    Array<tir::Stmt> fragments;
    Array<String> fragment_names;
    if (stmt_index != 0) {
      tir::Stmt before =
          stmt_index == 1
              ? stages[0]
              : tir::SeqStmt(Array<tir::Stmt>{stages.begin(), stages.begin() + stmt_index});
      fragments.push_back(before);
      fragment_names.push_back(next_id(before, "_pre"));
    }

    fragments.push_back(stmt_to_extract);
    fragment_names.push_back(extracted_id);

    if (stmt_index != stages.size() - 1) {
      tir::Stmt after =
          stmt_index == stages.size() - 2
              ? stages[stages.size() - 1]
              : tir::SeqStmt(Array<tir::Stmt>{stages.begin() + (stmt_index + 1), stages.end()});
      fragments.push_back(after);
      fragment_names.push_back(next_id(after, "_post"));
    }
    return std::tuple{fragments, fragment_names};
  }();

  Array<tir::PrimFunc> split_functions;
  Array<GlobalVar> fragment_ids;
  for (size_t i = 0; i < fragments.size(); i++) {
    const auto& fragment = fragments[i];
    auto fragment_primfunc = tir::StmtToPrimFunc(fragment).ToPrimFunc();
    GlobalVar fragment_id(fragment_names[i]);
    UpdateStructInfo(fragment_id, PrimFuncSignature(fragment_primfunc));
    fragment_ids.push_back(fragment_id);
    split_functions.push_back(fragment_primfunc);
  }

  ArgumentMapping arg_mapping(original, tir_primfunc, split_functions, fragment_ids);

  Map<GlobalVar, BaseFunc> updates;
  for (const auto& [global_var, func] : self->mod->functions) {
    if (auto* ptr = func.as<relax::FunctionNode>()) {
      auto func = GetRef<relax::Function>(ptr);

      if (auto opt = Mutator::Apply(tir_primfunc, arg_mapping, func)) {
        updates.Set(global_var, opt.value());
      }
    }
  }

  self->mod->Remove(tir_primfunc);
  for (size_t i = 0; i < split_functions.size(); i++) {
    self->mod->Add(fragment_ids[i], ReplaceAllVariables(split_functions[i]));
  }
  for (const auto& [global_var, updated] : updates) {
    self->mod->Update(global_var, updated);
  }

  return fragment_ids;
}

}  // namespace relax
}  // namespace tvm
