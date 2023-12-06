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
 * \file tvm/relax/transform/inject_lora.cc
 * \brief Mutate module to accept LoRA parameters
 */

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include <optional>
#include <regex>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "../op/tensor/binary.h"
#include "../op/tensor/linear_algebra.h"
#include "utils.h"

namespace tvm {
namespace relax {

namespace {
enum class LoraParamOrder {
  // Insert (lora_A, lora_B) after the corresponding base weight.
  AfterCorrespondingBaseWeight,

  // Append (lora_A, lora_B) after all other parameters.
  AfterAllParams,

  // Insert (lora_A, lora_B) after the location specified by
  // `attr::kNumInput` ("num_input"), incrementing `attr::kNumInput`
  // to mark the LoRA as being provided at runtime.  If
  // `attr::kNumInput` is absent, this is equivalent to
  // `AfterAllParams`.
  EndOfRuntimeParams,
};

class LoraInjector : public ExprMutator {
 public:
  LoraInjector(Array<Variant<Var, String>> params_to_update, std::optional<int> lora_r,
               LoraParamOrder lora_param_order)
      : lora_r_(lora_r), lora_param_order_(lora_param_order) {
    for (auto variant : params_to_update) {
      if (auto var = variant.as<Var>()) {
        param_vars_.insert(var.value());
      } else if (auto str = variant.as<String>()) {
        param_patterns_.push_back(std::regex(std::string(str.value())));
      } else {
        LOG(FATAL) << "InternalError: "
                   << "Expected Variant<Var, String> to contain either Var or String, "
                   << "but instead contained " << variant->GetTypeKey();
      }
    }
  }

  using ExprMutator::VisitExpr_;
  Expr VisitExpr_(const FunctionNode* op) override {
    auto func = GetRef<Function>(op);

    auto opt_num_input = func->attrs.GetAttr<Integer>(attr::kNumInput);
    size_t num_input = opt_num_input.value_or(Integer(func->params.size()))->value;
    CHECK_LE(num_input, func->params.size())
        << "ValueError: "
        << "Attribute attr::kNumInput (\"" << attr::kNumInput << "\") specified that " << num_input
        << " parameters would be unknown until runtime.  "
        << "However, the function only has " << func->params.size() << " parameters total.";

    Map<ObjectRef, ObjectRef> var_upper_bound_attr =
        func->GetAttr<Map<ObjectRef, ObjectRef>>("tir_var_upper_bound")
            .value_or(Map<ObjectRef, ObjectRef>());

    Array<Var> new_runtime_params;
    Array<Var> new_compile_time_params;
    Array<Var> append_after_all_params;
    Array<Var> append_after_runtime_params;
    Array<Binding> new_bindings;

    for (size_t i = 0; i < func->params.size(); i++) {
      const auto& param = func->params[i];

      auto base_param_output = [&]() {
        if (i < num_input) {
          return &new_runtime_params;
        } else {
          return &new_compile_time_params;
        }
      }();

      if (RequiresLoraParams(param)) {
        const auto* base_sinfo = param->struct_info_.as<TensorStructInfoNode>();
        CHECK(base_sinfo) << "TypeError: "
                          << "Parameters being updated to acceept a LoRA must be tensors, "
                          << "but parameter " << param << " has struct info "
                          << param->struct_info_;

        auto opt_base_shape = base_sinfo->GetShape();
        CHECK(opt_base_shape) << "TypeError: "
                              << "Tensor being updated to accept a LoRA must have a known shape, "
                              << "but parameter " << param << " has struct info "
                              << param->struct_info_ << ", without a known shape.";
        auto base_shape = opt_base_shape.value();

        CHECK_GE(base_shape.size(), 2)
            << "TypeError: "
            << "Tensor being updated to accept a LoRA must be at least two-dimensional, "
            << "but parameter " << param << " has struct info " << param->struct_info_
            << ", and is " << base_shape.size() << "-dimensional.";

        Array<PrimExpr> batch(base_shape.begin(), base_shape.end() - 2);
        PrimExpr outfeatures = base_shape[base_shape.size() - 2];
        PrimExpr infeatures = base_shape[base_shape.size() - 1];

        auto lora_r_dim = [&]() -> PrimExpr {
          if (lora_r_.has_value()) {
            return IntImm(DataType::Int(64), lora_r_.value());
          } else {
            return tir::Var(param->name_hint() + "_lora_r", DataType::Int(64));
          }
        }();

        if (auto lora_r_var = lora_r_dim.as<tir::VarNode>()) {
          std::optional<int64_t> upper_bound = std::nullopt;
          if (auto outfeatures_int = outfeatures.as<IntImmNode>()) {
            upper_bound = outfeatures_int->value;
          }
          if (auto infeatures_int = infeatures.as<IntImmNode>()) {
            upper_bound = std::min(infeatures_int->value,
                                   upper_bound.value_or(std::numeric_limits<int64_t>::max()));
          }

          if (upper_bound.has_value()) {
            var_upper_bound_attr.Set(lora_r_var->name_hint, Integer(upper_bound.value()));
          }
        }

        auto make_struct_info = [&](PrimExpr a, PrimExpr b) -> TensorStructInfo {
          Array<PrimExpr> shape = batch;
          shape.push_back(a);
          shape.push_back(b);
          return TensorStructInfo(ShapeExpr(shape), base_sinfo->dtype);
        };

        auto [base_weights_name, lora_A_name,
              lora_B_name] = [&]() -> std::tuple<String, String, String> {
          std::string old_name = param->name_hint();
          std::string suffix = ".weight";

          if (suffix.size() <= old_name.size() &&
              std::equal(suffix.rbegin(), suffix.rend(), old_name.rbegin())) {
            // If this parameter follows pytorch naming conventions,
            // maintain the same naming conventions for the LoRA.
            old_name = old_name.substr(0, old_name.size() - suffix.size());
            return {old_name + ".base_layer.weight", old_name + ".lora_A.weight",
                    old_name + ".lora_B.weight"};
          } else {
            return {old_name + "_base", old_name + "_LA", old_name + "_LB"};
          }
        }();

        Var base(base_weights_name, make_struct_info(outfeatures, infeatures));
        Var lora_a(lora_A_name, make_struct_info(lora_r_dim, infeatures));
        Var lora_b(lora_B_name, make_struct_info(outfeatures, lora_r_dim));

        base_param_output->push_back(base);

        auto lora_param_loc = [&]() {
          switch (lora_param_order_) {
            case LoraParamOrder::AfterCorrespondingBaseWeight:
              return base_param_output;
            case LoraParamOrder::AfterAllParams:
              return &append_after_all_params;

            case LoraParamOrder::EndOfRuntimeParams:
              return &append_after_runtime_params;

            default:
              LOG(FATAL) << "InternalError: "
                         << "Invalid lora param order: " << static_cast<int>(lora_param_order_);
          }
        }();
        lora_param_loc->push_back(lora_a);
        lora_param_loc->push_back(lora_b);

        Var lora_offset(param->name_hint() + "_lora_offset",
                        make_struct_info(outfeatures, infeatures));

        new_bindings.push_back(VarBinding(lora_offset, matmul(lora_b, lora_a, DataType::Void())));
        new_bindings.push_back(VarBinding(param, add(base, lora_offset)));

      } else {
        base_param_output->push_back(param);
      }
    }

    if (new_bindings.size()) {
      Array<Var> new_params;
      new_params.insert(new_params.end(), new_runtime_params.begin(), new_runtime_params.end());
      new_params.insert(new_params.end(), append_after_runtime_params.begin(),
                        append_after_runtime_params.end());
      new_params.insert(new_params.end(), new_compile_time_params.begin(),
                        new_compile_time_params.end());
      new_params.insert(new_params.end(), append_after_all_params.begin(),
                        append_after_all_params.end());

      SeqExpr new_body({DataflowBlock(new_bindings)}, func->body);
      func = Function(new_params, VisitExpr(new_body), NullOpt, func->is_pure, func->attrs,
                      func->span);
      if (opt_num_input) {
        func = WithAttr(func, attr::kNumInput,
                        Integer(num_input + append_after_runtime_params.size()));
      }
      if (var_upper_bound_attr.size()) {
        func = WithAttr(func, "tir_var_upper_bound", var_upper_bound_attr);
      }
      func = Downcast<Function>(CanonicalizeBindings(func));
    }

    return std::move(func);
  }

 private:
  bool RequiresLoraParams(const Var& var) const {
    if (param_vars_.count(var)) {
      return true;
    }

    std::string name = var->name_hint();
    return std::any_of(param_patterns_.begin(), param_patterns_.end(),
                       [&name](const std::regex& regex) { return std::regex_match(name, regex); });
  }

  /* \brief The dimension of the LoRA
   *
   * For weights `W` with shape `[outfeatures, infeatures]`, LoRA
   * replaces `W` with `(W + LB*LA)`, where `LA` has shape `[lora_r,
   * infeatures]` and `LB` has shape `[outfeatures, lora_r]`.  The
   * `lora_r` parameter is small relative to either `infeatures` or
   * `outfeatures`, but can vary between implementations
   *
   * If explicitly specified, `lora_r` will be set to the specified
   * integer.  Otherwise, `lora_r` will be a unique symbolic variable
   * for each weight that is updated.
   */
  std::optional<int> lora_r_;

  /* \brief Variables to replace */
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> param_vars_;

  /* \brief Patterns that match variables to replace */
  std::vector<std::regex> param_patterns_;

  /* \brief Where to place new parameters */
  LoraParamOrder lora_param_order_;
};
}  // namespace

namespace transform {
Pass InjectLora(Array<Variant<Var, String>> params_to_update, Optional<IntImm> ffi_lora_r,
                String ffi_lora_param_order) {
  LoraParamOrder lora_param_order = [&]() {
    if (ffi_lora_param_order == "after_corresponding_base_weight") {
      return LoraParamOrder::AfterCorrespondingBaseWeight;
    } else if (ffi_lora_param_order == "after_all_params") {
      return LoraParamOrder::AfterAllParams;
    } else if (ffi_lora_param_order == "end_of_runtime_params") {
      return LoraParamOrder::EndOfRuntimeParams;
    } else {
      LOG(FATAL) << "ValueError: "
                 << "Expected lora_param_order to be one of "
                 << ", but received \"" << ffi_lora_param_order << "\"";
    }
  }();

  std::optional<int> lora_r = std::nullopt;
  if (ffi_lora_r.defined()) {
    lora_r = ffi_lora_r.value()->value;
  }

  auto pass_func = [=](IRModule mod, PassContext pc) {
    LoraInjector mutator(params_to_update, lora_r, lora_param_order);

    std::unordered_set<GlobalVar, ObjectPtrHash, ObjectPtrEqual> to_remove;
    std::unordered_map<GlobalVar, Function, ObjectPtrHash, ObjectPtrEqual> to_add;

    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto func = base_func.as<Function>()) {
        auto updated = Downcast<Function>(mutator(func.value()));
        if (!updated.same_as(base_func)) {
          GlobalVar new_gvar(gvar->name_hint);
          UpdateStructInfo(new_gvar, GetStructInfo(updated));
          to_add.insert({new_gvar, updated});
          to_remove.insert(gvar);
        }
      }
    }

    if (to_remove.size() || to_add.size()) {
      auto write_ptr = mod.CopyOnWrite();

      for (const auto& gvar : to_remove) {
        write_ptr->Remove(gvar);
      }
      for (const auto& [gvar, func] : to_add) {
        write_ptr->Add(gvar, func);
      }
    }

    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 1, "InjectLora", {});
}

TVM_REGISTER_GLOBAL("relax.transform.InjectLora").set_body_typed(InjectLora);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
