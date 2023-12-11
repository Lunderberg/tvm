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
 * \file tvm/relax/transform/inject_routing_table.cc
 * \brief Mutate module to look up parameter(s) in a table
 */

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include <optional>
#include <regex>
#include <unordered_set>
#include <vector>

#include "utils.h"

namespace tvm {
namespace relax {

namespace {
class RoutingTableInjector : public ExprMutator {
 public:
  RoutingTableInjector(Array<Variant<Var, String>> params_to_update,
                       Optional<PrimExpr> routing_table_size)
      : routing_table_size_(routing_table_size) {
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

    PrimExpr routing_table_size = [&]() -> PrimExpr {
      if (routing_table_size_.defined()) {
        return routing_table_size_.value();
      } else {
        return tir::Var("routing_table_size", DataType::Int(64));
      }
    }();
    CHECK(routing_table_size->dtype == DataType::Int(64))
        << "TypeError: "
        << "Routing table must have Int64 dtype, "
        << "but value " << routing_table_size << " has dtype " << routing_table_size;

    Optional<Var> routing_table;

    Array<Var> new_params;
    Array<Binding> new_bindings;
    for (const auto& param : func->params) {
      if (RequiresRoutingTable(param)) {
        const auto* base_sinfo = param->struct_info_.as<TensorStructInfoNode>();
        CHECK(base_sinfo) << "TypeError: "
                          << "Parameters being updated to use a routing table must be tensors, "
                          << "but parameter " << param << " has struct info "
                          << param->struct_info_;

        auto opt_base_shape = base_sinfo->GetShape();
        CHECK(opt_base_shape)
            << "TypeError: "
            << "Tensor being updated to use a routing table must have a known shape, "
            << "but parameter " << param << " has struct info " << param->struct_info_
            << ", without a known shape.";
        auto base_shape = opt_base_shape.value();

        CHECK_GE(base_shape.size(), 1)
            << "TypeError: "
            << "Tensor being updated to use a routing table must have a batch dimension, "
            << "but parameter " << param << " has struct info " << param->struct_info_
            << ", and is zero-dimensional.";

        PrimExpr batch_size = base_shape[0];
        if (routing_table) {
          Array<PrimExpr> routing_table_shape =
              Downcast<TensorStructInfo>(routing_table.value()->struct_info_)->GetShape().value();
          CHECK(StructuralEqual()(batch_size, routing_table_shape[0]))
              << "All tensors being updated to use a routing table must have the same batch "
                 "dimension, "
              << "but parameter " << param << " has struct info " << param->struct_info_
              << ", and its batch size " << batch_size << " differs from the batch size "
              << routing_table_shape[0] << " defined by earlier parameters.";
        } else {
          batch_size = base_shape[0];
          routing_table =
              Var("routing_table", TensorStructInfo(ShapeExpr({batch_size}), DataType::Int(64)));
        }

        base_shape.Set(0, routing_table_size);
        Var indexed_param(
            "indexed_" + param->name_hint(),
            TensorStructInfo(ShapeExpr(base_shape), base_sinfo->dtype, base_sinfo->vdevice));
        new_params.push_back(indexed_param);
        new_bindings.push_back(
            VarBinding(param, relax::take(indexed_param, routing_table.value(), Integer(0))));

      } else {
        new_params.push_back(param);
      }
    }

    if (new_bindings.size()) {
      auto opt_num_input = func->attrs.GetAttr<Integer>(attr::kNumInput);
      if (opt_num_input) {
        // Insert the binding table at the end of the runtime parameters
        new_params.insert(new_params.begin() + opt_num_input.value()->value, routing_table.value());
      } else {
        // Insert the binding table at the end of the runtime parameters
        new_params.push_back(routing_table.value());
      }
      SeqExpr new_body({DataflowBlock(new_bindings)}, func->body);

      func = Function(new_params, VisitExpr(new_body), NullOpt, func->is_pure, func->attrs,
                      func->span);
      if (opt_num_input) {
        func = WithAttr(func, attr::kNumInput, Integer(opt_num_input.value()->value + 1));
      }

      func = Downcast<Function>(CanonicalizeBindings(func));
    }

    return std::move(func);
  }

 private:
  bool RequiresRoutingTable(const Var& var) const {
    if (param_vars_.count(var)) {
      return true;
    }

    std::string name = var->name_hint();
    return std::any_of(param_patterns_.begin(), param_patterns_.end(),
                       [&name](const std::regex& regex) { return std::regex_match(name, regex); });
  }

  /* \brief The dimension of the routing table
   *
   * A parameter of shape `[batch_size, *shape]` will be replaced with
   * a new parameter of shape `[routing_table_size, *shape]`.
   */
  Optional<PrimExpr> routing_table_size_;

  /* \brief Variables to replace */
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> param_vars_;

  /* \brief Patterns that match variables to replace */
  std::vector<std::regex> param_patterns_;
};
}  // namespace

namespace transform {
Pass InjectRoutingTable(Array<Variant<Var, String>> params_to_update,
                        Optional<PrimExpr> routing_table_size) {
  auto pass_func = [=](IRModule mod, PassContext pc) {
    RoutingTableInjector mutator(params_to_update, routing_table_size);

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
  return tvm::transform::CreateModulePass(pass_func, 1, "InjectRoutingTable", {});
}

TVM_REGISTER_GLOBAL("relax.transform.InjectRoutingTable").set_body_typed(InjectRoutingTable);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
