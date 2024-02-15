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

/*! \file src/relax/transform/lazy_transform_params.cc */

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include <optional>
#include <unordered_map>

#include "utils.h"

namespace tvm {
namespace relax {

namespace {
class LazyInputMutator : public ExprMutator {
 public:
  Expr VisitExpr_(const FunctionNode* func) override {
    if (plan_.has_value()) {
      return ExprMutator::VisitExpr_(func);
    }

    int64_t num_input_params = func->GetAttr<IntImm>(attr::kNumInput).value_or(Integer(0))->value;
    CHECK_GE(num_input_params, 0) << "ValueError: "
                                  << "Annotation for attr::kNumInput (\"" << attr::kNumInput
                                  << "\") must be non-negative, but was " << num_input_params;
    CHECK_LE(static_cast<size_t>(num_input_params), func->params.size())
        << "ValueError: "
        << "Annotation for attr::kNumInput (\"" << attr::kNumInput << "\") specifies "
        << num_input_params << " parameters to be provided at runtime, "
        << "but the function only accepts " << func->params.size() << " parameters in total";

    std::unordered_map<Var, size_t, ObjectPtrHash, ObjectPtrEqual> lookup;
    for (size_t i = 0; i < func->params.size(); i++) {
      lookup.insert({func->params[i], i});
    }

    Var fget_param("fget_param",
                   FuncStructInfo({ObjectStructInfo(), PrimStructInfo(DataType::Int(64))},
                                  ObjectStructInfo()));

    Array<Var> new_params(func->params.begin(), func->params.begin() + num_input_params);
    new_params.push_back(fget_param);

    auto node = GetRef<Function>(func);
    node.CopyOnWrite()->params = new_params;
    node = WithAttr(node, attr::kNumInput, Integer(num_input_params + 1));

    plan_ = FunctionPlan{std::move(lookup), fget_param};
    auto output = Downcast<Function>(ExprMutator::VisitExpr_(node.get()));
    plan_.reset();
    return output;
  }

  Expr VisitExpr_(const VarNode* op) override {
    if (plan_) {
      Var var = GetRef<Var>(op);
      if (auto it = plan_->param_lookup.find(var); it != plan_->param_lookup.end()) {
        auto untyped =
            builder_->Emit(relax::Call(plan_->fget_param,
                                       {
                                           StringImm(var->name_hint()),
                                           PrimValue(IntImm(DataType::Int(64), it->second)),
                                       }),
                           var->name_hint() + "_untyped");
        return builder_->EmitMatchCast(untyped, GetStructInfo(var), var->name_hint());
      }
    }

    return ExprMutator::VisitExpr_(op);
  }

 private:
  struct FunctionPlan {
    std::unordered_map<Var, size_t, ObjectPtrHash, ObjectPtrEqual> param_lookup;
    Expr fget_param;
  };
  std::optional<FunctionPlan> plan_;
};
}  // namespace

Function WithLazyInputs(Function func) {
  LazyInputMutator mutator;

  func = Downcast<Function>(mutator.VisitExpr(func));
  func = Downcast<Function>(EliminateCommonSubexpr(func));
  return func;
}

namespace transform {

Pass LazyGetInput() {
  auto pass_func = [](Function func, IRModule, PassContext) -> Function {
    if (!func->GetAttr<String>(tvm::attr::kGlobalSymbol).defined()) {
      return func;
    }
    return WithLazyInputs(func);
  };
  return CreateFunctionPass(/*pass_function=*/pass_func,
                            /*opt_level=*/0,
                            /*pass_name=*/"LazyGetInput",
                            /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.LazyGetInput").set_body_typed(LazyGetInput);

// Pass LazyTransformParams() {
//   auto pass_func = [](Function func, IRModule, PassContext) -> Function {
//     LazyInput mutator;
//     return Downcast<Function>(mutator(func));
//   };
//   return CreateFunctionPass(/*pass_function=*/pass_func,
//                             /*opt_level=*/0,
//                             /*pass_name=*/"MutateOpsForTraining",
//                             /*required=*/{});
// }

}  // namespace transform
}  // namespace relax
}  // namespace tvm
