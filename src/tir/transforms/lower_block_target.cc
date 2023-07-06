/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file lower_block_target.cc
 */

#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "ir_utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief Move target annotation from Block/For into AttrStmt
 *
 * Blocks may be annotated with the target on which they should run.
 * Prior to removing blocks, the target should be preserved in a
 * AttrStmt.  For consistency, the target annotation of a loop is also
 * moved to an AttrStmt.
 */
class TargetAnnotationLowerer : public StmtExprMutator {
 public:
  static Stmt Apply(Stmt body) {
    TargetAnnotationLowerer mutator;
    return mutator(std::move(body));
  }

 private:
  Stmt VisitStmt_(const BlockNode* op) override {
    auto node = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    return MoveTargetAnnotation(node);
  }

  Stmt VisitStmt_(const ForNode* op) override {
    auto node = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    return MoveTargetAnnotation(node);
  }

  template <typename Node>
  Stmt MoveTargetAnnotation(Node node) {
    if (auto opt = node->annotations.Get(tvm::attr::kTarget)) {
      auto writer = node.CopyOnWrite();
      writer->body = AttrStmt(opt.value(), tvm::attr::kTarget, 0, std::move(writer->body));
      writer->annotations.erase(tvm::attr::kTarget);
    }
    return std::move(node);
  }
};

namespace transform {

Pass LowerScheduleableTargetAnnotation() {
  auto pass_func = [=](PrimFunc func, IRModule, PassContext) {
    if (auto new_body = TargetAnnotationLowerer::Apply(func->body); !new_body.same_as(func->body)) {
      func.CopyOnWrite()->body = std::move(new_body);
    }
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerScheduleableTargetAnnotation", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerScheduleableTargetAnnotation")
    .set_body_typed(LowerScheduleableTargetAnnotation);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
