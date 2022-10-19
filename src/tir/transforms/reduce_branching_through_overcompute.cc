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
 * \file reduce_branching_through_overcompute.cc
 *
 * \brief Attempt to remove conditional statements by introducing
 * extra computations that do not impact the final results.
 */

#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "../analysis/control_flow_graph.h"
#include "remove_no_op.h"
#include "simplify.h"

namespace tvm {
namespace tir {

class BranchReducer : public arith::IRMutatorWithAnalyzer {
 public:
  static Stmt Apply(Stmt stmt) {
    arith::Analyzer analyzer;
    arith::ControlFlowGraph touch_pattern(stmt);
    BranchReducer visitor(&analyzer, std::move(touch_pattern));
    return visitor(std::move(stmt));
  }

 private:
  using Parent = IRMutatorWithAnalyzer;
  using Parent::VisitStmt;
  using Parent::VisitStmt_;

  BranchReducer(arith::Analyzer* analyzer, arith::ControlFlowGraph touch_pattern)
      : Parent(analyzer), touch_pattern_(touch_pattern) {}

  Stmt VisitStmt_(const IfThenElseNode* op) final {
    IfThenElse cond = Downcast<IfThenElse>(Parent::VisitStmt_(op));

    auto is_special_case = [this](PrimExpr condition, Stmt general_case,
                                  Stmt special_case) -> bool {
      condition = analyzer_->rewrite_simplify(condition);
      With<arith::ConstraintContext> constraint(analyzer_, condition);
      Stmt stmt = general_case;
      // stmt = Simplify(stmt, analyzer_);
      stmt = RemoveNoOp(stmt, analyzer_, &touch_pattern_);
      return StructuralEqual()(stmt, special_case);
    };

    Stmt else_case = cond->else_case.defined() ? cond->else_case : Evaluate(0);

    if (is_special_case(cond->condition, else_case, cond->then_case)) {
      return else_case;
    } else if (is_special_case(!cond->condition, cond->then_case, else_case)) {
      return cond->then_case;
    } else {
      return std::move(cond);
    }
  }

 private:
  arith::ControlFlowGraph touch_pattern_;
};

Stmt ReduceBranchingThroughOvercompute(Stmt stmt) { return BranchReducer::Apply(std::move(stmt)); }

namespace transform {

Pass ReduceBranchingThroughOvercompute() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    arith::Analyzer analyzer;

    auto* n = f.CopyOnWrite();
    n->body = BranchReducer::Apply(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.ReduceBranchingThroughOvercompute", {});
}

TVM_REGISTER_GLOBAL("tir.transform.ReduceBranchingThroughOvercompute")
    .set_body_typed(ReduceBranchingThroughOvercompute);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
