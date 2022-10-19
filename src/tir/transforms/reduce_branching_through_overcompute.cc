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

#include <optional>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "../analysis/control_flow_graph.h"
#include "remove_no_op.h"
#include "simplify.h"

namespace tvm {
namespace tir {

struct ReduceBranchingThroughOvercomputeConfigNode
    : public tvm::AttrsNode<ReduceBranchingThroughOvercomputeConfigNode> {
  bool use_dataflow_analysis;

  TVM_DECLARE_ATTRS(ReduceBranchingThroughOvercomputeConfigNode,
                    "tir.transform.ReduceBranchingThroughOvercomputeConfig") {
    TVM_ATTR_FIELD(use_dataflow_analysis)
        .describe(
            "If true, known buffer values are propagated and used "
            "to statically prove that overcompute is valid.")
        .set_default(false);
  }
};

class ReduceBranchingThroughOvercomputeConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ReduceBranchingThroughOvercomputeConfig, Attrs,
                                            ReduceBranchingThroughOvercomputeConfigNode);
};

TVM_REGISTER_NODE_TYPE(ReduceBranchingThroughOvercomputeConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.ReduceBranchingThroughOvercompute",
                                ReduceBranchingThroughOvercomputeConfig);

class BranchReducer : public arith::IRMutatorWithAnalyzer {
 public:
  static Stmt Apply(Stmt stmt, const arith::ControlFlowGraph* touch_pattern) {
    arith::Analyzer analyzer;
    BranchReducer visitor(&analyzer, touch_pattern);
    return visitor(std::move(stmt));
  }

 private:
  using Parent = IRMutatorWithAnalyzer;
  using Parent::VisitStmt;
  using Parent::VisitStmt_;

  BranchReducer(arith::Analyzer* analyzer, const arith::ControlFlowGraph* touch_pattern)
      : Parent(analyzer), touch_pattern_(touch_pattern) {}

  Stmt VisitStmt_(const IfThenElseNode* op) final {
    IfThenElse cond = Downcast<IfThenElse>(Parent::VisitStmt_(op));

    auto is_special_case = [this](PrimExpr condition, Stmt general_case,
                                  Stmt special_case) -> bool {
      condition = analyzer_->rewrite_simplify(condition);
      With<arith::ConstraintContext> constraint(analyzer_, condition);
      Stmt stmt = general_case;
      stmt = RemoveNoOp(stmt, analyzer_, touch_pattern_);
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
  const arith::ControlFlowGraph* touch_pattern_;
};

namespace transform {

Pass ReduceBranchingThroughOvercompute() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    arith::Analyzer analyzer;

    std::optional<arith::ControlFlowGraph> touch_pattern = std::nullopt;

    ReduceBranchingThroughOvercomputeConfig config =
        ctx->GetConfig<ReduceBranchingThroughOvercomputeConfig>(
               "tir.ReduceBranchingThroughOvercompute")
            .value_or(AttrsWithDefaultValues<ReduceBranchingThroughOvercomputeConfig>());
    if (config->use_dataflow_analysis) {
      touch_pattern.emplace(f->body);
    }
    auto touch_pattern_ptr = touch_pattern.has_value() ? &touch_pattern.value() : nullptr;

    auto* n = f.CopyOnWrite();
    n->body = BranchReducer::Apply(std::move(n->body), touch_pattern_ptr);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.ReduceBranchingThroughOvercompute", {});
}

TVM_REGISTER_GLOBAL("tir.transform.ReduceBranchingThroughOvercompute")
    .set_body_typed(ReduceBranchingThroughOvercompute);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
