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
 * \file remove_no_op.cc
 * \brief Remove no op from the stmt
 */
#include "remove_no_op.h"

#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <optional>

#include "../../arith/const_fold.h"
#include "../../arith/ir_mutator_with_analyzer.h"
#include "../analysis/control_flow_graph.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

struct RemoveNoOpConfigNode : public tvm::AttrsNode<RemoveNoOpConfigNode> {
  bool use_dataflow_analysis;

  TVM_DECLARE_ATTRS(RemoveNoOpConfigNode, "tir.transform.RemoveNoOpConfig") {
    TVM_ATTR_FIELD(use_dataflow_analysis)
        .describe(
            "If true, known buffer values are propagated and used "
            "to statically prove statements as no-ops.")
        .set_default(false);
  }
};

class RemoveNoOpConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(RemoveNoOpConfig, Attrs, RemoveNoOpConfigNode);
};

TVM_REGISTER_NODE_TYPE(RemoveNoOpConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.RemoveNoOp", RemoveNoOpConfig);

// Mark the statement of each stage.
class NoOpRemover : public arith::IRMutatorWithAnalyzer {
 public:
  static Stmt Apply(Stmt stmt, arith::Analyzer* analyzer,
                    const arith::ControlFlowGraph* touch_pattern) {
    NoOpRemover visitor(analyzer, touch_pattern);
    return visitor(std::move(stmt));
  }

 private:
  using Parent = IRMutatorWithAnalyzer;
  using Parent::VisitStmt;
  using Parent::VisitStmt_;

  NoOpRemover(arith::Analyzer* analyzer, const arith::ControlFlowGraph* touch_pattern)
      : Parent(analyzer), touch_pattern_(touch_pattern) {}

  Stmt VisitStmt_(const LetStmtNode* op) final {
    Stmt stmt = Parent::VisitStmt_(op);
    op = stmt.as<LetStmtNode>();
    if (is_no_op(op->body)) {
      return MakeEvaluate(op->value);
    }

    bool body_uses_bound_variable =
        !UsesVar(op->body, [&](const VarNode* var) { return var == op->var.get(); });
    if (body_uses_bound_variable && HasSideEffect(op->value)) {
      return SeqStmt({MakeEvaluate(op->value), op->body});
    } else if (body_uses_bound_variable) {
      return op->body;
    } else {
      return stmt;
    }
  }
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == "pragma_debug_skip_region") {
      return MakeEvaluate(0);
    } else if (op->attr_key == attr::async_wait_queue_scope) {
      auto wait_attrs = GetAsyncWaitAttributes(op);
      auto wait_cnt = wait_attrs.second;
      arith::Analyzer ana;
      if (ana.CanProve(wait_cnt < 0)) {
        // A negative wait count can arise if it depends on a loop variable.
        // For example, a wait count 1 - i can be negative after loop unrolling.
        // We assume that such wait is a nop.
        auto inner = op->body.as<AttrStmtNode>();
        ICHECK(inner);
        return StmtMutator::VisitStmt(inner->body);
      }
    }
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<AttrStmtNode>();
    return is_no_op(op->body) ? MakeEvaluate(op->value) : stmt;
  }
  Stmt VisitStmt_(const IfThenElseNode* op) final {
    Stmt stmt = Parent::VisitStmt_(op);
    op = stmt.as<IfThenElseNode>();
    if (op->else_case.defined()) {
      bool no_op_else = is_no_op(op->else_case);
      bool no_op_then = is_no_op(op->then_case);
      if (no_op_else && no_op_then) {
        return MakeEvaluate(op->condition);
      } else if (no_op_else) {
        return IfThenElse(op->condition, op->then_case);
      } else if (no_op_then) {
        return IfThenElse(Not(op->condition), op->else_case);
      } else {
        return stmt;
      }
    } else {
      if (is_no_op(op->then_case)) {
        return MakeEvaluate(op->condition);
      } else {
        return stmt;
      }
    }
  }
  Stmt VisitStmt_(const ForNode* op) final {
    var_range_map_[op->loop_var.get()] = arith::IntSet::FromMinExtent(op->min, op->extent);
    auto extent_range = arith::EvalSet(op->extent, var_range_map_);
    if (!arith::is_neg_inf(extent_range.max()) && !arith::is_pos_inf(extent_range.max()) &&
        analyzer_->CanProve(extent_range.max() <= 0)) {
      return Evaluate(0);
    }
    Stmt stmt = StmtMutator::VisitStmt_(op);
    var_range_map_.erase(op->loop_var.get());
    op = stmt.as<ForNode>();
    if (is_zero(op->extent)) {
      return Evaluate(0);
    }
    return is_no_op(op->body) ? MakeEvaluate({op->min, op->extent}) : stmt;
  }
  Stmt VisitStmt_(const AllocateNode* op) final {
    Stmt stmt = Parent::VisitStmt_(op);
    op = stmt.as<AllocateNode>();
    return is_no_op(op->body) ? MakeEvaluate(op->extents) : stmt;
  }

  Stmt VisitStmt_(const ProducerRealizeNode* op) final {
    Stmt stmt = Parent::VisitStmt_(op);
    op = stmt.as<ProducerRealizeNode>();
    return is_no_op(op->body) ? op->body : stmt;
  }
  Stmt VisitStmt_(const EvaluateNode* op) final {
    if (HasSideEffect(op->value)) {
      return GetRef<Stmt>(op);
    } else {
      return Evaluate(0);
    }
  }

  Stmt VisitStmt_(const SeqStmtNode* op) final {
    auto ret = Downcast<SeqStmt>(StmtMutator::VisitSeqStmt_(op, true));

    bool need_compact = std::any_of(ret->seq.begin(), ret->seq.end(),
                                    [](const auto& stmt) { return is_no_op(stmt); });

    if (need_compact) {
      Array<Stmt> filtered;
      for (Stmt stmt : ret->seq) {
        if (!is_no_op(stmt)) {
          filtered.push_back(std::move(stmt));
        }
      }
      ret = SeqStmt(filtered);
    }

    if (ret->size() == 0) {
      return Evaluate(0);
    } else if (ret->size() == 1) {
      return ret->seq[0];
    } else {
      return std::move(ret);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore store = GetRef<BufferStore>(op);

    // Helper function that returns a statement containing only the
    // side effects of evaluating this BufferStore, but not the store
    // itself.
    auto only_side_effects = [&]() {
      Array<Stmt> statements;
      statements.push_back(MakeEvaluate(store->value));
      for (const auto& index : store->indices) {
        statements.push_back(MakeEvaluate(index));
      }
      return this->VisitStmt(SeqStmt(statements));
    };

    if (touch_pattern_) {
      // A write that is later overwritten is a no-op.
      if (touch_pattern_->IsOverwrittenWithoutEffect(store, analyzer_)) {
        return only_side_effects();
      }

      // A write whose destination is known to already contain the
      // values to be written is a no-op.
      PrimExpr stores_existing_value = store->value == BufferLoad(store->buffer, store->indices);

      PrimExpr simplified =
          touch_pattern_->SimplifyInContext(stores_existing_value, store, analyzer_);
      if (auto* as_int = as_const_int(simplified); as_int && *as_int) {
        return only_side_effects();
      }
    }

    // If the stored value is a load from the same location, the
    // statement is a no-op, regardless of contextual information.
    if (const BufferLoadNode* load = store->value.as<BufferLoadNode>()) {
      if (load->buffer->data.same_as(store->buffer->data) &&
          analyzer_->CanProveEqual(load->buffer->elem_offset, store->buffer->elem_offset) &&
          ArrayValueEqual(load->buffer->shape, store->buffer->shape) &&
          ArrayValueEqual(load->buffer->strides, store->buffer->strides) &&
          ArrayValueEqual(load->indices, store->indices)) {
        return only_side_effects();
      }
    }

    return std::move(store);
  }

 private:
  bool ArrayValueEqual(const Array<PrimExpr>& a, const Array<PrimExpr>& b) {
    if (a.size() != b.size()) {
      return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
      if (!analyzer_->CanProveEqual(a[i], b[i])) {
        return false;
      }
    }
    return true;
  }

  bool HasSideEffect(const PrimExpr& value) {
    return SideEffect(value) > CallEffectKind::kReadState;
  }

  Stmt MakeEvaluate(PrimExpr value) {
    if (HasSideEffect(value)) {
      return Evaluate(value);
    } else {
      return Evaluate(0);
    }
  }
  Stmt MakeEvaluate(const Array<PrimExpr>& values) {
    Stmt stmt;
    for (PrimExpr e : values) {
      if (HasSideEffect(e)) {
        if (stmt.defined()) {
          stmt = SeqStmt({stmt, Evaluate(e)});
        } else {
          stmt = Evaluate(e);
        }
      }
    }
    return stmt.defined() ? stmt : Evaluate(0);
  }

  std::unordered_map<const VarNode*, arith::IntSet> var_range_map_;
  const arith::ControlFlowGraph* touch_pattern_;
};

Stmt RemoveNoOp(Stmt stmt, arith::Analyzer* analyzer,
                const arith::ControlFlowGraph* touch_pattern) {
  return NoOpRemover::Apply(std::move(stmt), analyzer, touch_pattern);
}

namespace transform {

Pass RemoveNoOp() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    std::optional<arith::ControlFlowGraph> touch_pattern = std::nullopt;

    RemoveNoOpConfig config = ctx->GetConfig<RemoveNoOpConfig>("tir.RemoveNoOp")
                                  .value_or(AttrsWithDefaultValues<RemoveNoOpConfig>());
    if (config->use_dataflow_analysis) {
      touch_pattern.emplace(f->body);
    }
    auto touch_pattern_ptr = touch_pattern.has_value() ? &touch_pattern.value() : nullptr;

    arith::Analyzer analyzer;
    analyzer.rewrite_simplify.SetEnabledExtensions(
        arith::RewriteSimplifier::kTransitivelyProveInequalities);

    auto* n = f.CopyOnWrite();
    n->body = NoOpRemover::Apply(std::move(n->body), &analyzer, touch_pattern_ptr);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.RemoveNoOp", {});
}

TVM_REGISTER_GLOBAL("tir.transform.RemoveNoOp").set_body_typed(RemoveNoOp);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
