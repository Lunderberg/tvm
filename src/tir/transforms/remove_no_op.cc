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

#include "../../arith/buffer_touch_pattern.h"
#include "../../arith/const_fold.h"
#include "../../arith/ir_mutator_with_analyzer.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

// Mark the statement of each stage.
class NoOpRemover : public arith::IRMutatorWithAnalyzer {
 public:
  static Stmt Apply(Stmt stmt) {
    arith::Analyzer analyzer;
    analyzer.rewrite_simplify.SetEnabledFeatures(
        arith::RewriteSimplifier::kTransitivelyProveInequalities);
    arith::BufferTouchPattern touch_pattern(stmt);
    NoOpRemover visitor(&analyzer, std::move(touch_pattern));
    return visitor(std::move(stmt));
  }

 private:
  using Parent = IRMutatorWithAnalyzer;
  using Parent::VisitStmt;
  using Parent::VisitStmt_;

  NoOpRemover(arith::Analyzer* analyzer, arith::BufferTouchPattern touch_pattern)
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
    Stmt ret = StmtMutator::VisitSeqStmt_(op, true);
    op = ret.as<SeqStmtNode>();
    ICHECK(op != nullptr);
    bool need_compact = false;
    for (size_t i = 0; i < op->size(); ++i) {
      if (is_no_op(op->seq[i])) need_compact = true;
    }
    if (need_compact) {
      auto n = CopyOnWrite(op);
      size_t top = 0;
      for (size_t i = 0; i < n->seq.size(); ++i) {
        if (!is_no_op(n->seq[i])) {
          n->seq.Set(top++, n->seq[i]);
        }
      }
      if (top == 1) {
        return n->seq[0];
      } else {
        n->seq.resize(top);
        return Stmt(n);
      }
    } else {
      if (op->size() == 1) {
        return op->seq[0];
      } else {
        return ret;
      }
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
      touch_pattern_.RemoveTouches(store);
      return this->VisitStmt(SeqStmt(statements));
    };

    // A write that is later overwritten is a no-op.
    if (touch_pattern_.IsOverwrittenWithoutEffect(store, analyzer_)) {
      return only_side_effects();
    }

    // A write whose destination is known to already contain the
    // values to be written is a no-op.
    if (auto opt = touch_pattern_.KnownValue(store)) {
      PrimExpr known_value = opt.value();
      if (analyzer_->CanProveEqual(store->value, known_value)) {
        return only_side_effects();
      }
    }

    store = Downcast<BufferStore>(Parent::VisitStmt_(store.get()));

    if (const BufferLoadNode* load = store->value.as<BufferLoadNode>()) {
      if (load->buffer->data.same_as(store->buffer->data) &&
          analyzer_->CanProveEqual(load->buffer->elem_offset, store->buffer->elem_offset) &&
          ArrayValueEqual(load->buffer->shape, store->buffer->shape) &&
          ArrayValueEqual(load->buffer->strides, store->buffer->strides) &&
          ArrayValueEqual(load->indices, store->indices)) {
        return Evaluate(0);
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
  arith::BufferTouchPattern touch_pattern_;
};

Stmt RemoveNoOp(Stmt stmt) { return NoOpRemover::Apply(std::move(stmt)); }

Stmt RemoveNoOp(Stmt stmt, arith::Analyzer* analyzer, arith::BufferTouchPattern* touch_pattern) {
  return NoOpRemover::Apply(std::move(stmt));
}

namespace transform {

Pass RemoveNoOp() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    arith::Analyzer analyzer;

    auto* n = f.CopyOnWrite();
    n->body = NoOpRemover::Apply(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.RemoveNoOp", {});
}

TVM_REGISTER_GLOBAL("tir.transform.RemoveNoOp").set_body_typed(RemoveNoOp);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
