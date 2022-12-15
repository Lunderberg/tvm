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
 * \file control_flow_graph.cc
 * \brief Utility to deduce bound of expression
 */

#include "control_flow_graph.h"

#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <unordered_set>

#include "../../arith/conjunctive_normal_form.h"
#include "../../arith/const_fold.h"
#include "../../arith/constraint_extract.h"
#include "../../arith/ir_mutator_with_analyzer.h"
#include "../../arith/ir_visitor_with_analyzer.h"
#include "../../arith/narrow_predicate_expression.h"
#include "../../arith/pattern_match.h"
#include "../../arith/rewrite_simplify.h"
#include "../../arith/unwrap_vector_expr.h"
#include "../../support/debug_timer.h"

namespace tvm {
namespace tir {

using namespace arith;
using tvm::support::DebugTimer;

namespace {

bool HasBufferLoad(PrimExpr expr) {
  struct Visitor : public ExprVisitor {
    void VisitExpr_(const BufferLoadNode* node) override { found_buffer_load = true; }
    bool found_buffer_load{false};
  };

  Visitor visitor;
  visitor(expr);
  return visitor.found_buffer_load;
}

class RemoveLTNodes : public ExprMutator {
 public:
  static PrimExpr Apply(PrimExpr expr) { return RemoveLTNodes()(std::move(expr)); }

 private:
  PrimExpr VisitExpr_(const GTNode* op) final { return VisitExpr(LT(op->b, op->a)); }

  PrimExpr VisitExpr_(const LTNode* op) final {
    LT ret = Downcast<LT>(ExprMutator::VisitExpr_(op));
    if (!IsIndexType(ret->a->dtype) || !IsIndexType(ret->b->dtype)) {
      return std::move(ret);
    }

    PVar<PrimExpr> x, y;
    PVar<IntImm> c1;
    if ((x + c1 < y).Match(ret) || (x < y - c1).Match(ret)) {
      return (x <= y - (c1 + 1)).Eval();
    }
    if ((x - c1 < y).Match(ret) || (x < y + c1).Match(ret)) {
      return (x <= y + (c1 - 1)).Eval();
    }
    if ((x < c1).Match(ret)) {
      return (x <= (c1 - 1)).Eval();
    }
    if ((c1 < x).Match(ret)) {
      return ((c1 + 1) <= x).Eval();
    }

    return std::move(ret);
  }
};

Optional<PrimExpr> SubstituteParamValues(const Array<Var>& param_vars,
                                         const Array<PrimExpr>& param_values,
                                         const PrimExpr& expr) {
  ICHECK_EQ(param_vars.size(), param_values.size())
      << "Expression was defined as having " << param_vars.size() << " parameters, but received "
      << param_values.size() << " arguments.";

  Map<tir::Var, PrimExpr> var_map;
  for (size_t i = 0; i < param_values.size(); i++) {
    var_map.Set(param_vars[i], param_values[i]);
  }

  return Substitute(expr, var_map);
}
}  // namespace

PrimExpr BufferTouch::BeforeLoopIteration() const {
  PrimExpr loop_predicate = Bool(true);
  for (auto it = loop_var_expressions.rbegin(); it != loop_var_expressions.rend(); it++) {
    const Var& loop_var = it->first;
    const PrimExpr& loop_expr = it->second;
    loop_predicate = (loop_var < loop_expr) || ((loop_var == loop_expr) && loop_predicate);
  }
  return loop_predicate;
}

PrimExpr BufferTouch::AtLoopIteration() const {
  PrimExpr loop_predicate = Bool(true);
  for (auto it = loop_var_expressions.rbegin(); it != loop_var_expressions.rend(); it++) {
    const Var& loop_var = it->first;
    const PrimExpr& loop_expr = it->second;
    loop_predicate = (loop_var == loop_expr) && loop_predicate;
  }
  return loop_predicate;
}

PrimExpr BufferTouch::AfterLoopIteration() const {
  PrimExpr loop_predicate = Bool(true);
  for (auto it = loop_var_expressions.rbegin(); it != loop_var_expressions.rend(); it++) {
    const Var& loop_var = it->first;
    const PrimExpr& loop_expr = it->second;
    loop_predicate = (loop_var > loop_expr) || ((loop_var == loop_expr) && loop_predicate);
  }
  return loop_predicate;
}

bool BufferTouch::IsSubsetOf(const BufferTouch& other, Analyzer* analyzer) const {
  if (this->buffer.same_as(other.buffer)) {
    With<ConstraintContext> constraint(analyzer, predicate);

    return analyzer->CanProve(other.predicate);
  } else {
    return false;
  }
}

bool BufferTouch::IsDistinctFrom(const BufferTouch& other, Analyzer* analyzer) const {
  if (this->buffer.same_as(other.buffer)) {
    With<ConstraintContext> constraint(analyzer, predicate);

    return analyzer->CanProve(!other.predicate);
  } else {
    return true;
  }
}

std::ostream& operator<<(std::ostream& os, const BufferTouch& tp) {
  auto touch_type = [&]() {
    if (tp.touch_type == BufferTouch::AccessType::Read) {
      return "read";
    } else if (tp.touch_type == BufferTouch::AccessType::Write) {
      return "write";
    } else if (tp.touch_type == BufferTouch::AccessType::Assume) {
      return "assume";
    } else {
      return "???";
    }
  }();

  os << "BufferTouch(" << tp.buffer->name << ", " << touch_type << ", " << tp.predicate
     << ", value = " << tp.value << ")";
  return os;
}

class BufferConstraintApply : public IRMutatorWithAnalyzer {
 public:
  using Parent = IRMutatorWithAnalyzer;

  BufferConstraintApply(const Map<Buffer, Array<Var>>& axis_var_lookup,
                        const std::vector<BufferTouch>& knowns, Analyzer* analyzer)
      : Parent(analyzer), axis_var_lookup_(axis_var_lookup), knowns_(knowns) {}

  using Parent::VisitExpr_;

  PrimExpr VisitExpr_(const BufferLoadNode* op) override {
    for (const auto& known : knowns_) {
      if (!op->buffer.same_as(known.buffer)) {
        continue;
      }

      Optional<Var> lane_var = NullOpt;
      IntImm num_lanes;

      Array<PrimExpr> indices = op->indices.Map([&](const auto& index) {
        if (index.dtype().lanes() == 1) {
          return index;
        } else {
          ICHECK(!lane_var) << "Multiple indices found with non-scalar values";
          lane_var = Var("lane", index.dtype().element_of());
          num_lanes = IntImm(index.dtype().element_of(), index.dtype().lanes());
          return UnwrapVectorExpr(index, lane_var.value());
        }
      });

      auto axis_vars = axis_var_lookup_.at(op->buffer);
      PrimExpr predicate = SubstituteParamValues(axis_vars, indices, known.predicate).value();

      std::optional<With<ConstraintContext>> context;
      if (lane_var.defined()) {
        Var lanes = lane_var.value();
        PrimExpr known = (IntImm(lanes.dtype(), 0) <= lanes) && (lanes < num_lanes);
        context.emplace(analyzer_, known);
      }

      if (analyzer_->CanProve(predicate)) {
        return SubstituteParamValues(axis_vars, op->indices, known.value).value();
      }
    }

    return GetRef<PrimExpr>(op);
  }

 private:
  const Map<Buffer, Array<Var>>& axis_var_lookup_;
  const std::vector<BufferTouch>& knowns_;
};

/*! \brief Extract the control-flow graph
 *
 * Walk through a statement, populating the control-flow graph.
 */
class ControlFlowGraphBuilder final : public IRVisitorWithAnalyzer {
 public:
  static void Build(ControlFlowGraph* out, const Stmt& stmt) {
    ControlFlowGraphBuilder extractor(out);
    extractor.AppendControlBlock();
    extractor(stmt);
  }

 private:
  ControlFlowGraphBuilder(ControlFlowGraph* out) : out_(out) {}

  using Parent = IRVisitorWithAnalyzer;
  using Parent::VisitExpr_;
  using Parent::VisitStmt_;

  void VisitStmt(const Stmt& stmt) override {
    // Update the lookup table to determine which control-flow block
    // contains the start of the specified statement.  This is used
    // later to determine which set of known values should be used to
    // simplify a statement.
    out_->control_flow_lookup_[stmt.get()] = CurrentControlBlock();
    Stmt prev_stmt = current_stmt_;
    current_stmt_ = stmt;
    Parent::VisitStmt(stmt);
    current_stmt_ = prev_stmt;
  }

  void VisitStmt_(const EvaluateNode* op) override {
    if (auto* call = op->value.as<CallNode>()) {
      if (call->op.same_as(builtin::assume())) {
        Assume(call->args[0], true);
        return;
      }
    }

    Parent::VisitStmt_(op);
  }

  void Assume(PrimExpr assumption, bool from_assume_statement) {
    for (const auto& expr : ExtractConstraints(assumption, false)) {
      AssumeConstraintComponent(expr, from_assume_statement);
    }
  }

  void AssumeConstraintComponent(PrimExpr assumption, bool from_assume_statement) {
    PrimExpr additional_predicate = Bool(true);

    std::vector<PrimExpr> buffer_exprs;
    for (const auto& expr : ExtractComponents(assumption)) {
      auto side_effect = tir::SideEffect(expr);
      if (side_effect <= tir::CallEffectKind::kPure) {
        // Pulling out portions of the assumption that do not depend
        // on a buffer value allows the following two forms to be
        // treated identically.
        //
        // Option 1: if i < 3: T.assume(buf[i] == value)
        // Option 2: T.assume(i>=3 or buf[i] == value)
        additional_predicate = additional_predicate && logical_not(expr);
      } else if (side_effect == tir::CallEffectKind::kReadState) {
        buffer_exprs.push_back(expr);
      } else {
        LOG(FATAL) << "Assumption must be pure or read-only, but contained expression " << expr
                   << " with side-effect \'" << side_effect << "\'";
      }
    }

    if (buffer_exprs.empty()) {
      out_->non_buffer_assumptions_.push_back(!CurrentScopePredicate() || assumption);
      return;
    }

    CHECK_EQ(buffer_exprs.size(), 1) << "T.assume must contain only a single buffer expression";

    auto* as_equal_node = buffer_exprs[0].as<tir::EQNode>();
    CHECK(as_equal_node || !from_assume_statement)
        << "T.assume buffer constraint must be of the form 'buffer[indices] == "
           "value', but received "
        << assumption;
    if (!as_equal_node) {
      // This assumption is an inequality on a data-dependent
      // conditional.  Not an error for this to occur, but also not
      // something that is currently supported.
      return;
    }

    tir::BufferLoad load;
    PrimExpr value;
    if (auto* as_load = as_equal_node->a.as<tir::BufferLoadNode>()) {
      load = GetRef<tir::BufferLoad>(as_load);
      value = as_equal_node->b;
    } else if (auto* as_load = as_equal_node->b.as<tir::BufferLoadNode>()) {
      load = GetRef<tir::BufferLoad>(as_load);
      value = as_equal_node->a;
    } else if (!from_assume_statement) {
      return;
    } else {
      LOG(FATAL) << "T.assume buffer constraint must be of the form 'buffer[indices] == value'";
    }

    auto has_side_effect = tir::SideEffect(value) > tir::CallEffectKind::kPure;
    CHECK(!has_side_effect || !from_assume_statement)
        << "Buffer value in constraint must be pure expression, but was " << value;
    if (has_side_effect) {
      return;
    }

    {
      InternalConstraintContext context(this, additional_predicate);
      VisitAccess(load, BufferTouch::AccessType::Assume, value);
    }
    // Appending a control block ensures that all control blocks have
    // at most one statement that changes the known buffer contents.
    auto prev_block = CurrentControlBlock();
    auto new_block = AppendControlBlock();
    MarkControlFlow(prev_block, new_block);
  }

  void VisitExpr_(const LetNode* op) override {
    std::optional<BindLetVar> binding;
    if (UsesLoopVar(op->value)) {
      binding.emplace(this, op->var, op->value);
    }
    Parent::VisitExpr_(op);
  }

  void VisitStmt_(const LetStmtNode* op) override {
    std::optional<BindLetVar> binding;
    if (UsesLoopVar(op->value)) {
      binding.emplace(this, op->var, op->value);
    }
    Parent::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode* op) override {
    Parent::VisitExpr_(op);
    BufferLoad load = GetRef<BufferLoad>(op);
    VisitAccess(load, BufferTouch::AccessType::Read, load);
  }

  void VisitStmt_(const BufferStoreNode* op) override {
    Parent::VisitStmt_(op);
    VisitAccess(GetRef<BufferStore>(op), BufferTouch::AccessType::Write, op->value);
    // Appending a control block ensures that all control blocks have
    // at most one statement that changes the buffer contents.
    auto prev_block = CurrentControlBlock();
    auto new_block = AppendControlBlock();
    MarkControlFlow(prev_block, new_block);
  }

  void VisitStmt_(const ForNode* op) override {
    out_->iterator_ranges_.Set(op->loop_var, Range::FromMinExtent(op->min, op->extent));

    auto before_loop = CurrentControlBlock();
    size_t loop_start = -1;

    std::vector<const ForNode*> loopnest = {op};

    {
      std::vector<std::unique_ptr<BindActiveLoopVar>> bindings;

      Stmt body = op->body;

      while (auto* loop_body = body.as<ForNode>()) {
        loopnest.push_back(loop_body);
        out_->iterator_ranges_.Set(loop_body->loop_var,
                                   Range::FromMinExtent(loop_body->min, loop_body->extent));
        body = loop_body->body;
      }

      for (const ForNode* loop : loopnest) {
        bindings.push_back(
            std::make_unique<BindActiveLoopVar>(this, loop->loop_var, loop->min, loop->extent));
        analyzer_.Bind(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
      }

      loop_start = AppendControlBlock();

      VisitStmt(body);

      while (bindings.size()) {
        bindings.pop_back();
      }
    }

    PrimExpr first_loopnest_iter = Bool(true);
    PrimExpr last_loopnest_iter = Bool(true);
    Map<Var, PrimExpr> first_iter_remap;
    Map<Var, PrimExpr> last_iter_remap;

    for (const ForNode* loop : loopnest) {
      PrimExpr max_iterator_value = analyzer_.Simplify(loop->min + loop->extent - 1);
      first_loopnest_iter = first_loopnest_iter && (loop->loop_var == loop->min);
      last_loopnest_iter = last_loopnest_iter && (loop->loop_var == max_iterator_value);
      first_iter_remap.Set(loop->loop_var, loop->min);
      last_iter_remap.Set(loop->loop_var, max_iterator_value);
    }

    PrimExpr fused_iter = 0;
    for (const ForNode* loop : loopnest) {
      fused_iter = fused_iter * loop->extent + (loop->loop_var - loop->min);
    }
    Map<Var, PrimExpr> forward_iter_remap;
    Map<Var, PrimExpr> backward_iter_remap;
    PrimExpr stride = 1;
    for (size_t i = loopnest.size(); i > 0; i--) {
      const ForNode* loop = loopnest[i - 1];
      PrimExpr forward = floordiv(fused_iter + 1, stride);
      PrimExpr backward = floordiv(fused_iter - 1, stride);
      // Floormod is only required for inner loopnests.  If the
      // outermost loop would wrap around, the entire loopnest is
      // about to be exited.
      if (i > 1) {
        forward = floormod(forward, loop->extent);
        backward = floormod(backward, loop->extent);
      }
      forward = forward + loop->min;
      backward = backward + loop->min;
      forward_iter_remap.Set(loop->loop_var, analyzer_.Simplify(forward));
      backward_iter_remap.Set(loop->loop_var, analyzer_.Simplify(backward));

      stride = stride * loop->extent;
    }

    auto loop_end = CurrentControlBlock();
    auto after_loop = AppendControlBlock();
    {
      auto [forward, backward] = MarkControlFlow(before_loop, loop_start);
      backward.post_condition = first_loopnest_iter;
      forward.var_remap = first_iter_remap;
    }
    {
      auto [forward, backward] = MarkControlFlow(loop_end, after_loop);
      backward.var_remap = last_iter_remap;
      forward.post_condition = last_loopnest_iter;
    }
    {
      auto [forward, backward] = MarkControlFlow(loop_end, loop_start);
      backward.var_remap = backward_iter_remap;
      forward.var_remap = forward_iter_remap;
      backward.post_condition = RemoveLTNodes::Apply(analyzer_.Simplify(!first_loopnest_iter));
      forward.post_condition = RemoveLTNodes::Apply(analyzer_.Simplify(!last_loopnest_iter));
    }
  }

  void VisitStmt_(const IfThenElseNode* op) override {
    this->VisitExpr(op->condition);

    PrimExpr real_condition =
        RemoveLTNodes::Apply(analyzer_.Simplify(ExtractRealCondition(op->condition)));

    auto before_branching = CurrentControlBlock();

    auto branch_start = AppendControlBlock();
    MarkControlFlow(before_branching, branch_start);

    {
      InternalConstraintContext context(this, real_condition);
      auto then_start = AppendControlBlock();
      if (context.assume.defined()) {
        Assume(context.assume.value(), false);
      }
      auto [forward, backward] = MarkControlFlow(branch_start, then_start);
      backward.post_condition = real_condition;
      forward.post_condition = real_condition;
      this->VisitStmt(op->then_case);
    }
    auto then_end = CurrentControlBlock();

    auto negation = RemoveLTNodes::Apply(analyzer_.rewrite_simplify(!real_condition));
    {
      InternalConstraintContext context(this, negation);
      auto else_start = AppendControlBlock();
      if (context.assume.defined()) {
        Assume(context.assume.value(), false);
      }
      auto [forward, backward] = MarkControlFlow(branch_start, else_start);
      backward.post_condition = negation;
      forward.post_condition = negation;

      if (op->else_case.defined()) {
        this->VisitStmt(op->else_case.value());
      }
    }

    auto else_end = CurrentControlBlock();
    auto after_branching = AppendControlBlock();

    if (HasBufferLoad(real_condition)) {
      // The buffer value may have changed during the body of the
      // condition, so we can't provide it as a post-condition.
      MarkControlFlow(then_end, after_branching);
      MarkControlFlow(else_end, after_branching);
    } else {
      {
        auto [forward, backward] = MarkControlFlow(then_end, after_branching);
        backward.post_condition = real_condition;
        forward.post_condition = real_condition;
      }
      {
        auto [forward, backward] = MarkControlFlow(else_end, after_branching);
        backward.post_condition = negation;
        forward.post_condition = negation;
      }
    }
  }

  /*! \brief Internal utility, returns true if the expression depends
   *  on a loop iterator
   */
  bool UsesLoopVar(const PrimExpr& expr) {
    return UsesVar(expr, [&](const VarNode* expr_var) {
      return loop_dependent_vars_.find(expr_var) != loop_dependent_vars_.end();
    });
  }

  /*! \brief Record the interaction with the buffer.
   *
   * \param node The TIR node that accesses the buffer.  Should be
   * either a BufferLoad or BufferStore node.
   *
   * \param touch_type The type of buffer access being performed.  A
   * BufferStore should always use AccessType::Write.  A BufferLoad
   * may use either AccessType::Read or AccessType::Assume, depending
   * on whether the BufferLoad occurs within `builtin::assume`.
   *
   * \param known_value_expr The value in the buffer following the access.
   */
  template <typename BufferAccess>
  void VisitAccess(const BufferAccess& node, BufferTouch::AccessType touch_type,
                   PrimExpr known_value_expr) {
    auto& current_block = out_->control_flow_.back();
    std::vector<BufferTouch> buffer_touches = current_block.MakeBufferTouch(
        out_, node->buffer, node->indices, touch_type, known_value_expr);

    for (const auto& buffer_touch : buffer_touches) {
      current_block.touch_points.push_back(buffer_touch);
    }
  }

  /*! \brief Return a predicate for having reached the current
   *  control-flow block
   *
   * For example, while inside an IfThenElse, will return the
   * IfThenElse's condition.
   */
  PrimExpr CurrentScopePredicate() const {
    PrimExpr predicate = Bool(true);
    for (const auto& condition : conditions_) {
      predicate = predicate && condition;
    }
    return predicate;
  }

  /* \brief Add a new control block, returning its index */
  size_t AppendControlBlock() {
    size_t index = out_->control_flow_.size();
    auto& block = out_->control_flow_.emplace_back();
    block.active_loop_iterators = active_loop_iterators_;
    block.let_bindings_using_loop = let_bindings_using_loop_;
    block.scope_predicate = CurrentScopePredicate();
    return index;
  }

  /* \brief The index of the current control block */
  size_t CurrentControlBlock() { return out_->control_flow_.size() - 1; }

  /* \brief Mark a possible control from one block to another
   *
   * \param from_block The block from which control leaves
   *
   * \param to_block The block to which control enters
   *
   * \param var_remap Variable replacements that should be made in
   * known expression while traversing this edge.  For example,
   * replacing `i` with `i-1` when entering the next loop iteration,
   * or replacing `i` with `n-1` when concluding a loop.
   */
  std::pair<ControlFlowGraph::ControlFlowEdge&, ControlFlowGraph::ControlFlowEdge&> MarkControlFlow(
      size_t from_block, size_t to_block) {
    ICHECK_LE(from_block, out_->control_flow_.size());
    ICHECK_LE(to_block, out_->control_flow_.size());

    auto& forward = out_->control_flow_[from_block].successors.emplace_back(
        ControlFlowGraph::ControlFlowEdge{to_block, {}, NullOpt});
    auto& backward = out_->control_flow_[to_block].predecessors.emplace_back(
        ControlFlowGraph::ControlFlowEdge{from_block, {}, NullOpt});
    return {forward, backward};
  }

  // Internal utility, context manager for entering/leaving a scoped constraint
  struct InternalConstraintContext {
    InternalConstraintContext(ControlFlowGraphBuilder* self, PrimExpr constraint)
        : self(self), analyzer_context(&self->analyzer_, constraint) {
      old_num_constraints = self->conditions_.size();

      auto side_effect = tir::SideEffect(constraint);
      if (side_effect <= tir::CallEffectKind::kPure) {
        self->conditions_.push_back(constraint);
      } else if (side_effect <= tir::CallEffectKind::kReadState) {
        assume = constraint;
      }

      new_num_constraints = self->conditions_.size();
    }
    ~InternalConstraintContext() {
      ICHECK_EQ(self->conditions_.size(), new_num_constraints)
          << "Internal error: Each condition should only be popped once.";
      self->conditions_.erase(self->conditions_.begin() + old_num_constraints,
                              self->conditions_.end());
    }

    ControlFlowGraphBuilder* self{nullptr};
    With<ConstraintContext> analyzer_context;
    size_t old_num_constraints{0};
    size_t new_num_constraints{0};
    Optional<PrimExpr> assume{NullOpt};

    // Disable default-generated copy/move assignment and constructors
    InternalConstraintContext(const InternalConstraintContext&) = delete;
    InternalConstraintContext& operator=(const InternalConstraintContext&) = delete;
    InternalConstraintContext(InternalConstraintContext&&) = delete;
    InternalConstraintContext& operator=(InternalConstraintContext&&) = delete;
  };

  // Internal utility, context manager for tracking a loop
  struct BindActiveLoopVar {
    BindActiveLoopVar(ControlFlowGraphBuilder* self, Var var, PrimExpr loop_min,
                      PrimExpr loop_extent)
        : self(self), var(var) {
      PrimExpr loop_max = loop_min + (loop_extent - 1);
      auto loop_range = Range::FromMinExtent(loop_min, loop_extent);
      self->active_loop_iterators_.push_back({var, loop_min, loop_max, loop_range});
      self->loop_dependent_vars_.insert(var.get());
    }
    ~BindActiveLoopVar() { self->active_loop_iterators_.pop_back(); }

    ControlFlowGraphBuilder* self;
    Var var;

    // Disable default-generated copy/move assignment and constructors
    BindActiveLoopVar(const BindActiveLoopVar&) = delete;
    BindActiveLoopVar& operator=(const BindActiveLoopVar&) = delete;
    BindActiveLoopVar(BindActiveLoopVar&&) = delete;
    BindActiveLoopVar& operator=(BindActiveLoopVar&&) = delete;
  };

  // Internal utility, context manager for tracking a variable binding
  struct BindLetVar {
    BindLetVar(ControlFlowGraphBuilder* self, Var var, PrimExpr value) : self(self), var(var) {
      self->let_bindings_using_loop_.Set(var, value);
      self->loop_dependent_vars_.insert(var.get());
    }
    ~BindLetVar() {
      self->loop_dependent_vars_.erase(var.get());
      self->let_bindings_using_loop_.erase(var);
    }
    ControlFlowGraphBuilder* self;
    Var var;

    // Disable default-generated copy/move assignment and constructors
    BindLetVar(const BindLetVar&) = delete;
    BindLetVar& operator=(const BindLetVar&) = delete;
    BindLetVar(BindLetVar&&) = delete;
    BindLetVar& operator=(BindLetVar&&) = delete;
  };

  struct LoopEntry {
    Var loop_var;
    PrimExpr loop_min;
    PrimExpr loop_max;
    Range loop_range;
  };

  // Track in order to know which Vars to write in terms of the buffer
  // indices and substitute out of the predicate.
  std::vector<ControlFlowGraph::ControlFlowBlock::LoopEntry> active_loop_iterators_;

  // Track all loop iterators, along with values derived from loop iterators.
  std::unordered_set<const VarNode*> loop_dependent_vars_;

  // Any let binding that depends, directly or indirectly, on a loop
  // binding.  When making a predicate in terms of the buffer indices,
  // these need to be substituted out.
  // std::unordered_map<const VarNode*, PrimExpr> let_bindings_using_loop_;
  Map<Var, PrimExpr> let_bindings_using_loop_;

  // Track in order to know what conditions limit the buffer access
  std::vector<PrimExpr> conditions_;

  // Track in order to know what statement initiated the buffer access
  Stmt current_stmt_;

  // Output data structure
  ControlFlowGraph* out_;
};

class IfThenElseCollector : public ExprMutator {
 public:
  /* \brief Hoist and collect tir::if_then_else conditionals
   *
   * \param expr The expression to be analyzed, which may contain
   * tir::if_then_else.
   *
   * \param analyzer The analyzer to be used for simplification
   *
   * \returns A map from boolean expression to the value of the
   * expression where that boolean expression is true.  The values of
   * this Map may not contain tir::if_then_else.
   */
  static Map<PrimExpr, PrimExpr> Collect(PrimExpr expr, Analyzer* analyzer) {
    IfThenElseCollector collector(analyzer);
    expr = collector(expr);

    if (collector.info_.has_value()) {
      auto& info = *collector.info_;
      Map<PrimExpr, PrimExpr> output;
      for (const auto [cond, value] : info.replacements) {
        output.Set(cond, analyzer->Simplify(Substitute(expr, {{info.dummy_var, value}})));
      }
      return output;
    } else {
      return {{Bool(true), expr}};
    }
  }

 private:
  IfThenElseCollector(Analyzer* analyzer) : analyzer_(analyzer) {}

  PrimExpr VisitExpr_(const CallNode* op) {
    if (op->op.same_as(tir::builtin::if_then_else())) {
      PrimExpr cond = analyzer_->Simplify(op->args[0]);

      const PrimExpr& then_expr = op->args[1];
      const PrimExpr& else_expr = op->args[2];

      if (is_one(cond)) {
        return VisitExpr(then_expr);
      } else if (is_zero(cond)) {
        return VisitExpr(else_expr);
      } else {
        PrimExpr negation = analyzer_->Simplify(!cond);

        Map<PrimExpr, PrimExpr> replacements;
        for (auto [region_predicate, then_expr_region] : Collect(then_expr, analyzer_)) {
          region_predicate = analyzer_->Simplify(cond && region_predicate);
          if (!is_zero(region_predicate)) {
            replacements.Set(region_predicate, then_expr_region);
          }
        }
        for (auto [region_predicate, else_expr_region] : Collect(else_expr, analyzer_)) {
          region_predicate = analyzer_->Simplify(cond && region_predicate);
          if (!is_zero(region_predicate)) {
            replacements.Set(region_predicate, else_expr_region);
          }
        }

        Var dummy_var("dummy", op->dtype);

        info_ = {dummy_var, replacements};
        return std::move(dummy_var);
      }
    }

    return ExprMutator::VisitExpr_(op);
  }

  struct IfThenElseInfo {
    // The variable representing the tir::if_then_else CallNode to be replaced
    Var dummy_var;

    // A map defining the possible values of the tir::if_then_else
    // CallNode.  Each key is a boolean expression, defining a region
    // where that boolean is true.  Each value is the value taken by
    // the expression when the boolean expression is true.
    Map<PrimExpr, PrimExpr> replacements;
  };

  Analyzer* analyzer_;
  std::optional<IfThenElseInfo> info_{std::nullopt};
};

std::pair<std::vector<BufferTouch>, Map<Var, Range>>
ControlFlowGraph::ControlFlowBlock::MakeBufferTouch(const tir::Buffer& buf,
                                                    Array<Var> index_variables,
                                                    Array<PrimExpr> indices,
                                                    BufferTouch::AccessType touch_type,
                                                    PrimExpr known_value_expr) const {
  auto timer = DebugTimer("MakeBufferTouch")
                   .always_print_short_timers()
                   .on_start([&](auto& out) {
                     out << "buf = " << buf->name << indices << " type = "
                         << (touch_type == BufferTouch::AccessType::Read     ? "Read"
                             : touch_type == BufferTouch::AccessType::Write  ? "Write"
                             : touch_type == BufferTouch::AccessType::Assume ? "Assume"
                                                                             : "???");
                   })
                   .start();

  const auto& current_block = *this;

  Analyzer local_analyzer;
  local_analyzer.rewrite_simplify.SetEnabledExtensions(arith::RewriteSimplifier::Extension(
      arith::RewriteSimplifier::kTransitivelyProveInequalities |
      arith::RewriteSimplifier::kConvertBooleanToAndOfOrs |
      arith::RewriteSimplifier::kApplyConstraintsToBooleanBranches));

  Optional<Var> lane_var = NullOpt;
  IntImm num_lanes;

  Array<PrimExpr> index_expressions = indices.Map([&](const auto& index) {
    if (index.dtype().lanes() == 1) {
      return index;
    } else {
      ICHECK(!lane_var) << "Multiple indices found with non-scalar values";
      lane_var = Var("lane", index.dtype().element_of());
      num_lanes = IntImm(index.dtype().element_of(), index.dtype().lanes());
      return UnwrapVectorExpr(index, lane_var.value());
    }
  });

  Array<Var> loop_vars;

  Map<Var, Range> loop_ranges;
  for (const auto& loop_entry : current_block.active_loop_iterators) {
    loop_vars.push_back(loop_entry.loop_var);
    loop_ranges.Set(loop_entry.loop_var, loop_entry.loop_range);
  }

  // If the indices contain multiple lanes, treat the lane variable
  // as an additional loop iterator to be solved for and substituted
  // out.
  if (lane_var) {
    loop_vars.push_back(lane_var.value());
    loop_ranges.Set(lane_var.value(), Range::FromMinExtent(0, num_lanes));
  }

  IntConstraintsTransform transform = [&]() {
    ICHECK_EQ(index_variables.size(), index_expressions.size());

    Array<PrimExpr> relations;

    for (size_t i = 0; i < index_expressions.size(); i++) {
      PrimExpr expr = index_expressions[i];
      Var var = index_variables[i];

      expr = Substitute(expr, current_block.let_bindings_using_loop);
      relations.push_back(var == expr);
    }

    IntConstraints system(loop_vars, loop_ranges, relations);
    return arith::SolveLinearEquations(system);
  }();

  Map<Var, PrimExpr> loop_var_to_axis_var = transform->src_to_dst;
  Map<Var, Range> free_params = transform->dst->ranges;
  PrimExpr transform_predicate =
      std::accumulate(transform->dst->relations.begin(), transform->dst->relations.end(),
                      PrimExpr(Bool(true)), [](PrimExpr a, PrimExpr b) { return a && b; });

  // transform_predicate = SimplifyAsAndOfOrs(transform_predicate, &local_analyzer);
  transform_predicate = local_analyzer.Simplify(transform_predicate);

  auto find_removable_params = [&]() -> Map<Var, PrimExpr> {
    Map<Var, PrimExpr> removable_params;

    // The arith::SolveLinearEquations is more general than the
    // utilities in iter_affine_map.h, but can introduce free
    // parameters that could later be determined with the known
    // constraints.  This step removes all such free parameters.
    for (const auto& expr : ExtractConstraints(transform_predicate)) {
      if (auto* as_equal = expr.as<EQNode>()) {
        auto check_expr = [&](const PrimExpr& a, const PrimExpr& b) {
          auto* var_ptr = a.as<VarNode>();
          if (!var_ptr) {
            return;
          }

          Var var = GetRef<Var>(var_ptr);
          if (free_params.count(var) == 0) {
            return;
          }

          bool uses_free_param =
              UsesVar(b, [&](const VarNode* v) { return free_params.count(GetRef<Var>(v)) > 0; });
          if (uses_free_param) {
            return;
          }
          removable_params.Set(var, b);
        };
        check_expr(as_equal->a, as_equal->b);
        check_expr(as_equal->b, as_equal->a);
      }
    }

    // In addition, the arith::SolveLinearEquation can introduce
    // free parameters with an extent of one.  Filtering them out here
    // avoids needing to track them through later simplifications.
    for (const auto [var, range] : free_params) {
      if (is_one(range->extent)) {
        removable_params.Set(var, range->min);
      }
    }

    return removable_params;
  };
  for (auto removable_params = find_removable_params(); removable_params.size() > 0;
       removable_params = find_removable_params()) {
    auto update = [&](PrimExpr expr) {
      auto timer = DebugTimer("Removing extent=1 parameters")
                       .on_finish([before = expr, &expr](auto& out) {
                         out << "Changed from " << before << " to " << expr;
                       })
                       .start();
      expr = local_analyzer.Simplify(Substitute(expr, removable_params));
      return expr;
    };

    Map<Var, PrimExpr> new_map;
    for (const auto [loop_var, expr] : loop_var_to_axis_var) {
      static_cast<void>(expr);  // gcc 7.x bug, https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81767
      new_map.Set(loop_var, update(expr));
    }
    loop_var_to_axis_var = new_map;

    transform_predicate = update(transform_predicate);

    for (const auto [var, expr] : removable_params) {
      static_cast<void>(expr);  // gcc 7.x bug, https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81767
      free_params.erase(var);
    }
  }

  {
    auto timer = DebugTimer("Binding free parameters")
                     .on_start([&](auto& out) { out << free_params; })
                     .always_print_short_timers()
                     .start();
    local_analyzer.Bind(free_params);
  }

  // Collect the current loop variables, along with an expression for
  // the loop variables in terms of the buffer axis variables.  This
  // is used during forward/backward propagation to generate predicate
  // tracking whether a loop iteration has been reached.
  std::vector<std::pair<Var, PrimExpr>> loop_var_expressions;
  for (const auto& entry : current_block.active_loop_iterators) {
    auto expr_it = loop_var_to_axis_var.find(entry.loop_var);
    ICHECK(expr_it != loop_var_to_axis_var.end());
    loop_var_expressions.push_back({entry.loop_var, (*expr_it).second});
  }

  // Normalization function, applied to both the predicate and the
  // known value.  Converts from an expression in terms of loop
  // iterators to an expression in terms of buffer indices.
  auto normalize_expr = [&](PrimExpr expr) -> PrimExpr {
    expr = Substitute(expr, current_block.let_bindings_using_loop);

    if (lane_var) {
      expr = UnwrapVectorExpr(expr, lane_var.value());
    }
    expr = Substitute(expr, loop_var_to_axis_var);

    return expr;
  };

  // The full predicate is composed of the values required to reach
  // the scope of the BufferStore or builtin::assume(), any bounds
  // implied by solving for the axis variables, and any additional
  // statements resulting from unpacking the expression contained in
  // builtin::assume().
  PrimExpr scope_predicate = normalize_expr(current_block.scope_predicate);
  transform_predicate = normalize_expr(transform_predicate);

  // Map<PrimExpr, PrimExpr> regions = IfThenElseCollector::Collect(known_value_expr,
  // &local_analyzer);
  Map<PrimExpr, PrimExpr> regions = [&]() {
    Map<PrimExpr, PrimExpr> regions;
    auto timer = DebugTimer("Collecting IfThenElse regions")
                     .always_print_short_timers()
                     .on_finish([&regions](auto& out) { out << "regions = " << regions; })
                     .start();
    regions = IfThenElseCollector::Collect(known_value_expr, &local_analyzer);
    return regions;
  }();

  std::vector<BufferTouch> buffer_touches;
  for (auto [region_predicate, region_known_value] : regions) {
    // std::cout << "\t"
    //           << "In region " << region_predicate << ", known value: " << region_known_value
    //           << std::endl;

    ICHECK_EQ(IfThenElseCollector::Collect(region_known_value, &local_analyzer).size(), 1);

    {
      auto timer = DebugTimer("Simplifying region-specific known value expr")
                       .always_print_short_timers()
                       .on_finish([before = region_known_value, &region_known_value](auto& out) {
                         out << "Changed from " << before << " to " << region_known_value;
                       })
                       .start();
      region_known_value = local_analyzer.Simplify(normalize_expr(region_known_value));
    }

    // Deliberately use an analyzer without scope-based information,
    // to avoid simplifying `scope_predicate` to True.
    PrimExpr predicate_expr = transform_predicate && scope_predicate && region_predicate;
    predicate_expr = RemoveLTNodes::Apply(predicate_expr);
    {
      auto timer = DebugTimer("Simplifying predicate expr")

                       .always_print_short_timers()
                       .on_finish([before = predicate_expr, &predicate_expr](auto& out) {
                         out << "Changed from " << before << " to " << predicate_expr;
                       })
                       .start();
      predicate_expr = local_analyzer.Simplify(predicate_expr);
    }

    if (!is_zero(predicate_expr)) {
      BufferTouch buffer_touch = {buf, predicate_expr, region_known_value, loop_var_expressions,
                                  touch_type};
      buffer_touches.push_back(buffer_touch);
    }
  }

  return {buffer_touches, free_params};
}

std::vector<BufferTouch> ControlFlowGraph::ControlFlowBlock::MakeBufferTouch(
    ControlFlowGraph* graph, const tir::Buffer& buf, const Array<PrimExpr>& indices,
    BufferTouch::AccessType touch_type, PrimExpr known_value_expr) const {
  ICHECK(graph);
  auto [buffer_touches, free_params] = MakeBufferTouch(buf, graph->GetIndexVariables(buf, indices),
                                                       indices, touch_type, known_value_expr);
  for (const auto& pair : free_params) {
    graph->free_predicate_parameters_.Set(pair.first, pair.second);
  }
  return buffer_touches;
}

ControlFlowGraph::ControlFlowGraph(const tir::Stmt& stmt, size_t max_revisits)
    : max_revisits_(max_revisits) {
  {
    auto timer = DebugTimer("Collecting info")
                     .on_finish([this](auto& out) {
                       out << "collected " << control_flow_.size() << " blocks";
                     })
                     .start();
    ControlFlowGraphBuilder::Build(this, stmt);
  }
  std::cout << "-------------- Blocks ------------------" << std::endl;
  std::cout << *this << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  // std::terminate();

  // {
  //   auto timer = DebugTimer("Forward propagation").start();
  //   ForwardPropagateKnownValues();
  // }
  {
    auto timer = DebugTimer("Backward propagation").start();
    BackwardPropagateUnusedValues();
  }
  {
    auto timer = DebugTimer("Forward propagation").start();
    ForwardPropagateKnownValues();
  }
}

void ControlFlowGraph::RemoveStore(const tir::BufferStore& store) {
  size_t context_index = [&]() {
    auto it = control_flow_lookup_.find(store.get());
    ICHECK(it != control_flow_lookup_.end())
        << "BufferStore did not occur in the Stmt provided to BufferTouchPattern's constructor";
    return it->second;
  }();

  auto& touch_points = control_flow_[context_index].touch_points;

  touch_points.erase(std::remove_if(touch_points.begin(), touch_points.end(),
                                    [](const BufferTouch& touch) {
                                      return touch.touch_type == BufferTouch::AccessType::Write;
                                    }),
                     touch_points.end());
  ForwardPropagateKnownValues(context_index);
  BackwardPropagateUnusedValues(context_index);
}

std::ostream& operator<<(std::ostream& os, const ControlFlowGraph::ControlFlowEdge& edge) {
  os << edge.index;
  if (edge.var_remap.size()) {
    os << " with remap " << edge.var_remap;
  }
  if (edge.post_condition) {
    os << " with postcondition " << edge.post_condition;
  }

  return os;
}

std::ostream& operator<<(std::ostream& os, const ControlFlowGraph::ControlFlowBlock& block) {
  if (block.predecessors.size()) {
    os << "\t"
       << "Predecessors: [";
    for (size_t i = 0; i < block.predecessors.size(); i++) {
      if (i) {
        os << ", ";
      }
      os << block.predecessors[i];
    }
    os << "]\n";
  }

  // os << "\t" << "Active loop iterators: [";
  // for (size_t i = 0; i < block.active_loop_iterators.size(); i++) {
  //   if (i) {
  //     os << ", ";
  //   }
  //   os << block.active_loop_iterators[i].loop_var;
  // }
  // os << "]\n";

  if (!block.known_at_block_start.empty()) {
    os << "\t"
       << "Before block knowns: " << block.known_at_block_start << "\n";
  }

  if (!block.unused_at_block_start.empty()) {
    os << "\t"
       << "Before block unused: " << block.unused_at_block_start << "\n";
  }

  for (size_t i = 0; i < block.touch_points.size(); i++) {
    os << "\t"
       << "Touch[" << i << "] = " << block.touch_points[i] << "\n";
  }
  if (!block.known_at_block_end.empty()) {
    os << "\t"
       << "After block: " << block.known_at_block_end << "\n";
  }

  if (!block.unused_at_block_end.empty()) {
    os << "\t"
       << "After block unused: " << block.unused_at_block_end << "\n";
  }

  if (block.successors.size()) {
    os << "\t"
       << "Successors: [";
    for (size_t i = 0; i < block.successors.size(); i++) {
      if (i) {
        os << ", ";
      }
      os << block.successors[i];
    }
    os << "]";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const ControlFlowGraph& pattern) {
  os << "Touch pattern contains " << pattern.control_flow_.size() << " control blocks."
     << (pattern.control_flow_.size() ? "\n" : "");
  for (size_t i = 0; i < pattern.control_flow_.size(); i++) {
    os << "ControlBlock[" << i << "]: \n" << pattern.control_flow_[i] << "\n";
  }

  return os;
}

bool BufferTouch::IsEquivalentTo(const BufferTouch& other, Analyzer* analyzer) const {
  // Constraints must apply to the same buffer to be equivalent
  if (!buffer.same_as(other.buffer) || touch_type != other.touch_type) {
    return false;
  }

  ExprDeepEqual deep_equal;

  auto implies = [&](const PrimExpr& a, const PrimExpr& b) -> bool {
    With<ConstraintContext> context(analyzer, a);
    return analyzer->CanProve(b);
  };

  // Predicates must be equivalent expressions, or must both be undefined
  bool equivalent_predicates =
      deep_equal(predicate, other.predicate) ||
      (implies(predicate, other.predicate) && implies(other.predicate, predicate));
  if (!equivalent_predicates) {
    return false;
  }

  // The known value must be equal
  if (!deep_equal(value, other.value) && !analyzer->CanProveEqual(value, other.value)) {
    return false;
  }

  return true;
}

std::ostream& operator<<(std::ostream& os, const BufferState& state) {
  for (size_t i = 0; i < state.constraints_.size(); i++) {
    os << "constraints[" << i << "] = " << state.constraints_[i]
       << (i + 1 == state.constraints_.size() ? "" : "\n");
  }
  return os;
}

PrimExpr BufferState::SubstituteKnownBufferValues(
    PrimExpr expr, const Map<tir::Buffer, Array<tir::Var>>& axis_var_lookup,
    Analyzer* analyzer) const {
  BufferConstraintApply mutator(axis_var_lookup, constraints_, analyzer);
  return mutator(std::move(expr));
}

void BufferState::AddCondition(const PrimExpr& condition) {
  for (auto& constraint : constraints_) {
    constraint.predicate = constraint.predicate && condition;
  }
}

void BufferState::Substitute(const Map<Var, PrimExpr>& var_remap, Analyzer* analyzer) {
  if (var_remap.size()) {
    for (auto& prior : constraints_) {
      PrimExpr updated = tvm::tir::Substitute(prior.predicate, var_remap);
      if (!updated.same_as(prior.predicate)) {
        // prior.predicate = SimplifyAsAndOfOrs(updated, analyzer);
        auto timer = DebugTimer("Post-substitution simplification")
                         .always_print_short_timers()
                         .on_start([&](auto& out) { out << "before = " << updated; })
                         .on_finish([&](auto& out) { out << "after = " << prior.predicate; })
                         .start();
        prior.predicate = analyzer->Simplify(updated);
      }
    }
  }
  RemoveEmptyConstraints();
}

void BufferState::Simplify(Analyzer* analyzer) {
  for (auto& constraint : constraints_) {
    // constraint.predicate = SimplifyAsAndOfOrs(constraint.predicate, analyzer);
    constraint.predicate = analyzer->Simplify(constraint.predicate);
  }
  RemoveEmptyConstraints();
}

void BufferState::RemoveEmptyConstraints() {
  constraints_.erase(
      std::remove_if(constraints_.begin(), constraints_.end(),
                     [](const BufferTouch& touch) { return is_zero(touch.predicate); }),
      constraints_.end());
}

void BufferState::Union(const BufferState& b, Analyzer* analyzer) {
  for (const auto& b_constraint : b.constraints_) {
    bool used = false;
    for (auto& a_constraint : constraints_) {
      if (a_constraint.buffer.same_as(b_constraint.buffer) &&
          analyzer->CanProveEqual(a_constraint.value, b_constraint.value)) {
        // a_constraint.predicate =
        //     SimplifyAsAndOfOrs(a_constraint.predicate || b_constraint.predicate, analyzer);
        a_constraint.predicate =
            analyzer->Simplify(a_constraint.predicate || b_constraint.predicate);
        used = true;
        break;
      }
    }
    if (!used) {
      constraints_.push_back(b_constraint);
    }
  }
}

void BufferState::Intersection(const BufferState& b, Analyzer* analyzer) {
  // For a constraint to be in the output, it must be present in both
  // inputs.

  std::vector<BufferTouch> new_constraints;
  for (const auto& ai : constraints_) {
    for (const auto& bi : b.constraints_) {
      if (ai.buffer.same_as(bi.buffer)) {
        // PrimExpr predicate = SimplifyAsAndOfOrs(ai.predicate && bi.predicate, analyzer);
        PrimExpr predicate = analyzer->Simplify(ai.predicate && bi.predicate);
        if (!is_zero(predicate)) {
          With<ConstraintContext> context(analyzer, predicate);
          PrimExpr known_value_a = ai.value;
          PrimExpr known_value_b = bi.value;

          bool is_consistent = analyzer->CanProveEqual(known_value_a, known_value_b);
          if (is_consistent) {
            new_constraints.push_back({ai.buffer, predicate, known_value_a});
          }
        }
      }
    }
  }

  constraints_ = std::move(new_constraints);
}

class BufferRegionCollector : public ExprVisitor {
 public:
  struct Region {
    PrimExpr region_predicate;
    std::unordered_map<const BufferLoadNode*, Optional<PrimExpr>> known_values;
  };

  static std::vector<Region> Collect(const Map<Buffer, Array<Var>>& axis_var_lookup,
                                     const std::vector<BufferTouch>& knowns,
                                     const std::vector<Optional<PrimExpr>>& exprs,
                                     Analyzer* analyzer, bool verbose = false) {
    BufferRegionCollector collector(axis_var_lookup, knowns, analyzer, verbose);
    for (const auto& expr : exprs) {
      if (expr) {
        collector(expr.value());
      }
    }

    return collector.regions_;
  }

 private:
  using Parent = ExprVisitor;

  BufferRegionCollector(const Map<Buffer, Array<Var>>& axis_var_lookup,
                        const std::vector<BufferTouch>& knowns, Analyzer* analyzer, bool verbose)
      : analyzer_(analyzer), axis_var_lookup_(axis_var_lookup), knowns_(knowns), verbose_(verbose) {
    regions_.push_back(Region{Bool(true), {}});
  }

  using Parent::VisitExpr_;

  void VisitExpr_(const BufferLoadNode* op) override {
    auto timer = DebugTimer("Extracting regions")
                     .on_start([&](auto& out) { out << " for " << GetRef<PrimExpr>(op); })
                     .always_print_short_timers(verbose_)
                     .start();

    // Helper struct for the known values of this BufferLoad
    struct Known {
      PrimExpr predicate;
      Optional<PrimExpr> value;
    };

    std::vector<Known> new_regions;

    PrimExpr unknown_region = Bool(true);

    for (const BufferTouch& constraint : knowns_) {
      if (!op->buffer.same_as(constraint.buffer)) {
        // This is a different buffer, so continue searching.
        continue;
      }

      auto timer = DebugTimer("Collecting regions for touch")
                       .on_start([&](auto& out) { out << " at " << constraint.predicate; })
                       .always_print_short_timers(verbose_)
                       .start();

      auto axis_vars = axis_var_lookup_.at(op->buffer);
      PrimExpr touch_predicate =
          SubstituteParamValues(axis_vars, op->indices, constraint.predicate).value();
      {
        auto timer = DebugTimer("Simplifying touch predicate")
                         .on_start([&](auto& out) { out << touch_predicate; })
                         .always_print_short_timers(verbose_)
                         .start();
        // touch_predicate = SimplifyAsAndOfOrs(touch_predicate, analyzer_);
        touch_predicate = analyzer_->Simplify(touch_predicate);
      }

      if (!is_zero(touch_predicate)) {
        Optional<PrimExpr> known_value =
            SubstituteParamValues(axis_vars, op->indices, constraint.value);
        new_regions.push_back(Known{touch_predicate, known_value});

        unknown_region = unknown_region && !touch_predicate;
        {
          auto timer = DebugTimer("Simplifying unknown region")
                           .on_start([&](auto& out) { out << unknown_region; })
                           .always_print_short_timers(verbose_)
                           .start();
          // unknown_region = SimplifyAsAndOfOrs(unknown_region, analyzer_);
          unknown_region = analyzer_->Simplify(unknown_region);
        }
      }
    }

    if (new_regions.size()) {
      Analyzer local_analyzer;

      auto timer = DebugTimer("Updaing new regions").always_print_short_timers(verbose_).start();

      if (!is_zero(unknown_region)) {
        new_regions.insert(new_regions.begin(), Known{unknown_region, NullOpt});
      }

      std::vector<Region> updated_regions;
      for (const auto& prev_region : regions_) {
        for (const auto& new_region : new_regions) {
          // PrimExpr intersection =
          //     SimplifyAsAndOfOrs(prev_region.region_predicate && new_region.predicate,
          //     analyzer_);
          PrimExpr intersection =
              analyzer_->Simplify(prev_region.region_predicate && new_region.predicate);

          if (!is_zero(intersection)) {
            Region merged{intersection, prev_region.known_values};
            merged.known_values[op] = new_region.value;
            updated_regions.push_back(std::move(merged));
          }
        }
      }
      regions_ = updated_regions;
    }
  }

  Analyzer* analyzer_;
  std::vector<Region> regions_;
  const Map<Buffer, Array<Var>>& axis_var_lookup_;
  const std::vector<BufferTouch>& knowns_;
  bool verbose_{false};
};

class BufferRegionValueReplacer : public IRMutatorWithAnalyzer {
 public:
  static PrimExpr Apply(
      const std::unordered_map<const BufferLoadNode*, Optional<PrimExpr>>& known_values,
      PrimExpr expr, Analyzer* analyzer) {
    BufferRegionValueReplacer mutator(known_values, analyzer);

    PrimExpr result = [&]() {
      auto timer = DebugTimer("BufferRegionValueReplacer::Apply, running mutator").start();
      return mutator(expr);
    }();
    // Simplification must occur after the substitution, as known
    // values may provide enable simplifications.  Also, cannot track
    // whether a BufferLoad was
    // {
    // auto timer = DebugTimer("BufferRegionValueReplacer::Apply, simplifying after replacement")
    //                  .on_finish([&result, orig = result](auto& out) {
    //                    out << "Simplified from " << orig << " to " << result;
    //                  })
    //                  .start();
    //   result = analyzer->Simplify(result);
    // };
    return result;
  }

 private:
  using Parent = IRMutatorWithAnalyzer;

  BufferRegionValueReplacer(
      const std::unordered_map<const BufferLoadNode*, Optional<PrimExpr>>& known_values,
      Analyzer* analyzer)
      : Parent(analyzer), known_values_(known_values) {}

  using Parent::VisitExpr_;

  PrimExpr VisitExpr_(const BufferLoadNode* op) override {
    auto it = known_values_.find(op);
    if (it != known_values_.end() && it->second) {
      return it->second.value();
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  const std::unordered_map<const BufferLoadNode*, Optional<PrimExpr>>& known_values_;
};

void BufferState::ApplyTouches(const Map<Buffer, Array<Var>>& axis_var_lookup,
                               const std::vector<BufferTouch>& touch_points, Analyzer* analyzer,
                               bool verbose) {
  std::vector<BufferTouch> new_knowns;
  Map<Buffer, PrimExpr> keep_prior_known_at;

  std::optional<DebugTimer::Impl> timer;
  DebugTimer("Looping over touch points").always_print_short_timers(verbose).start(timer);

  for (auto& touch : touch_points) {
    if (touch.touch_type == BufferTouch::AccessType::Read) {
      continue;
    }

    PrimExpr known_value = touch.value;

    PrimExpr predicate = touch.predicate && touch.AfterLoopIteration();

    {
      auto timer = DebugTimer("Simplifying touch/loop predicate")
                       .always_print_short_timers(verbose)
                       .on_finish([before = predicate, &predicate](auto& out) {
                         out << "simplified from " << before << " to " << predicate;
                       })
                       .start();
      predicate = analyzer->Simplify(predicate);
    }

    auto regions = [&]() {
      auto timer = DebugTimer("Collecting regions").always_print_short_timers(verbose).start();
      return BufferRegionCollector::Collect(axis_var_lookup, constraints_, {predicate, touch.value},
                                            analyzer, verbose);
    }();

    for (const auto& region : regions) {
      auto timer =
          DebugTimer("Examining region")
              .on_start([&](auto& out) { out << "region predicate = " << region.region_predicate; })
              .always_print_short_timers(verbose)
              .start();

      PrimExpr updated_predicate = region.region_predicate && predicate;

      {
        auto timer = DebugTimer("Updating region-specific predicate")
                         .always_print_short_timers(verbose)
                         .on_start([&updated_predicate, &region](auto& out) {
                           out << "in region = " << region.region_predicate
                               << ", before = " << updated_predicate;
                         })
                         .on_finish([&updated_predicate, &region](auto& out) {
                           out << "in region = " << region.region_predicate
                               << ", after = " << updated_predicate;
                         })
                         .start();
        updated_predicate =
            BufferRegionValueReplacer::Apply(region.known_values, updated_predicate, analyzer);
      };

      {
        auto timer = DebugTimer("Simplifying region-specific predicate")
                         .always_print_short_timers(verbose)
                         .on_start([&updated_predicate](auto& out) {
                           out << "before = " << updated_predicate;
                         })
                         .on_finish([&updated_predicate](auto& out) {
                           out << "after = " << updated_predicate;
                         })
                         .start();
        // updated_predicate = SimplifyAsAndOfOrs(updated_predicate, analyzer);
        updated_predicate = analyzer->Simplify(updated_predicate);
      };

      PrimExpr updated_value = known_value;
      {
        auto timer =
            DebugTimer("Updating value")
                .always_print_short_timers(verbose)
                .on_start([&updated_value](auto& out) { out << "before = " << updated_value; })
                .on_finish([&updated_value](auto& out) { out << "after = " << updated_value; })
                .start();
        updated_value =
            BufferRegionValueReplacer::Apply(region.known_values, updated_value, analyzer);
      };

      if (!is_zero(updated_predicate)) {
        auto timer =
            DebugTimer("Finalizing changes from region").always_print_short_timers(verbose).start();
        if (auto it = keep_prior_known_at.find(touch.buffer); it != keep_prior_known_at.end()) {
          keep_prior_known_at.Set(touch.buffer, (*it).second && !updated_predicate);
        } else {
          keep_prior_known_at.Set(touch.buffer, !updated_predicate);
        }

        if (!HasBufferLoad(updated_value)) {
          BufferTouch new_constraint{touch.buffer, updated_predicate, updated_value};
          new_knowns.push_back(new_constraint);
        }
      }
    }
  }

  DebugTimer("Reducing extent of prior knowns").always_print_short_timers(verbose).start(timer);

  if (keep_prior_known_at.size()) {
    for (auto& constraint : constraints_) {
      if (auto it = keep_prior_known_at.find(constraint.buffer); it != keep_prior_known_at.end()) {
        // constraint.predicate = SimplifyAsAndOfOrs(constraint.predicate && (*it).second,
        // analyzer);
        constraint.predicate = analyzer->Simplify(constraint.predicate && (*it).second);
      }
    }
  }

  DebugTimer("Extending extent of new knowns").always_print_short_timers(verbose).start(timer);

  if (new_knowns.size()) {
    std::vector<bool> used(new_knowns.size(), false);

    for (auto& constraint : constraints_) {
      PrimExpr expand_known_at = Bool(false);

      PrimExpr prev_value = constraint.value;

      for (size_t i = 0; i < new_knowns.size(); i++) {
        if (new_knowns[i].buffer.same_as(constraint.buffer)) {
          Optional<PrimExpr> overwritten_with = new_knowns[i].value;
          if (overwritten_with && analyzer->CanProveEqual(prev_value, overwritten_with.value())) {
            // expand_known_at =
            //     SimplifyAsAndOfOrs(expand_known_at || new_knowns[i].predicate, analyzer);
            expand_known_at = analyzer->Simplify(expand_known_at || new_knowns[i].predicate);
            used[i] = true;
          }
        }
      }

      if (!is_zero(expand_known_at)) {
        // constraint.predicate =
        //     SimplifyAsAndOfOrs(constraint.predicate || expand_known_at, analyzer);
        constraint.predicate = analyzer->Simplify(constraint.predicate || expand_known_at);
      }
    }

    for (size_t i = 0; i < new_knowns.size(); i++) {
      if (!used[i]) {
        constraints_.push_back(new_knowns[i]);
      }
    }
  }

  DebugTimer("Removing empty constraints").always_print_short_timers(verbose).start(timer);

  constraints_.erase(
      std::remove_if(constraints_.begin(), constraints_.end(),
                     [&](const auto& constraint) { return is_zero(constraint.predicate); }),
      constraints_.end());
}

void BufferState::BackpropUnusedIndices(const Map<Buffer, Array<Var>>& axis_var_lookup,
                                        const std::vector<BufferTouch>& touch_points,
                                        Analyzer* analyzer) {
  std::optional<DebugTimer::Impl> timer;
  DebugTimer("Looping over touch points").start(timer);

  std::vector<BufferTouch> new_knowns;
  Map<Buffer, PrimExpr> keep_prior_known_at;

  Map<Buffer, PrimExpr> regions_written;
  Map<Buffer, PrimExpr> regions_read;
  for (auto it = touch_points.rbegin(); it != touch_points.rend(); it++) {
    const auto& touch = *it;

    Map<Buffer, PrimExpr>* to_update{nullptr};
    if (touch.touch_type == BufferTouch::AccessType::Write) {
      to_update = &regions_written;

    } else if (touch.touch_type == BufferTouch::AccessType::Read) {
      to_update = &regions_read;
    } else {
      continue;
    }

    PrimExpr prev = to_update->Get(touch.buffer).value_or(Bool(false));
    PrimExpr new_predicate = touch.predicate && touch.BeforeLoopIteration();
    to_update->Set(touch.buffer, prev || new_predicate);
  }

  auto update_map = [&](auto& map) {
    Map<Buffer, PrimExpr> new_map;
    for (auto [buffer, predicate] : map) {
      // new_map.Set(buffer, SimplifyAsAndOfOrs(predicate, analyzer));
      auto timer = DebugTimer("Simplifying backprop region predicate")
                       .on_start([predicate = PrimExpr(predicate)](auto& out) { out << predicate; })
                       .on_finish([&predicate](std::ostream& out) { out << predicate; })
                       .start();
      predicate = analyzer->Simplify(predicate);
      PrimExpr as_or_of_ands = NormalizeBooleanOperators(!analyzer->Simplify(!predicate));
      new_map.Set(buffer, predicate);
    }
    map = std::move(new_map);
  };
  DebugTimer("Update regions_written map").start(timer);
  update_map(regions_written);

  DebugTimer("Update regions_read map").start(timer);
  update_map(regions_read);

  DebugTimer("Widening predicate for buffers that already have unused indices").start(timer);

  // If buffer is already in used, widen the predicate
  for (auto& prev_unused : constraints_) {
    if (auto opt_predicate = regions_written.Get(prev_unused.buffer)) {
      PrimExpr new_predicate = prev_unused.predicate || opt_predicate.value();
      // prev_unused.predicate = SimplifyAsAndOfOrs(new_predicate, analyzer);
      prev_unused.predicate = analyzer->Simplify(new_predicate);
      regions_written.erase(prev_unused.buffer);
    }
  }

  DebugTimer("Inserting predicate for buffers that already have unused indices").start(timer);

  // Otherwise, add new "touch" to represent the unused values
  for (auto [buffer, predicate] : regions_written) {
    constraints_.push_back(
        BufferTouch{buffer, predicate, tir::Call(buffer->dtype, builtin::undef(), {})});
  }

  DebugTimer("Inserting predicate for buffers that already have unused indices").start(timer);

  // If buffer is read out, narrow the predicate
  for (auto& prev_unused : constraints_) {
    if (auto opt_pred = regions_read.Get(prev_unused.buffer)) {
      PrimExpr predicate = opt_pred.value();
      // prev_unused.predicate = SimplifyAsAndOfOrs(prev_unused.predicate && !predicate, analyzer);
      prev_unused.predicate = analyzer->Simplify(prev_unused.predicate && !predicate);
    }
  }

  DebugTimer("Clean-up empty constraints").start(timer);

  // Clean-up and remove any empty constraints
  constraints_.erase(
      std::remove_if(constraints_.begin(), constraints_.end(),
                     [](const auto& constraint) { return is_zero(constraint.predicate); }),
      constraints_.end());
}

void BufferState::RemoveFreeParameters(const Map<Var, Range>& free_predicate_parameters,
                                       Analyzer* analyzer) {
  for (auto& known : constraints_) {
    known.predicate = NarrowPredicateExpression(known.predicate, free_predicate_parameters);
    // known.predicate = SimplifyAsAndOfOrs(known.predicate, analyzer);
    known.predicate = analyzer->Simplify(known.predicate);
  }
}

bool BufferState::IsEquivalentTo(const BufferState& other, Analyzer* analyzer) const {
  if (constraints_.size() != other.constraints_.size()) {
    return false;
  }

  for (size_t i = 0; i < constraints_.size(); i++) {
    if (!constraints_[i].IsEquivalentTo(other.constraints_[i], analyzer)) {
      return false;
    }
  }

  return true;
}

Optional<Array<Var>> ControlFlowGraph::GetIndexVariables(const Buffer& buf) const {
  if (auto it = axis_var_lookup_.find(buf); it != axis_var_lookup_.end()) {
    return (*it).second;
  } else {
    return NullOpt;
  }
}

Array<Var> ControlFlowGraph::GetIndexVariables(const Buffer& buf, const Array<PrimExpr>& indices) {
  if (auto it = axis_var_lookup_.find(buf); it != axis_var_lookup_.end()) {
    return (*it).second;
  }

  Array<Var> vars;
  for (size_t i = 0; i < indices.size(); i++) {
    std::stringstream ss;
    ss << buf->name << "_axis_" << i;
    vars.push_back(Var(ss.str(), indices[i].dtype().element_of()));
  }

  axis_var_lookup_.Set(buf, vars);
  return vars;
}

void ControlFlowGraph::ForwardPropagateKnownValues(std::optional<size_t> flow_from) {
  // Values to visit when searching.  Using a std::set to
  // preferentially visit nodes near the start of the control flow.
  std::set<size_t> to_visit;

  if (flow_from.has_value()) {
    to_visit.insert(flow_from.value());
  } else {
    // Initiatize the locations to search from, propagating values
    // forward from all locations that have a known value.
    for (size_t i = 0; i < control_flow_.size(); i++) {
      bool has_known_value = false;
      for (const auto& touch : control_flow_[i].touch_points) {
        if (!HasBufferLoad(touch.value)) {
          has_known_value = true;
          break;
        }
      }

      if (has_known_value) {
        to_visit.insert(i);
      }
    }
  }

  // Map from a block's index
  std::unordered_map<size_t, size_t> visit_count_lookup;

  Analyzer analyzer;
  analyzer.rewrite_simplify.SetEnabledExtensions(arith::RewriteSimplifier::Extension(
      arith::RewriteSimplifier::kTransitivelyProveInequalities |
      arith::RewriteSimplifier::kConvertBooleanToAndOfOrs |
      arith::RewriteSimplifier::kApplyConstraintsToBooleanBranches));
  // analyzer.rewrite_simplify.SetEnabledExtensions(
  //     arith::RewriteSimplifier::Extension(arith::RewriteSimplifier::kTransitivelyProveInequalities
  //     |
  //                                         arith::RewriteSimplifier::kConvertBooleanToAndOfOrs));

  analyzer.Bind(iterator_ranges_);
  analyzer.Bind(free_predicate_parameters_);

  while (to_visit.size()) {
    size_t visiting = *to_visit.begin();
    to_visit.erase(visiting);

    auto timer = DebugTimer([&]() {
      std::stringstream ss;
      ss << "Visiting " << visiting;
      return ss.str();
    }());

    size_t num_previous_visits = visit_count_lookup[visiting]++;

    ControlFlowBlock& block = control_flow_[visiting];

    // std::cout << "----------- Control block " << visiting << " -------------" << std::endl;

    // std::cout << "Visiting control block " << visiting << " / " << control_flow_.size() << ", "
    //           << num_previous_visits << " previous visits" << std::endl;

    // Step 1: Collect known values provided from each precedessor
    block.known_at_block_start = [&]() -> BufferState {
      if (num_previous_visits >= max_revisits_) {
        return BufferState();
      }
      auto timer = DebugTimer("Collecting priors knowns")
                       .subcategory([&](auto& out) { out << " for " << visiting; })
                       .start();

      // Validate internal constraint.  This should be true by
      // construction, as ControlFlowGraphBuilder only builds graphs
      // that have two or fewer predecessors.
      ICHECK_LE(block.predecessors.size(), 2)
          << "InternalError: Each block should have at most two predecessors.  "
          << "Graph constructed in ControlFlowGraphBuilder did not satisfy this constraint.";

      std::vector<BufferState> states;
      for (const auto& pred : block.predecessors) {
        const auto& pred_block = control_flow_[pred.index];
        BufferState state = pred_block.known_at_block_end;
        if (pred.var_remap.size()) {
          state.Substitute(pred.var_remap, &analyzer);
          // std::cout << "Pred " << pred.index << " has " << pred.var_remap.size()
          //           << " replacements, remaps to: " << state << std::endl;
        }
        states.push_back(state);
      }

      if (std::all_of(block.predecessors.begin(), block.predecessors.end(),
                      [&](const auto& pred) { return visit_count_lookup[pred.index] == 0; })) {
        // Predecessors, if any, are unvisited.
        // std::cout << "Neither predecessor has been visited, starting with a blank slate"
        //           << std::endl;
        return {};
      } else if (block.predecessors.size() == 1) {
        // std::cout << "Block " << block.predecessors[0].index
        //           << " is the only predecessor, using its known values" << std::endl;
        // Block has only a single predecessor
        return states[0];
      }

      const auto& pred_a = block.predecessors[0];
      const auto& pred_b = block.predecessors[1];

      auto& priors_a = states[0];
      auto& priors_b = states[1];

      // During the first visit of a block, predecessor blocks may be
      // unvisited, even though we preferentially visit earlier blocks
      // first.  (e.g. During the first visit of the start of a For
      // loop, the end of the For loop has not yet been visited.)  If
      // this is the case, assume the best-case scenario that all
      // knowns are consistent, and rely on a later visit to
      // resolve/remove any conflicts.
      if (visit_count_lookup[pred_a.index] == 0) {
        // std::cout << "Haven't visited block " << pred_a.index
        //           << " before, assuming that it produces the same knowns as block " <<
        //           pred_b.index
        //           << ": "
        //           << "\n"
        //           << priors_b << std::endl;
        return priors_b;
      } else if (visit_count_lookup[pred_b.index] == 0) {
        // std::cout << "Haven't visited block " << pred_b.index
        //           << " before, assuming that it produces the same knowns as block " <<
        //           pred_a.index
        //           << ": "
        //           << "\n"
        //           << priors_a << std::endl;
        return priors_a;
      }

      if (pred_a.post_condition && pred_b.post_condition) {
        // The predicate can identify which predecessor block applies
        // (e.g. i==0 for the first loop iteration, i>0 for remaining
        // loop iterations).  Therefore, we can use all buffer
        // constraints, conditional on having come from the
        // predecessor that provides it.
        priors_a.AddCondition(pred_a.post_condition.value());
        // std::cout << "First pred with conditions: " << priors_a << std::endl;
        priors_b.AddCondition(pred_b.post_condition.value());
        // std::cout << "Second pred with conditions: " << priors_b << std::endl;
        {
          auto timer = DebugTimer("Taking/simplifying union of priors")
                           .subcategory([&](auto& out) { out << "from  " << visiting; })
                           .start();
          priors_a.Union(priors_b, &analyzer);
        }

        // std::cout << "From merging 2 predecessors, known at block start: "
        //           << "\n"
        //           << priors_a << std::endl;

        return priors_a;
      } else {
        // We don't know which predecessor applies.  Therefore, the
        // only buffer constraints that can be used are those that
        // appear in both predecessors.
        priors_a.Intersection(priors_b, &analyzer);
        // std::cout
        //     << "From merging 2 predecessors with a data-dependent branch, known at block
        //     start: "
        //     << "\n"
        //     << priors_a << std::endl;
        return priors_a;
      }
    }();

    // Step 2: Collect knowns provided as a result of executing this block
    auto post_state = [&]() {
      if (num_previous_visits >= max_revisits_) {
        return BufferState();
      }

      auto post_state = block.known_at_block_start;
      {
        auto timer = DebugTimer("Applying touches")
                         .subcategory([&](auto& out) { out << "for block " << visiting; })
                         .start();
        post_state.ApplyTouches(axis_var_lookup_, block.touch_points, &analyzer, visiting == 4);
      }
      // std::cout << "Afer applying " << block.touch_points.size() << " touches: " << post_state
      //           << std::endl;
      {
        auto timer = DebugTimer("Removing free parameters")
                         .subcategory([&](auto& out) { out << "for block " << visiting; })
                         .start();
        post_state.RemoveFreeParameters(free_predicate_parameters_, &analyzer);
      }
      // std::cout << "Afer removing free parameters: " << post_state << std::endl;
      return post_state;
    }();

    // if (block.touch_points.size()) {
    //   std::cout << "After applying touches: "
    //             << "\n"
    //             << post_state << std::endl;
    // } else {
    //   std::cout << "No touches in this block." << std::endl;
    // }

    // Step 3: If any changes are made to the post knowns since the
    // previous time we visited this block, mark the successor block
    // as needing to be visited.
    if (num_previous_visits == 0 ||
        !post_state.IsEquivalentTo(block.known_at_block_end, &analyzer)) {
      block.known_at_block_end = std::move(post_state);
      for (const auto& successor : block.successors) {
        // std::cout << "Queueing successor " << successor.index << " to be visited" << std::endl;
        to_visit.insert(successor.index);
      }
    }

    // std::cout << "---------------------------------------" << std::endl;
  }
}

void ControlFlowGraph::BackwardPropagateUnusedValues(std::optional<size_t> flow_from) {
  // Values to visit when searching.  Using a std::set to
  // preferentially visit nodes near the end of the control flow.
  std::set<size_t> to_visit;

  if (flow_from.has_value()) {
    to_visit.insert(flow_from.value());
  } else {
    // Initiatize the locations to search from, propagating values
    // backward from anywhere that performs a write.
    for (size_t i = 0; i < control_flow_.size(); i++) {
      const auto& touch_points = control_flow_[i].touch_points;
      bool performs_write = std::any_of(
          touch_points.begin(), touch_points.end(),
          [](const auto& touch) { return touch.touch_type == BufferTouch::AccessType::Write; });
      if (performs_write) {
        to_visit.insert(i);
      }
    }
  }

  // Map from a block's index
  std::unordered_map<size_t, size_t> visit_count_lookup;

  Analyzer analyzer;
  analyzer.rewrite_simplify.SetEnabledExtensions(arith::RewriteSimplifier::Extension(
      arith::RewriteSimplifier::kTransitivelyProveInequalities |
      arith::RewriteSimplifier::kConvertBooleanToAndOfOrs |
      arith::RewriteSimplifier::kApplyConstraintsToBooleanBranches));

  analyzer.Bind(iterator_ranges_);
  analyzer.Bind(free_predicate_parameters_);

  while (to_visit.size()) {
    size_t visiting = *to_visit.rbegin();
    to_visit.erase(visiting);

    auto timer = DebugTimer("Visiting block")
                     .subcategory([&](auto& out) { out << "block " << visiting; })
                     .start();

    size_t num_previous_visits = visit_count_lookup[visiting]++;

    ControlFlowBlock& block = control_flow_[visiting];

    // Step 1: Collect known unused indices provided by each successor
    block.unused_at_block_end = [&]() -> BufferState {
      if (num_previous_visits >= max_revisits_) {
        return BufferState();
      }

      auto timer = DebugTimer("Collecting post unused")
                       .subcategory([&](auto& out) { out << "for block " << visiting; })
                       .start();

      ICHECK_LE(block.successors.size(), 2)
          << "Each block should have at most two successors, but block " << visiting
          << " breaks this requirement";

      std::vector<BufferState> states;
      for (const auto& successor : block.successors) {
        const auto& successor_block = control_flow_[successor.index];
        BufferState state = successor_block.unused_at_block_start;
        auto timer = DebugTimer("Substituting from successor")
                         .subcategory([&](auto& out) {
                           out << "for block " << visiting << " from successor " << successor.index;
                         })
                         .start();
        state.Substitute(successor.var_remap, &analyzer);
        states.push_back(state);
      }

      if (std::all_of(block.successors.begin(), block.successors.end(), [&](const auto& successor) {
            return visit_count_lookup[successor.index] == 0;
          })) {
        // Successors, if any, are unvisited.
        return {};
      } else if (block.successors.size() == 1) {
        // Block has only a single successor
        return states[0];
      }

      const auto& successor_a = block.successors[0];
      const auto& successor_b = block.successors[1];

      auto& post_a = states[0];
      auto& post_b = states[1];

      // During the first visit of a block, successor blocks may be
      // unvisited, even though we preferentially visit later blocks
      // first.  (e.g. During the first visit of the end of a For
      // loop, the start of the For loop has not yet been visited.)
      // If this is the case, assume the best-case scenario that all
      // knowns are consistent, and rely on a later visit to
      // resolve/remove any conflicts.
      if (visit_count_lookup[successor_a.index] == 0) {
        return post_b;
      } else if (visit_count_lookup[successor_b.index] == 0) {
        return post_a;
      }

      if (successor_a.post_condition && successor_b.post_condition) {
        // The predicate can identify which successor block applies
        // (e.g. i==n-1 for the last loop iteration, i<n-1 for earlier
        // loop iterations).  Therefore, we can use all buffer
        // constraints, conditional on having come from the
        // successor that provides it.
        {
          auto timer = DebugTimer("Adding condition to post_a")
                           .subcategory([&](auto& out) { out << "for block " << visiting; })
                           .always_print_short_timers()
                           .start();
          post_a.AddCondition(successor_a.post_condition.value());
          post_a.Simplify(&analyzer);
        }
        {
          auto timer = DebugTimer("Adding condition to post_b")
                           .subcategory([&](auto& out) { out << "for block " << visiting; })
                           .always_print_short_timers()
                           .start();
          post_b.AddCondition(successor_b.post_condition.value());
          post_b.Simplify(&analyzer);
        }
        {
          auto timer = DebugTimer("Taking union and simplifying of post unused from post_a/post_b")
                           .subcategory([&](auto& out) { out << "for block " << visiting; })
                           .always_print_short_timers()
                           .start();
          post_a.Union(post_b, &analyzer);
        }
        return post_a;
      } else {
        // We don't know which successor applies.  Therefore, the
        // only buffer constraints that can be used are those that
        // appear in both successors.
        post_a.Intersection(post_b, &analyzer);
        return post_a;
      }
    }();

    // Step 2: Collect knowns provided as a result of executing this block
    auto unused_at_block_start = [&]() {
      if (num_previous_visits >= max_revisits_) {
        return BufferState();
      }
      auto prior_state = block.unused_at_block_end;
      {
        auto timer = DebugTimer("Backprop touches")
                         .subcategory([&](auto& out) { out << "for block " << visiting; })
                         .start();
        prior_state.BackpropUnusedIndices(axis_var_lookup_, block.touch_points, &analyzer);
      }
      {
        auto timer = DebugTimer("Remove free parameters")
                         .subcategory([&](auto& out) { out << "for block " << visiting; })
                         .start();
        prior_state.RemoveFreeParameters(free_predicate_parameters_, &analyzer);
      }
      return prior_state;
    }();

    // Step 3: If any changes are made to the post knowns since the
    // previous time we visited this block, mark the successor block
    // as needing to be visited.
    if (num_previous_visits == 0 ||
        !unused_at_block_start.IsEquivalentTo(block.unused_at_block_start, &analyzer)) {
      block.unused_at_block_start = std::move(unused_at_block_start);
      for (const auto& pred : block.predecessors) {
        to_visit.insert(pred.index);
      }
    }
  }
}

bool ControlFlowGraph::IsOverwrittenWithoutEffect(const tir::BufferStore& store,
                                                  const Stmt& context) const {
  Optional<Array<Var>> index_variables = GetIndexVariables(store->buffer);
  if (!index_variables) {
    return false;
  }

  auto it = control_flow_lookup_.find(context.get());
  ICHECK(it != control_flow_lookup_.end())
      << "Context " << PrettyPrint(context) << " did not occur within analyzed statement";
  const auto& context_block = control_flow_[it->second];

  auto [store_touches, free_params] = context_block.MakeBufferTouch(
      store->buffer, index_variables.value(), store->indices, BufferTouch::AccessType::Write,
      BufferLoad(store->buffer, store->indices));

  ICHECK_EQ(store_touches.size(), 1) << "Internal Error, MakeBufferTouch should return "
                                     << "a single touch-point for known_value=BufferLoad(...), "
                                     << "because it does not include tir::if_then_else";
  auto store_touch = store_touches[0];

  Analyzer local_analyzer;
  local_analyzer.Bind(free_predicate_parameters_);
  local_analyzer.Bind(iterator_ranges_);
  local_analyzer.Bind(free_params);
  local_analyzer.rewrite_simplify.SetEnabledExtensions(arith::RewriteSimplifier::Extension(
      arith::RewriteSimplifier::kTransitivelyProveInequalities |
      arith::RewriteSimplifier::kConvertBooleanToAndOfOrs |
      arith::RewriteSimplifier::kApplyConstraintsToBooleanBranches));

  PrimExpr predicate = store_touch.predicate && store_touch.AtLoopIteration();

  // predicate = SimplifyAsAndOfOrs(predicate, &local_analyzer);
  predicate = local_analyzer.Simplify(predicate);

  for (const auto& unused : context_block.unused_at_block_end.constraints_) {
    if (store_touch.buffer.same_as(unused.buffer)) {
      // PrimExpr difference = SimplifyAsAndOfOrs(predicate && !unused.predicate,
      // &local_analyzer);
      PrimExpr difference = local_analyzer.Simplify(predicate && !unused.predicate);
      if (is_zero(difference)) {
        return true;
      }
    }
  }
  return false;
}

PrimExpr ControlFlowGraph::SimplifyInContext(PrimExpr expr, const tir::Stmt& context,
                                             Analyzer* analyzer) const {
  size_t context_index = [&]() {
    auto it = control_flow_lookup_.find(context.get());
    ICHECK(it != control_flow_lookup_.end())
        << "Context did not occur in the Stmt provided to BufferTouchPattern's constructor";
    return it->second;
  }();

  const auto& control_flow_block = control_flow_[context_index];

  PrimExpr constraint = Bool(true);
  for (const auto& known : non_buffer_assumptions_) {
    constraint = constraint && known;
  }
  With<ConstraintContext> constraint_context(analyzer, constraint);
  With<ConstraintContext> control_flow_scope(analyzer, control_flow_block.scope_predicate);

  expr = control_flow_block.known_at_block_start.SubstituteKnownBufferValues(
      std::move(expr), axis_var_lookup_, analyzer);

  expr = analyzer->Simplify(std::move(expr));
  return expr;
}

}  // namespace tir
}  // namespace tvm
