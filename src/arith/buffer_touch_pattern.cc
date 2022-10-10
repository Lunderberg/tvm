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
 * \file buffer_touch_pattern.cc
 * \brief Utility to deduce bound of expression
 */

#include "buffer_touch_pattern.h"

#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <numeric>
#include <queue>
#include <sstream>
#include <vector>

#include "conjunctive_disjunctive_form.h"
#include "constraint_extract.h"
#include "ir_mutator_with_analyzer.h"
#include "ir_visitor_with_analyzer.h"
#include "narrow_expression_to_true.h"
#include "unwrap_vector_expr.h"

namespace tvm {
namespace arith {

using namespace tir;

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
}  // namespace

ParametrizedExpression::ParametrizedExpression(Array<Var> parameter_vars,
                                               Optional<PrimExpr> expression)
    : parameter_vars_(parameter_vars), expression_(expression) {}

namespace {
template <typename T>
Optional<PrimExpr> SubstituteParamValues(const Array<Var>& param_vars, const Array<T>& param_values,
                                         const Optional<PrimExpr>& expr) {
  ICHECK_EQ(param_vars.size(), param_values.size())
      << "Expression was defined as having " << param_vars.size() << " parameters, but received "
      << param_values.size() << " arguments.";

  if (!expr) {
    return NullOpt;
  }

  Map<tir::Var, PrimExpr> var_map;
  for (size_t i = 0; i < param_values.size(); i++) {
    var_map.Set(param_vars[i], param_values[i]);
  }

  return Substitute(expr.value(), var_map);
}
}  // namespace

Optional<PrimExpr> ParametrizedExpression::operator()(const Array<Var>& args) const {
  return SubstituteParamValues(parameter_vars_, args, expression_);
}

Optional<PrimExpr> ParametrizedExpression::operator()(const Array<PrimExpr>& args) const {
  return SubstituteParamValues(parameter_vars_, args, expression_);
}

bool ParametrizedExpression::IsConstant() const {
  if (!IsDefined()) {
    return true;
  }

  std::unordered_set<const VarNode*> vars;
  for (const auto& var : parameter_vars_) {
    vars.insert(var.get());
  }
  return !UsesVar(expression_.value(), [&](const VarNode* var) { return vars.count(var); });
}

std::ostream& operator<<(std::ostream& os, const ParametrizedExpression& expr) {
  if (!expr.IsConstant()) {
    os << expr.parameter_vars_ << " => ";
  }

  os << expr.expression_;
  return os;
}

Predicate::Predicate(Array<tir::Var> parameter_vars, Optional<PrimExpr> expression)
    : ParametrizedExpression(parameter_vars, expression) {
  ICHECK(!expression || expression.value().dtype().is_bool())
      << "Predicate should be boolean expression, but received " << expression;
}

bool Predicate::IsSubsetOf(const Predicate& other, Analyzer* analyzer) const {
  ICHECK_EQ(parameter_vars_.size(), other.parameter_vars_.size())
      << "Predicates must be over the same number of parameters to be comparable";

  if (!IsDefined() || !other.IsDefined()) {
    return false;
  }

  PrimExpr other_predicate = other(parameter_vars_).value();

  With<ConstraintContext> constraint(analyzer, expression_.value());

  return analyzer->CanProve(other_predicate);
}

bool Predicate::IsDistinctFrom(const Predicate& other, Analyzer* analyzer) const {
  ICHECK_EQ(parameter_vars_.size(), other.parameter_vars_.size())
      << "Predicates must be over the same number of parameters to be comparable";

  if (!IsDefined() || !other.IsDefined()) {
    return false;
  }

  PrimExpr other_predicate = other(parameter_vars_).value();

  With<ConstraintContext> constraint(analyzer, expression_.value());

  return analyzer->CanProve(logical_not(other_predicate));
}

Predicate Predicate::Difference(const Predicate& other, Analyzer* analyzer) const {
  ICHECK_EQ(parameter_vars_.size(), other.parameter_vars_.size())
      << "Predicates must be over the same number of parameters to be comparable";

  if (!IsDefined() || !other.IsDefined()) {
    return Predicate(parameter_vars_, NullOpt);
  }

  PrimExpr other_predicate = other(parameter_vars_).value();

  PrimExpr new_predicate_expr =
      SimplifyAsAndOfOrs(expression_.value() && !other_predicate, analyzer);

  return Predicate(parameter_vars_, new_predicate_expr);
}

Predicate Predicate::Intersection(const Predicate& other, Analyzer* analyzer) const {
  ICHECK_EQ(parameter_vars_.size(), other.parameter_vars_.size())
      << "Predicates must be over the same number of parameters to be comparable";

  if (!IsDefined() || !other.IsDefined()) {
    return Predicate(parameter_vars_, NullOpt);
  }

  if (this->IsSubsetOf(other, analyzer)) {
    return (*this);
  } else if (other.IsSubsetOf(*this, analyzer)) {
    return other;
  }

  PrimExpr other_predicate = other(parameter_vars_).value();

  PrimExpr new_predicate_expr = analyzer->Simplify(expression_.value() && other_predicate);

  return Predicate(parameter_vars_, new_predicate_expr);
}

Predicate Predicate::Union(const Predicate& other, Analyzer* analyzer) const {
  ICHECK_EQ(parameter_vars_.size(), other.parameter_vars_.size())
      << "Predicates must be over the same number of parameters to be comparable";

  if (!IsDefined() || !other.IsDefined()) {
    return Predicate(parameter_vars_, NullOpt);
  }

  PrimExpr other_predicate = other(parameter_vars_).value();

  PrimExpr new_predicate_expr =
      SimplifyAsAndOfOrs(expression_.value() || other_predicate, analyzer);

  Map<tir::Var, Range> new_free_params;

  std::unordered_set<const VarNode*> undefined;
  auto undefined_var_arr = UndefinedVars(new_predicate_expr);
  for (const auto& var : undefined_var_arr) {
    undefined.insert(var.get());
  }

  return Predicate(parameter_vars_, new_predicate_expr);
}

void Predicate::Remap(const Map<Var, PrimExpr>& var_remap) {
  if (var_remap.empty() || !expression_) {
    return;
  }

  expression_ = Substitute(expression_.value(), var_remap);
}

void Predicate::Simplify(Analyzer* analyzer) {
  if (!expression_) {
    return;
  }

  // TODO: Are the extra simplification rounds necessary?
  PrimExpr expr = analyzer->Simplify(expression_.value(), 5);

  expression_ = expr;
}

bool Predicate::CanProve(Array<PrimExpr> args, Analyzer* analyzer) const {
  Optional<PrimExpr> expr = (*this)(std::move(args));
  return expr && analyzer->CanProve(expr.value());
}

bool Predicate::CanDisprove(Array<PrimExpr> args, Analyzer* analyzer) const {
  Optional<PrimExpr> expr = (*this)(std::move(args));
  return expr && analyzer->CanProve(!expr.value());
}

bool Predicate::IsAlwaysFalse() const {
  Analyzer analyzer;
  return expression_ && analyzer.CanProve(!expression_.value());
}

Predicate Predicate::WithoutFreeParameters(const Map<Var, Range>& free_params) const {
  if (!expression_) {
    return *this;
  }

  PrimExpr expr = NarrowExpressionToTrue(expression_.value(), free_params);
  return Predicate(parameter_vars_, expr);
}

std::ostream& operator<<(std::ostream& os, const Predicate& expr) {
  os << "predicate(" << static_cast<const ParametrizedExpression&>(expr);
  os << ")";
  return os;
}

BufferTouch::BufferTouch(tir::Buffer buffer, Array<Var> axis_vars, PrimExpr predicate,
                         AccessType touch_type, PrimExpr value)
    : buffer(buffer),
      axis_vars(axis_vars),
      predicate(predicate),
      value(value),
      touch_type(touch_type) {}

bool BufferTouch::IsSubsetOf(const BufferTouch& other, Analyzer* analyzer) const {
  if (this->buffer.same_as(other.buffer)) {
    CheckSameAxisVars(other);
    With<ConstraintContext> constraint(analyzer, predicate);

    return analyzer->CanProve(other.predicate);
  } else {
    return false;
  }
}

bool BufferTouch::IsDistinctFrom(const BufferTouch& other, Analyzer* analyzer) const {
  if (this->buffer.same_as(other.buffer)) {
    CheckSameAxisVars(other);
    With<ConstraintContext> constraint(analyzer, predicate);

    return analyzer->CanProve(!other.predicate);
  } else {
    return true;
  }
}

void BufferTouch::CheckSameAxisVars(const BufferTouch& other) const {
  ICHECK_EQ(axis_vars.size(), other.axis_vars.size());
  for (size_t i = 0; i < axis_vars.size(); i++) {
    ICHECK(axis_vars[i].same_as(other.axis_vars[i]));
  }
}

void BufferConstraint::CheckSameAxisVars(const BufferConstraint& other) const {
  ICHECK_EQ(axis_vars.size(), other.axis_vars.size());
  for (size_t i = 0; i < axis_vars.size(); i++) {
    ICHECK(axis_vars[i].same_as(other.axis_vars[i]));
  }
}

std::ostream& operator<<(std::ostream& os, const BufferTouch& tp) {
  auto touch_type = (tp.touch_type == BufferTouch::AccessType::Read)     ? "read"
                    : (tp.touch_type == BufferTouch::AccessType::Write)  ? "write"
                    : (tp.touch_type == BufferTouch::AccessType::Assume) ? "assume"
                                                                         : "???";
  os << "BufferTouch(" << tp.buffer->name << ", " << touch_type << ", " << tp.predicate
     << ", value = " << tp.value << ")";
  return os;
}

class BufferConstraintApply : public IRMutatorWithAnalyzer {
 public:
  using Parent = IRMutatorWithAnalyzer;

  BufferConstraintApply(const std::vector<BufferConstraint>& knowns, Analyzer* analyzer)
      : Parent(analyzer), knowns_(knowns) {}

  using Parent::VisitExpr_;

  PrimExpr VisitExpr_(const BufferLoadNode* op) override {
    for (const auto& known : knowns_) {
      if (!op->buffer.same_as(known.buffer)) {
        continue;
      }

      // TODO: De-dup this lane-handling section with similar code in
      // VisitBufferAccess.
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

      PrimExpr predicate = SubstituteParamValues(known.axis_vars, indices, known.predicate).value();
      std::optional<With<ConstraintContext>> context;
      if (lane_var.defined()) {
        Var lanes = lane_var.value();
        PrimExpr known = (IntImm(lanes.dtype(), 0) <= lanes) && (lanes < num_lanes);
        context.emplace(analyzer_, known);
      }
      if (analyzer_->CanProve(predicate)) {
        return SubstituteParamValues(known.axis_vars, op->indices, known.value).value();
      }
    }

    return GetRef<PrimExpr>(op);
  }

 private:
  const std::vector<BufferConstraint>& knowns_;
};

// Find Read region of the tensor in the stmt.
class BufferTouchExtractor final : public IRVisitorWithAnalyzer {
 public:
  static void Extract(BufferTouchPattern* out, const Stmt& stmt) {
    BufferTouchExtractor extractor(out);
    extractor.AppendControlBlock("start");
    extractor(stmt);
  }

 private:
  BufferTouchExtractor(BufferTouchPattern* out) : out_(out) {}

  using Parent = IRVisitorWithAnalyzer;
  using Parent::VisitExpr_;
  using Parent::VisitStmt_;

  void VisitStmt(const Stmt& stmt) override {
    // Point from the statement to the first touch point that occurs
    // at or after the statement.
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
        // if i < 3: T.assume(buf[i] == value)
        // T.assume(i>=3 or buf[i] == value)
        additional_predicate = additional_predicate && logical_not(expr);
      } else if (side_effect == tir::CallEffectKind::kReadState) {
        buffer_exprs.push_back(expr);
      } else {
        LOG(FATAL) << "Assumption must be pure or read-only";
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
      // This assumption is an inequality a data-dependent
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

    // TODO: An assumption shouldn't remove previously known
    // constraints.  Will need to split out the BufferConstraint from
    // the clearing of previous in KnownBufferValue.

    {
      InternalConstraintContext context(this, additional_predicate);
      VisitAccess(load, BufferTouch::AccessType::Assume, value);
    }
    // Appending a control block ensures that all control blocks have
    // at most one statement that changes the known buffer contents.
    auto prev_block = CurrentControlBlock();
    std::stringstream ss;
    ss << "after T.assume of " << assumption;
    auto new_block = AppendControlBlock(ss.str());
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
    auto new_block = AppendControlBlock("after bufferstore into " + op->buffer->name);
    MarkControlFlow(prev_block, new_block);
  }

  // TODO: tvm_access_ptr and address_of both act as opaque access of
  // entire buffer.

  void VisitStmt_(const ForNode* op) override {
    out_->iterator_ranges_.Set(op->loop_var, Range::FromMinExtent(op->min, op->extent));

    auto before_loop = CurrentControlBlock();
    auto loop_start = AppendControlBlock("start of loop over " + op->loop_var->name_hint);
    MarkControlFlow(before_loop, loop_start, {}, op->loop_var == op->min);

    BindActiveLoopVar binding(this, op->loop_var, op->min, op->extent);
    Parent::VisitStmt_(op);

    auto loop_end = CurrentControlBlock();
    auto after_loop = AppendControlBlock("after loop over " + op->loop_var->name_hint);
    MarkControlFlow(loop_end, after_loop,
                    {{op->loop_var, analyzer_.Simplify(op->min + op->extent - 1)}});

    std::vector<size_t> to_visit = {loop_start};
    std::unordered_set<size_t> marked = {loop_start};

    std::vector<const BufferTouch*> touches;

    while (to_visit.size()) {
      size_t visiting = to_visit.back();
      to_visit.pop_back();

      const auto& block = out_->control_flow_[visiting];

      for (const auto& touch : block.touch_points) {
        touches.push_back(&touch);
      }

      for (size_t successor : block.successors) {
        if (!marked.count(successor)) {
          to_visit.push_back(successor);
          marked.insert(successor);
        }
      }
    }

    bool depends_on_other_iterations = touches.size() > 1;

    if (depends_on_other_iterations) {
      std::stringstream d_name;
      d_name << op->loop_var->name_hint << "_delta";
      Var delta(d_name.str(), op->loop_var.dtype());
      MarkControlFlow(loop_end, loop_start, {{op->loop_var, op->loop_var - 1}},
                      op->loop_var > op->min);
    }
  }

  void VisitStmt_(const IfThenElseNode* op) override {
    this->VisitExpr(op->condition);

    PrimExpr real_condition = ExtractRealCondition(op->condition);

    auto before_branching = CurrentControlBlock();

    auto branch_start = AppendControlBlock([&]() {
      std::stringstream ss;
      ss << "before branch on " << real_condition;
      return ss.str();
    }());
    MarkControlFlow(before_branching, branch_start);

    auto then_start = AppendControlBlock([&]() {
      std::stringstream ss;
      ss << "then_case on " << real_condition;
      return ss.str();
    }());
    MarkControlFlow(branch_start, then_start);
    {
      InternalConstraintContext context(this, real_condition);
      this->VisitStmt(op->then_case);
    }
    auto then_end = CurrentControlBlock();

    size_t else_start = -1;
    size_t else_end = -1;
    if (op->else_case.defined()) {
      else_start = AppendControlBlock([&]() {
        std::stringstream ss;
        ss << "else_case on " << real_condition;
        return ss.str();
      }());
      MarkControlFlow(branch_start, else_start);

      auto negation = analyzer_.rewrite_simplify(Not(real_condition));
      InternalConstraintContext context(this, real_condition);
      this->VisitStmt(op->else_case);

      else_end = CurrentControlBlock();
    }

    auto after_branching = AppendControlBlock([&]() {
      std::stringstream ss;
      ss << "after branch on " << real_condition;
      return ss.str();
    }());
    MarkControlFlow(then_end, after_branching);

    if (op->else_case.defined()) {
      MarkControlFlow(else_end, after_branching);
    }
  }

  bool UsesLoopVar(const PrimExpr& expr) {
    return UsesVar(expr, [&](const VarNode* expr_var) {
      return loop_dependent_vars_.find(expr_var) != loop_dependent_vars_.end();
    });
  }

  template <typename BufferAccess>
  void VisitAccess(const BufferAccess& node, BufferTouch::AccessType touch_type,
                   PrimExpr known_value_expr) {
    auto index_variables = MakeIndexVariables(node->buffer, node->indices);

    Optional<Var> lane_var = NullOpt;
    IntImm num_lanes;

    Array<PrimExpr> index_expressions = node->indices.Map([&](const auto& index) {
      if (index.dtype().lanes() == 1) {
        return index;
      } else {
        ICHECK(!lane_var) << "Multiple indices found with non-scalar values";
        lane_var = Var("lane", index.dtype().element_of());
        num_lanes = IntImm(index.dtype().element_of(), index.dtype().lanes());
        return UnwrapVectorExpr(index, lane_var.value());
      }
    });

    // If the indices contain multiple lanes, treat the lane variable
    // as an additional loop iterator to be solved for and substituted
    // out.
    IntConstraintsTransform transform;
    if (lane_var) {
      BindActiveLoopVar binding(this, lane_var.value(), 0, num_lanes);
      transform = SolveForBufferIndices(index_variables, index_expressions);
    } else {
      transform = SolveForBufferIndices(index_variables, index_expressions);
    }

    Map<Var, PrimExpr> loop_var_to_axis_var = transform->src_to_dst;
    const Map<Var, Range>& free_params = transform->dst->ranges;

    for (const auto& pair : free_params) {
      out_->free_predicate_parameters_.Set(pair.first, pair.second);
    }

    // Constraints imposed on the axis variable (min/max bounds) and
    // on free parameters (relationships relative to axis vars).
    // These are used as part of the access predicate.

    // The arith::SolveLinearEquation sometimes introduces free
    // parameters with extent of one.  Filtering them out here avoids
    // needing to track them through later simplifications.
    bool has_extent_one_params =
        std::any_of(free_params.begin(), free_params.end(),
                    [](const auto& pair) { return is_one(pair.second->extent); });
    if (has_extent_one_params) {
      Analyzer analyzer;
      analyzer.Bind(free_params);
      Map<Var, PrimExpr> new_map;
      for (const auto& pair : loop_var_to_axis_var) {
        new_map.Set(pair.first, analyzer.Simplify(pair.second));
      }
      loop_var_to_axis_var = new_map;
    }

    // Normalization function, applied to both the predicate and the
    // known value.  Converts from an expression in terms of loop
    // iterators to an expression in terms of buffer indices.
    auto normalize_expr = [&](PrimExpr expr) -> PrimExpr {
      expr = Substitute(expr, let_bindings_using_loop_);

      if (lane_var) {
        expr = UnwrapVectorExpr(expr, lane_var.value());
      }
      expr = Substitute(expr, loop_var_to_axis_var);

      return expr;
    };

    known_value_expr = analyzer_.Simplify(normalize_expr(known_value_expr));

    // The full predicate is composed of the values required to reach
    // the scope of the BufferStore or builtin::assume(), any bounds
    // implied by solving for the axis variables, and any additional
    // statements resulting from unpacking the expression contained in
    // builtin::assume().
    PrimExpr scope_predicate = normalize_expr(CurrentScopePredicate());
    PrimExpr transform_predicate = normalize_expr(
        std::accumulate(transform->dst->relations.begin(), transform->dst->relations.end(),
                        PrimExpr(Bool(true)), [](PrimExpr a, PrimExpr b) { return a && b; }));
    PrimExpr loop_predicate = CurrentLoopPredicate(loop_var_to_axis_var);

    // Deliberately use an analyzer without scope-based information,
    // to avoid simplifying `scope_predicate` to True.
    Analyzer local_analyzer;
    PrimExpr predicate_expr =
        local_analyzer.Simplify(transform_predicate && scope_predicate && loop_predicate);

    BufferTouch buffer_touch(node->buffer, index_variables, predicate_expr, touch_type,
                             known_value_expr);

    out_->control_flow_.back().touch_points.push_back(buffer_touch);
  }

  std::function<void()> EnterConstraint(const PrimExpr& constraint) override {
    auto side_effect = tir::SideEffect(constraint);
    if (side_effect <= tir::CallEffectKind::kPure) {
      conditions_.push_back(constraint);

      return [this]() {
        ICHECK(conditions_.size()) << "Internal error: Each condition should only be popped once.";
        conditions_.pop_back();
      };
    } else if (side_effect <= tir::CallEffectKind::kReadState) {
      Assume(constraint, false);
      return []() {};
    } else {
      return []() {};
    }
  }

  Array<Var> MakeIndexVariables(const Buffer& buf, const Array<PrimExpr>& indices) {
    if (auto it = axis_var_lookup_.find(buf.get()); it != axis_var_lookup_.end()) {
      return it->second;
    }

    Array<Var> vars;
    for (size_t i = 0; i < indices.size(); i++) {
      std::stringstream ss;
      ss << buf->name << "_axis_" << i;
      vars.push_back(Var(ss.str(), indices[i].dtype().element_of()));
    }

    axis_var_lookup_[buf.get()] = vars;
    return vars;
  }

  IntConstraintsTransform SolveForBufferIndices(const Array<Var>& index_variables,
                                                const Array<PrimExpr>& index_expressions) {
    ICHECK_EQ(index_variables.size(), index_expressions.size());

    Array<PrimExpr> relations;

    for (size_t i = 0; i < index_expressions.size(); i++) {
      PrimExpr index = index_expressions[i];
      Var var = index_variables[i];

      relations.push_back(var == Substitute(index, let_bindings_using_loop_));
    }

    Array<Var> loop_vars;

    Map<Var, Range> ranges;
    for (const auto& loop_entry : active_loop_iterators_) {
      loop_vars.push_back(loop_entry.loop_var);
      ranges.Set(loop_entry.loop_var, loop_entry.loop_range);
    }

    IntConstraints system(loop_vars, ranges, relations);
    IntConstraintsTransform solution = arith::SolveLinearEquations(system);

    return solution;
  }

  PrimExpr CurrentScopePredicate() const {
    PrimExpr predicate = Bool(true);
    for (const auto& condition : conditions_) {
      predicate = predicate && condition;
    }
    return predicate;
  }

  // Generate a boolean expression represents having reached the
  // current loop iteration.
  PrimExpr CurrentLoopPredicate(Map<Var, PrimExpr>& loop_var_to_axis_var) const {
    PrimExpr loop_predicate = Bool(true);
    for (auto it = active_loop_iterators_.rbegin(); it != active_loop_iterators_.rend(); it++) {
      auto expr_it = loop_var_to_axis_var.find(it->loop_var);
      ICHECK(expr_it != loop_var_to_axis_var.end());
      PrimExpr loop_expr = (*expr_it).second;

      loop_predicate =
          (it->loop_var >= loop_expr) || ((it->loop_var == loop_expr) && loop_predicate);
    }
    return loop_predicate;
  }

  /* \brief Add a new control block, returning its index */
  size_t AppendControlBlock(std::string name) {
    size_t index = out_->control_flow_.size();
    out_->control_flow_.emplace_back();
    return index;
  }

  /* \brief The index of the current control block */
  size_t CurrentControlBlock() { return out_->control_flow_.size() - 1; }

  /* \brief Mark a possible control from one block to another */
  void MarkControlFlow(size_t from_block, size_t to_block, Map<Var, PrimExpr> var_remap = {},
                       Optional<PrimExpr> predicate = NullOpt) {
    ICHECK_LE(from_block, out_->control_flow_.size());
    ICHECK_LE(to_block, out_->control_flow_.size());

    out_->control_flow_[from_block].successors.push_back(to_block);
    out_->control_flow_[to_block].predecessors.push_back(
        BufferTouchPattern::ControlFlowEdge{from_block, var_remap, predicate});
  }

  // Internal utility, context manager for entering/leaving a scoped constraint
  struct InternalConstraintContext {
    InternalConstraintContext(BufferTouchExtractor* self, PrimExpr constraint)
        : self(self), analyzer_context(&self->analyzer_, constraint) {
      old_num_constraints = self->conditions_.size();

      auto side_effect = tir::SideEffect(constraint);
      if (side_effect <= tir::CallEffectKind::kPure) {
        self->conditions_.push_back(constraint);
      } else if (side_effect <= tir::CallEffectKind::kReadState) {
        self->Assume(constraint, false);
      }

      new_num_constraints = self->conditions_.size();
    }
    ~InternalConstraintContext() {
      ICHECK_EQ(self->conditions_.size(), new_num_constraints)
          << "Internal error: Each condition should only be popped once.";
      self->conditions_.erase(self->conditions_.begin() + old_num_constraints,
                              self->conditions_.end());
    }

    BufferTouchExtractor* self{nullptr};
    With<ConstraintContext> analyzer_context;
    size_t old_num_constraints{0};
    size_t new_num_constraints{0};

    // Disable default-generated copy/move assignment and constructors
    InternalConstraintContext(const InternalConstraintContext&) = delete;
    InternalConstraintContext& operator=(const InternalConstraintContext&) = delete;
    InternalConstraintContext(InternalConstraintContext&&) = delete;
    InternalConstraintContext& operator=(InternalConstraintContext&&) = delete;
  };

  // Internal utility, context manager for tracking a loop
  struct BindActiveLoopVar {
    BindActiveLoopVar(BufferTouchExtractor* self, Var var, PrimExpr loop_min, PrimExpr loop_extent)
        : self(self), var(var) {
      PrimExpr loop_max = loop_min + (loop_extent - 1);
      auto loop_range = Range::FromMinExtent(loop_min, loop_extent);
      self->active_loop_iterators_.push_back({var, loop_min, loop_max, loop_range});
      self->loop_dependent_vars_.insert(var.get());
    }
    ~BindActiveLoopVar() { self->active_loop_iterators_.pop_back(); }

    BufferTouchExtractor* self;
    Var var;

    // Disable default-generated copy/move assignment and constructors
    BindActiveLoopVar(const BindActiveLoopVar&) = delete;
    BindActiveLoopVar& operator=(const BindActiveLoopVar&) = delete;
    BindActiveLoopVar(BindActiveLoopVar&&) = delete;
    BindActiveLoopVar& operator=(BindActiveLoopVar&&) = delete;
  };

  // Internal utility, context manager for tracking a variable binding
  struct BindLetVar {
    BindLetVar(BufferTouchExtractor* self, Var var, PrimExpr value) : self(self), var(var) {
      self->let_bindings_using_loop_[var.get()] = value;
      self->loop_dependent_vars_.insert(var.get());
    }
    ~BindLetVar() {
      self->loop_dependent_vars_.erase(var.get());
      self->let_bindings_using_loop_.erase(var.get());
    }
    BufferTouchExtractor* self;
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
  std::vector<LoopEntry> active_loop_iterators_;

  // Track all loop iterators, along with values derived from loop iterators.
  std::unordered_set<const VarNode*> loop_dependent_vars_;

  // Any let binding that depends, directly or indirectly, on a loop
  // binding.  When making a predicate in terms of the buffer indices,
  // these need to be substituted out.
  std::unordered_map<const VarNode*, PrimExpr> let_bindings_using_loop_;

  // Track in order to know what conditions limit the buffer access
  std::vector<PrimExpr> conditions_;

  // Cache of previously used axis vars
  std::unordered_map<const BufferNode*, Array<Var>> axis_var_lookup_;

  // Track in order to know what statement initiated the buffer access
  Stmt current_stmt_;

  // Output data structure
  BufferTouchPattern* out_;
};

BufferTouchPattern::BufferTouchPattern(const tir::Stmt& stmt) {
  BufferTouchExtractor::Extract(this, stmt);
  ForwardPropagateKnownValues();
}

std::ostream& operator<<(std::ostream& os, const BufferTouchPattern::ControlFlowBlock& block) {
  os << "Control block with " << block.touch_points.size() << " touch points"
     << "\n";

  for (size_t i = 0; i < block.known_at_block_start.constraints.size(); i++) {
    os << "\t\t"
       << "PriorKnown[" << i << "] = " << block.known_at_block_start.constraints[i] << "\n";
  }
  for (size_t i = 0; i < block.touch_points.size(); i++) {
    os << "\t\t"
       << "Touch[" << i << "] = " << block.touch_points[i] << "\n";
  }
  for (size_t i = 0; i < block.known_at_block_end.constraints.size(); i++) {
    os << "\t\t"
       << "PostKnown[" << i << "] = " << block.known_at_block_end.constraints[i] << "\n";
  }

  os << "\t\t"
     << "Predecessors: [";
  for (size_t i = 0; i < block.predecessors.size(); i++) {
    if (i) {
      os << ", ";
    }
    os << block.predecessors[i].from_index;
    if (block.predecessors[i].var_remap.size()) {
      os << " with remap " << block.predecessors[i].var_remap;
    }
    if (block.predecessors[i].predicate) {
      os << " with postcondition " << block.predecessors[i].predicate;
    }
  }
  os << "]\n";

  os << "\t\t"
     << "Successors: [";
  for (size_t i = 0; i < block.successors.size(); i++) {
    if (i) {
      os << ", ";
    }
    os << block.successors[i];
  }
  os << "]";
  return os;
}

std::ostream& operator<<(std::ostream& os, const BufferTouchPattern& pattern) {
  os << "Touch pattern contains " << pattern.control_flow_.size() << " control blocks."
     << (pattern.control_flow_.size() ? "\n" : "");
  for (size_t i = 0; i < pattern.control_flow_.size(); i++) {
    os << "\t"
       << "ControlBlock[" << i << "] = " << pattern.control_flow_[i] << "\n";
  }

  return os;
}

BufferConstraint::BufferConstraint(tir::Buffer buffer, Array<tir::Var> axis_vars,
                                   PrimExpr predicate, PrimExpr value)
    : buffer(buffer), axis_vars(axis_vars), predicate(predicate), value(value) {}

bool BufferConstraint::IsDistinctFrom(const BufferConstraint& other, Analyzer* analyzer) const {
  if (!buffer.same_as(other.buffer)) {
    return true;
  }

  With<ConstraintContext> constraint(analyzer, predicate);
  return analyzer->CanProve(!other.predicate);
}

void BufferConstraint::OverwriteBy(const BufferConstraint& other, Analyzer* analyzer) {
  if (IsDistinctFrom(other, analyzer)) {
    return;
  }

  predicate = SimplifyAsAndOfOrs(other.predicate, analyzer);
}

bool BufferConstraint::IsEquivalentTo(const BufferConstraint& other, Analyzer* analyzer) const {
  // Constraints must apply to the same buffer to be equivalent
  if (!buffer.same_as(other.buffer)) {
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

void BufferState::AddCondition(const PrimExpr& condition) {
  for (auto& constraint : constraints) {
    constraint.predicate = constraint.predicate && condition;
  }
}

void BufferState::Substitute(const Map<Var, PrimExpr>& var_remap) {
  if (var_remap.size()) {
    for (auto& prior : constraints) {
      prior.predicate = tvm::tir::Substitute(prior.predicate, var_remap);
    }
  }
}

void BufferState::Simplify(Analyzer* analyzer) {
  for (auto& constraint : constraints) {
    constraint.predicate = SimplifyAsAndOfOrs(constraint.predicate, analyzer);
  }
}

std::ostream& operator<<(std::ostream& os, const BufferConstraint& obj) {
  return os << "Buffer " << obj.buffer->name << " is " << obj.value << " if " << obj.predicate;
}

BufferState BufferState::Union(const BufferState& a, const BufferState& b, Analyzer* analyzer) {
  BufferState output_state = a;
  for (const auto& b_constraint : b.constraints) {
    bool used = false;
    for (auto& a_constraint : output_state.constraints) {
      if (a_constraint.buffer.same_as(b_constraint.buffer) &&
          analyzer->CanProveEqual(a_constraint.value, b_constraint.value)) {
        a_constraint.predicate =
            SimplifyAsAndOfOrs(a_constraint.predicate || b_constraint.predicate, analyzer);
        used = true;
        break;
      }
    }
    if (!used) {
      output_state.constraints.push_back(b_constraint);
    }
  }

  return output_state;
}

BufferState BufferState::Intersection(const BufferState& a, const BufferState& b,
                                      Analyzer* analyzer) {
  // For a constraint to be in the output, it must be present in both
  // inputs.

  BufferState output_state;
  for (const auto& ai : a.constraints) {
    for (const auto& bi : b.constraints) {
      if (ai.buffer.same_as(bi.buffer)) {
        PrimExpr predicate = SimplifyAsAndOfOrs(ai.predicate && bi.predicate, analyzer);
        if (!is_zero(predicate)) {
          auto axis_vars = ai.axis_vars;
          With<ConstraintContext> context(analyzer, predicate);
          PrimExpr known_value_a = ai.value;
          PrimExpr known_value_b = bi.value;

          bool is_consistent = analyzer->CanProveEqual(known_value_a, known_value_b);
          if (is_consistent) {
            output_state.constraints.push_back({ai.buffer, axis_vars, predicate, known_value_a});
          }
        }
      }
    }
  }

  return output_state;
}

class BufferRegionCollector : public ExprVisitor {
 public:
  struct Region {
    PrimExpr region_predicate;
    std::unordered_map<const BufferLoadNode*, Optional<PrimExpr>> known_values;
  };

  static std::vector<Region> Collect(const std::vector<BufferConstraint>& knowns,
                                     const std::vector<Optional<PrimExpr>>& exprs,
                                     Analyzer* analyzer) {
    BufferRegionCollector collector(knowns, analyzer);
    for (const auto& expr : exprs) {
      if (expr) {
        collector(expr.value());
      }
    }

    return collector.regions_;
  }

 private:
  using Parent = ExprVisitor;

  BufferRegionCollector(const std::vector<BufferConstraint>& knowns, Analyzer* analyzer)
      : analyzer_(analyzer), knowns_(knowns) {
    regions_.push_back(Region{Bool(true), {}});
  }

  using Parent::VisitExpr_;

  void VisitExpr_(const BufferLoadNode* op) override {
    // Helper struct for the known values of this BufferLoad
    struct Known {
      PrimExpr predicate;
      Optional<PrimExpr> value;
    };

    std::vector<Known> new_regions;

    PrimExpr unknown_region = Bool(true);

    for (const BufferConstraint& constraint : knowns_) {
      if (!op->buffer.same_as(constraint.buffer)) {
        // This is a different buffer, so continue searching.
        continue;
      }

      PrimExpr touch_predicate =
          SubstituteParamValues(constraint.axis_vars, op->indices, constraint.predicate).value();
      touch_predicate = SimplifyAsAndOfOrs(touch_predicate, analyzer_);

      if (!is_zero(touch_predicate)) {
        Optional<PrimExpr> known_value =
            SubstituteParamValues(constraint.axis_vars, op->indices, constraint.value);
        new_regions.push_back(Known{touch_predicate, known_value});

        unknown_region = unknown_region && !touch_predicate;
        unknown_region = SimplifyAsAndOfOrs(unknown_region, analyzer_);
      }
    }

    if (new_regions.size()) {
      Analyzer local_analyzer;

      if (!is_zero(unknown_region)) {
        new_regions.insert(new_regions.begin(), Known{unknown_region, NullOpt});
      }

      std::vector<Region> updated_regions;
      for (const auto& prev_region : regions_) {
        for (const auto& new_region : new_regions) {
          PrimExpr intersection =
              SimplifyAsAndOfOrs(prev_region.region_predicate && new_region.predicate, analyzer_);

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
  const std::vector<BufferConstraint>& knowns_;
};

class BufferRegionValueReplacer : public IRMutatorWithAnalyzer {
 public:
  static PrimExpr Apply(
      const std::unordered_map<const BufferLoadNode*, Optional<PrimExpr>>& known_values,
      PrimExpr expr, Analyzer* analyzer) {
    BufferRegionValueReplacer mutator(known_values, analyzer);
    PrimExpr result = mutator(expr);
    // Simplification must occur after the substitution, as known
    // values may provide enable simplifications.  Also, cannot track
    // whether a BufferLoad was
    result = analyzer->Simplify(result);
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

void BufferState::ApplyTouches(const std::vector<BufferTouch>& touch_points,
                               const Map<Var, Range>& free_predicate_parameters,
                               Analyzer* analyzer) {
  std::vector<BufferConstraint> new_knowns;
  Map<Buffer, PrimExpr> keep_prior_known_at;

  for (auto& touch : touch_points) {
    if (touch.touch_type == BufferTouch::AccessType::Read) {
      continue;
    }

    Array<Var> axis_vars = touch.axis_vars;
    PrimExpr predicate = touch.predicate;

    PrimExpr known_value = touch.value;
    auto regions =
        BufferRegionCollector::Collect(constraints, {touch.predicate, touch.value}, analyzer);

    for (const auto& region : regions) {
      PrimExpr updated_predicate = BufferRegionValueReplacer::Apply(
          region.known_values, region.region_predicate && predicate, analyzer);

      updated_predicate = SimplifyAsAndOfOrs(updated_predicate, analyzer);
      PrimExpr updated_value =
          BufferRegionValueReplacer::Apply(region.known_values, known_value, analyzer);

      if (!is_zero(updated_predicate)) {
        if (auto it = keep_prior_known_at.find(touch.buffer); it != keep_prior_known_at.end()) {
          keep_prior_known_at.Set(touch.buffer, (*it).second && !updated_predicate);
        } else {
          keep_prior_known_at.Set(touch.buffer, !updated_predicate);
        }

        if (!HasBufferLoad(updated_value)) {
          BufferConstraint new_constraint{touch.buffer, axis_vars, updated_predicate,
                                          updated_value};
          new_knowns.push_back(new_constraint);
        }
      }
    }
  }

  if (keep_prior_known_at.size()) {
    for (auto& constraint : constraints) {
      if (auto it = keep_prior_known_at.find(constraint.buffer); it != keep_prior_known_at.end()) {
        constraint.predicate = SimplifyAsAndOfOrs(constraint.predicate && (*it).second, analyzer);
      }
    }
  }

  if (new_knowns.size()) {
    std::vector<bool> used(new_knowns.size(), false);

    for (auto& constraint : constraints) {
      PrimExpr expand_known_at = Bool(false);

      PrimExpr prev_value = constraint.value;

      for (size_t i = 0; i < new_knowns.size(); i++) {
        if (new_knowns[i].buffer.same_as(constraint.buffer)) {
          Optional<PrimExpr> overwritten_with = new_knowns[i].value;
          if (overwritten_with && analyzer->CanProveEqual(prev_value, overwritten_with.value())) {
            expand_known_at =
                SimplifyAsAndOfOrs(expand_known_at || new_knowns[i].predicate, analyzer);
            used[i] = true;
          }
        }
      }

      if (!is_zero(expand_known_at)) {
        constraint.predicate =
            SimplifyAsAndOfOrs(constraint.predicate || expand_known_at, analyzer);
      }
    }

    for (size_t i = 0; i < new_knowns.size(); i++) {
      if (!used[i]) {
        constraints.push_back(new_knowns[i]);
      }
    }
  }

  constraints.erase(
      std::remove_if(constraints.begin(), constraints.end(),
                     [&](const auto& constraint) { return is_zero(constraint.predicate); }),
      constraints.end());
}

void BufferState::RemoveFreeParameters(const Map<Var, Range>& free_predicate_parameters,
                                       Analyzer* analyzer) {
  for (auto& known : constraints) {
    known.predicate = NarrowExpressionToTrue(known.predicate, free_predicate_parameters);
    known.predicate = SimplifyAsAndOfOrs(known.predicate, analyzer);
  }
}

bool BufferState::IsEquivalentTo(const BufferState& other, Analyzer* analyzer) const {
  if (constraints.size() != other.constraints.size()) {
    return false;
  }

  for (size_t i = 0; i < constraints.size(); i++) {
    if (!constraints[i].IsEquivalentTo(other.constraints[i], analyzer)) {
      return false;
    }
  }

  return true;
}

void BufferTouchPattern::ForwardPropagateKnownValues() {
  // Values to visit when searching.  Using a std::set to
  // preferentially visit nodes near the start of the control flow.
  std::set<size_t> to_visit;

  // Track whether a buffer has been visited at least once.
  std::unordered_set<size_t> visited_once;

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

  Analyzer analyzer;
  analyzer.rewrite_simplify.SetEnabledExtensions(
      arith::RewriteSimplifier::kTransitivelyProveInequalities);

  analyzer.Bind(iterator_ranges_);
  analyzer.Bind(free_predicate_parameters_);

  while (to_visit.size()) {
    size_t visiting = *to_visit.begin();
    to_visit.erase(visiting);
    ControlFlowBlock& block = control_flow_[visiting];

    // Step 1: Collect known values provided from each precedessor
    block.known_at_block_start = [&]() -> BufferState {
      ICHECK_LE(block.predecessors.size(), 2) << "Each block should have at most two predecessors";

      std::vector<BufferState> states;
      for (const auto& pred : block.predecessors) {
        const auto& pred_block = control_flow_[pred.from_index];
        BufferState state = pred_block.known_at_block_end;
        state.Substitute(pred.var_remap);
        states.push_back(state);
      }

      if (std::all_of(block.predecessors.begin(), block.predecessors.end(),
                      [&](const auto& pred) { return !visited_once.count(pred.from_index); })) {
        // Predecessors, if any, are unvisited.
        return {};
      } else if (block.predecessors.size() == 1) {
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
      if (!visited_once.count(pred_a.from_index)) {
        return priors_b;
      } else if (!visited_once.count(pred_b.from_index)) {
        return priors_a;
      }

      if (pred_a.predicate && pred_b.predicate) {
        // The predicate can identify which predecessor block applies
        // (e.g. i==0 for the first loop iteration, i>0 for remaining
        // loop iterations).  Therefore, we can use all buffer
        // constraints, conditional on having come from the
        // predecessor that provides it.
        priors_a.AddCondition(pred_a.predicate.value());
        priors_b.AddCondition(pred_b.predicate.value());
        return BufferState::Union(priors_a, priors_b, &analyzer);
      } else {
        // We don't know which predecessor applies.  Therefore, the
        // only buffer constraints that can be used are those that
        // appear in both predecessors.
        return BufferState::Intersection(priors_a, priors_b, &analyzer);
      }
    }();

    // Step 2: Collect knowns provided as a result of executing this block
    auto post_state = block.known_at_block_start;
    post_state.ApplyTouches(block.touch_points, free_predicate_parameters_, &analyzer);
    post_state.RemoveFreeParameters(free_predicate_parameters_, &analyzer);

    // Step 3: If any changes are made to the post knowns since the
    // previous time we visited this block, mark the successor block
    // as needing to be visited.
    //
    // TODO: Have a maximum number of times that blocks may be
    // visited, to guard against infinite loops.
    // if (has_updated_post) {
    if (!visited_once.count(visiting) ||
        !post_state.IsEquivalentTo(block.known_at_block_end, &analyzer)) {
      block.known_at_block_end = std::move(post_state);
      for (size_t successor : block.successors) {
        to_visit.insert(successor);
      }
    }

    visited_once.insert(visiting);
  }
}

bool BufferTouchPattern::IsOverwrittenWithoutEffect(const tir::BufferStore& store,
                                                    Analyzer* analyzer) const {
  auto it = control_flow_lookup_.find(store.get());
  ICHECK(it != control_flow_lookup_.end()) << "BufferStore did not occur within analyzed statement";

  const auto& store_block = control_flow_[it->second];
  ICHECK_GE(store_block.touch_points.size(), 1);

  const auto& store_touch = store_block.touch_points.back();

  std::unordered_set<size_t> seen;
  std::vector<size_t> to_visit = store_block.successors;

  while (to_visit.size()) {
    size_t visiting = to_visit.back();
    to_visit.pop_back();

    const auto& block = control_flow_[visiting];

    for (auto& touch : block.touch_points) {
      if (touch.touch_type == BufferTouch::AccessType::Read &&
          !store_touch.IsDistinctFrom(touch, analyzer)) {
        // The buffer location is being read out, has an effect.
        return false;
      } else if (touch.touch_type == BufferTouch::AccessType::Write &&
                 store_touch.IsSubsetOf(touch, analyzer)) {
        // The buffer is entirely overwritten, so the earlier store
        // has no effect.
        return true;
      } else {
        // Pass along to next block to visit.
        for (size_t next_index : block.successors) {
          if (!seen.count(next_index)) {
            to_visit.push_back(next_index);
          }
        }
      }
    }
  }

  return false;
}

PrimExpr BufferTouchPattern::SimplifyInContext(PrimExpr expr, const tir::Stmt& context,
                                               Analyzer* analyzer) const {
  size_t context_index = [&]() {
    auto it = control_flow_lookup_.find(context.get());
    ICHECK(it != control_flow_lookup_.end())
        << "Context did not occur in the Stmt provided to BufferTouchPattern's constructor";
    return it->second;
  }();

  PrimExpr constraint = Bool(true);
  for (const auto& known : non_buffer_assumptions_) {
    constraint = constraint && known;
  }
  With<ConstraintContext> constraint_context(analyzer, constraint);

  const auto& knowns = control_flow_[context_index].known_at_block_start.constraints;

  BufferConstraintApply mutator(knowns, analyzer);
  expr = mutator(expr);
  expr = analyzer->Simplify(expr);
  return expr;
}

void BufferTouchPattern::RemoveTouches(const tir::BufferStore& store) {
  // TODO: Update control_flow_
}

}  // namespace arith
}  // namespace tvm
