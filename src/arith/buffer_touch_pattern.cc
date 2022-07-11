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

#include <tvm/arith/int_solver.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <sstream>
#include <vector>

#include "constraint_extract.h"
#include "ir_visitor_with_analyzer.h"

namespace tvm {
namespace arith {

using namespace tir;

ParametrizedExpression::ParametrizedExpression(Array<Var> parameter_vars, PrimExpr expression)
    : parameter_vars_(parameter_vars), expression_(expression) {}

PrimExpr ParametrizedExpression::operator()(Array<PrimExpr> args) const {
  ICHECK_EQ(parameter_vars_.size(), args.size())
      << "Expression was defined as having " << parameter_vars_.size()
      << " parameters, but received " << args.size() << " arguments.";

  if (!expression_.defined()) {
    return expression_;
  }

  Map<tir::Var, PrimExpr> var_map;
  for (size_t i = 0; i < args.size(); i++) {
    var_map.Set(parameter_vars_[i], args[i]);
  }

  return Substitute(expression_, var_map);
}

bool ParametrizedExpression::IsConstant() const {
  std::unordered_set<const VarNode*> vars;
  for (const auto& var : parameter_vars_) {
    vars.insert(var.get());
  }
  return !UsesVar(expression_, [&](const VarNode* var) { return vars.count(var); });
}

std::ostream& operator<<(std::ostream& os, const ParametrizedExpression& expr) {
  if (!expr.IsConstant()) {
    os << expr.parameter_vars_ << " => ";
  }

  os << expr.expression_;
  return os;
}

Predicate::Predicate(Array<tir::Var> parameter_vars, PrimExpr expression,
                     Map<tir::Var, Range> free_parameters)
    : ParametrizedExpression(parameter_vars, expression), free_parameters_(free_parameters) {}

bool Predicate::IsSubsetOf(const Predicate& other) const {
  ICHECK_EQ(parameter_vars_.size(), other.parameter_vars_.size())
      << "Predicates must be over the same number of parameters to be comparable";

  Array<PrimExpr> vars_as_primexpr;
  for (const auto& var : parameter_vars_) {
    vars_as_primexpr.push_back(var);
  }

  PrimExpr other_predicate = other(vars_as_primexpr);

  arith::Analyzer analyzer;

  With<ConstraintContext> this_params(&analyzer, this->FreeParameterConstraints());
  With<ConstraintContext> other_params(&analyzer, other.FreeParameterConstraints());
  With<ConstraintContext> constraint(&analyzer, expression_);

  return analyzer.CanProve(other_predicate);
}

bool Predicate::IsDistinctFrom(const Predicate& other) const {
  ICHECK_EQ(parameter_vars_.size(), other.parameter_vars_.size())
      << "Predicates must be over the same number of parameters to be comparable";

  Array<PrimExpr> vars_as_primexpr;
  for (const auto& var : parameter_vars_) {
    vars_as_primexpr.push_back(var);
  }

  PrimExpr other_predicate = other(vars_as_primexpr);

  arith::Analyzer analyzer;

  With<ConstraintContext> this_params(&analyzer, this->FreeParameterConstraints());
  With<ConstraintContext> other_params(&analyzer, other.FreeParameterConstraints());
  With<ConstraintContext> constraint(&analyzer, expression_);

  return analyzer.CanProve(logical_not(other_predicate));
}

Predicate Predicate::Difference(const Predicate& other) const {
  ICHECK_EQ(parameter_vars_.size(), other.parameter_vars_.size())
      << "Predicates must be over the same number of parameters to be comparable";

  Array<PrimExpr> vars_as_primexpr;
  for (const auto& var : parameter_vars_) {
    vars_as_primexpr.push_back(var);
  }

  PrimExpr other_predicate = other(vars_as_primexpr);

  arith::Analyzer analyzer;

  With<ConstraintContext> this_params(&analyzer, this->FreeParameterConstraints());
  With<ConstraintContext> other_params(&analyzer, other.FreeParameterConstraints());

  PrimExpr new_predicate_expr = analyzer.Simplify(expression_ && !other_predicate);

  Map<tir::Var, Range> new_free_params = free_parameters_;
  for (const auto& pair : other.free_parameters_) {
    new_free_params.Set(pair.first, pair.second);
  }

  return Predicate(parameter_vars_, new_predicate_expr, new_free_params);
}

bool Predicate::CanProve(Array<PrimExpr> args, Analyzer* analyzer) const {
  With<ConstraintContext> constraint(analyzer, FreeParameterConstraints());
  PrimExpr expr = (*this)(std::move(args));
  bool result = analyzer->CanProve(expr);
  return result;
}

bool Predicate::CanDisprove(Array<PrimExpr> args, Analyzer* analyzer) const {
  With<ConstraintContext> constraint(analyzer, FreeParameterConstraints());
  PrimExpr expr = (*this)(std::move(args));
  bool result = analyzer->CanProve(!expr);
  return result;
}

PrimExpr Predicate::FreeParameterConstraints() const {
  PrimExpr constraint = Bool(true);
  for (const auto& pair : free_parameters_) {
    const Var& var = pair.first;
    const Range& range = pair.second;
    if (is_const_int(range->extent, 1)) {
      constraint = constraint && (var == range->min);
    } else {
      constraint = constraint && (var >= range->min);
      constraint = constraint && (var < range->min + range->extent);
    }
  }
  return constraint;
}

std::ostream& operator<<(std::ostream& os, const Predicate& expr) {
  os << "predicate(" << static_cast<const ParametrizedExpression&>(expr);
  if (expr.free_parameters_.size()) {
    for (const auto& pair : expr.free_parameters_) {
      os << ", for all " << pair.first << " in [" << pair.second->min << ", "
         << pair.second->min + pair.second->extent << ")";
    }
  }
  os << ")";
  return os;
}

BufferTouch::BufferTouch(Buffer buffer, Predicate predicate, AccessType touch_type,
                         ParametrizedExpression known_value, ObjectRef node)
    : buffer(buffer),
      predicate(predicate),
      touch_type(touch_type),
      known_value(known_value),
      node(node) {}

bool BufferTouch::IsSubsetOf(const BufferTouch& other) const {
  if (!this->buffer.same_as(other.buffer)) {
    return false;
  } else {
    return this->predicate.IsSubsetOf(other.predicate);
  }
}

std::ostream& operator<<(std::ostream& os, const BufferTouch& tp) {
  auto touch_type = (tp.touch_type == BufferTouch::AccessType::Read)    ? "read"
                    : (tp.touch_type == BufferTouch::AccessType::Write) ? "write"
                                                                        : "opaque";
  os << "BufferTouch(" << tp.buffer->name << ", " << touch_type << ", " << tp.predicate;
  if (tp.known_value.IsDefined()) {
    os << ", known_value = " << tp.known_value;
  }

  os << ")";
  return os;
}

// Find Read region of the tensor in the stmt.
class BufferTouchExtractor final : public IRVisitorWithAnalyzer {
 public:
  using Parent = IRVisitorWithAnalyzer;
  using Parent::VisitExpr_;
  using Parent::VisitStmt_;

  void VisitStmt(const Stmt& stmt) override {
    // Point from the statement to the first touch point that occurs
    // at or after the statement.
    context_lookup_[stmt.get()] = touch_points_.size();
    Stmt prev_stmt = current_stmt_;
    current_stmt_ = stmt;
    Parent::VisitStmt(stmt);
    current_stmt_ = prev_stmt;
  }

  void VisitStmt_(const EvaluateNode* op) override {
    if (auto* call = op->value.as<CallNode>()) {
      if (call->op.same_as(builtin::assume())) {
        Assume(call->args[0]);
        return;
      }
    }

    Parent::VisitStmt_(op);
  }

  void Assume(PrimExpr assumption) {
    for (const auto& expr : ExtractConstraints(assumption, false)) {
      AssumeConstraintComponent(expr);
    }
  }

  void AssumeConstraintComponent(PrimExpr assumption) {
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
      non_buffer_assumptions_.push_back(!CurrentScopePredicate() || assumption);
      return;
    }

    CHECK_EQ(buffer_exprs.size(), 1) << "T.assume must contain only a single buffer expression";

    auto* as_equal_node = buffer_exprs[0].as<tir::EQNode>();
    CHECK(as_equal_node)
        << "T.assume buffer constraint must be of the form 'buffer[indices] == value'";

    tir::BufferLoad load;
    PrimExpr value;
    if (auto* as_load = as_equal_node->a.as<tir::BufferLoadNode>()) {
      load = GetRef<tir::BufferLoad>(as_load);
      value = as_equal_node->b;
    } else if (auto* as_load = as_equal_node->b.as<tir::BufferLoadNode>()) {
      load = GetRef<tir::BufferLoad>(as_load);
      value = as_equal_node->a;
    } else {
      LOG(FATAL) << "T.assume buffer constraint must be of the form 'buffer[indices] == value'";
    }

    CHECK(tir::SideEffect(value) <= tir::CallEffectKind::kPure)
        << "Buffer value in constraint must be pure expression, but was " << value;

    // TODO: An assumption shouldn't remove previously known
    // constraints.  Will need to split out the BufferConstraint from
    // the clearing of previous in KnownBufferValue.

    VisitAccess(load, BufferTouch::AccessType::Read, value, additional_predicate);
  }

  void VisitExpr_(const LetNode* op) override {
    if (UsesLoopVar(op->value)) {
      let_bindings_using_loop_[op->var.get()] = op->value;
      loop_dependent_vars_.insert(op->var.get());
    }
    Parent::VisitExpr_(op);
    loop_dependent_vars_.erase(op->var.get());
    let_bindings_using_loop_.erase(op->var.get());
  }

  void VisitStmt_(const LetStmtNode* op) override {
    if (UsesLoopVar(op->value)) {
      let_bindings_using_loop_[op->var.get()] = op->value;
      loop_dependent_vars_.insert(op->var.get());
    }
    Parent::VisitStmt_(op);
    loop_dependent_vars_.erase(op->var.get());
    let_bindings_using_loop_.erase(op->var.get());
  }

  void VisitExpr_(const BufferLoadNode* op) override {
    Parent::VisitExpr_(op);
    VisitAccess(GetRef<BufferLoad>(op), BufferTouch::AccessType::Read);
  }

  void VisitStmt_(const BufferStoreNode* op) override {
    Parent::VisitStmt_(op);
    VisitAccess(GetRef<BufferStore>(op), BufferTouch::AccessType::Write, op->value);
  }

  // TODO: tvm_access_ptr and address_of both act as opaque access of
  // entire buffer.

  void VisitStmt_(const ForNode* op) override {
    active_loop_iterators_.push_back(op->loop_var);
    loop_dependent_vars_.insert(op->loop_var.get());
    Parent::VisitStmt_(op);
    loop_dependent_vars_.erase(op->loop_var.get());
    active_loop_iterators_.pop_back();
  }

  bool UsesLoopVar(const PrimExpr& expr) {
    return UsesVar(expr, [&](const VarNode* expr_var) {
      return loop_dependent_vars_.find(expr_var) != loop_dependent_vars_.end();
    });
  }

  template <typename BufferAccess>
  void VisitAccess(const BufferAccess& node, BufferTouch::AccessType touch_type,
                   Optional<PrimExpr> opt_known_value = NullOpt,
                   Optional<PrimExpr> additional_predicate = NullOpt) {
    auto index_variables = MakeIndexVariables(node->indices);
    auto transform = SolveForBufferIndices(index_variables, node->indices);
    auto predicate =
        CurrentPredicate(index_variables, node->indices, transform, additional_predicate);
    auto known_value = KnownValue(index_variables, transform, opt_known_value);
    touch_points_.push_back(BufferTouch(node->buffer, predicate, touch_type, known_value, node));
  }

  std::function<void()> EnterConstraint(const PrimExpr& constraint) override {
    conditions_.push_back(constraint);

    return [this]() {
      ICHECK(conditions_.size()) << "Internal error: Each condition should only be popped once.";
      conditions_.pop_back();
    };
  }

  Array<Var> MakeIndexVariables(const Array<PrimExpr>& indices) {
    Array<Var> vars;
    for (size_t i = 0; i < indices.size(); i++) {
      std::stringstream ss;
      ss << "i_" << i;
      vars.push_back(Var(ss.str()));
    }
    return vars;
  }

  IntConstraintsTransform SolveForBufferIndices(const Array<Var>& index_variables,
                                                const Array<PrimExpr>& index_expressions) {
    Map<Var, Range> ranges;
    Array<PrimExpr> relations;

    for (size_t i = 0; i < index_expressions.size(); i++) {
      PrimExpr index = index_expressions[i];
      Var var = index_variables[i];

      relations.push_back(var == Substitute(index, let_bindings_using_loop_));

      IntSet interval = analyzer_.int_set(index);
    }

    Array<Var> loop_vars;

    Map<Var, Range> loop_ranges;
    for (const auto& loop_var : active_loop_iterators_) {
      loop_vars.push_back(loop_var);
      IntSet loop_set = analyzer_.int_set(loop_var);
      Range loop_range = Range(loop_set.min(), loop_set.max());
      loop_ranges.Set(loop_var, loop_range);
    }

    IntConstraints system(loop_vars, loop_ranges, relations);
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

  Predicate CurrentPredicate(const Array<Var>& index_variables,
                             const Array<PrimExpr>& index_expressions,
                             const IntConstraintsTransform& transform,
                             const Optional<PrimExpr>& additional_predicate) {
    PrimExpr predicate = additional_predicate.value_or(Bool(true));
    predicate = predicate && CurrentScopePredicate();
    predicate = Substitute(predicate, let_bindings_using_loop_);

    for (size_t i = 0; i < index_expressions.size(); i++) {
      PrimExpr index = index_expressions[i];
      Var var = index_variables[i];

      IntSet interval = analyzer_.int_set(index);

      if (interval.IsSinglePoint()) {
        predicate = predicate && (var == interval.PointValue());
      } else {
        if (interval.HasLowerBound()) {
          predicate = predicate && (var >= interval.min());
        }
        if (interval.HasUpperBound()) {
          predicate = predicate && (var <= interval.max());
        }
      }
    }

    predicate = Substitute(predicate, transform->src_to_dst);
    predicate = analyzer_.Simplify(predicate);

    ICHECK(!UsesLoopVar(predicate))
        << "Internal error: Loop variable still used after substituting out the loop variable";

    return Predicate(index_variables, predicate, transform->dst->ranges);
  }

  ParametrizedExpression KnownValue(const Array<Var>& index_variables,
                                    const IntConstraintsTransform& transform,
                                    Optional<PrimExpr> opt_known_value) {
    if (!opt_known_value) {
      return ParametrizedExpression(index_variables, PrimExpr());
    }

    PrimExpr value = opt_known_value.value();
    value = Substitute(value, let_bindings_using_loop_);
    value = Substitute(value, transform->src_to_dst);
    value = analyzer_.Simplify(value);

    auto free_params = transform->dst->ranges;
    bool uses_free_param = UsesVar(value, [&](const VarNode* var) {
      return free_params.find(GetRef<Var>(var)) != free_params.end();
    });
    if (uses_free_param) {
      return ParametrizedExpression(index_variables, PrimExpr());
    } else {
      return ParametrizedExpression(index_variables, value);
    }
  }

  // Track in order to know which Vars to write in terms of the buffer
  // indices and substitute out of the predicate.
  std::vector<Var> active_loop_iterators_;

  // Track all loop iterators, along with values derived from loop iterators.
  std::unordered_set<const VarNode*> loop_dependent_vars_;

  // Any let binding that depends, directly or indirectly, on a loop
  // binding.  When making a predicate in terms of the buffer indices,
  // these need to be substituted out.
  std::unordered_map<const VarNode*, PrimExpr> let_bindings_using_loop_;

  // Track in order to know what conditions limit the buffer access
  std::vector<PrimExpr> conditions_;

  // Track in order to know what statement initiated the buffer access
  Stmt current_stmt_;

  // Output data structure
  std::vector<BufferTouch> touch_points_;

  std::vector<PrimExpr> non_buffer_assumptions_;

  // Map into touch_points_
  std::unordered_map<const tir::StmtNode*, size_t> context_lookup_;
};

class BufferConstraintSubstituter : public ExprMutator {
 public:
  BufferConstraintSubstituter(const BufferTouchPattern& touch_pattern, size_t ignore_touches_after_,
                              Analyzer* analyzer)
      : touch_pattern_(touch_pattern),
        analyzer_(analyzer),
        ignore_touches_after_(ignore_touches_after_) {}

  PrimExpr VisitExpr_(const BufferLoadNode* op) override {
    auto it = touch_pattern_.touch_points_.rbegin();
    if (ignore_touches_after_ < touch_pattern_.touch_points_.size()) {
      it += ignore_touches_after_;
    }

    PrimExpr access_predicate = Bool(true);

    auto implies = [this](const PrimExpr& known, const PrimExpr& conjecture) -> bool {
      With<ConstraintContext> constraint(analyzer_, known);
      return analyzer_->CanProve(conjecture);
    };

    std::vector<std::pair<PrimExpr, PrimExpr>> known_subregion;
    PrimExpr free_parameter_constraints = Bool(true);

    for (; it != touch_pattern_.touch_points_.rend(); it++) {
      const BufferTouch& touch = *it;

      PrimExpr touch_predicate = analyzer_->Simplify(touch.predicate(op->indices));

      // With<ConstraintContext> isn't safe to use in a std::vector,
      // so instead we collect a single expression with all the extra
      // constraints.
      free_parameter_constraints =
          free_parameter_constraints && touch.predicate.FreeParameterConstraints();
      With<ConstraintContext> constraint(analyzer_, free_parameter_constraints);

      if (!op->buffer.same_as(touch.buffer)) {
        // This is a different buffer, ignore
      } else if (touch.known_value.IsDefined() && implies(access_predicate, touch_predicate)) {
        // This access resulted in a known value, return it.
        PrimExpr value = touch.known_value(op->indices);
        for (auto it = known_subregion.rbegin(); it != known_subregion.rend(); it++) {
          value = if_then_else(it->first, it->second, value);
        }
        return value;
      } else if (implies(access_predicate, logical_not(touch_predicate))) {
        // The previous access didn't change the values we're
        // interested in, so continue searching.
      } else if (touch.known_value.IsDefined()) {
        // The previous access resulted in a known value, but only for
        // some of the indices we are interested in.  It's still
        // possible that the same value
        known_subregion.push_back({touch.predicate(op->indices), touch.known_value(op->indices)});
        access_predicate = access_predicate && !touch_predicate;
      } else if (touch.touch_type == BufferTouch::AccessType::Read) {
        // This access didn't change the buffer's contents, so
        // continue backtracking.
      } else {
        // This BufferTouch writes values to the buffer that we might
        // use, and we don't know what those values are.  Therefore,
        // cannot simplify out the buffer access.
        return GetRef<PrimExpr>(op);
      }
    }

    return GetRef<PrimExpr>(op);
  }

  const BufferTouchPattern& touch_pattern_;
  Analyzer* analyzer_;
  size_t ignore_touches_after_;
};

BufferTouchPattern::BufferTouchPattern(const tir::Stmt& stmt) {
  BufferTouchExtractor extractor;
  extractor(stmt);
  touch_points_ = std::move(extractor.touch_points_);
  context_lookup_ = std::move(extractor.context_lookup_);
  non_buffer_assumptions_ = std::move(extractor.non_buffer_assumptions_);
}

std::ostream& operator<<(std::ostream& os, const BufferTouchPattern& pattern) {
  os << "Touch pattern contains " << pattern.touch_points_.size() << " touches."
     << (pattern.touch_points_.size() ? "\n" : "");
  for (size_t i = 0; i < pattern.touch_points_.size(); i++) {
    os << "\t"
       << "Touch[" << i << "] = " << pattern.touch_points_[i];
    if (i + 1 < pattern.touch_points_.size()) {
      os << "\n";
    }
  }
  return os;
}

bool BufferTouchPattern::IsOverwrittenWithoutEffect(const tir::BufferStore& store) const {
  bool write_occurred = false;

  for (auto it = touch_points_.begin(); it != touch_points_.end(); it++) {
    if (it->node.same_as(store)) {
      write_occurred = true;
      if (!IsOverwrittenWithoutEffect(it)) {
        return false;
      }
    }
  }

  ICHECK(write_occurred) << "BufferStore did not occur within analyzed statement";

  return true;
}

bool BufferTouchPattern::IsOverwrittenWithoutEffect(
    std::vector<BufferTouch>::const_iterator write_iter) const {
  for (auto it = write_iter + 1; it != touch_points_.end(); it++) {
    // If the write_iter was a subset of another write, then it was entirely overwritten.
    if (it->touch_type == BufferTouch::AccessType::Write && write_iter->IsSubsetOf(*it)) {
      return true;
    }
    // If the written values are later read out, then this write had an effect.
    if (it->touch_type == BufferTouch::AccessType::Read && it->IsSubsetOf(*write_iter)) {
      return false;
    }
  }

  return false;
}

PrimExpr BufferTouchPattern::SimplifyInContext(PrimExpr expr, const tir::Stmt& context,
                                               Analyzer* analyzer) const {
  size_t context_index = [&]() {
    auto it = context_lookup_.find(context.get());
    ICHECK(it != context_lookup_.end())
        << "Context did not occur in the Stmt provided to BufferTouchPattern's constructor";
    return it->second;
  }();

  BufferConstraintSubstituter mutator(*this, context_index, analyzer);
  expr = mutator(expr);

  PrimExpr constraint = Bool(true);
  for (const auto& known : non_buffer_assumptions_) {
    constraint = constraint && known;
  }
  With<ConstraintContext> constraint_context(analyzer, constraint);
  return analyzer->Simplify(expr);
}

Optional<PrimExpr> BufferTouchPattern::KnownValue(const tir::BufferStore& store) const {
  Array<PrimExpr> values;

  for (auto it = touch_points_.rbegin(); it != touch_points_.rend(); it++) {
    if (it->node.same_as(store)) {
      if (auto opt = KnownValue(it, store->indices)) {
        values.push_back(opt.value());
      } else {
        // If a store based on this statement doesn't have a known
        // value, then the store overall doesn't have a known value.
        return NullOpt;
      }
    }
  }

  // For the store to have a known value, all touches resulting from
  // this statement must result in the same value.
  //
  // TODO: Handle multiple access from a single statement
  // (e.g. start/finish of while loop) that may have the same result.
  // Should attempt to prove that each touch was preceded by the same
  // known value.
  if (values.size() == 1) {
    return values[0];
  } else {
    return NullOpt;
  }
}

Optional<PrimExpr> BufferTouchPattern::KnownValue(const tir::BufferLoad& load) const {
  return NullOpt;
}

Optional<PrimExpr> BufferTouchPattern::KnownValue(
    std::vector<BufferTouch>::const_reverse_iterator access_iter,
    const Array<PrimExpr>& indices) const {
  for (auto it = access_iter + 1; it != touch_points_.rend(); it++) {
    // If a previous write touched the same indices, then we can use
    // the recorded values at those indices.
    if (it->touch_type == BufferTouch::AccessType::Write && access_iter->IsSubsetOf(*it)) {
      return it->known_value(indices);
    }
  }
  return NullOpt;
}

void BufferTouchPattern::RemoveTouches(const tir::BufferStore& store) {
  touch_points_.erase(std::remove_if(touch_points_.begin(), touch_points_.end(),
                                     [&](const auto& touch) { return touch.node.same_as(store); }));
  // TODO: Update context_lookup_
}

}  // namespace arith
}  // namespace tvm
