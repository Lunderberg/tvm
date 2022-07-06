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
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <sstream>
#include <vector>

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

  PrimExpr target_expression = other(vars_as_primexpr);

  arith::Analyzer analyzer;

  With<ConstraintContext> this_params(&analyzer, this->FreeParameterConstraints());
  With<ConstraintContext> other_params(&analyzer, other.FreeParameterConstraints());
  With<ConstraintContext> constraint(&analyzer, expression_);

  return analyzer.CanProve(target_expression);
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
  static std::vector<BufferTouch> Extract(const Stmt& stmt) {
    BufferTouchExtractor extractor;
    extractor(stmt);
    return extractor.touch_points_;
  }

 private:
  using Parent = IRVisitorWithAnalyzer;
  using Parent::VisitExpr_;
  using Parent::VisitStmt_;

  void VisitStmt(const Stmt& stmt) override {
    Stmt prev_stmt = current_stmt_;
    current_stmt_ = stmt;
    Parent::VisitStmt(stmt);
    current_stmt_ = prev_stmt;
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
    VisitAccess(GetRef<BufferStore>(op), BufferTouch::AccessType::Write);
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
  void VisitAccess(const BufferAccess& node, BufferTouch::AccessType touch_type) {
    auto index_variables = MakeIndexVariables(node->indices);
    auto transform = SolveForBufferIndices(index_variables, node->indices);
    auto predicate = CurrentPredicate(index_variables, node->indices, transform);
    auto known_value = KnownValue(node, index_variables, transform);
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

  Predicate CurrentPredicate(const Array<Var>& index_variables,
                             const Array<PrimExpr>& index_expressions,
                             const IntConstraintsTransform& transform) {
    PrimExpr predicate = Bool(true);
    for (const auto& condition : conditions_) {
      predicate = predicate && condition;
    }
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

  ParametrizedExpression KnownValue(const BufferStore& store, const Array<Var>& index_variables,
                                    const IntConstraintsTransform& transform) {
    PrimExpr value = store->value;
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

  ParametrizedExpression KnownValue(const BufferLoad& load, const Array<Var>& index_variables,
                                    const IntConstraintsTransform& transform) {
    // TODO: Track if a buffer load has a known value
    return ParametrizedExpression(index_variables, PrimExpr());
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
};

BufferTouchPattern::BufferTouchPattern(const tir::Stmt& stmt)
    : touches_(BufferTouchExtractor::Extract(stmt)) {}

std::ostream& operator<<(std::ostream& os, const BufferTouchPattern& pattern) {
  os << "Touch pattern contains " << pattern.touches_.size() << " touches."
     << (pattern.touches_.size() ? "\n" : "");
  for (size_t i = 0; i < pattern.touches_.size(); i++) {
    os << "\t"
       << "Touch[" << i << "] = " << pattern.touches_[i];
    if (i + 1 < pattern.touches_.size()) {
      os << "\n";
    }
  }
  return os;
}

bool BufferTouchPattern::IsOverwrittenWithoutEffect(const tir::BufferStore& store) const {
  bool write_occurred = false;

  for (auto it = touches_.begin(); it != touches_.end(); it++) {
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
  for (auto it = write_iter + 1; it != touches_.end(); it++) {
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

Optional<PrimExpr> BufferTouchPattern::KnownValue(const tir::BufferStore& store) const {
  Array<PrimExpr> values;

  for (auto it = touches_.rbegin(); it != touches_.rend(); it++) {
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
  for (auto it = access_iter + 1; it != touches_.rend(); it++) {
    // If a previous write touched the same indices, then we can use
    // the recorded values at those indices.
    if (it->touch_type == BufferTouch::AccessType::Write && access_iter->IsSubsetOf(*it)) {
      return it->known_value(indices);
    }
  }
  return NullOpt;
}

}  // namespace arith
}  // namespace tvm
