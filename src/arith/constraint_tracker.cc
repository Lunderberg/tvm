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
 * \file constraint_tracker.cc
 * \brief Utility for tracking currently active constraints
 */
#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_solver.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "buffer_touch_pattern.h"
#include "constraint_extract.h"

namespace tvm {
namespace arith {

class ConstraintTracker::Impl {
 public:
  explicit Impl(Analyzer* parent) : parent_(parent) {}

  // Enter a scoped constraint.  This constraint may be exited by
  // calling the provided callback.
  std::function<void()> EnterScopedConstraint(PrimExpr constraint);

  // Assume the statement is true, given any currently active scoped
  // constraints.
  void Assume(PrimExpr constraint);

  // Return a collection of expressions that are known to be true,
  // containing any scoped and global constraints.
  std::vector<PrimExpr> CurrentlyKnown() const;

  // Provide a known value at the specified indices in a buffer,
  // removing any previous assumptions about the value at these
  // indices.
  void KnownBufferValue(tir::Buffer buf, Array<PrimExpr> indices, PrimExpr value);

  // Return the value known to be stored in the specified buffer, at
  // the specified indices.  If the value is not known, returns
  // NullOpt.
  Optional<PrimExpr> KnownBufferValue(tir::Buffer buf, Array<PrimExpr> indices) const;

  /* \brief Enter a context in which the buffer values may be used for simplifications
   *
   * To avoid unexpected conflicts or accidental inlining,
   * simplifications that use a known buffer value must be explicitly
   * enabled.  Currently, they are enabled when attempting to prove an
   * expression, but not for general simplifications.
   *
   * \returns A callback to restore the previous state
   */
  std::function<void()> EnableBufferValueSimplifications();

 private:
  /* \brief An expression that must be true based on scope-implied constraints. */
  PrimExpr CurrentScopeConstraints() const;

  void AssumeIndependentConstraint(PrimExpr assumption);

  /* \brief Perform any context-free simplifications
   *
   * Deliberately not using the parent analyzer, because we don't want
   * to simplify out any scoped constraints already provided.
   */
  static PrimExpr Simplify(PrimExpr expr);

  struct Constraint {
    Constraint(PrimExpr expr);

    PrimExpr expr;
    PrimExpr negation;
    tir::CallEffectKind side_effect;
  };

  struct BufferConstraint {
    BufferConstraint(tir::Buffer buf, Predicate predicate, ParametrizedExpression value)
        : buf(buf), predicate(predicate), value(value) {}

    Optional<PrimExpr> KnownValue(const tir::Buffer& buf, const Array<PrimExpr>& indices,
                                  Analyzer* analyzer) const {
      if (!buf.same_as(this->buf)) {
        return NullOpt;
      }
      if (!predicate.CanProve(indices, analyzer)) {
        return NullOpt;
      }

      PrimExpr value = this->value(indices);
      if (value.defined()) {
        return value;
      } else {
        return NullOpt;
      }
    }

    tir::Buffer buf;
    Predicate predicate;
    ParametrizedExpression value;
  };

  Analyzer* parent_;
  std::vector<Constraint> scoped_constraints_;
  std::vector<Constraint> global_constraints_;
  std::vector<BufferConstraint> scoped_buffer_constraints_;
  std::vector<BufferConstraint> global_buffer_constraints_;

  // If enabled, allow simplifications that rely on propagating the
  // buffer value.  Currently disabled by default in order to
  // gradually test the implications of this change.
  bool allow_buffer_value_simplifications_{false};
};

///////////////////////////////////////////////////////////////////
// Exposing ConstraintTracker::Impl methods to ConstraintTracker //
///////////////////////////////////////////////////////////////////

ConstraintTracker::ConstraintTracker(Analyzer* parent) {
  impl_ = new ConstraintTracker::Impl(parent);
}

ConstraintTracker::~ConstraintTracker() {
  if (impl_) {
    delete impl_;
  }
}

std::function<void()> ConstraintTracker::EnterScopedConstraint(PrimExpr constraint) {
  return impl_->EnterScopedConstraint(std::move(constraint));
}

void ConstraintTracker::Assume(PrimExpr constraint) { impl_->Assume(std::move(constraint)); }

void ConstraintTracker::KnownBufferValue(tir::Buffer buf, Array<PrimExpr> indices, PrimExpr value) {
  impl_->KnownBufferValue(std::move(buf), std::move(indices), std::move(value));
}

Optional<PrimExpr> ConstraintTracker::KnownBufferValue(tir::Buffer buf, Array<PrimExpr> indices) {
  return impl_->KnownBufferValue(std::move(buf), std::move(indices));
}

std::vector<PrimExpr> ConstraintTracker::CurrentlyKnown() const { return impl_->CurrentlyKnown(); }

std::function<void()> ConstraintTracker::EnableBufferValueSimplifications() {
  return impl_->EnableBufferValueSimplifications();
}

///////////////////////////////////////////////////////////////////
//        Implementation of ConstraintTracker::Impl methods      //
///////////////////////////////////////////////////////////////////

PrimExpr ConstraintTracker::Impl::Simplify(PrimExpr expr) {
  arith::Analyzer analyzer;
  return analyzer.rewrite_simplify(std::move(expr));
}

ConstraintTracker::Impl::Constraint::Constraint(PrimExpr expr)
    : expr(expr), side_effect(tir::SideEffect(expr)) {
  negation = Simplify(tir::Not(expr));
}

PrimExpr ConstraintTracker::Impl::CurrentScopeConstraints() const {
  PrimExpr constraint = Bool(true);
  for (const auto& scoped_constraint : scoped_constraints_) {
    constraint = constraint && scoped_constraint.expr;
  }
  return constraint;
}

void ConstraintTracker::Impl::Assume(PrimExpr assumption) {
  for (const auto& expr : ExtractConstraints(assumption, false)) {
    AssumeIndependentConstraint(expr);
  }
}

void ConstraintTracker::Impl::AssumeIndependentConstraint(PrimExpr assumption) {
  PrimExpr predicate = CurrentScopeConstraints();
  std::vector<PrimExpr> buffer_exprs;
  for (const auto& expr : ExtractComponents(assumption)) {
    auto side_effect = tir::SideEffect(expr);
    if (side_effect <= tir::CallEffectKind::kPure) {
      // Pulling out portions of the assumption that do not depend on a buffer value
      //
      // if i < 3: T.assume(buf[i] == value)
      // T.assume(i>=3 or buf[i] == value)
      predicate = predicate && logical_not(expr);
    } else if (side_effect == tir::CallEffectKind::kReadState) {
      buffer_exprs.push_back(expr);
    } else {
      LOG(FATAL) << "Assumption must be pure or read-only";
    }
  }

  if (buffer_exprs.empty()) {
    global_constraints_.push_back(Simplify(logical_not(predicate)));
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

  KnownBufferValue(load->buffer, load->indices, value);
}

void ConstraintTracker::Impl::KnownBufferValue(tir::Buffer buf, Array<PrimExpr> index_expressions,
                                               PrimExpr value) {
  // The buffer constraint is in terms of the buffer indices, in order
  // to be substituted in when used in a later context.

  Array<Var> index_variables;
  Array<PrimExpr> relations;
  Array<Var> to_solve_for;
  Map<Var, Range> ranges;

  for (size_t i = 0; i < index_expressions.size(); i++) {
    PrimExpr index_expr = parent_->rewrite_simplify(index_expressions[i]);

    std::ostringstream os;
    os << "i_" << i;
    Var index_var(os.str(), index_expr.dtype());

    index_variables.push_back(index_var);
    relations.push_back(index_var == index_expr);

    for (auto& loop_var : tir::UndefinedVars(index_expr)) {
      if (!ranges.count(loop_var)) {
        IntSet var_range = parent_->int_set(loop_var);
        to_solve_for.push_back(loop_var);
        ranges.Set(loop_var, Range(var_range.min(), var_range.max()));
      }
    }
  }

  IntConstraints system(to_solve_for, ranges, relations);
  IntConstraintsTransform transform = arith::SolveLinearEquations(system);

  PrimExpr predicate_expr = Substitute(CurrentScopeConstraints(), transform->src_to_dst);
  Predicate predicate = Predicate(index_variables, predicate_expr, transform->dst->ranges);

  // TODO: Clear previous buffer predicates

  PrimExpr new_value_expr = Substitute(value, transform->src_to_dst);

  // Only track known values that are a constant expression in terms
  // of buffer indices.  Do not track known values that would require
  // loading from a different buffer.
  if (tir::SideEffect(new_value_expr) > tir::CallEffectKind::kPure) {
    return;
  }

  ParametrizedExpression constraint_value(index_variables, new_value_expr);

  global_buffer_constraints_.emplace_back(buf, predicate, constraint_value);
}

Optional<PrimExpr> ConstraintTracker::Impl::KnownBufferValue(tir::Buffer buf,
                                                             Array<PrimExpr> indices) const {
  if (!allow_buffer_value_simplifications_) {
    return NullOpt;
  }

  for (const auto& constraint : global_buffer_constraints_) {
    if (Optional<PrimExpr> value = constraint.KnownValue(buf, indices, parent_)) {
      return value;
    }
  }

  return NullOpt;
}

std::function<void()> ConstraintTracker::Impl::EnterScopedConstraint(PrimExpr constraint) {
  size_t prev_scoped_constraints = scoped_constraints_.size();

  for (const PrimExpr& subconstraint : ExtractConstraints(constraint)) {
    scoped_constraints_.push_back(subconstraint);
  }

  size_t new_scoped_constraints = scoped_constraints_.size();
  return [this, prev_scoped_constraints, new_scoped_constraints]() {
    ICHECK_EQ(scoped_constraints_.size(), new_scoped_constraints);
    scoped_constraints_.erase(scoped_constraints_.begin() + prev_scoped_constraints,
                              scoped_constraints_.end());
  };
}

std::vector<PrimExpr> ConstraintTracker::Impl::CurrentlyKnown() const {
  std::vector<PrimExpr> output;

  auto process = [&](const auto& constraint) {
    if (constraint.side_effect <= tir::CallEffectKind::kPure) {
      output.push_back(constraint.expr);
      output.push_back(tir::Not(constraint.negation));
    }
  };

  for (const auto& constraint : global_constraints_) {
    process(constraint);
  }
  for (const auto& constraint : scoped_constraints_) {
    process(constraint);
  }
  return output;
}

std::function<void()> ConstraintTracker::Impl::EnableBufferValueSimplifications() {
  bool current_state = allow_buffer_value_simplifications_;
  allow_buffer_value_simplifications_ = true;
  return [this, current_state]() { allow_buffer_value_simplifications_ = current_state; };
}

}  // namespace arith
}  // namespace tvm
