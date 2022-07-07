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

 private:
  PrimExpr CurrentScopeConstraints() const;

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
      PrimExpr predicate = this->predicate(indices);
      if (!analyzer->CanProve(predicate)) {
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
  global_constraints_.push_back(logical_not(CurrentScopeConstraints()) || assumption);
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

}  // namespace arith
}  // namespace tvm
