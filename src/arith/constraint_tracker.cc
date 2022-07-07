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
#include <tvm/tir/analysis.h>

#include "constraint_extract.h"

namespace tvm {
namespace arith {

class ConstraintTracker::Impl {
 public:
  // Enter a scoped constraint.  This constraint may be exited by
  // calling the provided callback.
  std::function<void()> EnterScopedConstraint(PrimExpr constraint);

  // Assume the statement is true, given any currently active scoped
  // constraints.
  void Assume(PrimExpr constraint);

  // Return a collection of expressions that are known to be true,
  // containing any scoped and global constraints.
  std::vector<PrimExpr> CurrentlyKnown() const;

 private:
  struct Constraint {
    Constraint(PrimExpr expr);

    PrimExpr expr;
    tir::CallEffectKind side_effect;
  };

  std::vector<Constraint> scoped_constraints_;
  std::vector<Constraint> global_constraints_;
};

ConstraintTracker::ConstraintTracker() { impl_ = new ConstraintTracker::Impl(); }

ConstraintTracker::~ConstraintTracker() {
  if (impl_) {
    delete impl_;
  }
}

std::function<void()> ConstraintTracker::EnterScopedConstraint(PrimExpr constraint) {
  return impl_->EnterScopedConstraint(std::move(constraint));
}

void ConstraintTracker::Assume(PrimExpr constraint) { impl_->Assume(std::move(constraint)); }

std::vector<PrimExpr> ConstraintTracker::CurrentlyKnown() const { return impl_->CurrentlyKnown(); }

ConstraintTracker::Impl::Constraint::Constraint(PrimExpr expr)
    : expr(expr), side_effect(tir::SideEffect(expr)) {}

void ConstraintTracker::Impl::Assume(PrimExpr constraint) {
  global_constraints_.push_back(constraint);
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
  for (const auto& constraint : global_constraints_) {
    output.push_back(constraint.expr);
  }
  for (const auto& constraint : scoped_constraints_) {
    output.push_back(constraint.expr);
  }
  return output;
}

}  // namespace arith
}  // namespace tvm
