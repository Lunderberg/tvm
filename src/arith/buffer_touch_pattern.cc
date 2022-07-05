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

Predicate::Predicate(Array<Var> parameter_vars, PrimExpr expression,
                     Map<Var, Range> free_parameters)
    : parameter_vars_(parameter_vars), free_parameters_(free_parameters), expression_(expression) {}

PrimExpr Predicate::operator()(Array<PrimExpr> args) const {
  ICHECK_EQ(parameter_vars_.size(), args.size())
      << "Expression was defined as having " << parameter_vars_.size()
      << " parameters, but received " << args.size() << " arguments.";

  Map<tir::Var, PrimExpr> var_map;
  for (size_t i = 0; i < args.size(); i++) {
    var_map.Set(parameter_vars_[i], args[i]);
  }

  return Substitute(expression_, var_map);
}

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
  os << "predicate(" << expr.parameter_vars_ << " = " << expr.expression_;
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
                         Optional<PrimExpr> known_value, ObjectRef node)
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
  return os << "BufferTouch(" << tp.buffer->name << ", " << touch_type << ", " << tp.predicate
            << ")";
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
    Optional<PrimExpr> known_value = NullOpt;
    auto predicate = CurrentPredicate(node->indices);
    touch_points_.push_back(BufferTouch(node->buffer, predicate, touch_type, known_value, node));
  }

  std::function<void()> EnterConstraint(const PrimExpr& constraint) override {
    conditions_.push_back(constraint);

    return [this]() {
      ICHECK(conditions_.size()) << "Internal error: Each condition should only be popped once.";
      conditions_.pop_back();
    };
  }

  Predicate CurrentPredicate(const Array<PrimExpr>& indices) {
    PrimExpr predicate = Bool(true);
    for (const auto& condition : conditions_) {
      predicate = predicate && condition;
    }
    predicate = Substitute(predicate, let_bindings_using_loop_);

    Array<Var> index_variables;

    Map<Var, Range> ranges;
    Array<PrimExpr> relations;

    for (size_t i = 0; i < indices.size(); i++) {
      PrimExpr index = indices[i];

      std::stringstream ss;
      ss << "i_" << i;
      Var var(ss.str());
      index_variables.push_back(var);
      relations.push_back(var == Substitute(index, let_bindings_using_loop_));

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

    predicate = Substitute(predicate, solution->src_to_dst);
    predicate = analyzer_.Simplify(predicate);

    ICHECK(!UsesLoopVar(predicate))
        << "Internal error: Loop variable still used after substituting out the loop variable";

    return Predicate(index_variables, predicate, solution->dst->ranges);
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

Optional<PrimExpr> BufferTouchPattern::KnownValue(const tir::BufferLoad& load) const {
  return NullOpt;
}

Optional<PrimExpr> BufferTouchPattern::KnownValue(const tir::BufferStore& store) const {
  return NullOpt;
}

}  // namespace arith
}  // namespace tvm
