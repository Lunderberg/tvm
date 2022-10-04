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

Predicate::Predicate(Array<tir::Var> parameter_vars, Optional<PrimExpr> expression,
                     Map<tir::Var, Range> free_parameters)
    : ParametrizedExpression(parameter_vars, expression), free_parameters_(free_parameters) {
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

  With<ConstraintContext> this_params(analyzer, this->FreeParameterConstraints());
  With<ConstraintContext> other_params(analyzer, other.FreeParameterConstraints());
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

  With<ConstraintContext> this_params(analyzer, this->FreeParameterConstraints());
  With<ConstraintContext> other_params(analyzer, other.FreeParameterConstraints());
  With<ConstraintContext> constraint(analyzer, expression_.value());

  return analyzer->CanProve(logical_not(other_predicate));
}

Predicate Predicate::Difference(const Predicate& other, Analyzer* analyzer) const {
  ICHECK_EQ(parameter_vars_.size(), other.parameter_vars_.size())
      << "Predicates must be over the same number of parameters to be comparable";

  if (!IsDefined() || !other.IsDefined()) {
    return Predicate(parameter_vars_, NullOpt, {});
  }

  PrimExpr other_predicate = other(parameter_vars_).value();

  With<ConstraintContext> this_params(analyzer, this->FreeParameterConstraints());
  With<ConstraintContext> other_params(analyzer, other.FreeParameterConstraints());

  // PrimExpr new_predicate_expr = analyzer.Simplify(expression_.value() && !other_predicate);
  // new_predicate_expr = ConvertToAndOfOrs(new_predicate_expr);
  // new_predicate_expr = analyzer.Simplify(new_predicate_expr);

  PrimExpr new_predicate_expr =
      SimplifyAsAndOfOrs(expression_.value() && !other_predicate, analyzer);

  Map<tir::Var, Range> new_free_params = free_parameters_;
  for (const auto& pair : other.free_parameters_) {
    new_free_params.Set(pair.first, pair.second);
  }

  return Predicate(parameter_vars_, new_predicate_expr, new_free_params);
}

Predicate Predicate::Intersection(const Predicate& other, Analyzer* analyzer) const {
  ICHECK_EQ(parameter_vars_.size(), other.parameter_vars_.size())
      << "Predicates must be over the same number of parameters to be comparable";

  if (!IsDefined() || !other.IsDefined()) {
    return Predicate(parameter_vars_, NullOpt, {});
  }

  if (this->IsSubsetOf(other, analyzer)) {
    return (*this);
  } else if (other.IsSubsetOf(*this, analyzer)) {
    return other;
  }

  PrimExpr other_predicate = other(parameter_vars_).value();

  With<ConstraintContext> this_params(analyzer, this->FreeParameterConstraints());
  With<ConstraintContext> other_params(analyzer, other.FreeParameterConstraints());

  PrimExpr new_predicate_expr = analyzer->Simplify(expression_.value() && other_predicate);

  Map<tir::Var, Range> new_free_params = free_parameters_;
  for (const auto& pair : other.free_parameters_) {
    new_free_params.Set(pair.first, pair.second);
  }

  return Predicate(parameter_vars_, new_predicate_expr, new_free_params);
}

Predicate Predicate::Union(const Predicate& other, Analyzer* analyzer) const {
  ICHECK_EQ(parameter_vars_.size(), other.parameter_vars_.size())
      << "Predicates must be over the same number of parameters to be comparable";

  if (!IsDefined() || !other.IsDefined()) {
    return Predicate(parameter_vars_, NullOpt, {});
  }

  // if (this->IsSubsetOf(other, analyzer)) {
  //   return (*this);
  // } else if (other.IsSubsetOf(*this, analyzer)) {
  //   return other;
  // }

  PrimExpr other_predicate = other(parameter_vars_).value();

  With<ConstraintContext> this_params(analyzer, this->FreeParameterConstraints());
  With<ConstraintContext> other_params(analyzer, other.FreeParameterConstraints());

  // PrimExpr new_predicate_expr = analyzer->Simplify(expression_.value() || other_predicate);
  PrimExpr new_predicate_expr =
      SimplifyAsAndOfOrs(expression_.value() || other_predicate, analyzer);

  Map<tir::Var, Range> new_free_params;

  std::unordered_set<const VarNode*> undefined;
  auto undefined_var_arr = UndefinedVars(new_predicate_expr);
  for (const auto& var : undefined_var_arr) {
    undefined.insert(var.get());
  }

  for (const auto& pair : free_parameters_) {
    if (undefined.count(pair.first.get())) {
      new_free_params.Set(pair.first, pair.second);
    }
  }

  for (const auto& pair : other.free_parameters_) {
    if (undefined.count(pair.first.get())) {
      new_free_params.Set(pair.first, pair.second);
    }
  }

  return Predicate(parameter_vars_, new_predicate_expr, new_free_params);
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

  With<ConstraintContext> context(analyzer, FreeParameterConstraints());

  // TODO: Are the extra simplification rounds necessary?
  PrimExpr expr = analyzer->Simplify(expression_.value(), 5);

  // Remove any free parameters that are no longer needed.  Using
  // Map::erase instead of constructing a new Map, to allow
  // CopyOnWrite.
  if (free_parameters_.size()) {
    std::unordered_set<const VarNode*> undefined;
    for (const auto& var : UndefinedVars(expr)) {
      undefined.insert(var.get());
    }

    Array<Var> to_remove;
    for (const auto& pair : free_parameters_) {
      if (!undefined.count(pair.first.get())) {
        to_remove.push_back(pair.first);
      }
    }

    for (const auto& var : to_remove) {
      free_parameters_.erase(var);
    }
  }

  expression_ = expr;
}

bool Predicate::CanProve(Array<PrimExpr> args, Analyzer* analyzer) const {
  With<ConstraintContext> constraint(analyzer, FreeParameterConstraints());
  Optional<PrimExpr> expr = (*this)(std::move(args));
  return expr && analyzer->CanProve(expr.value());
}

bool Predicate::CanDisprove(Array<PrimExpr> args, Analyzer* analyzer) const {
  With<ConstraintContext> constraint(analyzer, FreeParameterConstraints());
  Optional<PrimExpr> expr = (*this)(std::move(args));
  return expr && analyzer->CanProve(!expr.value());
}

bool Predicate::IsAlwaysFalse() const {
  Analyzer analyzer;
  With<ConstraintContext> constraint(&analyzer, FreeParameterConstraints());
  return expression_ && analyzer.CanProve(!expression_.value());
}

Predicate Predicate::WithoutFreeParameters() const {
  if (!expression_) {
    return *this;
  }

  PrimExpr expr = NarrowExpressionToTrue(expression_.value(), free_parameters_);
  return Predicate(parameter_vars_, expr, {});
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
                         ParametrizedExpression known_value, Array<PrimExpr> original_indices,
                         Map<Var, PrimExpr> loop_var_to_axis_var, ObjectRef node)
    : buffer(buffer),
      predicate(predicate),
      touch_type(touch_type),
      known_value(known_value),
      original_indices(original_indices),
      loop_var_to_axis_var(loop_var_to_axis_var),
      node(node) {}

bool BufferTouch::IsSubsetOf(const BufferTouch& other, Analyzer* analyzer) const {
  if (!this->buffer.same_as(other.buffer)) {
    return false;
  } else {
    return this->predicate.IsSubsetOf(other.predicate, analyzer);
  }
}

bool BufferTouch::ProvablyCrossLoopIndependent(const BufferTouch& preceding_in_body,
                                               const Var& loop_var, Analyzer* analyzer) const {
  return false;
  if (touch_type != AccessType::Write ||
      (preceding_in_body.touch_type != AccessType::Read &&
       preceding_in_body.touch_type != AccessType::Assume) ||
      !buffer.same_as(preceding_in_body.buffer) ||
      predicate.IsDistinctFrom(preceding_in_body.predicate, analyzer)) {
    return true;
  }

  ICHECK_EQ(original_indices.size(), preceding_in_body.original_indices.size());

  Var delta("delta", loop_var.dtype());
  PrimExpr prev_iter = loop_var - delta;
  With<ConstraintContext> context(analyzer, 0 < delta);

  for (size_t i = 0; i < original_indices.size(); i++) {
    const PrimExpr& write_index = original_indices[i];
    PrimExpr read_index =
        Substitute(preceding_in_body.original_indices[i], [&](const Var& var) -> PrimExpr {
          if (var.same_as(loop_var)) {
            return prev_iter;
          } else {
            return var;
          }
        });

    if (!analyzer->CanProve(read_index != write_index)) {
      return false;
    }
  }

  return true;
}

std::ostream& operator<<(std::ostream& os, const BufferTouch& tp) {
  auto touch_type = (tp.touch_type == BufferTouch::AccessType::Read)     ? "read"
                    : (tp.touch_type == BufferTouch::AccessType::Write)  ? "write"
                    : (tp.touch_type == BufferTouch::AccessType::Assume) ? "assume"
                                                                         : "???";
  os << "BufferTouch(" << tp.buffer->name << ", " << touch_type << ", " << tp.predicate;
  if (tp.known_value.IsDefined()) {
    os << ", known_value = " << tp.known_value;
  }

  os << ")";
  return os;
}

class BufferConstraintSubstituter : public IRMutatorWithAnalyzer {
 public:
  using Parent = IRMutatorWithAnalyzer;

  BufferConstraintSubstituter(const BufferTouchPattern& touch_pattern, size_t context_index,
                              Analyzer* analyzer)
      : Parent(analyzer), touch_pattern_(touch_pattern), context_index_(context_index) {}

  BufferTouch SimplifyTouch(BufferTouch touch) {
    if (touch.known_value.expression_) {
      PrimExpr constraint_expr =
          touch.predicate(touch.known_value.parameter_vars_).value_or(Bool(true));
      With<ConstraintContext> constraint(analyzer_, constraint_expr);
      With<ConstraintContext> free_params(analyzer_, touch.predicate.FreeParameterConstraints());
      PrimExpr simplified_known_value = (*this)(touch.known_value.expression_.value());
    }
    return touch;
  }

  Optional<PrimExpr> WithoutBufferLoad(const PrimExpr& expr) {
    all_buffer_loads_removed_ = true;
    PrimExpr modified = (*this)(expr);
    if (all_buffer_loads_removed_) {
      return modified;
    } else {
      return NullOpt;
    }
  }

  using Parent::VisitExpr_;

  PrimExpr VisitExpr_(const BufferLoadNode* op) override {
    const auto& context_block = touch_pattern_.control_flow_[context_index_];
    if (auto opt = FollowPredecessorBlock(context_block, op, Bool(true))) {
      return opt.value();
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  Optional<PrimExpr> FollowPredecessorBlock(const BufferTouchPattern::ControlFlowBlock& block,
                                            const BufferLoadNode* op, PrimExpr access_predicate) {
    IncreaseDepth temp(this);

    Optional<PrimExpr> result = NullOpt;
    for (const auto& predecessor_edge : block.predecessors) {
      size_t predecessor_index = predecessor_edge.from_index;
      IncreaseDepth temp2(this);
      const auto& predecessor = touch_pattern_.control_flow_[predecessor_index];
      auto opt_value = ApplyKnownValue(predecessor, op, access_predicate);
      if (!opt_value) {
        return GetRef<PrimExpr>(op);
      } else if (!result) {
        result = opt_value;
      } else if (!analyzer_->CanProveEqual(opt_value.value(), result.value())) {
        return NullOpt;
      }
    }

    return result;
  }

  Optional<PrimExpr> ApplyKnownValue(const BufferTouchPattern::ControlFlowBlock& block,
                                     const BufferLoadNode* op, PrimExpr access_predicate) {
    IncreaseDepth temp(this);

    auto implies = [this](const PrimExpr& known, const PrimExpr& conjecture) -> bool {
      With<ConstraintContext> constraint(analyzer_, known);
      return analyzer_->CanProve(conjecture);
    };

    std::vector<std::pair<PrimExpr, PrimExpr>> known_subregion;
    PrimExpr free_parameter_constraints = Bool(true);

    for (auto it = block.touch_points.rbegin(); it != block.touch_points.rend(); it++) {
      const BufferTouch& touch = *it;

      if (!op->buffer.same_as(touch.buffer)) {
        // This is a different buffer, so continue searching.
        continue;
      } else if (!touch.predicate.IsDefined() &&
                 touch.touch_type == BufferTouch::AccessType::Write) {
        // This buffer touch occurred at an unknown location, which
        // could have overwritten the values we are looking for
        break;
      } else if (!touch.predicate.IsDefined() &&
                 touch.touch_type == BufferTouch::AccessType::Read) {
        // This buffer touch occurred at an unknown location, but
        // didn't do anything at that location.
        continue;
      }

      PrimExpr touch_predicate = analyzer_->Simplify(touch.predicate(op->indices).value());

      // With<ConstraintContext> isn't safe to use in a std::vector,
      // so instead we collect a single expression with all the extra
      // constraints.
      free_parameter_constraints =
          free_parameter_constraints && touch.predicate.FreeParameterConstraints();
      With<ConstraintContext> constraint(analyzer_, free_parameter_constraints);

      if (touch.known_value.IsDefined() && implies(access_predicate, touch_predicate)) {
        // This access resulted in a known value, return it.
        PrimExpr value = touch.known_value(op->indices).value();
        // {
        //   With<ConstraintContext> constraint(analyzer_, access_predicate);
        //   OverrideBlockIndex backtrack(this, block.index);
        //   value = this->VisitExpr(value);
        // }
        for (auto it = known_subregion.rbegin(); it != known_subregion.rend(); it++) {
          value = if_then_else(it->first, it->second, value);
        }
        return value;
      } else if (implies(access_predicate, logical_not(touch_predicate))) {
        // The previous access didn't change the values we're
        // interested in, so continue searching.
        continue;
      } else if (touch.known_value.IsDefined()) {
        // The previous access resulted in a known value, but only for
        // some of the indices we are interested in.  It's still
        // possible that the same value
        PrimExpr value = touch.known_value(op->indices).value();
        // {
        //   With<ConstraintContext> constraint(analyzer_, access_predicate && touch_predicate);
        //   OverrideBlockIndex backtrack(this, block.index);
        //   value = this->VisitExpr(value);
        // }
        known_subregion.push_back({touch_predicate, value});
        access_predicate = access_predicate && !touch_predicate;
      } else if (touch.touch_type == BufferTouch::AccessType::Read ||
                 touch.touch_type == BufferTouch::AccessType::Assume) {
        // This access didn't change the buffer's contents, so
        // continue backtracking.
      } else {
        // This BufferTouch writes values to the buffer that we might
        // use, and we don't know what those values are.  Therefore,
        // cannot simplify out the buffer access.
        break;
      }
    }

    // All known BufferTouch were examined without being able to
    // determine the value of this buffer load.  Therefore, continue
    // to predecessor blocks.
    // With<ConstraintContext> constraint(analyzer_, free_parameter_constraints);
    Optional<PrimExpr> from_predecessor = FollowPredecessorBlock(block, op, access_predicate);

    if (from_predecessor) {
      PrimExpr value = from_predecessor.value();
      for (auto it = known_subregion.rbegin(); it != known_subregion.rend(); it++) {
        value = if_then_else(it->first, it->second, value);
      }
      return value;
    }

    all_buffer_loads_removed_ = false;
    return NullOpt;
  }

  struct OverrideBlockIndex {
    OverrideBlockIndex(BufferConstraintSubstituter* self, size_t index)
        : self(self), saved_index(self->context_index_) {
      self->context_index_ = index;
    }
    ~OverrideBlockIndex() { self->context_index_ = saved_index; }
    OverrideBlockIndex(const OverrideBlockIndex& other) = delete;
    OverrideBlockIndex(OverrideBlockIndex&& other) = delete;
    OverrideBlockIndex& operator=(const OverrideBlockIndex& other) = delete;
    OverrideBlockIndex& operator=(OverrideBlockIndex&& other) = delete;
    BufferConstraintSubstituter* self;
    size_t saved_index;
  };

  const BufferTouchPattern& touch_pattern_;
  size_t context_index_;
  bool all_buffer_loads_removed_{true};

  int depth{-1};
  struct IncreaseDepth {
    IncreaseDepth(BufferConstraintSubstituter* self) : self(self) { self->depth++; }
    ~IncreaseDepth() { self->depth--; }
    IncreaseDepth(const IncreaseDepth& other) = delete;
    IncreaseDepth(IncreaseDepth&& other) = delete;
    IncreaseDepth& operator=(const IncreaseDepth& other) = delete;
    IncreaseDepth& operator=(IncreaseDepth&& other) = delete;
    BufferConstraintSubstituter* self;
  };
};

class BufferConstraintApply : public IRMutatorWithAnalyzer {
 public:
  using Parent = IRMutatorWithAnalyzer;

  BufferConstraintApply(const std::vector<BufferTouchPattern::BufferConstraint>& knowns,
                        Analyzer* analyzer)
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

      PrimExpr predicate = known.predicate(indices).value();
      std::optional<With<ConstraintContext>> context;
      if (lane_var.defined()) {
        Var lanes = lane_var.value();
        PrimExpr known = (IntImm(lanes.dtype(), 0) <= lanes) && (lanes < num_lanes);
        context.emplace(analyzer_, known);
      }
      if (analyzer_->CanProve(predicate)) {
        return known.known_value(op->indices).value();
      }
    }

    return GetRef<PrimExpr>(op);
  }

 private:
  const std::vector<BufferTouchPattern::BufferConstraint>& knowns_;
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
    out_->context_lookup_[stmt.get()] = out_->touch_points_.size();
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

    VisitAccess(load, BufferTouch::AccessType::Assume, value, additional_predicate);
    // Appending a control block ensures that all control blocks have
    // at most one statement that changes the known buffer contents.
    auto prev_block = CurrentControlBlock();
    std::stringstream ss;
    ss << "after T.assume of " << assumption;
    auto new_block = AppendControlBlock(ss.str());
    MarkControlFlow(prev_block, new_block);
  }

  void VisitExpr_(const LetNode* op) override {
    BindLetVar binding;
    if (UsesLoopVar(op->value)) {
      binding = BindLetVar(this, op->var, op->value);
    }
    Parent::VisitExpr_(op);
  }

  void VisitStmt_(const LetStmtNode* op) override {
    BindLetVar binding;
    if (UsesLoopVar(op->value)) {
      binding = BindLetVar(this, op->var, op->value);
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
    MarkControlFlow(before_loop, loop_start, {}, {}, op->loop_var == op->min);

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

    bool depends_on_other_iterations = false;

    for (size_t i = 0; i < touches.size(); i++) {
      for (size_t j = 0; j < i; j++) {
        const auto* read = touches[i];
        const auto* write = touches[j];

        if (!write->ProvablyCrossLoopIndependent(*read, op->loop_var, &analyzer_)) {
          depends_on_other_iterations = true;
          break;
        }
      }
      if (depends_on_other_iterations) {
        break;
      }
    }

    if (depends_on_other_iterations) {
      std::stringstream d_name;
      d_name << op->loop_var->name_hint << "_delta";
      Var delta(d_name.str(), op->loop_var.dtype());
      // PrimExpr predicate = op->loop_var > op->min;
      // MarkControlFlow(loop_end, loop_start, {{op->loop_var, op->loop_var - 1}}, {},
      //                 op->loop_var > op->min && op->loop_var < op->min + op->extent);

      MarkControlFlow(loop_end, loop_start, {{op->loop_var, op->loop_var - 1}}, {},
                      op->loop_var > op->min);

      // MarkControlFlow(loop_end, loop_start, {}, {}, op->loop_var > op->min);
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
      With<ConstraintContext> constraint(&analyzer_, real_condition);
      auto func = EnterConstraint(real_condition);
      this->VisitStmt(op->then_case);
      func();
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
      With<ConstraintContext> constraint(&analyzer_, negation);
      auto func = EnterConstraint(negation);
      this->VisitStmt(op->else_case);
      func();

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
                   Optional<PrimExpr> known_value_expr = NullOpt,
                   Optional<PrimExpr> additional_predicate = NullOpt) {
    auto index_variables = MakeIndexVariables(node->indices);

    Optional<Var> lane_var = NullOpt;
    IntImm num_lanes;

    Array<PrimExpr> index_expressions = node->indices;
    index_expressions.MutateByApply([&](const auto& index) {
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
    Map<Var, Range> free_params = transform->dst->ranges;

    // Constraints imposed on the axis variable (min/max bounds) and
    // on free parameters (relationships relative to axis vars).
    // These are used as part of the access predicate.
    PrimExpr axis_var_relations = Bool(true);
    for (const auto& expr : transform->dst->relations) {
      axis_var_relations = axis_var_relations && expr;
    }

    // The arith::SolveLinearEquation sometimes introduces free
    // parameters with extent of one.  Filtering them out here avoids
    // needing to track them through later simplifications.
    bool has_extent_one_params = false;
    for (const auto& pair : free_params) {
      if (is_const_int(pair.second->extent, 1)) {
        has_extent_one_params = true;
        break;
      }
    }
    if (has_extent_one_params) {
      Analyzer analyzer;
      analyzer.Bind(free_params);
      Map<Var, PrimExpr> new_map;
      for (const auto& pair : loop_var_to_axis_var) {
        new_map.Set(pair.first, analyzer.Simplify(pair.second));
      }
      loop_var_to_axis_var = new_map;

      Map<Var, Range> new_params;
      for (const auto& pair : free_params) {
        if (!is_const_int(pair.second->extent, 1)) {
          new_params.Set(pair.first, pair.second);
        }
      }
      free_params = new_params;
    }

    // Normalization function, applied to both the predicate and the
    // known value.  Converts from an expression in terms of loop
    // iterators which may contain BufferLoad to an expression in
    // terms of buffer indices which may not contain BufferLoad.  If
    // this conversion cannot be done, returns None.
    auto normalize_expr = [&](const Optional<PrimExpr>& opt,
                              Analyzer* arg_analyzer) -> Optional<PrimExpr> {
      if (!opt) {
        return NullOpt;
      }

      PrimExpr expr = opt.value();
      expr = Substitute(expr, let_bindings_using_loop_);

      if (lane_var) {
        expr = UnwrapVectorExpr(expr, lane_var.value());
      }
      expr = Substitute(expr, loop_var_to_axis_var);

      // if (Optional<PrimExpr> without_buffer_load =
      //         BufferConstraintSubstituter(out_->touch_points_, -1, &analyzer_)
      //             .WithoutBufferLoad(expr)) {
      //   expr = without_buffer_load.value();
      // } else {
      //   return NullOpt;
      // }

      expr = arg_analyzer->Simplify(expr);

      return expr;
    };

    // The full predicate is composed of the values required to reach
    // the scope of the BufferStore or builtin::assume(), any bounds
    // implied by the indices used to access the buffer, and any
    // additional statements resulting from unpacking the expression
    // contained in builtin::assume().
    Optional<PrimExpr> predicate_expr =
        axis_var_relations && additional_predicate.value_or(Bool(true));

    predicate_expr = normalize_expr(predicate_expr, &analyzer_);
    known_value_expr = normalize_expr(known_value_expr, &analyzer_);

    Analyzer local_analyzer;
    PrimExpr scope_predicate = normalize_expr(CurrentScopePredicate(), &local_analyzer).value();

    PrimExpr loop_predicate = Bool(true);
    for (auto it = active_loop_iterators_.rbegin(); it != active_loop_iterators_.rend(); it++) {
      auto expr_it = loop_var_to_axis_var.find(it->loop_var);
      ICHECK(expr_it != loop_var_to_axis_var.end());
      PrimExpr loop_expr = (*expr_it).second;

      loop_predicate =
          (it->loop_var >= loop_expr) || ((it->loop_var == loop_expr) && loop_predicate);
    }

    predicate_expr =
        local_analyzer.Simplify(predicate_expr.value() && scope_predicate && loop_predicate);

    Predicate predicate(index_variables, predicate_expr, free_params);
    ParametrizedExpression known_value(index_variables, known_value_expr);

    BufferTouch buffer_touch(node->buffer, predicate, touch_type, known_value, node->indices,
                             loop_var_to_axis_var, node);

    out_->touch_points_.push_back(buffer_touch);
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

  Array<Var> MakeIndexVariables(const Array<PrimExpr>& indices) {
    Array<Var> vars;
    for (size_t i = 0; i < indices.size(); i++) {
      std::stringstream ss;
      ss << "i_" << i;
      vars.push_back(Var(ss.str(), indices[i].dtype().element_of()));
    }
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

      // IntSet interval = analyzer_.int_set(index);
    }

    Array<Var> loop_vars;

    Map<Var, Range> ranges;
    for (const auto& loop_entry : active_loop_iterators_) {
      loop_vars.push_back(loop_entry.loop_var);
      ranges.Set(loop_entry.loop_var, loop_entry.loop_range);

      // IntSet loop_set = analyzer_.int_set(loop_entry.loop_var);
      // auto max = loop_set.HasUpperBound() ? loop_set.max() + 1 : loop_set.max();
      // Range loop_range = Range(loop_set.min(), max);
      // ranges.Set(loop_entry.loop_var, loop_range);
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

  PrimExpr IndexRangePredicate(const Array<Var>& index_variables,
                               const Array<PrimExpr>& index_expressions) {
    ICHECK_EQ(index_variables.size(), index_expressions.size());

    PrimExpr predicate = Bool(true);

    for (size_t i = 0; i < index_expressions.size(); i++) {
      PrimExpr index = index_expressions[i];
      Var var = index_variables[i];

      IntSet interval = analyzer_.int_set(index);

      if (interval.IsSinglePoint()) {
        predicate = predicate && (var == analyzer_.Simplify(interval.PointValue()));
      } else {
        if (interval.HasLowerBound()) {
          predicate = predicate && (var >= analyzer_.Simplify(interval.min()));
        }
        if (interval.HasUpperBound()) {
          predicate = predicate && (var <= analyzer_.Simplify(interval.max()));
        }
      }
    }

    return predicate;
  }

  /* \brief Add a new control block, returning its index */
  size_t AppendControlBlock(std::string name) {
    size_t index = out_->control_flow_.size();
    out_->control_flow_.emplace_back();
    out_->control_flow_.back().index = index;
    out_->control_flow_.back().name = name;
    return index;
  }

  /* \brief The index of the current control block */
  size_t CurrentControlBlock() { return out_->control_flow_.size() - 1; }

  /* \brief Mark a possible control from one block to another */
  void MarkControlFlow(size_t from_block, size_t to_block, Map<Var, PrimExpr> var_remap = {},
                       Map<Var, Range> remap_var_ranges = {},
                       Optional<PrimExpr> predicate = NullOpt) {
    ICHECK_LE(from_block, out_->control_flow_.size());
    ICHECK_LE(to_block, out_->control_flow_.size());

    out_->control_flow_[from_block].successors.push_back(to_block);
    out_->control_flow_[to_block].predecessors.push_back(BufferTouchPattern::ControlFlowPredecessor{
        from_block, var_remap, remap_var_ranges, predicate});
  }

  struct BindActiveLoopVar {
    BindActiveLoopVar() : self{nullptr} {}
    BindActiveLoopVar(BufferTouchExtractor* self, Var var, PrimExpr loop_min, PrimExpr loop_extent)
        : self(self), var(var) {
      PrimExpr loop_max = loop_min + (loop_extent - 1);
      auto loop_range = Range::FromMinExtent(loop_min, loop_extent);
      self->active_loop_iterators_.push_back({var, loop_min, loop_max, loop_range});
      self->loop_dependent_vars_.insert(var.get());
    }

    BindActiveLoopVar(const BindActiveLoopVar&) = delete;
    BindActiveLoopVar& operator=(const BindActiveLoopVar&) = delete;

    BindActiveLoopVar(BindActiveLoopVar&& other) : BindActiveLoopVar() {
      std::swap(self, other.self);
    }
    BindActiveLoopVar& operator=(BindActiveLoopVar&& other) {
      std::swap(self, other.self);
      return *this;
    }

    ~BindActiveLoopVar() {
      if (self) {
        self->active_loop_iterators_.pop_back();
      }
    }
    BufferTouchExtractor* self;
    Var var;
  };

  struct BindLetVar {
    BindLetVar() : self{nullptr} {}
    BindLetVar(BufferTouchExtractor* self, Var var, PrimExpr value) : self(self), var(var) {
      self->let_bindings_using_loop_[var.get()] = value;
      self->loop_dependent_vars_.insert(var.get());
    }

    BindLetVar(const BindLetVar&) = delete;
    BindLetVar& operator=(const BindLetVar&) = delete;

    BindLetVar(BindLetVar&& other) : BindLetVar() { std::swap(self, other.self); }
    BindLetVar& operator=(BindLetVar&& other) {
      std::swap(self, other.self);
      return *this;
    }

    ~BindLetVar() {
      if (self) {
        self->loop_dependent_vars_.erase(var.get());
        self->let_bindings_using_loop_.erase(var.get());
      }
    }
    BufferTouchExtractor* self;
    Var var;
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

  for (size_t i = 0; i < block.known_at_block_start.size(); i++) {
    os << "\t\t"
       << "PriorKnown[" << i << "] = " << block.known_at_block_start[i] << "\n";
  }
  for (size_t i = 0; i < block.touch_points.size(); i++) {
    os << "\t\t"
       << "Touch[" << i << "] = " << block.touch_points[i] << "\n";
  }
  for (size_t i = 0; i < block.known_at_block_end.size(); i++) {
    os << "\t\t"
       << "PostKnown[" << i << "] = " << block.known_at_block_end[i] << "\n";
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
  os << "Touch pattern contains " << pattern.touch_points_.size() << " touches."
     << (pattern.touch_points_.size() ? "\n" : "");
  for (size_t i = 0; i < pattern.touch_points_.size(); i++) {
    os << "\t"
       << "Touch[" << i << "] = " << pattern.touch_points_[i] << "\n";
  }

  os << "Touch pattern contains " << pattern.control_flow_.size() << " control blocks."
     << (pattern.control_flow_.size() ? "\n" : "");
  for (size_t i = 0; i < pattern.control_flow_.size(); i++) {
    os << "\t"
       << "ControlBlock[" << i << "] (name = '" << pattern.control_flow_[i].name
       << "') = " << pattern.control_flow_[i] << "\n";
  }

  return os;
}

bool BufferTouchPattern::BufferConstraint::IsDistinctFrom(
    const BufferTouchPattern::BufferConstraint& other, Analyzer* analyzer) const {
  if (!buffer.same_as(other.buffer)) {
    return true;
  }

  return predicate.IsDistinctFrom(other.predicate, analyzer);
}

void BufferTouchPattern::BufferConstraint::OverwriteBy(
    const BufferTouchPattern::BufferConstraint& other, Analyzer* analyzer) {
  if (IsDistinctFrom(other, analyzer)) {
    return;
  }

  predicate = predicate.Difference(other.predicate, analyzer);
}

bool BufferTouchPattern::BufferConstraint::IsEquivalentTo(
    const BufferTouchPattern::BufferConstraint& other, Analyzer* analyzer) const {
  // Constraints must apply to the same buffer to be equivalent
  if (!buffer.same_as(other.buffer)) {
    return false;
  }

  ExprDeepEqual deep_equal;

  With<ConstraintContext> context(
      analyzer, predicate.FreeParameterConstraints() && other.predicate.FreeParameterConstraints());

  auto implies = [&](const PrimExpr& a, const PrimExpr& b) -> bool {
    With<ConstraintContext> context(analyzer, a);
    return analyzer->CanProve(b);
  };

  // Predicates must be equivalent expressions, or must both be undefined
  if (predicate.IsDefined() && other.predicate.IsDefined()) {
    PrimExpr predicate_expr = predicate.expression_.value();
    PrimExpr other_predicate_expr = other.predicate(predicate.parameter_vars_).value();

    bool equivalent_predicates = deep_equal(predicate_expr, other_predicate_expr) ||
                                 (implies(predicate_expr, other_predicate_expr) &&
                                  implies(other_predicate_expr, predicate_expr));
    if (!equivalent_predicates) {
      return false;
    }
  } else if (predicate.IsDefined() ^ other.predicate.IsDefined()) {
    return false;
  }

  // The known value must be equal, or both must be undefined
  if (known_value.IsDefined() && other.known_value.IsDefined()) {
    PrimExpr known_expr = known_value.expression_.value();
    PrimExpr other_known_expr = other.known_value(known_value.parameter_vars_).value();
    if (!deep_equal(known_expr, other_known_expr) &&
        !analyzer->CanProveEqual(known_expr, other_known_expr)) {
      return false;
    }
  } else if (known_value.IsDefined() ^ other.known_value.IsDefined()) {
    return false;
  }

  return true;
}

std::vector<BufferTouchPattern::BufferConstraint>
BufferTouchPattern::BufferConstraint::SimplifyOverwrittenConstraints(
    std::vector<BufferTouchPattern::BufferConstraint> constraints, Analyzer* analyzer) {
  for (size_t i = 0; i < constraints.size(); i++) {
    for (size_t j = i + 1; j < constraints.size(); j++) {
      constraints[i].OverwriteBy(constraints[j], analyzer);
    }
  }

  constraints.erase(std::remove_if(constraints.begin(), constraints.end(),
                                   [](const auto& constraint) -> bool {
                                     return constraint.predicate.IsAlwaysFalse() ||
                                            !constraint.known_value.IsDefined();
                                   }),
                    constraints.end());

  constraints = MergeDisjointConstraints(constraints, analyzer);

  return constraints;
}

std::vector<BufferTouchPattern::BufferConstraint>
BufferTouchPattern::BufferConstraint::MergeDisjointConstraints(
    std::vector<BufferTouchPattern::BufferConstraint> constraints, Analyzer* analyzer) {
  for (size_t i = 0; i < constraints.size(); i++) {
    for (size_t j = i + 1; j < constraints.size(); j++) {
      auto& a = constraints[i];
      auto& b = constraints[j];
      if (a.buffer.same_as(b.buffer) && a.predicate.IsDefined() && b.predicate.IsDefined() &&
          a.known_value.IsDefined() && b.known_value.IsDefined()) {
        auto axis_vars = a.predicate.parameter_vars_;

        PrimExpr union_predicate =
            analyzer->Simplify(a.predicate.expression_.value() || b.predicate(axis_vars).value());
        PrimExpr value_a = a.known_value(axis_vars).value();
        PrimExpr value_b = b.known_value(axis_vars).value();

        Map<Var, Range> free_parameters = a.predicate.free_parameters_;
        for (const auto& pair : b.predicate.free_parameters_) {
          free_parameters.Set(pair.first, pair.second);
        }
        analyzer->Bind(free_parameters);

        bool provably_equal_value = [&]() {
          With<ConstraintContext> context(analyzer, union_predicate);
          return analyzer->CanProveEqual(value_a, value_b);
        }();

        if (provably_equal_value) {
          union_predicate = SimplifyAsAndOfOrs(union_predicate, analyzer);

          BufferTouchPattern::BufferConstraint new_constraint{
              a.buffer, Predicate(axis_vars, union_predicate, free_parameters),
              ParametrizedExpression(axis_vars, value_a)};
          a = std::move(new_constraint);
          b.predicate.expression_ = NullOpt;
        }
      }
    }
  }

  constraints.erase(std::remove_if(constraints.begin(), constraints.end(),
                                   [](const auto& constraint) -> bool {
                                     return constraint.predicate.IsAlwaysFalse() ||
                                            !constraint.predicate.IsDefined() ||
                                            !constraint.known_value.IsDefined();
                                   }),
                    constraints.end());

  return constraints;
}

std::ostream& operator<<(std::ostream& os, const BufferTouchPattern::BufferConstraint& obj) {
  return os << "Buffer " << obj.buffer->name << " is " << obj.known_value << " if "
            << obj.predicate;
}

/* \brief Merge constraints that may overwrite each other.
 *
 * Assumes that "before" and "after" sets of constraints are
 * internally consistent.
 */
std::vector<BufferTouchPattern::BufferConstraint>
BufferTouchPattern::BufferConstraint::MergeSequentialConstraints(
    const std::vector<BufferTouchPattern::BufferConstraint>& arg_before,
    const std::vector<BufferTouchPattern::BufferConstraint>& arg_after, Analyzer* analyzer) {
  auto before = arg_before;
  auto after = arg_after;

  std::vector<bool> used(after.size(), false);
  std::vector<BufferTouchPattern::BufferConstraint> merged;

  for (const auto& prev : before) {
    Predicate overwrite_at = prev.predicate;
    overwrite_at.expression_ = Bool(false);

    Predicate expand_known_at = prev.predicate;
    expand_known_at.expression_ = Bool(false);

    auto axis_vars = prev.known_value.parameter_vars_;
    PrimExpr prev_value = prev.known_value.expression_.value();

    for (size_t i = 0; i < after.size(); i++) {
      if (after[i].buffer.same_as(prev.buffer)) {
        Optional<PrimExpr> overwritten_with = after[i].known_value(axis_vars);
        if (overwritten_with && analyzer->CanProveEqual(prev_value, overwritten_with.value())) {
          expand_known_at = expand_known_at.Union(after[i].predicate, analyzer);
          used[i] = true;
        } else {
          overwrite_at = overwrite_at.Union(after[i].predicate, analyzer);
        }
      }
    }

    Predicate new_predicate = prev.predicate;
    if (!overwrite_at.IsAlwaysFalse()) {
      new_predicate = new_predicate.Difference(overwrite_at, analyzer);
    }
    if (!expand_known_at.IsAlwaysFalse()) {
      new_predicate = new_predicate.Union(expand_known_at, analyzer);
    }

    if (!new_predicate.IsAlwaysFalse()) {
      BufferTouchPattern::BufferConstraint post_constraint = prev;
      post_constraint.predicate = new_predicate;
      merged.push_back(post_constraint);
    }
  }

  for (size_t i = 0; i < after.size(); i++) {
    if (!used[i]) {
      if (after[i].known_value.IsDefined()) {
        merged.push_back(after[i]);
      }
    }
  }

  return merged;
}

std::vector<BufferTouchPattern::BufferConstraint>
BufferTouchPattern::BufferConstraint::MergePredecessorConstraintsWithPostcondition(
    const std::vector<BufferTouchPattern::BufferConstraint>& a_constraints,
    const std::vector<BufferTouchPattern::BufferConstraint>& b_constraints, PrimExpr a_condition,
    PrimExpr b_condition, Analyzer* analyzer) {
  std::vector<BufferTouchPattern::BufferConstraint> output;
  std::vector<bool> a_used(a_constraints.size(), false);
  std::vector<bool> b_used(b_constraints.size(), false);

  for (size_t i = 0; i < a_constraints.size(); i++) {
    if (!a_used[i]) {
      auto constraint = a_constraints[i];
      if (constraint.predicate.expression_) {
        constraint.predicate.expression_ = constraint.predicate.expression_.value() && a_condition;
        constraint.predicate.Simplify(analyzer);
      }
      output.push_back(std::move(constraint));
    }
  }

  for (size_t i = 0; i < b_constraints.size(); i++) {
    if (!b_used[i]) {
      auto constraint = b_constraints[i];
      if (constraint.predicate.expression_) {
        constraint.predicate.expression_ = constraint.predicate.expression_.value() && b_condition;
        constraint.predicate.Simplify(analyzer);
      }
      output.push_back(std::move(constraint));
    }
  }

  return MergeDisjointConstraints(std::move(output), analyzer);
}

std::vector<BufferTouchPattern::BufferConstraint>
BufferTouchPattern::BufferConstraint::MergePredecessorConstraints(
    const std::vector<BufferTouchPattern::BufferConstraint>& a,
    const std::vector<BufferTouchPattern::BufferConstraint>& b, Optional<PrimExpr> a_condition,
    Analyzer* analyzer) {
  // For a constraint to be in the output, it must be present in both
  // inputs.

  With<ConstraintContext> context(analyzer, a_condition.value_or(Bool(true)));

  std::vector<BufferTouchPattern::BufferConstraint> consistent_constraints;
  for (const auto& ai : a) {
    for (const auto& bi : b) {
      if (ai.buffer.same_as(bi.buffer)) {
        Predicate predicate = ai.predicate.Intersection(bi.predicate, analyzer);
        if (!predicate.IsAlwaysFalse()) {
          auto axis_vars = predicate.parameter_vars_;
          With<ConstraintContext> context(analyzer, predicate.FreeParameterConstraints() &&
                                                        predicate.expression_.value_or(Bool(true)));
          Optional<PrimExpr> known_value_a = ai.known_value(axis_vars);
          Optional<PrimExpr> known_value_b = bi.known_value(axis_vars);

          bool is_consistent =
              known_value_a && known_value_b &&
              analyzer->CanProveEqual(known_value_a.value(), known_value_b.value());
          if (is_consistent) {
            consistent_constraints.push_back({ai.buffer, predicate, ai.known_value});
          }
        }
      }
    }
  }

  return MergeDisjointConstraints(std::move(consistent_constraints), analyzer);
}

namespace {

bool is_const_false(const PrimExpr& expr) {
  auto* as_int = as_const_int(expr);
  return as_int && !(*as_int);
}

class BufferRegionCollector : public ExprVisitor {
 public:
  struct Region {
    PrimExpr region_predicate;
    std::unordered_map<const BufferLoadNode*, Optional<PrimExpr>> known_values;
  };

  static std::vector<Region> Collect(
      const std::vector<BufferTouchPattern::BufferConstraint>& knowns,
      const std::vector<Optional<PrimExpr>>& exprs, Analyzer* analyzer) {
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

  BufferRegionCollector(const std::vector<BufferTouchPattern::BufferConstraint>& knowns,
                        Analyzer* analyzer)
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

    for (const BufferTouchPattern::BufferConstraint& constraint : knowns_) {
      ICHECK(constraint.predicate.IsDefined());

      if (!op->buffer.same_as(constraint.buffer)) {
        // This is a different buffer, so continue searching.
        continue;
      }

      PrimExpr touch_predicate = constraint.predicate(op->indices).value();
      // touch_predicate = analyzer_->Simplify(touch_predicate;)
      touch_predicate = SimplifyAsAndOfOrs(touch_predicate, analyzer_);

      if (!is_const_false(touch_predicate)) {
        Optional<PrimExpr> known_value = constraint.known_value(op->indices);
        new_regions.push_back(Known{touch_predicate, known_value});

        unknown_region = unknown_region && !touch_predicate;
        unknown_region = SimplifyAsAndOfOrs(unknown_region, analyzer_);
      }
    }

    if (new_regions.size()) {
      Analyzer local_analyzer;

      if (!is_const_false(unknown_region)) {
        new_regions.insert(new_regions.begin(), Known{unknown_region, NullOpt});
      }

      std::vector<Region> updated_regions;
      for (const auto& prev_region : regions_) {
        for (const auto& new_region : new_regions) {
          PrimExpr intersection =
              SimplifyAsAndOfOrs(prev_region.region_predicate && new_region.predicate, analyzer_);

          if (!is_const_false(intersection)) {
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
  const std::vector<BufferTouchPattern::BufferConstraint>& knowns_;
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
}  // namespace

Map<Var, Range> BufferTouchPattern::GetAllFreeParameters() const {
  Map<Var, Range> ret;
  for (const auto& block : control_flow_) {
    for (const auto& touch : block.touch_points) {
      for (const auto& pair : touch.predicate.free_parameters_) {
        ret.Set(pair.first, pair.second);
      }
    }

    for (const auto& pred : block.predecessors) {
      for (const auto& pair : pred.remap_var_ranges) {
        ret.Set(pair.first, pair.second);
      }
    }
  }
  return ret;
}

void BufferTouchPattern::ForwardPropagateKnownValues() {
  // Values to visit when searching.  Using a std::set to
  // preferentially visit nodes near the start of the control flow.
  std::set<size_t> to_visit;

  // If a control block has not yet been visited, then an empty
  // constraints vector just means that we haven't filled it yet.  If
  // a control block has been visited, an empty constraints vector
  // means that we don't know anything about any vector.
  //
  // TODO: See if this is cleaner if written as an
  // Optional<Array<Constraint>>
  std::unordered_set<size_t> visited_once;

  // Initiatize the locations to search from, propagating values
  // forward from all locations that have a known value.
  for (size_t i = 0; i < control_flow_.size(); i++) {
    bool has_known_value = false;
    for (const auto& touch : control_flow_[i].touch_points) {
      if (touch.known_value.IsDefined() && !HasBufferLoad(touch.known_value.expression_.value())) {
        has_known_value = true;
        break;
      }
    }

    if (has_known_value) {
      to_visit.insert(i);
    }
  }

  Analyzer analyzer;
  analyzer.rewrite_simplify.SetEnabledFeatures(
      arith::RewriteSimplifier::kTransitivelyProveInequalities);

  Map<Var, Range> all_free_parameters = GetAllFreeParameters();
  analyzer.Bind(iterator_ranges_);
  analyzer.Bind(all_free_parameters);

  // Utility function to simplify a list of knowns
  auto normalize_simplify = [&analyzer](std::vector<BufferTouchPattern::BufferConstraint> priors) {
    for (auto& prior : priors) {
      prior.predicate.expression_ =
          SimplifyAsAndOfOrs(prior.predicate.expression_.value(), &analyzer);
    }
    return priors;
  };

  while (to_visit.size()) {
    size_t visiting = *to_visit.begin();
    to_visit.erase(visiting);
    ControlFlowBlock& block = control_flow_[visiting];

    // Step 1: Collect known values provided from each precedessor
    block.known_at_block_start = [&]() -> std::vector<BufferTouchPattern::BufferConstraint> {
      ICHECK_LE(block.predecessors.size(), 2) << "Each block should have at most two predecessors";

      auto remap_priors = [&](std::vector<BufferTouchPattern::BufferConstraint> priors,
                              Map<Var, PrimExpr> var_remap) {
        if (var_remap.size()) {
          for (auto& prior : priors) {
            PrimExpr before_remap = prior.predicate.expression_.value();
            PrimExpr after_remap = Substitute(before_remap, var_remap);
            if (!before_remap.same_as(after_remap)) {
              prior.predicate.expression_ = SimplifyAsAndOfOrs(after_remap, &analyzer);
            }
          }
        }
        return priors;
      };

      auto add_condition = [&](std::vector<BufferTouchPattern::BufferConstraint> priors,
                               PrimExpr condition) {
        for (auto& prior : priors) {
          prior.predicate.expression_ = prior.predicate.expression_.value() && condition;
          prior.predicate.Simplify(&analyzer);
        }
        return priors;
      };

      if (block.predecessors.size() == 0) {
        // Block has no predecessors, nothing is known initially
        return {};
      } else if (block.predecessors.size() == 1) {
        // Block has only a single predecessor
        const auto& pred = block.predecessors[0];
        size_t prev_index = pred.from_index;
        const auto& prev_block = control_flow_[prev_index];
        return remap_priors(prev_block.known_at_block_end, pred.var_remap);
      }

      ICHECK_EQ(block.predecessors.size(), 2);

      const auto& pred_a = block.predecessors[0];
      const auto& pred_b = block.predecessors[1];

      const auto& pred_a_block = control_flow_[pred_a.from_index];
      const auto& pred_b_block = control_flow_[pred_b.from_index];
      if (!visited_once.count(pred_a.from_index) && !visited_once.count(pred_b.from_index)) {
        return {};
      } else if (!visited_once.count(pred_a.from_index)) {
        auto out = pred_b_block.known_at_block_end;
        out = remap_priors(out, pred_b.var_remap);
        if (pred_a.predicate && pred_b.predicate) {
          out = add_condition(out, pred_a.predicate.value() || pred_b.predicate.value());
        }
        out = normalize_simplify(out);
        return out;
      } else if (!visited_once.count(pred_b.from_index)) {
        auto out = pred_a_block.known_at_block_end;
        out = remap_priors(out, pred_a.var_remap);
        if (pred_a.predicate && pred_b.predicate) {
          out = add_condition(out, pred_a.predicate.value() || pred_b.predicate.value());
        }

        out = normalize_simplify(out);

        return out;
      }

      auto priors_a = remap_priors(pred_a_block.known_at_block_end, pred_a.var_remap);
      auto priors_b = remap_priors(pred_b_block.known_at_block_end, pred_b.var_remap);

      std::vector<BufferTouchPattern::BufferConstraint> output;
      if (pred_a.predicate && pred_b.predicate) {
        output = BufferTouchPattern::BufferConstraint::MergePredecessorConstraintsWithPostcondition(
            priors_a, priors_b, pred_a.predicate.value(), pred_b.predicate.value(), &analyzer);
      } else if (pred_a.predicate) {
        output = BufferTouchPattern::BufferConstraint::MergePredecessorConstraints(
            priors_a, priors_b, pred_a.predicate, &analyzer);
      } else if (pred_b.predicate) {
        output = BufferTouchPattern::BufferConstraint::MergePredecessorConstraints(
            priors_b, priors_a, pred_b.predicate, &analyzer);
      } else {
        output = BufferTouchPattern::BufferConstraint::MergePredecessorConstraints(
            priors_a, priors_b, NullOpt, &analyzer);
      }

      return output;
    }();
    const auto& prior_knowns = block.known_at_block_start;

    // Step 2: Collect knowns provided as a result of executing this block
    std::vector<BufferTouchPattern::BufferConstraint> new_knowns = [&]() {
      std::vector<BufferTouchPattern::BufferConstraint> new_knowns;

      for (auto& touch : block.touch_points) {
        if (touch.touch_type == BufferTouch::AccessType::Read) {
          continue;
        }

        Array<Var> axis_vars = touch.known_value.parameter_vars_;
        PrimExpr predicate = touch.predicate(axis_vars).value();

        PrimExpr known_value = touch.known_value(axis_vars).value();
        auto regions =
            BufferRegionCollector::Collect(prior_knowns, {predicate, known_value}, &analyzer);

        for (const auto& region : regions) {
          PrimExpr updated_predicate = BufferRegionValueReplacer::Apply(
              region.known_values, region.region_predicate && predicate, &analyzer);

          updated_predicate = SimplifyAsAndOfOrs(updated_predicate, &analyzer);
          PrimExpr updated_value =
              BufferRegionValueReplacer::Apply(region.known_values, known_value, &analyzer);

          if (!is_const_false(updated_predicate)) {
            Map<tir::Var, Range> free_parameters;
            for (const Var& var : UndefinedVars(updated_predicate)) {
              auto it = touch.predicate.free_parameters_.find(var);
              if (it != touch.predicate.free_parameters_.end()) {
                free_parameters.Set((*it).first, (*it).second);
              }
            }

            if (HasBufferLoad(updated_value)) {
              BufferTouchPattern::BufferConstraint overwrite{
                  touch.buffer, Predicate(axis_vars, updated_predicate, free_parameters),
                  ParametrizedExpression(axis_vars, NullOpt)};
              new_knowns.push_back(overwrite);
            } else {
              BufferTouchPattern::BufferConstraint new_constraint{
                  touch.buffer, Predicate(axis_vars, updated_predicate, free_parameters),
                  ParametrizedExpression(axis_vars, updated_value)};
              new_knowns.push_back(new_constraint);
            }
          }
        }
      }
      return new_knowns;
    }();

    // Step 3: Generate the knowns at the end of the block
    auto post_knowns = [&]() {
      if (new_knowns.size() == 0) {
        return prior_knowns;
      }
      auto post_knowns = BufferTouchPattern::BufferConstraint::MergeSequentialConstraints(
          prior_knowns, new_knowns, &analyzer);

      for (auto& known : post_knowns) {
        known.predicate = known.predicate.WithoutFreeParameters();
        known.predicate.expression_ =
            SimplifyAsAndOfOrs(known.predicate.expression_.value(), &analyzer);
      }
      return post_knowns;
    }();

    // Step 4: If any changes are made to the post knowns since the
    // previous time we visited this block, mark the successor block
    // as needing to be visited.
    bool has_updated_post = [&]() -> bool {
      if (!visited_once.count(visiting)) {
        return true;
      }

      const auto& previous_post_knowns = block.known_at_block_end;

      if (post_knowns.size() != previous_post_knowns.size()) {
        return true;
      }

      for (size_t i = 0; i < post_knowns.size(); i++) {
        if (!post_knowns[i].IsEquivalentTo(previous_post_knowns[i], &analyzer)) {
          return true;
        }
      }

      return false;
    }();

    // TODO: Have a maximum number of times that blocks may be
    // visited, to guard against infinite loops.
    if (has_updated_post) {
      block.known_at_block_end = post_knowns;
      for (size_t successor : block.successors) {
        to_visit.insert(successor);
      }
    }

    visited_once.insert(visiting);
  }
}

bool BufferTouchPattern::IsOverwrittenWithoutEffect(const tir::BufferStore& store,
                                                    Analyzer* analyzer) const {
  bool write_occurred = false;

  for (auto it = touch_points_.begin(); it != touch_points_.end(); it++) {
    if (it->node.same_as(store)) {
      write_occurred = true;
      if (!IsOverwrittenWithoutEffect(it, analyzer)) {
        return false;
      }
    }
  }

  ICHECK(write_occurred) << "BufferStore did not occur within analyzed statement";

  return true;
}

bool BufferTouchPattern::IsOverwrittenWithoutEffect(
    std::vector<BufferTouch>::const_iterator write_iter, Analyzer* analyzer) const {
  // TODO: Walk through control flow graph
  for (auto it = write_iter + 1; it != touch_points_.end(); it++) {
    // If the write_iter was a subset of another write, then it was entirely overwritten.
    if (it->touch_type == BufferTouch::AccessType::Write && write_iter->IsSubsetOf(*it, analyzer)) {
      return true;
    }
    // If the written values are later read out, then this write had an effect.
    if (it->touch_type == BufferTouch::AccessType::Read && it->IsSubsetOf(*write_iter, analyzer)) {
      return false;
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

  const auto& knowns = control_flow_[context_index].known_at_block_start;

  BufferConstraintApply mutator(knowns, analyzer);
  expr = mutator(expr);
  expr = analyzer->Simplify(expr);
  return expr;
}

void BufferTouchPattern::RemoveTouches(const tir::BufferStore& store) {
  touch_points_.erase(std::remove_if(touch_points_.begin(), touch_points_.end(),
                                     [&](const auto& touch) { return touch.node.same_as(store); }));
  // TODO: Update context_lookup_
}

}  // namespace arith
}  // namespace tvm
