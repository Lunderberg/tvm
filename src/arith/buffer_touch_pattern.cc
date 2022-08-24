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
      SimplifyUsingAndOfOrs(expression_.value() && !other_predicate, analyzer);

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
      // std::cout << "Cannot prove that read index " << read_index
      //           << " doesn't depend on previous write index " << write_index << std::endl;
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
      // std::cout << "Simplified from " << touch.known_value.expression_ << " to "
      //           << simplified_known_value << std::endl;
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
      // std::cout << "Could not replace " << GetRef<PrimExpr>(op) << " with known value" <<
      // std::endl;
      return GetRef<PrimExpr>(op);
    }
  }

  Optional<PrimExpr> FollowPredecessorBlock(const BufferTouchPattern::ControlFlowBlock& block,
                                            const BufferLoadNode* op, PrimExpr access_predicate) {
    IncreaseDepth temp(this);
    // std::cout << std::string(depth * 2, ' ') << "Attempting to replace " << GetRef<PrimExpr>(op)
    //           << " by visiting the " << block.predecessors.size() << " predecessors"
    //           << " of block " << block.index << " (name = '" << block.name << "')" << std::endl;

    Optional<PrimExpr> result = NullOpt;
    for (const auto& predecessor_edge : block.predecessors) {
      size_t predecessor_index = predecessor_edge.from_index;
      IncreaseDepth temp2(this);
      const auto& predecessor = touch_pattern_.control_flow_[predecessor_index];
      // std::cout << std::string(depth * 2, ' ') << "Checking constraints provided by predecessor "
      //           << predecessor.index << " (name = '" << predecessor.name << "')" << std::endl;
      auto opt_value = ApplyKnownValue(predecessor, op, access_predicate);
      // std::cout << std::string(depth * 2, ' ') << "In context of predecessor " <<
      // predecessor_index
      //           << " value " << GetRef<PrimExpr>(op) << " is known to be " << opt_value
      //           << std::endl;
      if (!opt_value) {
        // std::cout << std::string(depth * 2, ' ') << "Unknown value for predecessor "
        //           << predecessor_index
        //           << ", therefore cannot prove to be the same for all predecessors" << std::endl;
        return GetRef<PrimExpr>(op);
      } else if (!result) {
        // std::cout << std::string(depth * 2, ' ')
        //           << "This was the first predecessor checked, so it is consistent by definition"
        //           << std::endl;
        result = opt_value;
      } else if (!analyzer_->CanProveEqual(opt_value.value(), result.value())) {
        // std::cout << std::string(depth * 2, ' ') << "Value " << opt_value << " for predecessor "
        //           << predecessor_index << " wasn't equal to previous value " << result
        //           << ", bailing out" << std::endl;
        return NullOpt;
        // return GetRef<PrimExpr>(op);
      }
    }

    return result;
  }

  Optional<PrimExpr> ApplyKnownValue(const BufferTouchPattern::ControlFlowBlock& block,
                                     const BufferLoadNode* op, PrimExpr access_predicate) {
    IncreaseDepth temp(this);
    // std::cout << std::string(depth * 2, ' ') << "Attempting to replace " << GetRef<PrimExpr>(op)
    //           << " using information in block " << block.index << ", accessed with predicate "
    //           << access_predicate << std::endl;

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

  BufferConstraintApply(const BufferTouchPattern& touch_pattern, size_t context_index,
                        Analyzer* analyzer)
      : Parent(analyzer), touch_pattern_(touch_pattern), context_index_(context_index) {}

  using Parent::VisitExpr_;

  PrimExpr VisitExpr_(const BufferLoadNode* op) override {
    auto it = touch_pattern_.constraint_lookup_.find(context_index_);
    ICHECK(it != touch_pattern_.constraint_lookup_.end())
        << "Cannot find constraints for context " << context_index_;
    const auto& knowns = it->second;

    // std::cout << "Found BufferLoad " << GetRef<PrimExpr>(op) << std::endl;

    for (const auto& known : knowns) {
      if (!op->buffer.same_as(known.buffer)) {
        // std::cout << "Known value applies to a different buffer, skipping" << std::endl;
        // This is a different buffer, so continue searching.
        continue;
      }

      PrimExpr predicate = known.predicate(op->indices).value();
      // std::cout << "Predicate " << known.predicate << " at indices " << op->indices << " is "
      //           << predicate << ", simplified = " << analyzer_->Simplify(predicate) << std::endl;
      if (analyzer_->CanProve(predicate)) {
        // std::cout << "Can prove the predicate, replacing with the known value "
        //           << known.known_value(op->indices).value() << std::endl;
        return known.known_value(op->indices).value();
      } else {
        // std::cout << "Cannot prove that the predicate is true" << std::endl;
      }
    }

    return GetRef<PrimExpr>(op);
  }

 private:
  const BufferTouchPattern& touch_pattern_;
  size_t context_index_;
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
      // std::cout << "Found non-buffer assumption in scope " << CurrentScopePredicate()
      //           << ", stating that " << assumption << std::endl;
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
    } else {
      LOG(FATAL) << "T.assume buffer constraint must be of the form 'buffer[indices] == value'";
    }

    CHECK(tir::SideEffect(value) <= tir::CallEffectKind::kPure)
        << "Buffer value in constraint must be pure expression, but was " << value;

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
          // std::cout << "Read " << *read << " depends on previous loop iteration writing " <<
          // *write
          //           << std::endl;
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

    // std::cout << "Transform of indices " << index_expressions << " is " << loop_var_to_axis_var
    //           << " with free variables " << free_params << std::endl;

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
    Optional<PrimExpr> predicate_expr = IndexRangePredicate(index_variables, index_expressions) &&
                                        additional_predicate.value_or(Bool(true));

    predicate_expr = normalize_expr(predicate_expr, &analyzer_);
    known_value_expr = normalize_expr(known_value_expr, &analyzer_);

    Analyzer local_analyzer;
    PrimExpr scope_predicate = normalize_expr(CurrentScopePredicate(), &local_analyzer).value();

    std::cout << "Initial predicate is " << predicate_expr << std::endl;

    PrimExpr loop_predicate = Bool(true);
    for (auto it = active_loop_iterators_.rbegin(); it != active_loop_iterators_.rend(); it++) {
      auto expr_it = loop_var_to_axis_var.find(it->loop_var);
      ICHECK(expr_it != loop_var_to_axis_var.end());
      PrimExpr loop_expr = (*expr_it).second;

      // loop_predicate = (it->loop_var >= it->loop_min && it->loop_var >= loop_expr) ||
      //                  ((it->loop_var == loop_expr) && loop_predicate);
      loop_predicate =
          (it->loop_var >= loop_expr) || ((it->loop_var == loop_expr) && loop_predicate);
    }
    std::cout << "\t"
              << "Loop-based predicate is " << loop_predicate << std::endl;

    // PrimExpr relation_predicate = Bool(true);
    // for (const auto& relation : transform->dst->relations) {
    //   relation_predicate = relation_predicate && relation;
    // }
    // predicate_expr = predicate_expr.value() && relation_predicate;

    predicate_expr =
        local_analyzer.Simplify(predicate_expr.value() && scope_predicate && loop_predicate);
    // predicate_expr = predicate_expr.value() && loop_predicate;
    // predicate_expr = analyzer_.Simplify(predicate_expr.value() && loop_predicate);
    std::cout << "\t"
              << "Updated predicate with loops is " << predicate_expr << std::endl;

    // Optional<PrimExpr> has_known_value_expr = Bool(false);
    // Optional<PrimExpr> known_untouched_expr = Bool(false);

    // if (predicate_expr) {
    //   PrimExpr expr = predicate_expr.value();
    //   {
    //     PrimExpr narrowed = arith::NarrowExpressionToTrue(expr, free_params);
    //     PrimExpr simplified = analyzer_.Simplify(narrowed);
    //     std::cout << "Predicate expression " << expr << " can be narrowed to parameter-less "
    //               << narrowed << ", which simplifies to " << simplified << std::endl;
    //     has_known_value_expr = simplified;
    //   }
    //   {
    //     PrimExpr narrowed = arith::NarrowExpressionToTrue(!expr, free_params);
    //     PrimExpr simplified = analyzer_.Simplify(narrowed);
    //     std::cout << "Untouched expression " << expr << " can be narrowed to parameter-less "
    //               << narrowed << ", which simplifies to " << simplified << std::endl;
    //     known_untouched_expr = simplified;
    //   }
    // }

    // if (known_value_expr) {
    //   const auto& free_params = free_params;
    //   bool uses_free_param = UsesVar(known_value_expr.value(), [&](const VarNode* var) {
    //     return free_params.find(GetRef<Var>(var)) != free_params.end();
    //   });
    //   if (uses_free_param) {
    //     known_value_expr = NullOpt;
    //   }
    // }

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
      IntSet loop_set = analyzer_.int_set(loop_entry.loop_var);
      auto max = loop_set.HasUpperBound() ? loop_set.max() + 1 : loop_set.max();
      Range loop_range = Range(loop_set.min(), max);
      ranges.Set(loop_entry.loop_var, loop_range);
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
    BindActiveLoopVar(BufferTouchExtractor* self, Var var, PrimExpr loop_min, PrimExpr loop_max)
        : self(self), var(var) {
      self->active_loop_iterators_.push_back({var, loop_min, loop_max});
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
  std::cout << "BeforePropagation: Touch pattern" << *this << std::endl;
  // std::cout << "Intentional segfault: " << *static_cast<char*>(nullptr) << std::endl;
  ForwardPropagateKnownValues();
  std::cout << "AfterPropagation: Touch pattern" << *this << std::endl;
}

std::ostream& operator<<(std::ostream& os, const BufferTouchPattern::ControlFlowBlock& block) {
  os << "Control block with " << block.touch_points.size() << " touch points"
     << "\n";
  for (size_t i = 0; i < block.touch_points.size(); i++) {
    os << "\t\t"
       << "Touch[" << i << "] = " << block.touch_points[i] << "\n";
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

  os << "Touch pattern contains " << pattern.constraint_lookup_.size()
     << " known locations through the control flow"
     << "\n";

  for (size_t i = 0; i < pattern.control_flow_.size(); i++) {
    auto it = pattern.constraint_lookup_.find(i);
    if (it == pattern.constraint_lookup_.end()) {
      os << "\t"
         << "ControlBlock[" << i << "] (name = '" << pattern.control_flow_[i].name << ") "
         << "was never visited and has no knowns" << std::endl;
      ;
    } else {
      os << "\t"
         << "ControlBlock[" << i << "] (name = '" << pattern.control_flow_[i].name << ") "
         << "has " << it->second.size() << " known conditions" << std::endl;
      ;
      for (const auto& constraint : it->second) {
        os << "\t\t"
           << "Buffer " << constraint.buffer->name << " is " << constraint.known_value << " when "
           << constraint.predicate << "\n";
      }
    }
  }

  // for (const auto& state : pattern.constraint_lookup_) {
  //   os << "\t"
  //      << "Location " << state.first << " has " << state.second.size() << " known conditions"
  //      << "\n";
  //   for (const auto& constraint : state.second) {
  //     os << "\t\t"
  //        << "Buffer " << constraint.buffer->name << " is " << constraint.known_value << " when "
  //        << constraint.predicate << "\n";
  //   }
  // }

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
    // std::cout << "\t\t\t"
    //           << "Constraint on " << buffer->name << " doesn't apply to buffer "
    //           << other.buffer->name << std::endl;
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

    // TODO: Remove these debug breakdowns of "equivalent_predicates"
    // bool is_deep_equal = deep_equal(predicate_expr, other_predicate_expr);
    // if (!is_deep_equal && !implies(predicate_expr, other_predicate_expr)) {
    //   std::cout << "\t\t\t"
    //             << "Cannot use " << predicate_expr << " to prove " << other_predicate_expr
    //             << std::endl;
    //   return false;
    // }

    // if (!is_deep_equal && !implies(other_predicate_expr, predicate_expr)) {
    //   std::cout << "\t\t\t"
    //             << "Cannot use " << other_predicate_expr << " to prove " << predicate_expr
    //             << std::endl;
    //   return false;
    // }
    bool equivalent_predicates = deep_equal(predicate_expr, other_predicate_expr) ||
                                 (implies(predicate_expr, other_predicate_expr) &&
                                  implies(other_predicate_expr, predicate_expr));
    if (!equivalent_predicates) {
      return false;
    }
  } else if (predicate.IsDefined() ^ other.predicate.IsDefined()) {
    // std::cout << "\t\t\t"
    //           << "Predicate is defined " << predicate.IsDefined() << ", but other predicate "
    //           << other.predicate.IsDefined() << std::endl;
    return false;
  }

  // The known value must be equal, or both must be undefined
  if (known_value.IsDefined() && other.known_value.IsDefined()) {
    PrimExpr known_expr = known_value.expression_.value();
    PrimExpr other_known_expr = other.known_value(known_value.parameter_vars_).value();
    if (!deep_equal(known_expr, other_known_expr) &&
        !analyzer->CanProveEqual(known_expr, other_known_expr)) {
      // std::cout << "\t\t\t"
      //           << "Can't prove that " << known_expr << " is equal to " << other_known_expr
      //           << std::endl;
      return false;
    }
  } else if (known_value.IsDefined() ^ other.known_value.IsDefined()) {
    // std::cout << "\t\t\t"
    //           << "known value is defined " << predicate.IsDefined() << ", but other known value
    //           "
    //           << other.predicate.IsDefined() << std::endl;
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
          std::cout << "Merging conditions for known value " << value_a << std::endl;
          std::cout << "\t"
                    << "Unioned predicate is " << union_predicate << std::endl;

          // union_predicate = ConvertToAndOfOrs(union_predicate);
          // std::cout << "\t"
          //           << "As AND of ORs: " << union_predicate << std::endl;
          // union_predicate = analyzer.Simplify(union_predicate);
          union_predicate = SimplifyUsingAndOfOrs(union_predicate, analyzer);
          std::cout << "\t"
                    << "Then simplified: " << union_predicate << std::endl;

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

/* \brief Merge constraints that may overwrite each other.
 *
 * Assumes that "before" and "after" sets of constraints are
 * internally consistent.
 */
std::vector<BufferTouchPattern::BufferConstraint>
BufferTouchPattern::BufferConstraint::MergeSequentialConstraints(
    const std::vector<BufferTouchPattern::BufferConstraint>& before,
    const std::vector<BufferTouchPattern::BufferConstraint>& after, Analyzer* analyzer) {
  std::vector<BufferTouchPattern::BufferConstraint> output;
  output.insert(output.end(), before.begin(), before.end());
  output.insert(output.end(), after.begin(), after.end());

  output = SimplifyOverwrittenConstraints(std::move(output), analyzer);

  return output;
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
      const std::vector<Optional<PrimExpr>>& exprs, const Map<Var, Range>& all_free_parameters,
      Analyzer* analyzer) {
    BufferRegionCollector collector(knowns, all_free_parameters, analyzer);
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
                        const Map<Var, Range>& all_free_parameters, Analyzer* analyzer)
      : analyzer_(analyzer), knowns_(knowns), all_free_parameters_(all_free_parameters) {
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

    std::cout << "Collecting regions, examining BufferLoad " << GetRef<PrimExpr>(op) << std::endl;

    PrimExpr unknown_region = Bool(true);

    for (const BufferTouchPattern::BufferConstraint& constraint : knowns_) {
      ICHECK(constraint.predicate.IsDefined());

      if (!op->buffer.same_as(constraint.buffer)) {
        // This is a different buffer, so continue searching.
        continue;
      }

      std::cout << "\t"
                << "Examining constraint with predicate " << constraint.predicate << std::endl;

      PrimExpr touch_predicate = constraint.predicate(op->indices).value();
      // touch_predicate = analyzer_->Simplify(touch_predicate;)
      touch_predicate = SimplifyUsingAndOfOrs(touch_predicate, analyzer_);

      std::cout << "\t\t"
                << "Substituting indices, constraint applies iff " << touch_predicate << std::endl;

      if (!is_const_false(touch_predicate)) {
        Optional<PrimExpr> known_value = constraint.known_value(op->indices);
        new_regions.push_back(Known{touch_predicate, known_value});

        std::cout << "\t\t"
                  << "Making new region with " << touch_predicate << " having known value "
                  << known_value << std::endl;

        unknown_region = unknown_region && !touch_predicate;
        unknown_region = SimplifyUsingAndOfOrs(unknown_region, analyzer_);

        std::cout << "\t\t"
                  << "Remaining untouched regions are " << unknown_region << std::endl;
      }

      // PrimExpr always_touched = NarrowExpressionToTrue(touch_predicate, all_free_parameters_);
      // std::cout << "\t\t"
      //           << "Removing free parameters, constraint applies if " << always_touched
      //           << std::endl;
      // always_touched = analyzer_->Simplify(always_touched);
      // std::cout << "\t\t\t"
      //           << "Simplified = " << always_touched << std::endl;
      // PrimExpr never_touched = NarrowExpressionToTrue(!touch_predicate, all_free_parameters_);
      // std::cout << "\t\t"
      //           << "Removing free parameters, constraint doesn't apply if " << never_touched
      //           << std::endl;
      // never_touched = analyzer_->Simplify(never_touched);
      // std::cout << "\t\t\t"
      //           << "Simplified = " << analyzer_->Simplify(never_touched) << std::endl;
      // PrimExpr partially_touched = !always_touched && !never_touched;
      // std::cout << "\t\t"
      //           << "Removing free parameters, constraint sometimes applies if " <<
      //           partially_touched
      //           << std::endl;
      // partially_touched = analyzer_->Simplify(partially_touched);
      // std::cout << "\t\t\t"
      //           << "Simplified = " << analyzer_->Simplify(partially_touched) << std::endl;

      // if (!is_const_false(always_touched)) {
      //   Optional<PrimExpr> known_value = constraint.known_value(op->indices);
      //   new_regions.push_back(Known{always_touched, known_value});
      // }
      // // If this constraint touches all locations, no need to check any additional constraints.
      // if (is_const_false(never_touched)) {
      //   break;
      // }
    }

    // std::cout << "\t"
    //           << "All remaining access is the same condition, " << access_predicate
    //           << ", which simplifies to " << analyzer_->Simplify(access_predicate) <<
    //           std::endl;

    std::cout << "Regions before update = [";
    {
      bool first = true;
      for (const auto& prev : regions_) {
        if (first) {
          first = false;
        } else {
          std::cout << ", ";
        }
        std::cout << prev.region_predicate;
      }
    }
    std::cout << "]" << std::endl;

    if (new_regions.size()) {
      Analyzer local_analyzer;

      if (!is_const_false(unknown_region)) {
        new_regions.insert(new_regions.begin(), Known{unknown_region, NullOpt});
      }

      std::vector<Region> updated_regions;
      for (const auto& prev_region : regions_) {
        for (const auto& new_region : new_regions) {
          // PrimExpr intersection =
          //     local_analyzer.Simplify(prev_region.region_predicate && new_region.predicate);

          // PrimExpr intersection = SimplifyUsingAndOfOrs(
          //     prev_region.region_predicate && new_region.predicate, &local_analyzer);
          PrimExpr intersection = SimplifyUsingAndOfOrs(
              prev_region.region_predicate && new_region.predicate, analyzer_);

          std::cout << "Updating region (" << prev_region.region_predicate << " && "
                    << new_region.predicate << ") = " << intersection << std::endl;

          if (!is_const_false(intersection)) {
            Region merged{intersection, prev_region.known_values};
            merged.known_values[op] = new_region.value;
            updated_regions.push_back(std::move(merged));
          }
        }
      }
      regions_ = updated_regions;
    }

    std::cout << "Regions after update = [";
    {
      bool first = true;
      for (const auto& prev : regions_) {
        if (first) {
          first = false;
        } else {
          std::cout << ", ";
        }
        std::cout << prev.region_predicate;
      }
    }
    std::cout << "]" << std::endl;
  }

  Analyzer* analyzer_;
  std::vector<Region> regions_;
  const std::vector<BufferTouchPattern::BufferConstraint>& knowns_;
  const Map<Var, Range>& all_free_parameters_;
};

class BufferRegionValueReplacer : public IRMutatorWithAnalyzer {
 public:
  static PrimExpr Apply(
      const std::unordered_map<const BufferLoadNode*, Optional<PrimExpr>>& known_values,
      PrimExpr expr, Analyzer* analyzer) {
    BufferRegionValueReplacer mutator(known_values, analyzer);
    // std::cout << "\t\t\t\t"
    //           << "Starting with expression " << expr << std::endl;
    PrimExpr result = mutator(expr);
    // std::cout << "\t\t\t\t"
    //           << "Replaced BufferLoad for " << result << std::endl;
    // Simplification must occur after the substitution, as known
    // values may provide enable simplifications.  Also, cannot track
    // whether a BufferLoad was
    result = analyzer->Simplify(result);
    // std::cout << "\t\t\t\t"
    //           << "Simplified to " << result << std::endl;
    return result;
    // if (HasBufferLoad(result)) {
    //   // std::cout << "\t\t\t\t"
    //   //           << "Simplified result is " << result << ", which contains a bufferload"
    //   //           << std::endl;
    //   return NullOpt;
    // } else {
    //   return result;
    // }
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
      // std::cout << "\t\t\t\t"
      //           << "Replacing BufferLoad " << GetRef<PrimExpr>(op) << " with known value of "
      //           << it->second << std::endl;
      return it->second.value();
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  // Optional<PrimExpr> ApplyKnownValue(const BufferLoadNode* op) {
  //   auto implies = [this](const PrimExpr& known, const PrimExpr& conjecture) -> bool {
  //     With<ConstraintContext> constraint(analyzer_, known);
  //     return analyzer_->CanProve(conjecture);
  //   };

  //   std::vector<std::pair<PrimExpr, PrimExpr>> known_subregion;
  //   PrimExpr free_parameter_constraints = Bool(true);

  //   PrimExpr access_predicate = Bool(true);

  //   std::cout << "Attempting to apply known values to BufferLoad " << GetRef<PrimExpr>(op)
  //             << std::endl;

  //   for (const BufferTouchPattern::BufferConstraint& constraint : knowns_) {
  //     ICHECK(constraint.predicate.IsDefined());

  //     std::cout << "\t"
  //               << "Attempting to apply known constraint " << constraint.buffer->name
  //               << ", predicate = " << constraint.predicate
  //               << ", known value = " << constraint.known_value << std::endl;

  //     if (!op->buffer.same_as(constraint.buffer)) {
  //       std::cout << "\t\t"
  //                 << "This touch is a different buffer, skipping" << std::endl;
  //       // This is a different buffer, so continue searching.
  //       continue;
  //     }

  //     PrimExpr touch_predicate =
  //     analyzer_->Simplify(constraint.predicate(op->indices).value());

  //     std::cout << "\t\t"
  //               << "Predicate to determine if this touch applies is " << touch_predicate
  //               << std::endl;

  //     // With<ConstraintContext> isn't safe to use in a std::vector,
  //     // so instead we collect a single expression with all the extra
  //     // constraints.
  //     free_parameter_constraints =
  //         free_parameter_constraints && constraint.predicate.FreeParameterConstraints();
  //     With<ConstraintContext> context(analyzer_, free_parameter_constraints);

  //     if (constraint.known_value.IsDefined() && implies(access_predicate, touch_predicate)) {
  //       std::cout << "\t\t"
  //                 << "The access predicate " << access_predicate
  //                 << " implies that the touch predicate is true, so this constraint applies"
  //                 << std::endl;
  //       // The value provided by the constraint is known, return it.
  //       PrimExpr value = constraint.known_value(op->indices).value();
  //       for (auto it = known_subregion.rbegin(); it != known_subregion.rend(); it++) {
  //         value = if_then_else(it->first, it->second, value);
  //       }
  //       return value;
  //     } else if (implies(access_predicate, logical_not(touch_predicate))) {
  //       std::cout << "\t\t"
  //                 << "The touch predicate is false whenever the access predicate "
  //                 << access_predicate << " is true, so this constraint doesn't apply" <<
  //                 std::endl;
  //       // The constraint applies to a region that we're not
  //       // interested in , so continue searching.
  //       continue;
  //     } else if (constraint.known_value.IsDefined()) {
  //       // The constraint provides a known known value, but only for
  //       // some of the indices we are interested in.  Other locations
  //       // may have a different constraint applied.
  //       PrimExpr value = constraint.known_value(op->indices).value();
  //       known_subregion.push_back({touch_predicate, value});

  //       std::cout << "\t\t"
  //                 << "The constraint applies sometimes, if "
  //                 << (access_predicate && touch_predicate)
  //                 << ", but we still need to determine the value when "
  //                 << (access_predicate && !touch_predicate) << ", which simplifies to "
  //                 << analyzer_->Simplify((access_predicate && !touch_predicate)) << std::endl;

  //       access_predicate = access_predicate && !touch_predicate;
  //     } else {
  //       // This BufferTouch writes values to the buffer that we might
  //       // use, and we don't know what those values are.  Therefore,
  //       // cannot simplify out the buffer access.
  //       break;
  //     }
  //   }

  //   std::cout << "\t"
  //             << "Couldn't find a constraint that applies for " << access_predicate
  //             << ", cannot construct an expression for this BufferLoad" << std::endl;

  //   return NullOpt;
  // }

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

  std::cout << "Beginning search from control blocks [";
  bool is_first = true;
  for (const auto& i : to_visit) {
    if (is_first) {
      is_first = false;
    } else {
      std::cout << ", ";
    }
    std::cout << i;
  }
  std::cout << "]" << std::endl;

  std::unordered_map<size_t, std::vector<BufferTouchPattern::BufferConstraint>> known_after_block;

  Analyzer analyzer;

  Map<Var, Range> all_free_parameters = GetAllFreeParameters();
  analyzer.Bind(iterator_ranges_);
  analyzer.Bind(all_free_parameters);

  while (to_visit.size()) {
    size_t visiting = *to_visit.begin();
    to_visit.erase(visiting);
    ControlFlowBlock& block = control_flow_[visiting];

    std::cout << "Visiting control block " << visiting << ", block.index = " << block.index
              << ", block.name = '" << block.name << "'" << std::endl;

    // Step 0: Pull in prior knowns from the predecessors

    std::cout << "\t"
              << "Constructing prior knowns from predecessors [";
    for (size_t i = 0; i < block.predecessors.size(); i++) {
      if (i) {
        std::cout << ", ";
      }
      const auto& pred = block.predecessors[i];
      if (visited_once.count(pred.from_index)) {
        std::cout << pred.from_index;
      } else {
        std::cout << "(" << pred.from_index << ")";
      }
    }
    std::cout << "]" << std::endl;

    // bool all_predecessors_visited = true;
    // for (const auto& predecessor : block.predecessors) {
    //   if (!visited_once.count(predecessor.from_index)) {
    //     all_predecessors_visited = false;
    //     break;
    //   }
    // }

    auto normalize_simplify = [&](std::vector<BufferTouchPattern::BufferConstraint> priors) {
      for (auto& prior : priors) {
        prior.predicate.expression_ =
            SimplifyUsingAndOfOrs(prior.predicate.expression_.value(), &analyzer);
      }
      return priors;
    };

    auto prior_knowns = [&]() -> std::vector<BufferTouchPattern::BufferConstraint> {
      ICHECK_LE(block.predecessors.size(), 2) << "Each block should have at most two predecessors";

      auto remap_priors = [&](std::vector<BufferTouchPattern::BufferConstraint> priors,
                              Map<Var, PrimExpr> var_remap) {
        if (var_remap.size()) {
          for (auto& prior : priors) {
            PrimExpr before_remap = prior.predicate.expression_.value();
            PrimExpr after_remap = Substitute(before_remap, var_remap);
            if (!before_remap.same_as(after_remap)) {
              prior.predicate.expression_ = SimplifyUsingAndOfOrs(after_remap, &analyzer);
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
        std::cout << "\t\t"
                  << "No predecessor blocks, no known values" << std::endl;
        return {};
      } else if (block.predecessors.size() == 1) {
        // Block has only a single predecessor
        const auto& pred = block.predecessors[0];
        size_t prev_index = pred.from_index;
        std::cout << "\t\t"
                  << "Only " << prev_index << " as predecessor, copying priors" << std::endl;
        auto it = known_after_block.find(prev_index);
        if (it != known_after_block.end()) {
          return remap_priors(it->second, pred.var_remap);
        } else {
          return {};
        }
      }

      const auto& pred_a = block.predecessors[0];
      const auto& pred_b = block.predecessors[1];

      auto it_a = known_after_block.find(pred_a.from_index);
      auto it_b = known_after_block.find(pred_b.from_index);
      if (it_a == known_after_block.end() && it_b == known_after_block.end()) {
        std::cout << "\t\t"
                  << "Neither predicate visited, no known values" << std::endl;
        return {};
      } else if (it_a == known_after_block.end()) {
        std::cout << "\t\t"
                  << "Pred " << pred_b.from_index << " has been visited, but not "
                  << pred_a.from_index << ".  Using knowns from " << pred_b.from_index << std::endl;
        auto out = it_b->second;
        out = remap_priors(out, pred_b.var_remap);
        if (pred_a.predicate && pred_b.predicate) {
          out = add_condition(out, pred_a.predicate.value() || pred_b.predicate.value());
        }

        std::cout << "\t\t"
                  << "After adjusting, predecessor #" << pred_a.from_index << " provides "
                  << out.size() << " knowns" << std::endl;
        for (const auto& prior : out) {
          std::cout << "\t\t\t"
                    << "Buffer " << prior.buffer->name << " is " << prior.known_value << " where "
                    << prior.predicate << std::endl;
        }

        out = normalize_simplify(out);

        std::cout << "\t\t"
                  << "After adjusting and normalizing, predecessor #" << pred_a.from_index
                  << " provides " << out.size() << " knowns" << std::endl;
        for (const auto& prior : out) {
          std::cout << "\t\t\t"
                    << "Buffer " << prior.buffer->name << " is " << prior.known_value << " where "
                    << prior.predicate << std::endl;
        }

        return out;
      } else if (it_b == known_after_block.end()) {
        std::cout << "\t\t"
                  << "Pred " << pred_a.from_index << " has been visited, but not "
                  << pred_b.from_index << ".  Using knowns from " << pred_a.from_index << std::endl;
        auto out = it_a->second;
        out = remap_priors(out, pred_a.var_remap);
        if (pred_a.predicate && pred_b.predicate) {
          out = add_condition(out, pred_a.predicate.value() || pred_b.predicate.value());
        }

        std::cout << "\t\t"
                  << "After adjusting, predecessor #" << pred_b.from_index << " provides "
                  << out.size() << " knowns" << std::endl;
        for (const auto& prior : out) {
          std::cout << "\t\t\t"
                    << "Buffer " << prior.buffer->name << " is " << prior.known_value << " where "
                    << prior.predicate << std::endl;
        }

        out = normalize_simplify(out);

        std::cout << "\t\t"
                  << "After adjusting and normalizing, predecessor #" << pred_b.from_index
                  << " provides " << out.size() << " knowns" << std::endl;
        for (const auto& prior : out) {
          std::cout << "\t\t\t"
                    << "Buffer " << prior.buffer->name << " is " << prior.known_value << " where "
                    << prior.predicate << std::endl;
        }

        return out;
      }

      // ICHECK(!(pred_a.predicate && pred_b.predicate))
      //     << "Predicated predecessor is attempted to be proven a special case of the other, "
      //     << "which is treated as the general case.  "
      //     << "Should not have a predicate specified for both predecessors.";

      std::cout << "\t\t"
                << "Predecessor #" << pred_a.from_index << " provides " << it_a->second.size()
                << " knowns" << std::endl;
      for (const auto& prior : it_a->second) {
        std::cout << "\t\t\t"
                  << "Buffer " << prior.buffer->name << " is " << prior.known_value << " where "
                  << prior.predicate << std::endl;
      }

      std::cout << "\t\t"
                << "Predecessor #" << pred_b.from_index << " provides " << it_b->second.size()
                << " knowns" << std::endl;
      for (const auto& prior : it_b->second) {
        std::cout << "\t\t\t"
                  << "Buffer " << prior.buffer->name << " is " << prior.known_value << " where "
                  << prior.predicate << std::endl;
      }

      auto priors_a = remap_priors(it_a->second, pred_a.var_remap);
      auto priors_b = remap_priors(it_b->second, pred_b.var_remap);

      std::cout << "\t\t"
                << "After remapping with " << pred_a.var_remap << ", predecessor #"
                << pred_a.from_index << " provides " << priors_a.size() << " knowns" << std::endl;
      for (const auto& prior : priors_a) {
        std::cout << "\t\t\t"
                  << "Buffer " << prior.buffer->name << " is " << prior.known_value << " where "
                  << prior.predicate << std::endl;
      }

      std::cout << "\t\t"
                << "After remapping with " << pred_b.var_remap << ", predecessor #"
                << pred_b.from_index << " provides " << priors_b.size() << " knowns" << std::endl;
      for (const auto& prior : priors_b) {
        std::cout << "\t\t\t"
                  << "Buffer " << prior.buffer->name << " is " << prior.known_value << " where "
                  << prior.predicate << std::endl;
      }

      std::vector<BufferTouchPattern::BufferConstraint> output;
      if (pred_a.predicate && pred_b.predicate) {
        std::cout << "\t\t"
                  << "Merging disjoint predecessors " << pred_a.from_index << " ("
                  << pred_a.predicate << ") and " << pred_b.from_index << " (" << pred_b.predicate
                  << ")" << std::endl;
        output = BufferTouchPattern::BufferConstraint::MergePredecessorConstraintsWithPostcondition(
            priors_a, priors_b, pred_a.predicate.value(), pred_b.predicate.value(), &analyzer);
      } else if (pred_a.predicate) {
        std::cout << "\t\t"
                  << "Merging with predecessor " << pred_a.from_index << " as a special case of "
                  << pred_b.from_index << " with predicate " << pred_a.predicate << std::endl;
        output = BufferTouchPattern::BufferConstraint::MergePredecessorConstraints(
            priors_a, priors_b, pred_a.predicate, &analyzer);
      } else if (pred_b.predicate) {
        std::cout << "\t\t"
                  << "Merging with predecessor " << pred_b.from_index << " as a special case of "
                  << pred_a.from_index << " with predicate " << pred_b.predicate << std::endl;
        output = BufferTouchPattern::BufferConstraint::MergePredecessorConstraints(
            priors_b, priors_a, pred_b.predicate, &analyzer);
      } else {
        std::cout << "\t\t"
                  << "Merging with predecessors " << pred_a.from_index << " and "
                  << pred_b.from_index << " as two distinct predecessors, no known post-conditions"
                  << std::endl;
        output = BufferTouchPattern::BufferConstraint::MergePredecessorConstraints(
            priors_a, priors_b, NullOpt, &analyzer);
      }

      std::cout << "\t\t"
                << "After merging predecessors, have " << output.size() << " known statements"
                << std::endl;
      for (const auto& prior : output) {
        std::cout << "\t\t\t"
                  << "Block " << visiting << ", Buffer " << prior.buffer->name << " is "
                  << prior.known_value << " where " << prior.predicate << std::endl;
      }

      return output;
    }();

    // std::vector<BufferTouchPattern::BufferConstraint> prior_knowns;
    // bool found_first_predecessor = false;
    // for (const auto& predecessor : block.predecessors) {
    //   if (visited_once.count(predecessor.from_index)) {
    //     auto it = known_after_block.find(predecessor.from_index);
    //     if (it != known_after_block.end()) {
    //       // TODO: Update predecessor with remap.
    //       std::vector<BufferTouchPattern::BufferConstraint> priors = it->second;

    //       std::cout << "\t\t"
    //                 << "Before remap, predecessor block #" << predecessor.from_index
    //                 << " provides " << priors.size() << " knowns" << std::endl;
    //       for (const auto& prior : priors) {
    //         std::cout << "\t\t\t"
    //                   << "Buffer " << prior.buffer->name << " is " << prior.known_value
    //                   << " where " << prior.predicate << std::endl;
    //         ;
    //       }

    //       // The first time through, we assume that the values of
    //       // the only known predecessor are true for all cases.  The
    //       // second time through, with a predecessor that may derive
    //       // from the initial known, we verify that the same known
    //       // value is determined from both the initial and
    //       // subsequent usage.
    //       if (all_predecessors_visited) {
    //         for (auto& prior : priors) {
    //           if (prior.predicate.expression_) {
    //             prior.predicate.Remap(predecessor.var_remap);
    //             // prior.predicate.expression_ =
    //             //     prior.predicate.expression_.value() && predecessor.predicate;
    //             prior.predicate.Simplify(&analyzer);
    //           }
    //         }
    //       }

    //       std::cout << "\t\t"
    //                 << "After remap, predecessor block #" << predecessor.from_index
    //                 << " provides " << priors.size() << " knowns" << std::endl;
    //       for (const auto& prior : priors) {
    //         std::cout << "\t\t\t"
    //                   << "Buffer " << prior.buffer->name << " is " << prior.known_value
    //                   << " where " << prior.predicate << std::endl;
    //         ;
    //       }

    //       if (found_first_predecessor) {
    //         prior_knowns = BufferTouchPattern::BufferConstraint::MergePredecessorConstraints(
    //             prior_knowns, priors);
    //       } else {
    //         prior_knowns = std::move(priors);
    //         found_first_predecessor = true;
    //       }
    //     }
    //   }
    // }

    std::cout << "\t"
              << "Visiting control block " << visiting << " starts with " << prior_knowns.size()
              << " prior-block statements" << std::endl;
    for (const auto& known : prior_knowns) {
      std::cout << "\t\t"
                << "Buffer " << known.buffer->name << " where " << known.predicate
                << " is equal to " << known.known_value << std::endl;
    }

    // Step 1: Propagate the known values from before the control
    // block into known values for the control block.

    std::vector<BufferTouchPattern::BufferConstraint> new_knowns;

    for (auto& touch : block.touch_points) {
      if (touch.touch_type == BufferTouch::AccessType::Read) {
        continue;
      }

      Array<Var> axis_vars = touch.known_value.parameter_vars_;
      PrimExpr predicate = touch.predicate(axis_vars).value();

      PrimExpr known_value = touch.known_value(axis_vars).value();
      auto regions = BufferRegionCollector::Collect(prior_knowns, {predicate, known_value},
                                                    all_free_parameters, &analyzer);

      std::cout << "\t"
                << "Regions of interest are [";
      for (size_t i = 0; i < regions.size(); i++) {
        if (i) {
          std::cout << ", ";
        }
        std::cout << regions[i].region_predicate;
      }
      std::cout << "]" << std::endl;

      for (const auto& region : regions) {
        std::cout << "\t\t"
                  << "Within region " << region.region_predicate << std::endl;
        PrimExpr updated_predicate = BufferRegionValueReplacer::Apply(
            region.known_values, region.region_predicate && predicate, &analyzer);
        std::cout << "\t\t\t"
                  << "Buffer predicate && region predicate simplifies from "
                  << (region.region_predicate && predicate) << " to " << updated_predicate
                  << std::endl;

        // updated_predicate = ConvertToAndOfOrs(updated_predicate);
        // updated_predicate = analyzer.Simplify(updated_predicate);
        updated_predicate = SimplifyUsingAndOfOrs(updated_predicate, &analyzer);
        std::cout << "\t\t\t"
                  << "Buffer predicate && region predicate is normalized and further simplified to "
                  << updated_predicate << std::endl;
        PrimExpr updated_value =
            BufferRegionValueReplacer::Apply(region.known_values, known_value, &analyzer);

        std::cout << "\t\t\t"
                  << "Known value simplifies from " << known_value << " to " << updated_value
                  << std::endl;

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

    std::cout << "\t"
              << "Visiting control block " << visiting << " resulted in " << new_knowns.size()
              << " new post-block statements" << std::endl;
    for (const auto& known : new_knowns) {
      std::cout << "\t\t"
                << "Buffer " << known.buffer->name << " where " << known.predicate
                << " is equal to " << known.known_value << std::endl;
    }

    // Step 2: Propagate all constraints through to the end of the
    // control block.

    // Step 2a: If pre-block constraints for successor blocks have no
    // known value for this predicate, copy the post-block known value
    // to it.

    // Step 2b: If pre-block constraints for successor blocks already have an
    // initially known value, and if that is not an input value
    // (either initial expression or T.assume), check if they are
    // compatible.  If incompatible, repalce with NullOpt.

    // Step 2c: If pre-block constraints for successor blocks already
    // have an initially known value, handle partial overlaps.

    auto post_knowns = BufferTouchPattern::BufferConstraint::MergeSequentialConstraints(
        prior_knowns, new_knowns, &analyzer);

    std::cout << "\t"
              << "Visiting control block " << visiting << " resulted in " << post_knowns.size()
              << " total post-block statements at the end" << std::endl;
    for (const auto& known : post_knowns) {
      std::cout << "\t\t"
                << "Buffer " << known.buffer->name << " where " << known.predicate
                << " is equal to " << known.known_value << std::endl;
    }

    for (auto& known : post_knowns) {
      known.predicate = known.predicate.WithoutFreeParameters();
      known.predicate.expression_ =
          SimplifyUsingAndOfOrs(known.predicate.expression_.value(), &analyzer);
    }

    std::cout << "\t"
              << "Visiting control block " << visiting << " resulted in " << post_knowns.size()
              << " total simplified post-block statements at the end" << std::endl;
    for (const auto& known : post_knowns) {
      std::cout << "\t\t"
                << "Buffer " << known.buffer->name << " where " << known.predicate
                << " is equal to " << known.known_value << std::endl;
    }

    // Step 4: If any changes are made to the pre- values of the
    // successor block, mark the successor block as needing to be
    // visited.

    bool has_updated_post = [&]() -> bool {
      auto it = known_after_block.find(visiting);

      // std::cout << "\t"
      //           << "Checking if visiting block " << visiting << " has resulting in new
      //           information"
      //           << std::endl;

      if (it == known_after_block.end()) {
        // std::cout << "\t\t"
        //           << "First time that " << visiting << " has been examined" << std::endl;
        return true;
      }

      const auto& previous_post_knowns = it->second;

      if (post_knowns.size() != previous_post_knowns.size()) {
        // std::cout << "\t\t"
        //           << "Found " << post_knowns.size() << " this time, but only "
        //           << previous_post_knowns.size() << " last time" << std::endl;
        return true;
      }

      for (size_t i = 0; i < post_knowns.size(); i++) {
        if (!post_knowns[i].IsEquivalentTo(previous_post_knowns[i], &analyzer)) {
          // std::cout << "\t\t"
          //           << "Found different constraint #" << i << " from last time" << std::endl;
          return true;
        }
      }

      // std::cout << "\t\t"
      //           << "Found same resulting constraints as last time" << std::endl;
      return false;
    }();

    // TODO: Have a maximum number of times that blocks may be
    // visited, to guard against infinite loops.
    if (has_updated_post) {
      known_after_block[visiting] = post_knowns;
      for (size_t successor : block.successors) {
        to_visit.insert(successor);
        std::cout << "\t"
                  << "Queuing " << successor << " to be visited, to_visit = [";
        bool first = true;
        for (const auto& index : to_visit) {
          if (first) {
            first = false;
          } else {
            std::cout << ", ";
          }
          std::cout << index;
        }
        std::cout << "]" << std::endl;
      }
    }

    visited_once.insert(visiting);

    // std::cout << "\t" << to_visit.size() << " remaining to be visited" << std::endl;
  }

  // for (size_t i = 0; i < control_flow_.size(); i++) {
  //   auto it = known_after_block.find(i);
  //   if (it != known_after_block.end()) {
  //     std::cout << "After block " << i << ", there are known facts about " << it->second.size()
  //               << " buffers" << std::endl;
  //     for (const auto& constraint : it->second) {
  //       std::cout << "\t"
  //                 << "Buffer " << constraint.buffer->name << " where " << constraint.predicate
  //                 << " is equal to " << constraint.known_value << std::endl;
  //     }
  //   } else {
  //     std::cout << "Block " << i << " was never visited." << std::endl;
  //   }
  // }

  // Fill in any unvisited blocks with empty constraint sets
  for (size_t i = 0; i < control_flow_.size(); i++) {
    auto it = known_after_block.find(i);
    if (it == known_after_block.end()) {
      known_after_block[i];
    }
  }

  constraint_lookup_ = known_after_block;
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
  // std::cout << "Attempting to simplify " << expr << std::endl;

  size_t context_index = [&]() {
    // auto it = context_lookup_.find(context.get());
    // ICHECK(it != context_lookup_.end())
    //     << "Context did not occur in the Stmt provided to BufferTouchPattern's constructor";

    auto it = control_flow_lookup_.find(context.get());
    ICHECK(it != control_flow_lookup_.end())
        << "Context did not occur in the Stmt provided to BufferTouchPattern's constructor";
    return it->second;
  }();

  // std::cout << "\t"
  //           << "Context index for this statement is " << context_index << std::endl;

  PrimExpr constraint = Bool(true);
  for (const auto& known : non_buffer_assumptions_) {
    constraint = constraint && known;
  }
  With<ConstraintContext> constraint_context(analyzer, constraint);

  // auto it = constraint_lookup_.find(context_index);
  // ICHECK(it != constraint_lookup_.end());
  // auto constraints = it->second;
  // std::cout << "\t"
  //           << "There are " << constraints.size() << " known constraints at this location"
  //           << std::endl;
  // for (const auto& constraint : constraints) {
  //   std::cout << "\t\t"
  //             << "Buffer " << constraint.buffer->name << " is " << constraint.known_value
  //             << " for predicate " << constraint.predicate << std::endl;
  // }

  BufferConstraintApply mutator(*this, context_index, analyzer);
  expr = mutator(expr);

  // auto it = constraint_lookup_.find(context_index);
  // if (it != constraint_lookup_.end()) {
  //   // TODO: Should the Bool(true) instead be the predicate necessary
  //   // to reach this statement?  Might need to track that in the
  //   // ControlFlowBlock.
  //   auto regions =
  //       BufferRegionCollector::Collect(it->second, {expr}, GetAllFreeParameters(), analyzer);

  //   std::cout << "\t"
  //             << "Regions of interest are [";
  //   for (size_t i = 0; i < regions.size(); i++) {
  //     if (i) {
  //       std::cout << ", ";
  //     }
  //     std::cout << regions[i].region_predicate;
  //   }
  //   std::cout << "]" << std::endl;

  //   if (regions.size() == 1) {
  //     std::cout << "\t"
  //               << "Only one region exists, " << regions[0].region_predicate << std::endl;
  //     if (analyzer->CanProve(regions[0].region_predicate)) {
  //       if (auto opt_expr =
  //               BufferRegionValueReplacer::Apply(regions[0].known_values, expr, analyzer)) {
  //         return opt_expr.value();
  //       }
  //     } else {
  //       std::cout << "\t"
  //                 << "Cannot prove predicate of the region of known values" << std::endl;
  //     }
  //   }
  // }

  // std::cout << "Attempting to simplify " << expr << " in the context it appears" << std::endl;

  // BufferConstraintSubstituter mutator(*this, context_index, analyzer);
  // expr = mutator(expr);

  // std::cout << "\t"
  //           << "After substituting known buffer information, expr = " << expr << std::endl;

  // PrimExpr constraint = Bool(true);
  // for (const auto& known : non_buffer_assumptions_) {
  //   constraint = constraint && known;
  // }
  // With<ConstraintContext> constraint_context(analyzer, constraint);

  // std::cout << "\t"
  //           << "In this context, we have an additional constraint that " << constraint <<
  //           std::endl;
  // std::cout << "\t"
  //           << "So the expression simplifies to " << analyzer->Simplify(expr) << std::endl;

  return analyzer->Simplify(expr);
}

Optional<PrimExpr> BufferTouchPattern::KnownValue(const tir::BufferStore& store) const {
  // TODO: Replace calls with SimplifyInContext
  return NullOpt;

  // Array<PrimExpr> values;

  // for (auto it = touch_points_.rbegin(); it != touch_points_.rend(); it++) {
  //   if (it->node.same_as(store)) {
  //     if (auto opt = KnownValue(it, store->indices)) {
  //       values.push_back(opt.value());
  //     } else {
  //       // If a store based on this statement doesn't have a known
  //       // value, then the store overall doesn't have a known value.
  //       return NullOpt;
  //     }
  //   }
  // }

  // // For the store to have a known value, all touches resulting from
  // // this statement must result in the same value.
  // //
  // // TODO: Handle multiple access from a single statement
  // // (e.g. start/finish of while loop) that may have the same result.
  // // Should attempt to prove that each touch was preceded by the same
  // // known value.
  // if (values.size() == 1) {
  //   return values[0];
  // } else {
  //   return NullOpt;
  // }
}

Optional<PrimExpr> BufferTouchPattern::KnownValue(const tir::BufferLoad& load) const {
  return NullOpt;
}

// Optional<PrimExpr> BufferTouchPattern::KnownValue(
//     std::vector<BufferTouch>::const_reverse_iterator access_iter,
//     const Array<PrimExpr>& indices) const {
//   for (auto it = access_iter + 1; it != touch_points_.rend(); it++) {
//     // If a previous write touched the same indices, then we can use
//     // the recorded values at those indices.
//     if ((it->touch_type == BufferTouch::AccessType::Write ||
//          it->touch_type == BufferTouch::AccessType::Assume) &&
//         access_iter->IsSubsetOf(*it)) {
//       return it->known_value(indices);
//     }
//   }
//   return NullOpt;
// }

void BufferTouchPattern::RemoveTouches(const tir::BufferStore& store) {
  touch_points_.erase(std::remove_if(touch_points_.begin(), touch_points_.end(),
                                     [&](const auto& touch) { return touch.node.same_as(store); }));
  // TODO: Update context_lookup_
}

}  // namespace arith
}  // namespace tvm
