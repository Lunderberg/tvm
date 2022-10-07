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
 * \file buffer_touch_pattern.h
 * \brief Utility for extracting and interacting with buffer touch points
 */

#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_solver.h>
#include <tvm/runtime/container/array.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/var.h>

#ifndef TVM_ARITH_BUFFER_TOUCH_PATTERN_H_
#define TVM_ARITH_BUFFER_TOUCH_PATTERN_H_

namespace tvm {
namespace arith {

// Utility for expressing a parametric expression
class ParametrizedExpression {
 public:
  ParametrizedExpression(Array<tir::Var> parameter_vars, Optional<PrimExpr> expression);

  /* \brief Evaluate the expression using the provided arguments
   */
  Optional<PrimExpr> operator()(const Array<PrimExpr>& args) const;

  /* \brief Evaluate the expression using the provided arguments
   */
  Optional<PrimExpr> operator()(const Array<Var>& args) const;

  bool IsDefined() const { return static_cast<bool>(expression_); }
  bool IsConstant() const;

  friend std::ostream& operator<<(std::ostream& os, const ParametrizedExpression& expr);

  // protected:
  Array<tir::Var> parameter_vars_;
  Optional<PrimExpr> expression_;

  // TODO: Avoid having this as a friend class everywhere.
  friend class BufferTouchPattern;
};

// Utility for expressing an boolean condition in terms of variable
// parameters.
class Predicate : public ParametrizedExpression {
 public:
  Predicate(Array<tir::Var> parameter_vars, Optional<PrimExpr> expression);

  /* \brief Checks if this Predicate is a subset of another predicate
   *
   * Returns true if this predicate can be proven to be a subset of
   * the other subset.  Returns false if it cannot be proven to be a
   * subset of ther other subset.
   */
  bool IsSubsetOf(const Predicate& other, Analyzer* analyzer) const;

  /* \brief Checks if this Predicate is distinct of another predicate
   *
   * Returns true if it can be proven that the two predicates cannot
   * be simultaneously true.  Returns false if it cannot be proven
   * that the two predicates are distinct.
   */
  bool IsDistinctFrom(const Predicate& other, Analyzer* analyzer) const;

  /* \brief The difference of two predicates
   *
   * Returns a predicate that is true whenever this predicate is true
   * and the other predicate is false.
   */
  Predicate Difference(const Predicate& other, Analyzer* analyzer) const;

  /* \brief The difference of two predicates
   *
   * Returns a predicate that is true whenever this predicate is true
   * and the other predicate is false.
   */
  Predicate Intersection(const Predicate& other, Analyzer* analyzer) const;

  Predicate Union(const Predicate& other, Analyzer* analyzer) const;

  /* \brief Remap variables within the predicate
   *
   * Remaps variables in the predicate according to the supplied map.
   *
   * \param var_remap A map of variables to the expression that
   * replaces them.
   */
  void Remap(const Map<Var, PrimExpr>& var_remap);

  /* \brief Simplify the predicate using the supplied analyzer */
  void Simplify(Analyzer* analyzer);

  friend std::ostream& operator<<(std::ostream& os, const Predicate& expr);

  /* \brief Checks if this Predicate can be statically proven
   *
   * This is preferred over using
   * `analyzer->CanProve(predicate(args))`, as it handles the free
   * parameters that may exist for the predicate.
   */
  bool CanProve(Array<PrimExpr> args, Analyzer* analyzer) const;

  /* \brief Checks if this Predicate can be statically proven
   *
   * This is preferred over using
   * `analyzer->CanProve(!predicate(args))`, as it handles the free
   * parameters that may exist for the predicate.
   */
  bool CanDisprove(Array<PrimExpr> args, Analyzer* analyzer) const;

  /* \brief Checks if this Predicate is always false */
  bool IsAlwaysFalse() const;

  /* \brief Generate a predicate without free parameters
   *
   * The returned predicate will be true for a subset of this predicate.
   */
  Predicate WithoutFreeParameters(const Map<Var, Range>& free_params) const;
};

class BufferTouch {
 public:
  enum class AccessType {
    Read,
    Write,
    Assume,
  };

  BufferTouch(tir::Buffer buffer, Array<Var> axis_vars, PrimExpr predicate, AccessType touch_type,
              PrimExpr known_value);

  /* \brief Checks if this Predicate is a subset of another predicate
   *
   * Returns true if the indices accessed by this touch are a subset
   * of predicate is true can be proven to be a subset of the other
   * subset.  Returns false if it cannot be proven to be a subset of
   * ther other subset.
   */
  bool IsSubsetOf(const BufferTouch& other, Analyzer* analyzer) const;

  /* \brief Checks if this Predicate is distinct of another predicate
   *
   * Returns true if it can be proven that the two predicates cannot
   * be simultaneously true.  Returns false if it cannot be proven
   * that the two predicates are distinct.
   */
  bool IsDistinctFrom(const BufferTouch& other, Analyzer* analyzer) const;

  friend std::ostream& operator<<(std::ostream& os, const BufferTouch& expr);

 private:
  void CheckSameAxisVars(const BufferTouch& other) const;

  tir::Buffer buffer;
  Array<tir::Var> axis_vars;
  PrimExpr predicate;
  PrimExpr value;

  AccessType touch_type;

  friend class BufferTouchPattern;
};

struct BufferConstraint {
  BufferConstraint(tir::Buffer buffer, Array<tir::Var> axis_vars, PrimExpr predicate,
                   Optional<PrimExpr> value);
  BufferConstraint(tir::Buffer buffer, Array<tir::Var> axis_vars, PrimExpr predicate,
                   PrimExpr value);

  tir::Buffer buffer;
  Array<tir::Var> axis_vars;
  PrimExpr predicate;
  Optional<PrimExpr> value;

  void CheckSameAxisVars(const BufferConstraint& other) const;

  friend std::ostream& operator<<(std::ostream& os, const BufferConstraint& obj);

  bool IsDistinctFrom(const BufferConstraint& other, Analyzer* analyzer) const;

  void OverwriteBy(const BufferConstraint& other, Analyzer* analyzer);

  bool IsEquivalentTo(const BufferConstraint& other, Analyzer* analyzer) const;
};

struct BufferState {
  std::vector<BufferConstraint> constraints;

  /* \brief Merge constraints from multiple disjoint predecessors */
  static BufferState MergePredecessorConstraintsWithPostcondition(const BufferState& a,
                                                                  const BufferState& b,
                                                                  PrimExpr a_condition,
                                                                  PrimExpr b_condition,
                                                                  Analyzer* analyzer);

  /* \brief Merge constraints from multiple possible-conflicting predecessors */
  static BufferState MergePredecessorConstraints(const BufferState& a, const BufferState& b,
                                                 Analyzer* analyzer);

  /* \brief Merge constraints that produce the same known value */
  static BufferState MergeDisjointConstraints(BufferState constraints, Analyzer* analyzer);

  /* \brief Merge constraints, where "after" may overwrite "before" */
  static BufferState MergeSequentialConstraints(const BufferState& before, const BufferState& after,
                                                Analyzer* analyzer);
};

class BufferTouchPattern {
 public:
  /* \brief Extract the touch pattern from a TIR statement
   */
  explicit BufferTouchPattern(const tir::Stmt& stmt);

  /* \brief Check if a write is overwritten without impacting final results
   *
   * \param store The store to be examined
   *
   * \return True if the specified store can be proven to be
   * overwritten without contributing to any later statements.
   * Returns false otherwise.
   */
  bool IsOverwrittenWithoutEffect(const tir::BufferStore& store, Analyzer* analyzer) const;

  /* \brief Simplify the expression, assuming it occurs within the given context
   *
   * \param expr The expression to be simplified.  Does not need to
   * have occurred within the statement used to construct this
   * BufferTouchPattern.
   *
   * \param context The statement where this expression occurred, or
   * is to be inserted.  Must occur within the statement used to
   * construct this BufferTouchPattern.
   *
   * \param analyzer The analyzer to be used for simplifications
   *
   * \returns The simplified statement
   *
   */
  PrimExpr SimplifyInContext(PrimExpr expr, const tir::Stmt& context, Analyzer* analyzer) const;

  /* \brief Remove all touches associated with a specific write
   *
   * If a pass removes a write from the statement being examined, then
   * it should no longer contribute to any analysis.  This function
   * allows a store to be marked as removed.
   *
   * \param store The buffer touch to be removed from future analysis.
   */
  void RemoveTouches(const tir::BufferStore& store);

 private:
  friend std::ostream& operator<<(std::ostream& os, const BufferTouchPattern& pattern);

  // private:
 public:
  void ForwardPropagateKnownValues();

  struct ControlFlowEdge {
    /* \brief The source block of the control flow edge
     *
     * Lookup index into `control_flow_`
     */
    size_t from_index;

    /*! \brief Variable remaps
     *
     * e.g. Replacing loop iterator `i` with `i-delta` when following an
     * edge from the end of a loop to the beginning of the loop.
     */
    Map<Var, PrimExpr> var_remap;

    /*! \brief Predicate that must to true when following this edge
     *
     * This is applied after variable remapping.  For example, `i >
     * loop_min` when following the an edge from the end of a loop to
     * the beginning of the loop.
     */
    Optional<PrimExpr> predicate;
  };

  struct ControlFlowBlock {
    BufferState known_at_block_start;
    BufferState known_at_block_end;

    /* \brief Buffer touches that occur within the block
     *
     * All buffer touches within a block can be treated as occurring
     * simultaneously.
     */
    std::vector<BufferTouch> touch_points;

    /* \brief The blocks that occur after this block
     *
     * Lookup index into `control_flow_`
     */
    std::vector<size_t> successors;

    /* \brief The blocks that occur before this block */
    std::vector<ControlFlowEdge> predecessors;
  };
  friend std::ostream& operator<<(std::ostream& os, const ControlFlowBlock& pattern);

  /* \brief The control flow that occurs within the analyzed statement */
  std::vector<ControlFlowBlock> control_flow_;

  /* \brief A lookup into control_flow_
   *
   * A map to look up the control flow block that contains the
   * statement.
   */
  std::unordered_map<const tir::StmtNode*, size_t> control_flow_lookup_;

  Map<Var, Range> free_predicate_parameters_;
  Map<Var, Range> iterator_ranges_;

  /* \brief Assumptions that do not depend on buffer values */
  std::vector<PrimExpr> non_buffer_assumptions_;

  friend class BufferTouchExtractor;
};

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_BUFFER_TOUCH_PATTERN_H_
