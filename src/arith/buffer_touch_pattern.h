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
  friend class BufferConstraintSubstituter;
  friend class BufferTouchPattern;
};

// Utility for expressing an boolean condition in terms of variable
// parameters.
class Predicate : public ParametrizedExpression {
 public:
  Predicate(Array<tir::Var> parameter_vars, Optional<PrimExpr> expression,
            Map<tir::Var, Range> free_parameters);

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
  Predicate WithoutFreeParameters() const;

  /* \brief Boolean expression defining ranges of free parameters */
  PrimExpr FreeParameterConstraints() const;

  // private:
  Map<tir::Var, Range> free_parameters_;
};

class BufferTouch {
 public:
  enum class AccessType {
    Read,
    Write,
    Assume,
  };

  BufferTouch(tir::Buffer buffer, Predicate predicate, AccessType touch_type,
              ParametrizedExpression known_value, Array<PrimExpr> original_indices,
              Map<Var, PrimExpr> loop_var_to_axis_var, ObjectRef node);

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

  /* \brief Checks if this BufferTouch is a subset of another predicate
   *
   * Returns true if this is a buffer write that alters the value of
   * another buffer touch, where that buffer touch occurs earlier in
   * the body of a loop.
   *
   * \param preceding_in_body A BufferTouch that occurs at a preceding
   * location within the body of a loop.
   *
   * \param loop_var The loop iteration variable.
   *
   * \return True if the these read/writes may introduce a dependency
   * on a previous loop iteration, false otherwise.
   */
  bool ProvablyCrossLoopIndependent(const BufferTouch& preceding_in_body, const Var& loop_var,
                                    Analyzer* analyzer) const;

  friend std::ostream& operator<<(std::ostream& os, const BufferTouch& expr);

 private:
  tir::Buffer buffer;

  // TODO: Merge predicate/known_value into this class?
  Predicate predicate;
  AccessType touch_type;
  ParametrizedExpression known_value;
  Array<PrimExpr> original_indices;

  // Map usable to substitute out loop variables, resulting in
  // expressions in terms of the buffer axis variables.
  Map<Var, PrimExpr> loop_var_to_axis_var;

  // The BufferLoad or BufferStore object that caused this touch.
  ObjectRef node;

  // The statement in which this touch occurred.
  tir::Stmt stmt;

  friend class BufferTouchPattern;
  friend class BufferConstraintSubstituter;
};

class BufferTouchPattern {
 public:
  /* \brief Extract the touch pattern from a TIR statement
   */
  explicit BufferTouchPattern(const tir::Stmt& stmt);

  const std::vector<BufferTouch>& GetTouches() const { return touch_points_; }

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

  struct BufferConstraint {
    tir::Buffer buffer;
    Predicate predicate;
    ParametrizedExpression known_value;

    friend std::ostream& operator<<(std::ostream& os, const BufferConstraint& obj);

    bool IsDistinctFrom(const BufferConstraint& other, Analyzer* analyzer) const;

    void OverwriteBy(const BufferConstraint& other, Analyzer* analyzer);

    bool IsEquivalentTo(const BufferConstraint& other, Analyzer* analyzer) const;

    /* \brief Merge constraints that may overwrite each other.
     *
     * Assumes that "before" and "after" sets of constraints are
     * internally consistent.
     */
    static std::vector<BufferConstraint> MergeSequentialConstraints(
        const std::vector<BufferConstraint>& before, const std::vector<BufferConstraint>& after,
        Analyzer* analyzer);

    /*! \brief Simplify and remove overwritten constraints
     *
     * Given a vector of constraints, where later constraints may
     * overwrite earlier constraints, produce a set of disjoint
     * constraints representing the final state after all constraints
     * have been applied.
     *
     * \param constraints The vector of constraints, from oldest to newest.
     *
     * \return A set of disjoint constraints
     */
    static std::vector<BufferConstraint> SimplifyOverwrittenConstraints(
        std::vector<BufferConstraint> constraints, Analyzer* analyzer);

    /*! \brief Simplify disjoint
     *
     * Given a vector of disjoint constraints, merge any constraints
     * that produce the same known value.
     *
     * \param constraints The initial disjoing constraints.
     *
     * \return A set of disjoint constraints
     */
    static std::vector<BufferConstraint> MergeDisjointConstraints(
        std::vector<BufferConstraint> constraints, Analyzer* analyzer);

    /* \brief Merge constraints that jointly apply
     *
     * If a constraint applies to the same indices in the same buffer,
     * but cannot be shown to be the same value, it will be tracked as
     * a NullOpt, with no additional information tracked.
     *
     * \param a_condition Condition that is known to be true when
     */
    static std::vector<BufferConstraint> MergePredecessorConstraints(
        const std::vector<BufferConstraint>& a, const std::vector<BufferConstraint>& b,
        Optional<PrimExpr> a_condition, Analyzer* analyzer);

    /* \brief Merge constraints that jointly apply
     *
     * If a constraint applies to the same indices in the same buffer,
     * but cannot be shown to be the same value, it will be tracked as
     * a NullOpt, with no additional information tracked.
     *
     * \param a_condition Condition that is known to be true when
     * block A was the predecessor.
     *
     * \param b_condition Condition that is known to be true when
     * block B was the predecessor.
     */
    static std::vector<BufferConstraint> MergePredecessorConstraintsWithPostcondition(
        const std::vector<BufferConstraint>& a, const std::vector<BufferConstraint>& b,
        PrimExpr a_condition, PrimExpr b_condition, Analyzer* analyzer);
  };

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
    std::vector<BufferConstraint> known_at_block_start;
    std::vector<BufferConstraint> known_at_block_end;

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

  /* \brief An ordered list of buffer touches
   *
   * For all indices i and j, if i<j, then either buffer touch i
   * occurs sequentially before buffer touch j (e.g. for sequential
   * writes of a buffer), or buffer touch i and j have mutually
   * exclusive predicates (e.g. for writes within an if/else).
   */
  std::vector<BufferTouch> touch_points_;

  /* \brief A lookup into control_flow_
   *
   * A map to look up the control flow block that contains the
   * statement.
   */
  std::unordered_map<const tir::StmtNode*, size_t> control_flow_lookup_;

  Map<Var, Range> iterator_ranges_;

  /*! \brief All free parameters across all constraint predicates */
  Map<Var, Range> GetAllFreeParameters() const;

  /* \brief Assumptions that do not depend on buffer values */
  std::vector<PrimExpr> non_buffer_assumptions_;

  friend class BufferConstraintSubstituter;
  friend class BufferTouchExtractor;
};

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_BUFFER_TOUCH_PATTERN_H_
