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

class BufferTouch {
 public:
  enum class AccessType {
    Read,
    Write,
    Assume,
  };

  BufferTouch(tir::Buffer buffer, PrimExpr predicate, AccessType touch_type, PrimExpr known_value);

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
  tir::Buffer buffer;
  PrimExpr predicate;
  PrimExpr value;

  AccessType touch_type;

  friend class BufferTouchPattern;
  friend class BufferState;
};

struct BufferConstraint {
  BufferConstraint(tir::Buffer buffer, PrimExpr predicate, PrimExpr value);

  tir::Buffer buffer;
  PrimExpr predicate;
  PrimExpr value;

  friend std::ostream& operator<<(std::ostream& os, const BufferConstraint& obj);

  bool IsEquivalentTo(const BufferConstraint& other, Analyzer* analyzer) const;
};

/*! \brief Represents the known state of buffers at a specific point */
class BufferState {
 public:
  /*! Default constructor
   *
   * Initialize the buffer state with no known information.
   */
  BufferState() {}

  /*! \brief Replace BufferLoad instances with known values
   *
   * \param expr The expression to be updated.
   *
   * \param axis_var_lookup A map from buffer to the variables
   * representing positions along the buffer's axes.
   *
   * \param analyzer The analyzer to use when validating a
   * constraint's predicate.
   *
   * \returns The modified expression.  If no substitutions are made,
   * the original expression is returned.
   */
  PrimExpr SubstituteKnownBufferValues(PrimExpr expr,
                                       const Map<tir::Buffer, Array<tir::Var>>& axis_var_lookup,
                                       Analyzer* analyzer) const;

  /*! \brief Apply a condition to all known constraints
   *
   * For example, when propagating pre-loop constraints into the body
   * of a loop, add a condition that the loop iterator is zero.
   *
   * \param condition The condition to apply
   */
  void AddCondition(const PrimExpr& condition);

  /*! \brief Perform a variable substitution for all constraints
   *
   * For example, when propagating constraints from the end of a loop
   * to the beginning, replace `i` with `i-1`.
   *
   * \param var_remap The variable remapping to apply.
   */
  void Substitute(const Map<Var, PrimExpr>& var_remap);

  /*! \brief Simplify the predicate of all constraints
   *
   * \param analyzer The analyzer with which to simplify
   */
  void Simplify(Analyzer* analyzer);

  /*! \brief Update the known buffer values based on buffer touches
   *
   * For any Write or Assume touches, update the known values.  For
   * any Read touches, ignore.  Used to determine known values at the
   * end of a series of a control flow block, given the known values
   * at the start.
   *
   * \param axis_var_lookup A map from buffer to the variables
   * representing positions along the buffer's axes.
   *
   * \param touch_points The buffer touch points to apply
   *
   * \param analyzer The analyzer to use for simplifications
   */
  void ApplyTouches(const Map<tir::Buffer, Array<tir::Var>>& axis_var_lookup,
                    const std::vector<BufferTouch>& touch_points,
                    const Map<Var, Range>& free_predicate_parameters, Analyzer* analyzer);

  /*! \brief Remove free parameters from the constraints
   *
   * \param free_predicate_parameters
   *
   * \param analyzer The analyzer with which to simplify after removal
   */
  void RemoveFreeParameters(const Map<Var, Range>& free_predicate_parameters, Analyzer* analyzer);

  /*! \brief Check if two buffer states are equivalent
   *
   * \param other
   *
   * \param analyzer The analyzer used to check equality of PrimExpr
   *
   * \return True if the two states are provably equivalent, false otherwise.
   */
  bool IsEquivalentTo(const BufferState& other, Analyzer* analyzer) const;

  /* \brief Merge constraints from multiple disjoint predecessors */
  static BufferState Union(const BufferState& a, const BufferState& b, Analyzer* analyzer);

  /* \brief Merge constraints from multiple possible-conflicting predecessors */
  static BufferState Intersection(const BufferState& a, const BufferState& b, Analyzer* analyzer);

  friend std::ostream& operator<<(std::ostream& os, const BufferState&);

 private:
  /*! \brief The known constraints */
  std::vector<BufferConstraint> constraints;
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

  Map<tir::Buffer, Array<tir::Var>> axis_var_lookup_;

  /* \brief Assumptions that do not depend on buffer values */
  std::vector<PrimExpr> non_buffer_assumptions_;

  friend class BufferTouchExtractor;
};

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_BUFFER_TOUCH_PATTERN_H_
