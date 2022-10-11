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
 * \file control_flow_graph.h
 * \brief Utility for extracting and interacting with buffer touch points
 */

#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_solver.h>
#include <tvm/runtime/container/array.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/var.h>

#ifndef TVM_ARITH_CONTROL_FLOW_GRAPH_H_
#define TVM_ARITH_CONTROL_FLOW_GRAPH_H_

namespace tvm {
namespace arith {

/*! \brief Represents an interaction with a buffer */
struct BufferTouch {
  enum class AccessType {
    Read,
    Write,
    Assume,
  };

  /*! \brief The buffer being touched */
  tir::Buffer buffer;

  /*! \brief A predicate that is true when this touch applies
   *
   * May be in terms of axis variables to indicate touches that impact
   * only a portion of a buffer.
   */
  PrimExpr predicate;

  /*! \brief The value in this buffer after the touch
   *
   * May be in terms of axis variables to indicate a known
   * non-constant value.  May be in terms of a BufferLoad to indicate
   * an unknown value.
   */
  PrimExpr value;

  /*! \brief How the buffer was interacted with
   *
   * When used as a constraint (e.g. in BufferState), should use
   * Assume.
   */
  AccessType touch_type{AccessType::Assume};

  /* \brief Checks if this touch affects a subset of indices of another
   *
   * Returns true if the indices accessed by this touch are a subset
   * of predicate is true can be proven to be a subset of the other
   * subset.  Returns false if it cannot be proven to be a subset of
   * ther other subset.
   */
  bool IsSubsetOf(const BufferTouch& other, Analyzer* analyzer) const;

  /* \brief Checks if this touch affects distinct indicates from another
   *
   * Returns true if it can be proven that the two predicates cannot
   * be simultaneously true.  Returns false if it cannot be proven
   * that the two predicates are distinct.
   */
  bool IsDistinctFrom(const BufferTouch& other, Analyzer* analyzer) const;

  /* \brief Checks if this touch affects distinct indicates from another
   *
   * Returns true if it can be proven that the two predicates cannot
   * be simultaneously true.  Returns false if it cannot be proven
   * that the two predicates are distinct.
   */
  bool IsEquivalentTo(const BufferTouch& other, Analyzer* analyzer) const;

  friend std::ostream& operator<<(std::ostream& os, const BufferTouch& expr);
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
                    const std::vector<BufferTouch>& touch_points, Analyzer* analyzer);

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

  /* \brief Add known values provided by another state
   *
   * \param other The state with which to merge constraints
   *
   * \param analyzer The analyzer with which to simplify the result
   */
  void Union(const BufferState& other, Analyzer* analyzer);

  /* \brief Remove all known values not consistent with another state
   *
   * \param other The state with which to merge constraints
   *
   * \param analyzer The analyzer with which to simplify the result
   */
  void Intersection(const BufferState& other, Analyzer* analyzer);

  friend std::ostream& operator<<(std::ostream& os, const BufferState&);

 private:
  /*! \brief The known constraints */
  std::vector<BufferTouch> constraints;
};

class ControlFlowGraph {
 public:
  /* \brief Extract the touch pattern from a TIR statement
   */
  explicit ControlFlowGraph(const tir::Stmt& stmt);

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
  friend std::ostream& operator<<(std::ostream& os, const ControlFlowGraph& pattern);

  /*! \brief Propagate known values from known BufferStore/assume
   *  subsequent control flow blocks
   */
  void ForwardPropagateKnownValues();

  struct ControlFlowEdge {
    /* \brief The source block of the control flow edge
     *
     * Lookup index into `control_flow_`
     */
    size_t from_index;

    /*! \brief Variable remaps
     *
     * e.g. Replacing loop iterator `i` with `i-1` when following an
     * edge from the end of a loop to the beginning of the loop.
     */
    Map<Var, PrimExpr> var_remap;

    /*! \brief Condition that must to true after following this edge
     *
     * This is applied after variable remapping.  For example, `i >
     * loop_min` when following the an edge from the end of a loop to
     * the beginning of the loop.
     */
    Optional<PrimExpr> post_condition;
  };

  struct ControlFlowBlock {
    /*! \brief All known values prior to executing the block */
    BufferState known_at_block_start;

    /*! \brief All known values after executing the block */
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

  /*! \brief A map from free parameters to their range
   *
   * A BufferStore/BufferLoad has indices in terms of loop iterators,
   * while the internal BufferTouch must have predicate in terms of
   * the buffer's axes.  While converting to the internal BufferTouch,
   * reduction axes show up as free parameters.  Tracking the range of
   * the free parameters allows them to be removed later, by requiring
   * a predicate to be true for all values of the free parameters.
   */
  Map<Var, Range> free_predicate_parameters_;

  /*! \brief Ranges of iterators found in the analyzed statement */
  Map<Var, Range> iterator_ranges_;

  /* \brief A map from buffer to the variables representing positions
   * along the buffer's axes.
   *
   * This is stored here, rather than as part of the BufferState or
   * BufferTouch, to ensure that all access of a buffer use the same
   * variables to represent the buffer's axes, reducing the amount of
   * variable substitution required.
   */
  Map<tir::Buffer, Array<tir::Var>> axis_var_lookup_;

  /* \brief Assumptions that do not depend on buffer values
   *
   * These may be collected as part of the handling of `builtin::assume()`, and do not depend on any
   * buffer.  Since TIR only allows mutable values as part of buffers, these assumptions may be used
   * anywhere the
   */
  std::vector<PrimExpr> non_buffer_assumptions_;

  friend class ControlFlowGraphBuilder;
};

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_CONTROL_FLOW_GRAPH_H_
