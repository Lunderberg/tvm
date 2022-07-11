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
  ParametrizedExpression(Array<tir::Var> parameter_vars, PrimExpr expression);

  /* \brief Evaluate the expression using the provided arguments
   */
  PrimExpr operator()(Array<PrimExpr> args) const;

  bool IsDefined() const { return expression_.defined(); }
  bool IsConstant() const;

  friend std::ostream& operator<<(std::ostream& os, const ParametrizedExpression& expr);

 protected:
  Array<tir::Var> parameter_vars_;
  PrimExpr expression_;
};

// Utility for expressing an boolean condition in terms of variable
// parameters.
class Predicate : public ParametrizedExpression {
 public:
  Predicate(Array<tir::Var> parameter_vars, PrimExpr expression,
            Map<tir::Var, Range> free_parameters);

  /* \brief Checks if this Predicate is a subset of another predicate
   *
   * Returns true if this predicate can be proven to be a subset of
   * the other subset.  Returns false if it cannot be proven to be a
   * subset of ther other subset.
   */
  bool IsSubsetOf(const Predicate& other) const;

  /* \brief Checks if this Predicate is distinct of another predicate
   *
   * Returns true if it can be proven that the two predicates cannot
   * be simultaneously true.  Returns false if it cannot be proven
   * that the two predicates are distinct.
   */
  bool IsDistinctFrom(const Predicate& other) const;

  /* \brief The difference of two predicates
   *
   * Returns a predicate that is true whenever this predicate is true
   * and the other predicate is false.
   */
  Predicate Difference(const Predicate& other) const;

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

 private:
  /* \brief Internal utility to express parameter ranges as boolean constraint */
  PrimExpr FreeParameterConstraints() const;

  Map<tir::Var, Range> free_parameters_;
};

class BufferTouch {
 public:
  enum class AccessType {
    Read,
    Write,
  };

  BufferTouch(tir::Buffer buffer, Predicate predicate, AccessType touch_type,
              ParametrizedExpression known_value, ObjectRef node);

  /* \brief Checks if this Predicate is a subset of another predicate
   *
   * Returns true if the indices accessed by this touch are a subset
   * of predicate is true can be proven to be a subset of the other
   * subset.  Returns false if it cannot be proven to be a subset of
   * ther other subset.
   */
  bool IsSubsetOf(const BufferTouch& other) const;

  friend std::ostream& operator<<(std::ostream& os, const BufferTouch& expr);

 private:
  tir::Buffer buffer;
  Predicate predicate;
  AccessType touch_type;
  ParametrizedExpression known_value;

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
  bool IsOverwrittenWithoutEffect(const tir::BufferStore& store) const;

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

  /* \brief Tests if a loaded value has a known value at the point of the load.
   *
   * \param load The load to be examined
   *
   * \return An expression for the value being loaded, if it can be
   * determined from previous writes.  Otherwise, return nullopt.
   */
  Optional<PrimExpr> KnownValue(const tir::BufferLoad& load) const;

  /* \brief Attempt to determine the value about to be overwritten.
   *
   * \param store The store to be examined
   *
   * \return An expression for the value located in the location about
   * to be written, if it can be determined from previous writes.
   * Otherwise, return nullopt.
   */
  Optional<PrimExpr> KnownValue(const tir::BufferStore& store) const;

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
  /* \brief Internal utility, checks if a specific write was overwritten
   *
   * \param write_iter An iterator into `touches_`, pointing to the
   * buffer touch to be examined.
   */
  bool IsOverwrittenWithoutEffect(std::vector<BufferTouch>::const_iterator write_iter) const;

  /* \brief Internal utility, checks if a specific access was overwritten
   *
   * \param write_iter An iterator into `touches_`, pointing to the
   * buffer touch to be examined.
   */
  Optional<PrimExpr> KnownValue(std::vector<BufferTouch>::const_reverse_iterator access_iter,
                                const Array<PrimExpr>& indices) const;

  friend std::ostream& operator<<(std::ostream& os, const BufferTouchPattern& pattern);

 private:
  /* \brief An ordered list of buffer touches
   *
   * For all indices i and j, if i<j, then either buffer touch i
   * occurs sequentially before buffer touch j (e.g. for sequential
   * writes of a buffer), or buffer touch i and j have mutually
   * exclusive predicates (e.g. for writes within an if/else).
   */
  std::vector<BufferTouch> touch_points_;

  /* \brief A lookup into touches_
   *
   * A map to look up the first buffer touch that is at or after a
   * given Stmt.
   */
  std::unordered_map<const tir::StmtNode*, size_t> context_lookup_;

  /* \brief Assumptions that do not depend on buffer values */
  std::vector<PrimExpr> non_buffer_assumptions_;

  friend class BufferConstraintSubstituter;
};

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_BUFFER_TOUCH_PATTERN_H_
