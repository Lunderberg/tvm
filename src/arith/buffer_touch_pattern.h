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

// Utility for expressing an expression in terms of variable
// parameters.
class Predicate {
 public:
  Predicate(Array<tir::Var> parameter_vars, PrimExpr expression,
            Map<tir::Var, Range> free_parameters);

  /* \brief Evaluate the predicate using the provided arguments
   */
  PrimExpr operator()(Array<PrimExpr> args) const;

  /* \brief Checks if this Predicate is a subset of another predicate
   *
   * Returns true if this predicate can be proven to be a subset of
   * the other subset.  Returns false if it cannot be proven to be a
   * subset of ther other subset.
   */
  bool IsSubsetOf(const Predicate& other) const;

  friend std::ostream& operator<<(std::ostream& os, const Predicate& expr);

 private:
  /* \brief Internal utility to express parameter ranges as boolean constraint */
  PrimExpr FreeParameterConstraints() const;

  Array<tir::Var> parameter_vars_;
  Map<tir::Var, Range> free_parameters_;
  PrimExpr expression_;
};

class BufferTouch {
 public:
  enum class AccessType {
    Read,
    Write,
  };

  BufferTouch(tir::Buffer buffer, Predicate predicate, AccessType touch_type,
              Optional<PrimExpr> known_value, ObjectRef node);

  /* \brief Checks if this Predicate is a subset of another predicate
   *
   * Returns true if the indices accessed by this touch are a subset of  predicate is true can be
   * proven to be a subset of the other subset.  Returns false if it cannot be proven to be a subset
   * of ther other subset.
   */
  bool IsSubsetOf(const BufferTouch& other) const;

  friend std::ostream& operator<<(std::ostream& os, const BufferTouch& expr);

 private:
  tir::Buffer buffer;
  Predicate predicate;
  AccessType touch_type;
  Optional<PrimExpr> known_value;

  // The BufferLoad or BufferStore object that caused this touch.
  ObjectRef node;

  friend class BufferTouchPattern;
};

class BufferTouchPattern {
 public:
  /* \brief Extract the touch pattern from a TIR statement
   */
  explicit BufferTouchPattern(const tir::Stmt& stmt);

  const std::vector<BufferTouch>& GetTouches() const { return touches_; }

  /* \brief Check if a write is overwritten without impacting final results
   *
   * \param store The store to be examined
   *
   * \return True if the specified store can be proven to be
   * overwritten without contributing to any later statements.
   * Returns false otherwise.
   */
  bool IsOverwrittenWithoutEffect(const tir::BufferStore& store) const;

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

 private:
  /* \brief Internal utility, checks if a specific write was overwritten
   *
   * \param write_iter An iterator into `touches_`, pointing to the
   * buffer touch to be examined.
   */
  bool IsOverwrittenWithoutEffect(std::vector<BufferTouch>::const_iterator write_iter) const;

  /* \brief An ordered list of buffer touches
   *
   * For all indices i and j, if i<j, then either buffer touch i
   * occurs sequentially before buffer touch j (e.g. for sequential
   * writes of a buffer), or buffer touch i and j have mutually
   * exclusive predicates (e.g. for writes within an if/else).
   */
  std::vector<BufferTouch> touches_;
};

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_BUFFER_TOUCH_PATTERN_H_
