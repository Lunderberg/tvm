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
 * \file contraint_extract.h
 *
 * \brief Centralized location for extraction of constraints from a boolean expression.
 */

#ifndef TVM_ARITH_PROPAGATE_CONSTRAINT_H_
#define TVM_ARITH_PROPAGATE_CONSTRAINT_H_

#include <tvm/tir/expr.h>

#include <vector>

namespace tvm {
namespace arith {

class Comparison {
 public:
  /*! \brief internal structure for comparison.
   *
   * Values are assigned to allow these flags to be used in bitwise
   * operations.
   */
  enum CompareResult {
    kInconsistent = 0,
    kEQ = 1,
    kLT = 2,
    kLE = 3,
    kGT = 4,
    kGE = 5,
    kNE = 6,
    kUnknown = 7
  };

  static CompareResult TryCompare(const std::vector<Comparison>& knowns, const PrimExpr& lhs,
                                  const PrimExpr& rhs);

  explicit Comparison(const PrimExpr& expr);

  bool IsValid() const;

  Comparison Reversed() const;

 private:
  Comparison() {}
  Comparison(const PrimExpr& lhs, const PrimExpr& rhs, CompareResult result);
  Comparison(const PrimExpr& lhs, const PrimExpr& rhs, int64_t offset, CompareResult result);

  void Normalize();
  Comparison NormalizedTo(const PrimExpr& expr) const;

  Optional<PrimExpr> debug_as_primexpr() const;

  Comparison IntersectAssumingExpressionsMatch(const Comparison& other) const;

  static std::pair<PrimExpr, int64_t> RemoveOffset(const PrimExpr& expr);

  PrimExpr lhs_;
  PrimExpr rhs_;

  Optional<PrimExpr> orig_expr_{NullOpt};

  // Additive offset on rhs
  int64_t offset_{0};
  CompareResult result_{kInconsistent};
};

}  // namespace arith
}  // namespace tvm

#endif  // TVM_ARITH_PROPAGATE_CONSTRAINT_H_
