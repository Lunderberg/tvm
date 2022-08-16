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
 * \file tvm/arith/constraint_extract.cc
 */

#include "constraint_extract.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>

#include "pattern_match.h"
#include "rewrite_simplify.h"

namespace tvm {
namespace arith {

template <typename F>
void CollectConstraints(const PrimExpr& expr, F callback, bool keep_composite_constraints) {
  if (keep_composite_constraints) {
    callback(expr);
  }

  PVar<PrimExpr> x, y;
  if ((x && y).Match(expr)) {
    CollectConstraints(x.Eval(), callback, keep_composite_constraints);
    CollectConstraints(y.Eval(), callback, keep_composite_constraints);
  } else if ((!(x || y)).Match(expr)) {
    CollectConstraints(RewriteBooleanOperators(tir::Not(x.Eval())), callback,
                       keep_composite_constraints);
    CollectConstraints(RewriteBooleanOperators(tir::Not(y.Eval())), callback,
                       keep_composite_constraints);
  } else if (!keep_composite_constraints) {
    callback(expr);
  }
}

std::vector<PrimExpr> ExtractConstraints(const PrimExpr& expr, bool keep_composite_constraints) {
  std::vector<PrimExpr> out;
  CollectConstraints(
      expr, [&](const PrimExpr& part) { out.push_back(part); }, keep_composite_constraints);
  return out;
}

void CollectConstraints2(const PrimExpr& expr, std::function<void(const PrimExpr&)> callback) {
  PVar<PrimExpr> x, y, z;
  if ((x && y).Match(expr)) {
    CollectConstraints2(x.Eval(), callback);
    CollectConstraints2(y.Eval(), callback);
  } else if ((x || y).Match(expr)) {
    CollectConstraints2(x.Eval(), [&](const PrimExpr& x_part) {
      CollectConstraints2(y.Eval(), [&](const PrimExpr& y_part) { callback(x_part || y_part); });
    });
    // } else if ((x - 1 == y).Match(expr) || (y == x - 1).Match(expr)) {
    //   callback(x.Eval() - 1 == y.Eval());
    //   callback(x.Eval() - 1 <= y.Eval());
    //   callback(x.Eval() > y.Eval());
    // } else if ((x + 1 == y).Match(expr) || (y == x + 1).Match(expr)) {
    //   callback(x.Eval() - 1 >= y.Eval());
    //   callback(x.Eval() < y.Eval());
    // } else if ((x <= y).Match(expr)) {
    //   callback(x.Eval() == y.Eval() || x.Eval() < y.Eval());
    // } else if ((y != x).Match(expr)) {
    //   callback(x.Eval() < y.Eval() || y.Eval() < x.Eval());
    //   callback(x.Eval() <= y.Eval());
    //   callback(y.Eval() <= x.Eval());
  } else {
    callback(expr);
  }
}

std::vector<PrimExpr> ExtractConstraints2(const PrimExpr& expr) {
  std::vector<PrimExpr> out;
  PrimExpr normalized = RewriteBooleanOperators(expr);
  CollectConstraints2(normalized, [&](const PrimExpr& part) { out.push_back(part); });
  return out;
}

std::vector<std::vector<PrimExpr>> ExtractAndOfOrs(const PrimExpr& expr) {
  std::vector<std::vector<PrimExpr>> out;
  CollectConstraints2(expr, [&](const PrimExpr& part) { out.push_back(ExtractComponents(part)); });
  return out;
}

PrimExpr ConvertToAndOfOrs(const PrimExpr& expr) {
  PrimExpr output = Bool(true);
  CollectConstraints2(expr, [&](const PrimExpr& part) { output = output && part; });
  return output;
}

template <typename F>
void CollectComponents(const PrimExpr& expr, F callback) {
  PVar<PrimExpr> x, y;
  if ((x || y).Match(expr)) {
    CollectComponents(x.Eval(), callback);
    CollectComponents(y.Eval(), callback);
  } else if ((!(x && y)).Match(expr)) {
    CollectComponents(RewriteBooleanOperators(tir::Not(x.Eval())), callback);
    CollectComponents(RewriteBooleanOperators(tir::Not(y.Eval())), callback);
  } else {
    callback(expr);
  }
}

std::vector<PrimExpr> ExtractComponents(const PrimExpr& expr) {
  std::vector<PrimExpr> out;
  CollectComponents(expr, [&](const PrimExpr& part) { out.push_back(part); });
  return out;
}

}  // namespace arith
}  // namespace tvm
