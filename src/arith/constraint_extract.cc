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

std::vector<std::vector<PrimExpr>> DedupExprList(
    const std::vector<std::vector<PrimExpr>>& vec_vec) {
  std::unordered_map<size_t, PrimExpr, StructuralHash, StructuralEqual> index_to_expr;
  std::unordered_map<PrimExpr, size_t, StructuralHash, StructuralEqual> expr_to_index;

  std::vector<std::vector<size_t>> collected;

  auto expr_to_key = [&](const PrimExpr& expr) -> size_t {
    auto it = expr_to_index.find(expr);
    if (it != expr_to_index.end()) {
      return it->second;
    }

    size_t index = expr_to_index.size();
    expr_to_index[expr] = index;
    index_to_expr[index] = expr;
    return index;
  };

  // Could throw everything into a set from the start, but that would
  // remove any ordering that is already present.
  auto vector_same_contents = [&](const std::vector<size_t>& a,
                                  const std::vector<size_t>& b) -> bool {
    if (a.size() != b.size()) {
      return false;
    }

    return std::is_permutation(a.begin(), a.end(), b.begin());
  };

  // Intermediate map of indices, to avoid repeated walking of each
  // expression.
  std::vector<std::vector<size_t>> indices;
  for (const auto& vec : vec_vec) {
    // Map from PrimExpr to size_t, de-duplicating
    std::vector<size_t> inner;
    for (const auto& expr : vec) {
      size_t index = expr_to_key(expr);
      if (std::all_of(inner.begin(), inner.end(), [&](size_t prev) { return prev != index; })) {
        inner.push_back(index);
      }
    }

    // Add to list of indices, de-duplicating
    if (std::all_of(indices.begin(), indices.end(), [&](const std::vector<size_t>& prev) {
          return !vector_same_contents(prev, inner);
        })) {
      indices.push_back(inner);
    }
  }

  std::vector<std::vector<PrimExpr>> out;
  for (const auto& vec : indices) {
    std::vector<PrimExpr> expr_vec;
    for (const auto& i : vec) {
      auto it = index_to_expr.find(i);
      ICHECK(it != index_to_expr.end());
      expr_vec.push_back(it->second);
    }
    out.push_back(std::move(expr_vec));
  }

  return out;
}

void CollectComponents2(const PrimExpr& expr, std::function<void(const PrimExpr&)> callback) {
  PVar<PrimExpr> x, y, z;
  if ((x || y).Match(expr)) {
    CollectComponents2(x.Eval(), callback);
    CollectComponents2(y.Eval(), callback);
  } else if ((x && y).Match(expr)) {
    CollectComponents2(x.Eval(), [&](const PrimExpr& x_part) {
      CollectComponents2(y.Eval(), [&](const PrimExpr& y_part) { callback(x_part && y_part); });
    });
  } else {
    callback(expr);
  }
}

std::vector<std::vector<PrimExpr>> ExtractOrOfAnds(const PrimExpr& expr) {
  std::vector<std::vector<PrimExpr>> out;
  CollectComponents2(expr,
                     [&](const PrimExpr& part) { out.push_back(ExtractConstraints(part, false)); });
  return DedupExprList(out);
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
