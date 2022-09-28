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
 * \file tvm/arith/conjunctive_disjunctive_form.cc
 */

#include "conjunctive_disjunctive_form.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>

#include "pattern_match.h"
#include "rewrite_simplify.h"

namespace tvm {
namespace arith {

namespace {
/* \brief A utility for simplifying expressions using conjunctive/disjuctive normal forms */
class BooleanSimplifier {
 public:
  enum class Rep {
    AndOfOrs,
    OrOfAnds,
  };

  BooleanSimplifier(const PrimExpr& expr, Rep rep);

  PrimExpr AsPrimExpr() const;

  void Simplify(Analyzer* analyzer);

 private:
  void SimplifyWithinChunks(Analyzer* analyzer);
  void SimplifyAcrossChunks(Analyzer* analyzer);
  void Cleanup();

  static void VisitAndExpressions(const PrimExpr& expr,
                                  std::function<void(const PrimExpr&)> callback);
  static void VisitOrExpressions(const PrimExpr& expr,
                                 std::function<void(const PrimExpr&)> callback);

  // Utility class to avoid mixing up indices and lookup keys.
  enum class Key : size_t {};
  Key GetKey(const PrimExpr& expr);

  void TrySimplifyOr(Key& a, Key& b, Analyzer* analyzer);
  void TrySimplifyAnd(Key& a, Key& b, Analyzer* analyzer);

  PrimExpr GetExpr(Key key) const;

  PrimExpr ChunkExpr(const std::vector<Key>& chunk) const;

  std::vector<std::vector<Key>> chunks;
  std::unordered_map<Key, PrimExpr, StructuralHash, StructuralEqual> key_to_expr;
  std::unordered_map<PrimExpr, Key, StructuralHash, StructuralEqual> expr_to_key;
  Key key_true;
  Key key_false;

  Rep rep;
};

BooleanSimplifier::BooleanSimplifier(const PrimExpr& expr, Rep rep)
    : key_true(GetKey(Bool(true))), key_false(GetKey(Bool(false))), rep(rep) {
  auto collect_outer = (rep == Rep::AndOfOrs) ? VisitAndExpressions : VisitOrExpressions;
  auto collect_inner = (rep == Rep::AndOfOrs) ? VisitOrExpressions : VisitAndExpressions;

  collect_outer(expr, [&](const PrimExpr& outer_expr) {
    std::vector<Key> or_components;
    collect_inner(outer_expr, [&](const PrimExpr& inner_expr) {
      Key key = GetKey(inner_expr);
      bool is_duplicate = std::any_of(or_components.begin(), or_components.end(),
                                      [&](Key prev) { return prev == key; });
      if (!is_duplicate) {
        or_components.push_back(key);
      }
    });

    bool is_permutation =
        std::any_of(chunks.begin(), chunks.end(), [&](const std::vector<Key>& prev_components) {
          return or_components.size() == prev_components.size() &&
                 std::is_permutation(prev_components.begin(), prev_components.end(),
                                     or_components.begin());
        });
    if (!is_permutation) {
      chunks.push_back(std::move(or_components));
    }
  });
}

void BooleanSimplifier::VisitAndExpressions(const PrimExpr& expr,
                                            std::function<void(const PrimExpr&)> callback) {
  PVar<PrimExpr> x, y, z;
  if ((x && y).Match(expr)) {
    VisitAndExpressions(x.Eval(), callback);
    VisitAndExpressions(y.Eval(), callback);
  } else if ((x || y).Match(expr)) {
    VisitAndExpressions(x.Eval(), [&](const PrimExpr& x_part) {
      VisitAndExpressions(y.Eval(), [&](const PrimExpr& y_part) { callback(x_part || y_part); });
    });
  } else {
    callback(expr);
  }
}

void BooleanSimplifier::VisitOrExpressions(const PrimExpr& expr,
                                           std::function<void(const PrimExpr&)> callback) {
  PVar<PrimExpr> x, y, z;
  if ((x || y).Match(expr)) {
    VisitOrExpressions(x.Eval(), callback);
    VisitOrExpressions(y.Eval(), callback);
  } else if ((x && y).Match(expr)) {
    VisitOrExpressions(x.Eval(), [&](const PrimExpr& x_part) {
      VisitOrExpressions(y.Eval(), [&](const PrimExpr& y_part) { callback(x_part && y_part); });
    });
  } else {
    callback(expr);
  }
}

BooleanSimplifier::Key BooleanSimplifier::GetKey(const PrimExpr& expr) {
  auto it = expr_to_key.find(expr);
  if (it != expr_to_key.end()) {
    return it->second;
  }

  Key key{expr_to_key.size()};
  expr_to_key[expr] = key;
  key_to_expr[key] = expr;
  return key;
}

PrimExpr BooleanSimplifier::GetExpr(BooleanSimplifier::Key key) const {
  auto it = key_to_expr.find(key);
  ICHECK(it != key_to_expr.end());
  return it->second;
}

PrimExpr BooleanSimplifier::AsPrimExpr() const {
  PrimExpr expr = Bool(rep == Rep::AndOfOrs);
  for (const auto& chunk : chunks) {
    if (rep == Rep::AndOfOrs) {
      expr = expr && ChunkExpr(chunk);
    } else {
      expr = expr || ChunkExpr(chunk);
    }
  }
  return expr;
}

void BooleanSimplifier::TrySimplifyOr(Key& a, Key& b, Analyzer* analyzer) {
  PrimExpr joint = GetExpr(a) || GetExpr(b);
  PrimExpr simplified = analyzer->Simplify(joint);
  if (!ExprDeepEqual()(simplified, joint)) {
    if (auto* simplified_or = simplified.as<OrNode>()) {
      a = GetKey(simplified_or->a);
      b = GetKey(simplified_or->b);
    } else {
      a = key_false;
      b = GetKey(simplified);
    }
  }
}

void BooleanSimplifier::TrySimplifyAnd(Key& a, Key& b, Analyzer* analyzer) {
  PrimExpr joint = GetExpr(a) && GetExpr(b);
  PrimExpr simplified = analyzer->Simplify(joint);
  if (!ExprDeepEqual()(simplified, joint)) {
    if (auto* simplified_and = simplified.as<AndNode>()) {
      a = GetKey(simplified_and->a);
      b = GetKey(simplified_and->b);
    } else {
      a = key_true;
      b = GetKey(simplified);
    }
  }
}

PrimExpr BooleanSimplifier::ChunkExpr(const std::vector<Key>& chunk) const {
  PrimExpr expr = Bool(rep != Rep::AndOfOrs);
  for (Key j : chunk) {
    if (rep == Rep::AndOfOrs) {
      expr = expr || GetExpr(j);
    } else {
      expr = expr && GetExpr(j);
    }
  }
  return expr;
}

void BooleanSimplifier::Simplify(Analyzer* analyzer) {
  SimplifyWithinChunks(analyzer);
  SimplifyAcrossChunks(analyzer);
}

void BooleanSimplifier::SimplifyWithinChunks(Analyzer* analyzer) {
  for (auto& chunk : chunks) {
    for (size_t expr_i = 0; expr_i < chunk.size(); expr_i++) {
      for (size_t expr_j = expr_i + 1; expr_j < chunk.size(); expr_j++) {
        Key& key_i = chunk[expr_i];
        Key& key_j = chunk[expr_j];

        if (rep == Rep::AndOfOrs) {
          TrySimplifyOr(key_i, key_j, analyzer);
        } else {
          TrySimplifyAnd(key_i, key_j, analyzer);
        }
      }
    }
  }
  Cleanup();
}

void BooleanSimplifier::SimplifyAcrossChunks(Analyzer* analyzer) {
  for (size_t i_and = 0; i_and < chunks.size(); i_and++) {
    for (size_t j_and = i_and + 1; j_and < chunks.size(); j_and++) {
      auto& i_chunk = chunks[i_and];
      auto& j_chunk = chunks[j_and];

      if (i_chunk.size() == 1 && j_chunk.size() == 1) {
        auto& key_i = i_chunk[0];
        auto& key_j = j_chunk[0];
        if (rep == Rep::AndOfOrs) {
          TrySimplifyAnd(key_i, key_j, analyzer);
        } else {
          TrySimplifyOr(key_i, key_j, analyzer);
        }
        continue;
      }
      std::unordered_set<Key> j_set(j_chunk.begin(), j_chunk.end());

      std::optional<size_t> i_distinct_index;
      for (size_t i = 0; i < i_chunk.size(); i++) {
        if (!j_set.count(i_chunk[i])) {
          i_distinct_index = i;
          break;
        }
      }

      if (!i_distinct_index.has_value()) {
        // For AndOfOrs
        // I = (i_0 || i_1 || ... || i_N)
        // J = (i_0 || i_1 || ... || i_N || j_0 || ... || j_N)
        // I && J == I == I && true

        // For OrOfAnds
        // I = (i_0 && i_1 && ... && i_N)
        // J = (i_0 && i_1 && ... && i_N && j_0 && ... && j_N)
        // I || J == I == I || false

        if (rep == Rep::AndOfOrs) {
          j_chunk = {key_true};
        } else {
          j_chunk = {key_false};
        }
        continue;
      }

      std::unordered_set<Key> i_set(i_chunk.begin(), i_chunk.end());

      std::optional<size_t> j_distinct_index;
      for (size_t j = 0; j < j_chunk.size(); j++) {
        if (!i_set.count(j_chunk[j])) {
          j_distinct_index = j;
          break;
        }
      }

      if (!j_distinct_index.has_value()) {
        // For AndOfOrs
        // I = (i_0 || ... || i_N || j_0 || ... || j_N)
        // J = (j_0 || ... || j_N)
        // I && J == J == true && J

        // For OrOfAnds
        // I = (i_0 && ... && i_N && j_0 && ... && j_N)
        // J = (j_0 && ... && j_N)
        // I || J == J == false || J

        if (rep == Rep::AndOfOrs) {
          i_chunk = {key_true};
        } else {
          i_chunk = {key_false};
        }
        continue;
      }

      if (i_chunk.size() == j_chunk.size()) {
        size_t num_shared_exprs = 0;
        for (const auto& j_key : j_chunk) {
          if (i_set.count(j_key)) {
            ++num_shared_exprs;
          }
        }

        if (num_shared_exprs + 1 == i_chunk.size()) {
          // All but one of the expressions are shared.  If the AND
          // of the distinct expressions can be simplified, we can
          // replace.
          //
          // In AndOfOrs:
          // (A or B) and (A or C) => A or (B and C)
          //
          // In OrOfAnds:
          // (A and B) or (A and C) => A and (B or C)
          auto& key_i = i_chunk[i_distinct_index.value()];
          auto& key_j = j_chunk[j_distinct_index.value()];
          if (rep == Rep::AndOfOrs) {
            TrySimplifyAnd(key_i, key_j, analyzer);
          } else {
            TrySimplifyOr(key_i, key_j, analyzer);
          }
        }
      }
    }
  }
  Cleanup();
}

void BooleanSimplifier::Cleanup() {
  Key outer_identity = (rep == Rep::AndOfOrs) ? key_true : key_false;
  Key inner_identity = (rep == Rep::AndOfOrs) ? key_false : key_true;

  for (auto& chunk : chunks) {
    // Any occurrence of True inside an OR makes the entire expression True.
    if (std::any_of(chunk.begin(), chunk.end(), [&](Key key) { return key == outer_identity; })) {
      chunk = {outer_identity};
    } else {
      // Any occurrence of False inside an OR can be removed
      chunk.erase(std::remove_if(chunk.begin(), chunk.end(),
                                 [&](Key key) { return key == inner_identity; }),
                  chunk.end());
    }
  }

  // Any occurence of False inside an AND makes the entire expression False.
  if (std::any_of(chunks.begin(), chunks.end(),
                  [&](const std::vector<Key>& chunk) { return chunk.size() == 0; })) {
    chunks = {{}};
  } else {
    // Any occurrence of True inside an AND can be removed.
    chunks.erase(std::remove_if(chunks.begin(), chunks.end(),
                                [&](const std::vector<Key>& chunk) {
                                  return chunk.size() == 1 && chunk[0] == outer_identity;
                                }),
                 chunks.end());
  }
}

}  // namespace

PrimExpr SimplifyAsAndOfOrs(const PrimExpr& orig, Analyzer* analyzer) {
  BooleanSimplifier repr(analyzer->Simplify(orig), BooleanSimplifier::Rep::AndOfOrs);
  repr.Simplify(analyzer);
  return repr.AsPrimExpr();
}
PrimExpr SimplifyAsOrOfAnds(const PrimExpr& orig, Analyzer* analyzer) {
  BooleanSimplifier repr(analyzer->Simplify(orig), BooleanSimplifier::Rep::AndOfOrs);
  repr.Simplify(analyzer);
  return repr.AsPrimExpr();
}

}  // namespace arith
}  // namespace tvm
