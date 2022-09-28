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

namespace {
/* \brief A utility for simplifying expressions using conjunctive/disjuctive normal forms
 *
 *
 */
class AndOfOrs {
 public:
  enum class Rep {
    AndOfOrs,
    OrOfAnds,
  };

  AndOfOrs(const PrimExpr& expr, Rep rep);

  PrimExpr AsPrimExpr() const;

  size_t NumTerms() const;
  bool IsOneLayer() const;

  void Simplify(Analyzer* analyzer);
  void SimplifyComponents(Analyzer* analyzer);
  void SimplifyWithinChunks(Analyzer* analyzer);
  void SimplifyAcrossChunks(Analyzer* analyzer);

 private:
  void Cleanup();

  // Utility class to avoid mixing up indices and lookup keys.
  enum class Key : size_t {};
  Key GetKey(const PrimExpr& expr);

  void TrySimplifyOr(Key& a, Key& b, Analyzer* analyzer);
  void TrySimplifyAnd(Key& a, Key& b, Analyzer* analyzer);

  PrimExpr GetExpr(Key key) const;

  PrimExpr ChunkExpr(const std::vector<Key>& chunk) const;
  PrimExpr KnownProvidedWhileInChunk(const std::vector<Key>& chunk) const;
  PrimExpr KnownProvidedByComponentToSiblings(Key key) const;

  std::vector<std::vector<Key>> expr_indices;
  std::unordered_map<Key, PrimExpr, StructuralHash, StructuralEqual> key_to_expr;
  std::unordered_map<PrimExpr, Key, StructuralHash, StructuralEqual> expr_to_key;
  Key key_true;
  Key key_false;

  Rep rep;
};

AndOfOrs::AndOfOrs(const PrimExpr& expr, Rep rep)
    : key_true(GetKey(Bool(true))), key_false(GetKey(Bool(false))), rep(rep) {
  auto collect_outer = (rep == Rep::AndOfOrs) ? CollectConstraints2 : CollectComponents2;
  auto collect_inner = (rep == Rep::AndOfOrs) ? CollectComponents2 : CollectConstraints2;

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

    bool is_permutation = std::any_of(
        expr_indices.begin(), expr_indices.end(), [&](const std::vector<Key>& prev_components) {
          return or_components.size() == prev_components.size() &&
                 std::is_permutation(prev_components.begin(), prev_components.end(),
                                     or_components.begin());
        });
    if (!is_permutation) {
      expr_indices.push_back(std::move(or_components));
    }
  });
}

AndOfOrs::Key AndOfOrs::GetKey(const PrimExpr& expr) {
  auto it = expr_to_key.find(expr);
  if (it != expr_to_key.end()) {
    return it->second;
  }

  Key key{expr_to_key.size()};
  expr_to_key[expr] = key;
  key_to_expr[key] = expr;
  return key;
}

PrimExpr AndOfOrs::GetExpr(AndOfOrs::Key key) const {
  auto it = key_to_expr.find(key);
  ICHECK(it != key_to_expr.end());
  return it->second;
}

PrimExpr AndOfOrs::AsPrimExpr() const {
  PrimExpr expr = Bool(rep == Rep::AndOfOrs);
  for (const auto& chunk : expr_indices) {
    if (rep == Rep::AndOfOrs) {
      expr = expr && ChunkExpr(chunk);
    } else {
      expr = expr || ChunkExpr(chunk);
    }
  }
  return expr;
}

bool AndOfOrs::IsOneLayer() const {
  if (expr_indices.size() <= 1) {
    return true;
  }

  for (const auto& chunk : expr_indices) {
    if (chunk.size() > 1) {
      return false;
    }
  }
  return true;
}

size_t AndOfOrs::NumTerms() const {
  size_t total = 0;
  for (const auto& chunk : expr_indices) {
    total += chunk.size();
  }
  return total;
}

void AndOfOrs::TrySimplifyOr(Key& a, Key& b, Analyzer* analyzer) {
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

void AndOfOrs::TrySimplifyAnd(Key& a, Key& b, Analyzer* analyzer) {
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

PrimExpr AndOfOrs::ChunkExpr(const std::vector<Key>& chunk) const {
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

PrimExpr AndOfOrs::KnownProvidedWhileInChunk(const std::vector<Key>& chunk) const {
  PrimExpr known = Bool(true);
  for (const auto& other_chunk : expr_indices) {
    if (&chunk != &other_chunk) {
      PrimExpr provides = ChunkExpr(other_chunk);
      if (rep == Rep::OrOfAnds) {
        provides = RewriteBooleanOperators(!provides);
      }
      known = known && provides;
    }
  }
  return known;
}

PrimExpr AndOfOrs::KnownProvidedByComponentToSiblings(Key key) const {
  PrimExpr provides = GetExpr(key);
  if (rep == Rep::AndOfOrs) {
    provides = RewriteBooleanOperators(!provides);
  }
  return provides;
}

void AndOfOrs::Simplify(Analyzer* analyzer) {
  SimplifyComponents(analyzer);
  SimplifyWithinChunks(analyzer);
  SimplifyAcrossChunks(analyzer);
}

void AndOfOrs::SimplifyComponents(Analyzer* analyzer) {
  std::vector<PrimExpr> known_from_other_chunks(expr_indices.size(), Bool(true));

  while (true) {
    bool updated = false;
    for (auto& chunk : expr_indices) {
      With<ConstraintContext> chunk_context(analyzer, KnownProvidedWhileInChunk(chunk));

      for (auto& key : chunk) {
        PrimExpr known = Bool(true);
        for (const auto& other_key : chunk) {
          if (&key != &other_key) {
            known = known && KnownProvidedByComponentToSiblings(other_key);
          }
        }
        With<ConstraintContext> component_context(analyzer, known);

        PrimExpr before = GetExpr(key);
        PrimExpr after = analyzer->Simplify(before);
        if (!ExprDeepEqual()(before, after)) {
          key = GetKey(after);
          updated = true;
        }
      }
    }
    if (!updated) {
      break;
    }
  }

  Cleanup();
}

void AndOfOrs::SimplifyWithinChunks(Analyzer* analyzer) {
  for (auto& chunk : expr_indices) {
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

void AndOfOrs::SimplifyAcrossChunks(Analyzer* analyzer) {
  for (size_t i_and = 0; i_and < expr_indices.size(); i_and++) {
    for (size_t j_and = i_and + 1; j_and < expr_indices.size(); j_and++) {
      auto& i_chunk = expr_indices[i_and];
      auto& j_chunk = expr_indices[j_and];

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

void AndOfOrs::Cleanup() {
  Key outer_identity = (rep == Rep::AndOfOrs) ? key_true : key_false;
  Key inner_identity = (rep == Rep::AndOfOrs) ? key_false : key_true;

  for (auto& chunk : expr_indices) {
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
  if (std::any_of(expr_indices.begin(), expr_indices.end(),
                  [&](const std::vector<Key>& chunk) { return chunk.size() == 0; })) {
    expr_indices = {{}};
  } else {
    // Any occurrence of True inside an AND can be removed.
    expr_indices.erase(std::remove_if(expr_indices.begin(), expr_indices.end(),
                                      [&](const std::vector<Key>& chunk) {
                                        return chunk.size() == 1 && chunk[0] == outer_identity;
                                      }),
                       expr_indices.end());
  }
}

}  // namespace

PrimExpr SimplifyUsingCNFAndDNF(const PrimExpr& orig, Analyzer* analyzer, int max_rounds) {
  ExprDeepEqual expr_equal;

  PrimExpr lookback = Bool(false);
  PrimExpr expr = orig;

  Optional<PrimExpr> best = NullOpt;
  size_t num_terms_in_best = -1;
  int rounds_since_improvement = 0;

  for (int i = 0; i < max_rounds; i++) {
    if (as_const_int(expr)) {
      break;
    }

    auto representation = (i % 2 == 0) ? AndOfOrs::Rep::AndOfOrs : AndOfOrs::Rep::OrOfAnds;
    AndOfOrs repr(orig, representation);
    repr.Simplify(analyzer);
    PrimExpr simplified = repr.AsPrimExpr();

    if (repr.NumTerms() < num_terms_in_best) {
      best = repr.AsPrimExpr();
      num_terms_in_best = repr.NumTerms();
      rounds_since_improvement = 0;
    } else {
      rounds_since_improvement++;
    }

    if (repr.IsOneLayer()) {
      break;
    }

    bool converged = expr_equal(simplified, lookback);
    lookback = expr;
    expr = simplified;
    if (converged) {
      break;
    }

    if (rounds_since_improvement >= 4) {
      break;
    }
  }

  PrimExpr result = best.value_or(expr);

  return result;
}

}  // namespace arith
}  // namespace tvm
