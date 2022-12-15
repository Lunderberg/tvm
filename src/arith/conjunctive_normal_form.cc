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
 * \file tvm/arith/conjunctive_normal_form.cc
 */

#include "conjunctive_normal_form.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>

#include <algorithm>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../support/debug_timer.h"
#include "pattern_match.h"
#include "rewrite_simplify.h"

namespace tvm {
namespace arith {

using tvm::support::DebugTimer;

namespace {

class PrimExprCache {
 public:
  /* \brief Type-safe wrapper class that represents an PrimExpr
   *
   * Because integer indices are used frequently through this class,
   * maintaining a separation between integer indices used to access
   * specific elements of the internal representation, and unique
   * identifiers used to represent expressions PrimExpr, is useful.
   */
  enum class Key : size_t {};

  PrimExprCache() {}

  /*! \brief Convert a PrimExpr to a Key */
  Key GetKey(const PrimExpr& expr);

  /*! \brief Convert a Key to a PrimExpr */
  PrimExpr GetExpr(Key key) const;

 private:
  /*! \brief Mapping from internal Key to PrimExpr */
  std::unordered_map<Key, PrimExpr, StructuralHash, StructuralEqual> key_to_expr_;

  /*! \brief Mapping from PrimExpr to internal Key */
  std::unordered_map<PrimExpr, Key, StructuralHash, StructuralEqual> expr_to_key_;
};

/* \brief A utility for simplifying expressions using conjunctive/disjuctive normal forms */
class AndOfOrs {
 public:
  /*! \brief Construct the simplifier
   *
   * Convert a PrimExpr to the internal representation.
   *
   * \param expr The PrimExpr to be simplified.
   */
  explicit AndOfOrs(const PrimExpr& expr);

  /*! \brief Convert internal representation to PrimExpr */
  PrimExpr AsPrimExpr() const;

  /*! \brief Simplify the internal representation */
  void Simplify(Analyzer* analyzer, bool recursive = false);

  size_t NumChunks() const { return chunks_.size(); }

  size_t NumTerms() const {
    size_t sum = 0;
    for (const auto& chunk : chunks_) {
      sum += chunk.size();
    }
    return sum;
  }

  struct RecursiveStepInfo {
    PrimExpr true_for_all;
    std::vector<std::pair<PrimExpr, AndOfOrs>> primexpr_or_sub;
  };
  RecursiveStepInfo MakeRecursiveStep(Analyzer* analyzer) const;

 private:
  using Key = PrimExprCache::Key;
  using InternalRepr = std::vector<std::vector<Key>>;

  struct BuildImpl {
    BuildImpl(InternalRepr chunks) : internal(chunks) {}
    InternalRepr internal;
  };
  friend BuildImpl operator&&(BuildImpl lhs, BuildImpl rhs);
  friend BuildImpl operator||(BuildImpl lhs, BuildImpl rhs);

  BuildImpl Build(const PrimExpr& expr);

  /*! \brief Internal utility, simplify within each group of expressions
   *
   * For each pair of values within a chunk, attempt to simplify them into
   * a single expression.
   *
   * For example,
   *    before = (a == 5) && ((b < 10) || (b > 10))
   *    after  = (a == 5) && ((b != 10) || false)
   */
  bool SimplifyIndividualExpressions(Analyzer* analyzer);

  /*! \brief Internal utility, simplify within each group of expressions
   *
   * For each pair of values within a chunk, attempt to simplify them into
   * a single expression.
   *
   * For example,
   *    before = (a == 5) && ((b < 10) || (b > 10))
   *    after  = (a == 5) && ((b != 10) || false)
   */
  bool SimplifyWithinChunks(Analyzer* analyzer);

  /*! \brief Internal utility, simplify across groups of expressions
   *
   * For each pair of chunks, if the two chunks differ by only a single
   * term, attempt to simplify those differing terms.
   *
   * For example,
   *    before = ((a == 5) || (b <= 10)) && ((a == 5) || (b >= 10))
   *    after  = ((a == 5) || (b == 10)) && ((a == 5) || true)
   */
  bool SimplifyAcrossChunks(Analyzer* analyzer);

  // bool SimplifyRecursively(Analyzer* analyzer);

  /*! \brief Remove instances of true/false from internal representation
   *
   * To avoid invalidating iterators, `SimplifyWithinChunks` and
   * `SimplifyAcrossChunks` may replace keys, but may not remove keys
   * from the internal representation.  For example, `(a < 5) && (a <
   * 10)` would be simplified to `(a < 5) && true`.  The
   * `RemoveTrueFalse` function removes these leftover instances of
   * true/false.
   */
  void RemoveTrueFalse();

  bool RemoveSupersetChunks();

  bool RemoveChunksContainingSingletons();

  bool RemoveNegatedSingletonsFromChunks(Analyzer* analyzer);

  bool RemoveNegationsFromAlmostSupersetChunks(Analyzer* analyzer);

  /*! \brief Internal utility function used to convert to internal form */
  static void VisitAndExpressions(const PrimExpr& expr,
                                  std::function<void(const PrimExpr&)> callback);
  /*! \brief Internal utility function used to convert to internal form */
  static void VisitOrExpressions(const PrimExpr& expr,
                                 std::function<void(const PrimExpr&)> callback);

  /*! \brief Convert a PrimExpr to a Key */
  Key GetKey(const PrimExpr& expr) { return cache_.GetKey(expr); }

  /*! \brief Convert a Key to a PrimExpr */
  PrimExpr GetExpr(Key key) const { return cache_.GetExpr(key); }

  /*! \brief Attempt to simplify (a && b)
   *
   * If successful, will overwrite the parameters `a` and `b` with the
   * simplified form.
   */
  bool TrySimplifyOr(Key* a, Key* b, Analyzer* analyzer);

  /*! \brief Attempt to simplify (a || b)
   *
   * If successful, will overwrite the parameters `a` and `b` with the
   * simplified form.
   */
  bool TrySimplifyAnd(Key* a, Key* b, Analyzer* analyzer);

  PrimExprCache cache_;

  /* \brief A cache for ImpliedExprs
   *
   * Used when checking whether one OR-group is a more restrictive
   * form of another.  For example, `(i
   *
   * For example, `(i != 5)` is equivalent to `(i < 5) or (5 < i)`.
   * Therefore, the OR chunk `A = (i < 5 or j==10)` is a more
   * restrictive form of `B = (i != 5 or j==10)`, and `A and B` could
   * be simplified to `B`.
   */
  std::unordered_map<Key, std::vector<Key>> implied_exprs_;
  const std::vector<Key>& ImpliedExprs(Key key);

  /*! A cache for NegateExpr */
  std::unordered_map<Key, Key> negated_exprs_;
  /*! Find an expression for the negation of a known expression
   *
   * \param analyzer The analyzer with which to simplify.  Should only
   * have constraints that are common to the entire AndOfOr.
   */
  Key NegateExpr(Key key, Analyzer* analyzer);

  /*! \brief The internal representation
   *
   * `chunks[i][j]` is the j-th expression in the i-th OR-group.
   */
  std::vector<std::vector<Key>> chunks_;

  /*! \brief Cached key representing tir::Bool(true) */
  Key key_true_;

  /*! \brief Cached key representing tir::Bool(false) */
  Key key_false_;
};

template <typename T, typename U>
std::vector<PrimExpr> CollectComponent(const PrimExpr& expr) {
  std::vector<PrimExpr> output;
  std::vector<PrimExpr> to_search = {expr};

  while (to_search.size()) {
    PrimExpr term = to_search.back();
    to_search.pop_back();

    if (auto* as_component = term.as<T>()) {
      to_search.push_back(as_component->b);
      to_search.push_back(as_component->a);
    } else if (auto* as_not = term.as<NotNode>(); as_not && as_not->a.as<U>()) {
      auto* as_other_component = as_not->a.as<U>();
      to_search.push_back(as_other_component->a);
      to_search.push_back(as_other_component->b);
    } else {
      output.push_back(term);
    }
  }

  ICHECK_GT(output.size(), 0) << "Error collecting components from " << expr;
  return output;
}

AndOfOrs::AndOfOrs(const PrimExpr& expr)
    : key_true_(GetKey(Bool(true))), key_false_(GetKey(Bool(false))) {
  chunks_ = Build(expr).internal;
}

AndOfOrs::BuildImpl AndOfOrs::Build(const PrimExpr& expr) {
  // Fast path, if it is already in the correct form
  using Key = PrimExprCache::Key;
  auto fast_path = [&]() -> std::optional<BuildImpl> {
    std::vector<std::vector<Key>> chunks;
    for (PrimExpr and_component : CollectComponent<AndNode, OrNode>(expr)) {
      std::vector<Key> chunk;
      for (PrimExpr or_component : CollectComponent<OrNode, AndNode>(and_component)) {
        if (or_component.as<AndNode>()) {
          return std::nullopt;
        }
        chunk.push_back(GetKey(or_component));
      }
      chunks.push_back(chunk);
    }
    return chunks;
  }();

  if (fast_path.has_value()) {
    return fast_path.value();
  }

  if (auto* as_and = expr.as<AndNode>()) {
    return Build(as_and->a) && Build(as_and->b);
  } else if (auto* as_or = expr.as<OrNode>()) {
    return Build(as_or->a) || Build(as_or->b);
  } else if (auto* as_not = expr.as<NotNode>()) {
    // No operator overloading, because we need to make new keys for the negations.

    auto inner = Build(as_not->a);

    BuildImpl out({});

    for (const auto& chunk : inner.internal) {
      std::vector<std::vector<Key>> or_component;
      for (const auto& key : chunk) {
        or_component.push_back({GetKey(!GetExpr(key))});
      }
      out = out || BuildImpl(or_component);
    }
    return out;

  } else {
    return BuildImpl{{{GetKey(expr)}}};
  }

  // std::vector<std::vector<Key>> chunks;
  // VisitAndExpressions(expr, [&](const PrimExpr& outer_expr) {
  //   std::vector<Key> or_components;
  //   VisitOrExpressions(outer_expr, [&](const PrimExpr& inner_expr) {
  //     Key key = GetKey(inner_expr);
  //     bool is_duplicate = std::any_of(or_components.begin(), or_components.end(),
  //                                     [&](Key prev) { return prev == key; });
  //     if (!is_duplicate) {
  //       or_components.push_back(key);
  //     }
  //   });

  //   bool is_permutation =
  //       std::any_of(chunks_.begin(), chunks_.end(), [&](const std::vector<Key>& prev_components)
  //       {
  //         return or_components.size() == prev_components.size() &&
  //                std::is_permutation(prev_components.begin(), prev_components.end(),
  //                                    or_components.begin());
  //       });
  //   if (!is_permutation) {
  //     chunks.push_back(std::move(or_components));
  //   }
  // });
  // return {chunks}
  ;
}

AndOfOrs::BuildImpl operator&&(AndOfOrs::BuildImpl lhs, AndOfOrs::BuildImpl rhs) {
  for (auto& chunk : rhs.internal) {
    lhs.internal.push_back(std::move(chunk));
  }
  return lhs;
}

AndOfOrs::BuildImpl operator||(AndOfOrs::BuildImpl lhs, AndOfOrs::BuildImpl rhs) {
  using Key = PrimExprCache::Key;

  std::unordered_set<const std::vector<Key>*> shared_lhs_chunks;
  std::unordered_set<const std::vector<Key>*> shared_rhs_chunks;

  // TODO: Pull out chunks that are common between both lhs and rhs
  for (const auto& lhs_chunk : lhs.internal) {
    for (const auto& rhs_chunk : rhs.internal) {
      bool is_shared = lhs_chunk.size() == rhs_chunk.size() &&
                       std::is_permutation(lhs_chunk.begin(), lhs_chunk.end(), rhs_chunk.begin());

      if (is_shared) {
        shared_lhs_chunks.insert(&lhs_chunk);
        shared_rhs_chunks.insert(&rhs_chunk);
      }
    }
  }

  if (shared_lhs_chunks.size()) {
    std::cout << "Found " << shared_lhs_chunks.size() << " shared chunks" << std::endl;
  }

  std::vector<std::vector<Key>> out_chunks;

  for (const auto& lhs_chunk : lhs.internal) {
    if (shared_lhs_chunks.count(&lhs_chunk)) {
      out_chunks.push_back(lhs_chunk);
    } else {
      for (const auto& rhs_chunk : rhs.internal) {
        if (!shared_rhs_chunks.count(&rhs_chunk)) {
          std::vector<Key> out_chunk = lhs_chunk;
          for (Key rhs_key : rhs_chunk) {
            out_chunk.push_back(rhs_key);
          }
          out_chunks.push_back(out_chunk);
        }
      }
    }
  }

  return {out_chunks};
}

void AndOfOrs::VisitAndExpressions(const PrimExpr& expr,
                                   std::function<void(const PrimExpr&)> callback) {
  PVar<PrimExpr> x, y, z;
  if ((x && y).Match(expr)) {
    // These are separate AND conditions, recurse into them in case
    // they contain AND internally.
    VisitAndExpressions(x.Eval(), callback);
    VisitAndExpressions(y.Eval(), callback);
  } else if ((x || y).Match(expr)) {
    // This may be the bottom-most breakdown, but either x or y may
    // themselves contain AND.  (e.g. (A && B) || (C && D) should be
    // split into (A || C), (A || D), (B || C), and (B || D).)
    // Recurse into each, then reconstruct an OR condition.
    VisitAndExpressions(x.Eval(), [&](const PrimExpr& x_part) {
      VisitAndExpressions(y.Eval(), [&](const PrimExpr& y_part) { callback(x_part || y_part); });
    });
  } else {
    // This is bottom-most breakdown.
    callback(expr);
  }
}

void AndOfOrs::VisitOrExpressions(const PrimExpr& expr,
                                  std::function<void(const PrimExpr&)> callback) {
  PVar<PrimExpr> x, y, z;
  if ((x || y).Match(expr)) {
    // These are separate OR conditions, recurse into them in case
    // they contain OR internally.
    VisitOrExpressions(x.Eval(), callback);
    VisitOrExpressions(y.Eval(), callback);
  } else if ((x && y).Match(expr)) {
    // This may be the bottom-most breakdown, but either x or y may
    // themselves contain OR.  (e.g. (A || B) && (C || D) should be
    // split into (A && C), (A && D), (B && C), and (B && D).)
    // Recurse into each, then reconstruct an AND condition.
    VisitOrExpressions(x.Eval(), [&](const PrimExpr& x_part) {
      VisitOrExpressions(y.Eval(), [&](const PrimExpr& y_part) { callback(x_part && y_part); });
    });
  } else {
    // This is bottom-most breakdown.
    callback(expr);
  }
}

PrimExprCache::Key PrimExprCache::GetKey(const PrimExpr& expr) {
  auto it = expr_to_key_.find(expr);
  if (it != expr_to_key_.end()) {
    return it->second;
  }

  Key key{expr_to_key_.size()};
  expr_to_key_[expr] = key;
  key_to_expr_[key] = expr;
  return key;
}

PrimExpr PrimExprCache::GetExpr(PrimExprCache::Key key) const {
  auto it = key_to_expr_.find(key);
  ICHECK(it != key_to_expr_.end());
  return it->second;
}

PrimExpr AndOfOrs::AsPrimExpr() const {
  PrimExpr expr = Bool(true);
  for (const auto& chunk : chunks_) {
    PrimExpr chunk_expr = Bool(false);
    for (Key j : chunk) {
      chunk_expr = chunk_expr || GetExpr(j);
    }
    expr = expr && chunk_expr;
  }
  return expr;
}

AndOfOrs::RecursiveStepInfo AndOfOrs::MakeRecursiveStep(Analyzer* analyzer) const {
  std::vector<Key> singletons;
  std::vector<std::vector<Key>> to_split;

  for (const auto& chunk : chunks_) {
    if (chunk.size() == 1) {
      singletons.push_back(chunk[0]);
    } else {
      to_split.push_back(chunk);
    }
  }

  std::vector<std::pair<PrimExpr, AndOfOrs>> primexpr_or_sub;

  while (to_split.size()) {
    std::unordered_map<Key, size_t> counts;
    for (const auto& chunk : to_split) {
      for (Key key : chunk) {
        counts[key]++;
      }
    }

    Key split_on = [&]() {
      Key max_key;
      size_t max_count = 0;
      for (const auto& [key, count] : counts) {
        if (count > max_count) {
          max_key = key;
          max_count = count;
        }
      }
      return max_key;
    }();

    std::vector<std::vector<Key>> this_split;
    std::vector<std::vector<Key>> next_split;
    for (auto chunk : to_split) {
      bool contains_split_key =
          std::any_of(chunk.begin(), chunk.end(), [&](Key key) { return key == split_on; });
      if (contains_split_key) {
        chunk.erase(
            std::remove_if(chunk.begin(), chunk.end(), [&](Key key) { return key == split_on; }),
            chunk.end());
        this_split.push_back(chunk);
      } else {
        next_split.push_back(chunk);
      }
    }

    AndOfOrs substep = *this;
    substep.chunks_ = this_split;
    primexpr_or_sub.push_back({GetExpr(split_on), substep});
    std::cout << "Separated out recursive step with OR " << GetExpr(split_on)
              << ", inner step = " << substep.AsPrimExpr() << std::endl;
    substep.Simplify(analyzer);
    std::cout << "\t"
              << "Which would simplify to " << substep.AsPrimExpr() << std::endl;

    to_split = next_split;
  }

  PrimExpr true_for_all = Bool(true);
  for (Key key : singletons) {
    true_for_all = true_for_all || GetExpr(key);
  }

  std::cout << "True across all branches: " << true_for_all << std::endl;
  return {true_for_all, primexpr_or_sub};
}

bool AndOfOrs::TrySimplifyOr(Key* a_ptr, Key* b_ptr, Analyzer* analyzer) {
  Key& a = *a_ptr;
  Key& b = *b_ptr;
  PrimExpr joint = GetExpr(a) || GetExpr(b);

  auto timer = DebugTimer("TrySimplifyOr").on_finish([&](auto& out) { out << joint; }).start();

  PrimExpr simplified = analyzer->rewrite_simplify(joint);

  if (!ExprDeepEqual()(simplified, joint)) {
    if (auto* simplified_or = simplified.as<OrNode>()) {
      a = GetKey(simplified_or->a);
      b = GetKey(simplified_or->b);
    } else {
      a = key_false_;
      b = GetKey(simplified);
    }
    return true;
  } else {
    return false;
  }
}

bool AndOfOrs::TrySimplifyAnd(Key* a_ptr, Key* b_ptr, Analyzer* analyzer) {
  Key& a = *a_ptr;
  Key& b = *b_ptr;
  PrimExpr joint = GetExpr(a) && GetExpr(b);
  PrimExpr simplified = analyzer->rewrite_simplify(joint);
  if (!ExprDeepEqual()(simplified, joint)) {
    if (auto* simplified_and = simplified.as<AndNode>()) {
      a = GetKey(simplified_and->a);
      b = GetKey(simplified_and->b);
    } else {
      a = key_true_;
      b = GetKey(simplified);
    }
    return true;
  } else {
    return false;
  }
}

void AndOfOrs::Simplify(Analyzer* analyzer, bool recursive) {
  auto timer = DebugTimer("AndOfOrs::Simplify").start();
  // RemoveTrueFalse();
  // RemoveSupersetChunks();
  // SimplifyWithinChunks(analyzer);
  // RemoveTrueFalse();
  // RemoveSupersetChunks();
  // SimplifyAcrossChunks(analyzer);
  // RemoveTrueFalse();
  // RemoveSupersetChunks();
  // SimplifyIndividualExpressions(analyzer);
  // RemoveTrueFalse();
  // RemoveSupersetChunks();

  size_t i = 0;

  while (true) {
    ICHECK_LT(i++, 100);
    int steps_completed = 0;
    auto timer =
        DebugTimer("AndOfOrs::Simplify, round")
            .always_print_short_timers()
            .on_finish([&](auto& out) { out << "Restart after " << steps_completed << " steps"; })
            .start();
    // Simplifications are arranged from least expensive to most
    // expensive.  If a simplification provides a simplification,
    // start again from the cheapest cases.
    RemoveTrueFalse();
    steps_completed++;
    if (RemoveChunksContainingSingletons()) continue;
    steps_completed++;
    if (RemoveNegatedSingletonsFromChunks(analyzer)) continue;
    steps_completed++;
    if (RemoveSupersetChunks()) continue;
    steps_completed++;
    if (RemoveNegationsFromAlmostSupersetChunks(analyzer)) continue;
    steps_completed++;
    if (SimplifyIndividualExpressions(analyzer)) continue;
    steps_completed++;
    if (SimplifyWithinChunks(analyzer)) continue;
    steps_completed++;
    if (SimplifyAcrossChunks(analyzer)) continue;
    steps_completed++;
    // if (recursive) {
    //   if (SimplifyRecursively(analyzer)) continue;
    // }

    break;
  }
}

bool AndOfOrs::SimplifyIndividualExpressions(Analyzer* analyzer) {
  bool made_change = false;

  Map<PrimExpr, PrimExpr> replacements;
  auto timer = DebugTimer("Simplifying individual expressions")
                   .always_print_short_timers()
                   .on_finish([&](auto& out) { out << "Replacements: " << replacements; })
                   .start();

  std::optional<DebugTimer::Impl> subtimer;

  for (auto& chunk : chunks_) {
    DebugTimer("Generating known from other chunks").start(subtimer);
    PrimExpr known_from_other_chunks = [&]() -> PrimExpr {
      PrimExpr known = Bool(true);
      for (const auto& other : chunks_) {
        if (&other != &chunk) {
          PrimExpr chunk_known = Bool(false);
          for (const auto& key : other) {
            chunk_known = chunk_known || GetExpr(key);
          }
          known = known && chunk_known;
        }
      }
      return known;
    }();

    DebugTimer("Entering chunk context")
        .on_start([&](auto& out) { out << "for chunk " << known_from_other_chunks; })
        .start(subtimer);
    With<ConstraintContext> chunk_context(analyzer, known_from_other_chunks);

    for (auto& key : chunk) {
      DebugTimer("Generating known from siblings").start(subtimer);
      PrimExpr known_from_siblings = [&]() -> PrimExpr {
        PrimExpr known = Bool(true);
        for (const auto& other : chunk) {
          if (&key != &other) {
            known = known && NormalizeBooleanOperators(!GetExpr(other));
          }
        }
        return known;
      }();

      DebugTimer("Entering expr context").start(subtimer);
      With<ConstraintContext> expr_context(analyzer, known_from_siblings);

      DebugTimer("Simplifying expr").start(subtimer);
      PrimExpr before = GetExpr(key);
      PrimExpr after = [&]() { return analyzer->Simplify(before); }();
      Key after_key = GetKey(after);
      if (after_key != key) {
        replacements.Set(before, after);
        made_change = true;
        key = after_key;
      }
    }
  }
  return made_change;
}

bool AndOfOrs::SimplifyWithinChunks(Analyzer* analyzer) {
  bool made_change = false;

  Map<PrimExpr, PrimExpr> replacements;
  auto timer = DebugTimer("Simplifying within chunks")
                   .always_print_short_timers()
                   .on_finish([&](auto& out) { out << "Replacements: " << replacements; })
                   .start();

  std::optional<DebugTimer::Impl> subtimer;
  for (auto& chunk : chunks_) {
    auto make_chunk_expr = [&]() -> Optional<PrimExpr> {
      Optional<PrimExpr> expr = NullOpt;
      for (const auto& key : chunk) {
        if (expr.defined()) {
          expr = tvm::tir::Or(expr.value(), GetExpr(key));
        } else {
          expr = GetExpr(key);
        }
      }
      return expr;
    };

    bool made_change_in_chunk = false;

    DebugTimer("Simplifying within chunk")
        .ms_required_to_print(50)
        .on_finish([&, before = make_chunk_expr()](auto& out) {
          out << "Simplified chunk of size " << chunk.size() << " from " << before;
          if (made_change_in_chunk) {
            out << " to " << make_chunk_expr();
          } else {
            out << ", no change";
          }
        })
        .start(subtimer);

    bool chunk_contains_true = false;
    for (size_t expr_i = 0; expr_i < chunk.size() && !chunk_contains_true; expr_i++) {
      Key& key_i = chunk[expr_i];

      if (key_i != key_false_) {
        for (size_t expr_j = expr_i + 1; expr_j < chunk.size() && !chunk_contains_true; expr_j++) {
          Key& key_j = chunk[expr_j];

          if (key_j != key_false_) {
            PrimExpr before_i = GetExpr(key_i);
            PrimExpr before_j = GetExpr(key_j);
            bool changed = TrySimplifyOr(&key_i, &key_j, analyzer);
            made_change = made_change || changed;
            made_change_in_chunk = made_change_in_chunk || changed;

            if (changed) {
              replacements.Set(Or(before_i, before_j), GetExpr(key_i) || GetExpr(key_j));
            }

            if (key_i == key_true_ || key_j == key_true_) {
              chunk_contains_true = true;
            }
          }
        }
      }
    }
  }
  return made_change;
}

bool AndOfOrs::SimplifyAcrossChunks(Analyzer* analyzer) {
  bool made_change = false;

  auto timer = DebugTimer("Simplifying across chunks")
                   .on_start([&](auto& out) { out << "before = " << AsPrimExpr(); })
                   .on_finish([&](auto& out) {
                     if (made_change) {
                       out << "no change";
                     } else {
                       out << "after = " << AsPrimExpr();
                     }
                   })
                   .start();

  for (size_t i_and = 0; i_and < chunks_.size(); i_and++) {
    for (size_t j_and = i_and + 1; j_and < chunks_.size(); j_and++) {
      auto& i_chunk = chunks_[i_and];
      auto& j_chunk = chunks_[j_and];

      if (i_chunk.size() == 1 && j_chunk.size() == 1) {
        auto& key_i = i_chunk[0];
        auto& key_j = j_chunk[0];
        bool change = TrySimplifyAnd(&key_i, &key_j, analyzer);
        made_change = made_change || change;
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
        // I = (i_0 || i_1 || ... || i_N)
        // J = (i_0 || i_1 || ... || i_N || j_0 || ... || j_N)
        // I && J == I == I && true

        j_chunk = {key_true_};
        made_change = true;
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
        // I = (i_0 || ... || i_N || j_0 || ... || j_N)
        // J = (j_0 || ... || j_N)
        // I && J == J == true && J

        i_chunk = {key_true_};
        made_change = true;
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
          // (A or B) and (A or C) => A or (B and C)
          auto& key_i = i_chunk[i_distinct_index.value()];
          auto& key_j = j_chunk[j_distinct_index.value()];

          // When attempting to simplify (B and C), the analyzer may
          // assume that A is false.
          PrimExpr known = [&]() {
            PrimExpr known = Bool(true);
            for (const auto& key : i_chunk) {
              if (&key != &key_i) {
                known = known && analyzer->Simplify(!GetExpr(key));
              }
            }
            return known;
          }();

          With<ConstraintContext> context(analyzer, known);

          PrimExpr known_from_other_chunks = [&]() -> PrimExpr {
            PrimExpr known = Bool(true);
            for (const auto& chunk : chunks_) {
              if (&chunk != &i_chunk && &chunk != &j_chunk) {
                PrimExpr chunk_expr = Bool(false);
                for (const auto& key : chunk) {
                  chunk_expr = chunk_expr || GetExpr(key);
                }
                known = known && chunk_expr;
              }
            }
            return known;
          }();
          With<ConstraintContext> context2(analyzer, known_from_other_chunks);

          // std::cout << "Attempting to simplify (" << GetExpr(key_i) << " AND " << GetExpr(key_j)
          //           << ")" << std::endl;
          // std::cout << "\t"
          //           << "Known1: " << known << std::endl;
          // std::cout << "\t"
          //           << "Known2: " << known_from_other_chunks << std::endl;

          bool change = TrySimplifyAnd(&key_i, &key_j, analyzer);
          made_change = made_change || change;

          // std::cout << "\t"
          //           << "Result: " << (GetExpr(key_i) && GetExpr(key_j)) << std::endl;
        }
      }
    }
  }

  return made_change;
}

// bool AndOfOrs::SimplifyRecursively(Analyzer* analyzer) {
//   return false;
// }

void AndOfOrs::RemoveTrueFalse() {
  auto timer = DebugTimer("Cleanup true/false").start();

  for (auto& chunk : chunks_) {
    // Any occurrence of True inside an OR makes the entire expression True.
    if (std::any_of(chunk.begin(), chunk.end(), [&](Key key) { return key == key_true_; })) {
      chunk = {key_true_};
    } else {
      // Any occurrence of False inside an OR can be removed
      chunk.erase(
          std::remove_if(chunk.begin(), chunk.end(), [&](Key key) { return key == key_false_; }),
          chunk.end());
    }
  }

  // Any occurence of False inside an AND makes the entire expression False.
  if (std::any_of(chunks_.begin(), chunks_.end(),
                  [&](const std::vector<Key>& chunk) { return chunk.size() == 0; })) {
    chunks_ = {{}};
  } else {
    // Any occurrence of True inside an AND can be removed.
    chunks_.erase(std::remove_if(chunks_.begin(), chunks_.end(),
                                 [&](const std::vector<Key>& chunk) {
                                   return chunk.size() == 1 && chunk[0] == key_true_;
                                 }),
                  chunks_.end());
  }
}

bool AndOfOrs::RemoveChunksContainingSingletons() {
  // Each branch of an AND may be simplified assuming that the other
  // branches are true.  Therefore, if an expression occurs on its
  // own, that expression may be assumed to be true in all other
  // branches of the AND.  If an OR-group includes any of these, the
  // entire OR group may be replaced with True.
  std::unordered_set<Key> singleton_keys;
  std::unordered_set<const std::vector<Key>*> singleton_chunks;
  for (const auto& chunk : chunks_) {
    if (chunk.size() == 1 && !singleton_keys.count(chunk[0])) {
      singleton_keys.insert(chunk[0]);
      singleton_chunks.insert(&chunk);
    }
  }
  if (singleton_keys.empty()) {
    return false;
  }

  bool made_change = false;
  for (auto& chunk : chunks_) {
    if (!singleton_chunks.count(&chunk)) {
      bool contains_singleton_key = std::any_of(
          chunk.begin(), chunk.end(), [&](Key key) -> bool { return singleton_keys.count(key); });
      if (contains_singleton_key) {
        chunk = {key_true_};
        made_change = true;
      }
    }
  }
  return made_change;
}

bool AndOfOrs::RemoveNegatedSingletonsFromChunks(Analyzer* analyzer) {
  // Each branch of an AND may be simplified assuming that the other
  // branches are true.  Therefore, if an expression occurs on its
  // own, that expression may be assumed to be true in all other
  // branches of the AND.  Therefore, keys corresponding to the
  // negated singleton may be replaced with False in other OR groups.
  std::unordered_set<Key> singleton_keys;
  std::unordered_set<const std::vector<Key>*> singleton_chunks;
  for (const auto& chunk : chunks_) {
    if (chunk.size() == 1 && !singleton_keys.count(chunk[0])) {
      singleton_keys.insert(chunk[0]);
      singleton_chunks.insert(&chunk);
    }
  }
  if (singleton_keys.empty()) {
    return false;
  }

  std::unordered_set<Key> known_false;
  for (Key singleton : singleton_keys) {
    Key negation = NegateExpr(singleton, analyzer);
    known_false.insert(negation);
    for (Key implied : ImpliedExprs(negation)) {
      known_false.insert(implied);
    }
  }

  bool made_change = false;
  for (auto& chunk : chunks_) {
    if (!singleton_chunks.count(&chunk)) {
      for (auto& key : chunk) {
        if (known_false.count(key)) {
          key = key_false_;
          made_change = true;
        }
      }
    }
  }
  return made_change;
}

bool AndOfOrs::RemoveNegationsFromAlmostSupersetChunks(Analyzer* analyzer) {
  bool made_change = false;
  for (const auto& chunk_a : chunks_) {
    if (chunk_a.size() < 2) continue;

    for (auto& chunk_b : chunks_) {
      if (&chunk_a == &chunk_b) continue;
      if (chunk_b.size() == 1) continue;

      auto single_difference = [&]() -> std::optional<Key> {
        std::vector<Key> difference;

        for (Key key_a : chunk_a) {
          bool in_intersection = std::any_of(chunk_b.begin(), chunk_b.end(),
                                             [&](Key key_b) { return key_a == key_b; });
          if (!in_intersection) {
            difference.push_back(key_a);
            if (difference.size() >= 2) {
              return std::nullopt;
            }
          }
        }

        if (difference.empty()) {
          return std::nullopt;
        } else {
          return difference[0];
        }
      }();

      if (single_difference.has_value()) {
        Key negated = NegateExpr(*single_difference, analyzer);
        std::unordered_set<Key> known_false{negated};
        for (Key implied : ImpliedExprs(negated)) {
          known_false.insert(implied);
        }

        for (auto& key_b : chunk_b) {
          if (known_false.count(key_b)) {
            key_b = key_false_;
            made_change = true;
          }
        }
      }
    }
  }

  return made_change;
}

bool AndOfOrs::RemoveSupersetChunks() {
  bool made_change = false;
  for (auto it = chunks_.rbegin(); it != chunks_.rend(); it++) {
    auto& superset_chunk = *it;
    if (superset_chunk.size() < 2) {
      continue;
    }

    // A chunk is a superset if every value in the subset is either contained in the superset, or
    // is an implied statement from
    std::unordered_set<Key> superset_lookup;
    for (Key key : superset_chunk) {
      superset_lookup.insert(key);
      for (Key implied : ImpliedExprs(key)) {
        superset_lookup.insert(implied);
      }
    }

    bool is_superset = std::any_of(chunks_.begin(), chunks_.end(), [&](const auto& subset_chunk) {
      return &subset_chunk != &superset_chunk &&
             std::all_of(subset_chunk.begin(), subset_chunk.end(),
                         [&](Key subset_key) { return superset_lookup.count(subset_key); });
    });

    if (is_superset) {
      made_change = true;
      superset_chunk = {key_true_};
    }
  }

  return made_change;
}

const std::vector<PrimExprCache::Key>& AndOfOrs::ImpliedExprs(Key key) {
  if (auto it = implied_exprs_.find(key); it != implied_exprs_.end()) {
    return it->second;
  }

  PrimExpr expr = GetExpr(key);

  auto as_key = [&](const auto& pat) { return GetKey(pat.Eval()); };

  PVar<PrimExpr> x, y;
  PVar<IntImm> c1;
  auto implied_keys = [&]() -> std::vector<Key> {
    if ((x < c1).Match(expr)) {
      return {as_key(x <= c1 - 1)};

    } else if ((c1 < x).Match(expr)) {
      return {as_key(c1 + 1 <= x)};

    } else if ((x <= c1).Match(expr)) {
      return {as_key(x < c1 + 1)};

    } else if ((c1 <= x).Match(expr)) {
      return {as_key(c1 - 1 < x)};

    } else if ((x != c1).Match(expr)) {
      return {as_key(x < c1), as_key(c1 < x), as_key(x <= c1 - 1), as_key(c1 + 1 <= x)};

    } else if ((x != y).Match(expr)) {
      return {as_key(x < y), as_key(y < x)};

    } else if ((x <= y).Match(expr)) {
      return {as_key(x < y), as_key(x == y)};

    } else {
      return {};
    }
  }();

  return implied_exprs_[key] = std::move(implied_keys);
}

PrimExprCache::Key AndOfOrs::NegateExpr(Key key, Analyzer* analyzer) {
  if (auto it = negated_exprs_.find(key); it != negated_exprs_.end()) {
    return it->second;
  }

  PrimExpr expr = GetExpr(key);
  PrimExpr negated = analyzer->Simplify(!expr);
  return negated_exprs_[key] = GetKey(negated);
}

// Helper utility for temporarily disabling the
// kConvertBooleanToAndOfOrs flag on an analyzer, to prevent infinite
// recursion.
class DisableAndOfOrRecursion {
 public:
  explicit DisableAndOfOrRecursion(Analyzer* analyzer)
      : analyzer_(analyzer), cached_flags_(analyzer->rewrite_simplify.GetEnabledExtensions()) {
    auto new_flags = static_cast<RewriteSimplifier::Extension>(
        cached_flags_ & (~RewriteSimplifier::kConvertBooleanToAndOfOrs));
    analyzer->rewrite_simplify.SetEnabledExtensions(new_flags);
  }
  ~DisableAndOfOrRecursion() { analyzer_->rewrite_simplify.SetEnabledExtensions(cached_flags_); }

  DisableAndOfOrRecursion(const DisableAndOfOrRecursion&) = delete;
  DisableAndOfOrRecursion& operator=(const DisableAndOfOrRecursion&) = delete;

 private:
  Analyzer* analyzer_;
  RewriteSimplifier::Extension cached_flags_;
};

}  // namespace

PrimExpr SimplifyAsAndOfOrs(const PrimExpr& expr, Analyzer* analyzer, bool recursive) {
  DisableAndOfOrRecursion context(analyzer);

  // static bool currently_calling = false;

  // if (currently_calling) {
  //   std::cout << "Recursively called SimplifyAsAndOfOrs, which shouldn't be possible" <<
  //   std::endl;
  // }

  // bool cache = currently_calling;
  // currently_calling = true;

  // auto timer = DebugTimer(
  //     [&]() {
  //       std::stringstream ss;
  //       ss << "Simplifying " << expr;
  //       return ss.str();
  //     }(),
  //     5);

  // PrimExpr pre_simplified = [&]() {
  //   auto timer = DebugTimer(
  //       [&]() {
  //         std::stringstream ss;
  //         ss << "Pre-simplification step for " << expr;
  //         return ss.str();
  //       }(),
  //       5);
  //   return analyzer->Simplify(expr);
  // }();
  PrimExpr pre_simplified = expr;

  auto timer =
      DebugTimer("Simplifying expr as AndOfOrs").on_start([&](auto& out) { out << expr; }).start();

  AndOfOrs repr(pre_simplified);

  {
    auto timer = DebugTimer("Calling AndOfOrs::Simplify")
                     .on_start([&](auto& out) { out << " on " << expr; })
                     .on_finish([&repr, orig = expr, converted = repr.AsPrimExpr()](auto& out) {
                       PrimExpr after = repr.AsPrimExpr();
                       if (StructuralEqual()(orig, after)) {
                         out << "no change";
                       } else {
                         out << "converted to " << converted << ", then simplified to " << after;
                       }
                     })
                     .start();
    repr.Simplify(analyzer, true);
    // repr.Simplify(analyzer, recursive);
  }

  // std::cout << "\t"
  //           << "After simplifying, have " << repr.NumTerms() << " in " << repr.NumChunks()
  //           << " chunks" << std::endl;

  // if (recursive || true) {
  //   auto info = repr.MakeRecursiveStep(analyzer);
  // }

  auto out = repr.AsPrimExpr();
  // currently_calling = cache;
  return out;
}

}  // namespace arith
}  // namespace tvm
