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

PrimExpr SimplifyUsingAndOfOrs(const PrimExpr& orig, Analyzer* analyzer) {
  auto vec_and = ExtractAndOfOrs(orig);
  ExprDeepEqual expr_equal;

  auto is_const_true = [](const PrimExpr& expr) {
    const auto* as_int = as_const_int(expr);
    return as_int && *as_int;
  };

  auto is_const_false = [](const PrimExpr& expr) {
    const auto* as_int = as_const_int(expr);
    return as_int && !*as_int;
  };

  auto known_from_and = [&](size_t i_and) {
    PrimExpr out = Bool(true);
    for (size_t j_and = 0; j_and < vec_and.size(); j_and++) {
      if (i_and != j_and && vec_and[j_and].size() == 1) {
        out = out && vec_and[j_and][0];
      }
    }
    return out;
  };

  auto known_from_or = [&](size_t i_and, size_t i_or) {
    PrimExpr out = Bool(true);
    const auto& vec_or = vec_and[i_and];
    for (size_t j_or = 0; j_or < vec_or.size(); j_or++) {
      if (i_or != j_or) {
        out = out && RewriteBooleanOperators(tir::Not(vec_or[j_or]));
      }
    }
    return out;
  };

  auto cleanup_or = [&](size_t i_and) {
    auto& vec_or = vec_and[i_and];
    if (std::any_of(vec_or.begin(), vec_or.end(), is_const_true)) {
      vec_or = {Bool(true)};
    } else {
      vec_or.erase(std::remove_if(vec_or.begin(), vec_or.end(), is_const_false), vec_or.end());
    }
  };

  auto cleanup_and = [&]() {
    auto is_vec_false = [](const std::vector<PrimExpr>& vec) { return vec.size() == 0; };
    auto is_vec_true = [is_const_true](const std::vector<PrimExpr>& vec) {
      return vec.size() == 1 && is_const_true(vec[0]);
    };

    if (std::any_of(vec_and.begin(), vec_and.end(), is_vec_false)) {
      vec_and = {{}};
    } else {
      vec_and.erase(std::remove_if(vec_and.begin(), vec_and.end(), is_vec_true), vec_and.end());
    }
  };

  auto try_merge_or = [&](const PrimExpr& a, const PrimExpr& b, const auto& callback) {
    PrimExpr joint = a || b;
    PrimExpr simplified = analyzer->Simplify(joint);
    if (!expr_equal(simplified, joint)) {
      if (auto* simplified_or = simplified.as<OrNode>()) {
        callback(simplified_or->a, simplified_or->b);
      } else {
        callback(Bool(false), simplified);
      }
    }
  };

  auto try_merge_and = [&](const PrimExpr& a, const PrimExpr& b, const auto& callback) {
    PrimExpr joint = a && b;
    PrimExpr simplified = analyzer->Simplify(joint);
    if (!expr_equal(simplified, joint)) {
      if (auto* simplified_and = simplified.as<AndNode>()) {
        callback(simplified_and->a, simplified_and->b);
      } else {
        callback(Bool(true), simplified);
      }
    }
  };

  auto print_current = [&](size_t num_tabs = 0) {
    std::string tabs(num_tabs, '\t');
    std::cout << tabs << "expr = (";
    for (size_t i_and = 0; i_and < vec_and.size(); i_and++) {
      const auto& vec_or = vec_and[i_and];

      std::cout << "\n" << tabs << "\t";
      if (i_and) {
        std::cout << " and ";
      } else {
        std::cout << "     ";
      }
      std::cout << "(";
      for (size_t i_or = 0; i_or < vec_or.size(); i_or++) {
        const PrimExpr& expr = vec_or[i_or];

        if (i_or) {
          std::cout << " or ";
        }
        std::cout << expr;
      }
      std::cout << ")";
    }
    if (vec_and.size()) {
      std::cout << "\n" << tabs;
    }
    std::cout << ")" << std::endl;
  };

  print_current();

  // Simplification within each individual expression
  std::cout << "Starting Step 1: "
            << "Local simplifications" << std::endl;
  while (true) {
    bool modified_and = false;
    for (size_t i_and = 0; i_and < vec_and.size(); i_and++) {
      With<ConstraintContext> context(analyzer, known_from_and(i_and));

      auto& vec_or = vec_and[i_and];

      while (true) {
        bool modified_or = false;
        for (size_t i_or = 0; i_or < vec_or.size(); i_or++) {
          With<ConstraintContext> context(analyzer, known_from_or(i_and, i_or));
          PrimExpr simplified = analyzer->Simplify(vec_or[i_or]);
          if (!expr_equal(simplified, vec_or[i_or])) {
            std::cout << "\t"
                      << "Simplified vec_and[" << i_and << "][" << i_or
                      << "] = " << vec_and[i_and][i_or] << " to " << simplified << std::endl;
            vec_or[i_or] = simplified;
            modified_or = true;
            modified_and = true;
          }
        }

        if (modified_or) {
          cleanup_or(i_and);
        } else {
          break;
        }
      }
    }

    if (modified_and) {
      cleanup_and();
    } else {
      break;
    }
  }
  std::cout << "Finished Step 1: "
            << "Local simplifications" << std::endl;

  print_current();

  std::cout << "Starting Step 2: "
            << "Simplifications within OR" << std::endl;

  // Simplification within pairs of OR statements
  {
    bool modified_and = false;
    for (size_t i_and = 0; i_and < vec_and.size(); i_and++) {
      auto& vec_or = vec_and[i_and];

      bool modified_or = false;
      for (size_t i_or = 0; i_or < vec_or.size(); i_or++) {
        for (size_t j_or = i_or + 1; j_or < vec_or.size(); j_or++) {
          auto& expr_i = vec_or[i_or];
          auto& expr_j = vec_or[j_or];
          try_merge_or(expr_i, expr_j, [&](PrimExpr new_i, PrimExpr new_j) {
            expr_i = new_i;
            expr_j = new_j;
            modified_or = true;
            modified_and = true;
          });
        }
      }

      if (modified_or) {
        cleanup_or(i_and);
      } else {
        break;
      }
    }

    if (modified_and) {
      cleanup_and();
    }
  }

  std::cout << "Finished Step 2: "
            << "Simplifications within OR" << std::endl;

  print_current();

  std::cout << "Starting Step 3: "
            << "Simplifications across OR" << std::endl;

  // Simplifications across AND pairs in related OR sets
  //
  // Requiring that (B and C) simplifies to a single expression
  // (A or B) and (A or C) => A or (B and C)
  while (true) {
    bool modified_and = false;

    for (size_t i_and = 0; i_and < vec_and.size(); i_and++) {
      for (size_t j_and = i_and + 1; j_and < vec_and.size(); j_and++) {
        auto& i_vec_or = vec_and[i_and];
        auto& j_vec_or = vec_and[j_and];

        if (i_vec_or.size() == 1 && j_vec_or.size() == 1) {
          auto& expr_i = i_vec_or[0];
          auto& expr_j = j_vec_or[0];
          try_merge_and(expr_i, expr_j, [&](PrimExpr new_i, PrimExpr new_j) {
            expr_i = new_i;
            expr_j = new_j;
            modified_and = true;
          });
          continue;
        }
        std::unordered_set<PrimExpr, StructuralHash, StructuralEqual> j_expr_set(j_vec_or.begin(),
                                                                                 j_vec_or.end());

        std::optional<size_t> i_distinct_expr;
        for (size_t k = 0; k < i_vec_or.size(); k++) {
          if (!j_expr_set.count(i_vec_or[k])) {
            i_distinct_expr = k;
            break;
          }
        }

        if (!i_distinct_expr.has_value()) {
          // i_vec_or is a subset of j_vec_or.  i_vec_or imposes a
          // stronger condition, so remove j_vec_or.

          j_vec_or = {Bool(true)};
          std::swap(i_vec_or, j_vec_or);
          modified_and = true;

          std::cout << "\t"
                    << "Removing vec_and[" << j_and << "], which is a subset of vec_and[" << i_and
                    << "], then swapping vec_and[" << i_and << "] into location of vec_and["
                    << j_and << "]" << std::endl;
          print_current(2);
          continue;
        }

        std::unordered_set<PrimExpr, StructuralHash, StructuralEqual> i_expr_set(i_vec_or.begin(),
                                                                                 i_vec_or.end());

        std::optional<size_t> j_distinct_expr;
        for (size_t k = 0; k < j_vec_or.size(); k++) {
          if (!i_expr_set.count(j_vec_or[k])) {
            j_distinct_expr = k;
            break;
          }
        }

        if (!j_distinct_expr.has_value()) {
          // j_vec_or is a subset of i_vec_or.  j_vec_or imposes a
          // stronger condition, so remove i_vec_or.

          j_vec_or = {Bool(true)};
          modified_and = true;

          std::cout << "\t"
                    << "Removing vec_and[" << i_and << "], which is a subset of vec_and[" << j_and
                    << "]" << std::endl;
          print_current(2);

          continue;
        }

        if (i_vec_or.size() == j_vec_or.size()) {
          size_t num_shared_exprs = 0;
          for (const auto& expr : j_vec_or) {
            if (i_expr_set.count(expr)) {
              ++num_shared_exprs;
            }
          }

          std::cout << "\t"
                    << "vec_and[" << i_and << "] and vec_and[" << j_and << "] are the same length "
                    << i_vec_or.size() << " and have " << num_shared_exprs
                    << " expresssions in common" << std::endl;

          if (num_shared_exprs + 1 == i_vec_or.size()) {
            // All but one of the expressions are shared.  If the AND
            // of the distinct expressions can be simplified, we can
            // replace
            //
            // (A or B) and (A or C) => A or (B and C)
            auto& expr_i = i_vec_or[i_distinct_expr.value()];
            auto& expr_j = j_vec_or[j_distinct_expr.value()];
            bool made_update = false;
            std::cout << "\t"
                      << "vec_and[" << i_and << "] and vec_and[" << j_and << "] differ only by "
                      << expr_i << " in vec_and[" << i_and << "] and " << expr_j << " in vec_and["
                      << j_and << "]" << std::endl;
            try_merge_and(expr_i, expr_j, [&](PrimExpr new_i, PrimExpr new_j) {
              std::cout << "\t\t"
                        << "Can simplify (" << expr_i << " && " << expr_j << ") to "
                        << (new_i && new_j) << ", "
                        << "replacing " << expr_i << " => " << new_i << " and " << expr_j << " => "
                        << new_j << std::endl;
              expr_i = new_i;
              expr_j = new_j;
              made_update = true;
              print_current(3);
            });
            if (made_update) {
              cleanup_or(i_and);
              cleanup_or(j_and);
              modified_and = true;
              continue;
            }
          }
        }
      }
    }

    if (modified_and) {
      cleanup_and();
    } else {
      break;
    }
  }

  std::cout << "Finished Step 3: "
            << "Simplifications across OR" << std::endl;

  print_current();

  // Make the final expression
  PrimExpr output = Bool(true);
  for (const auto& vec_or : vec_and) {
    PrimExpr constraint = Bool(false);
    for (const auto& expr : vec_or) {
      constraint = constraint || expr;
    }
    output = output && constraint;
  }
  return output;
}

}  // namespace arith
}  // namespace tvm
