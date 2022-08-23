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

PrimExpr ConvertToAndOfOrs(const PrimExpr& expr) {
  PrimExpr output = Bool(true);
  CollectConstraints2(expr, [&](const PrimExpr& part) { output = output && part; });
  return output;
}

std::vector<std::vector<PrimExpr>> ExtractAndOfOrs(const PrimExpr& expr) {
  std::vector<std::vector<PrimExpr>> out;
  CollectConstraints2(expr, [&](const PrimExpr& part) { out.push_back(ExtractComponents(part)); });
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
  return out;
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
// For debug, from https://stackoverflow.com/a/46455079
//
// TODO: Remove this
class NullStream : public std::ostream {
  class NullBuffer : public std::streambuf {
   public:
    int overflow(int c) { return c; }
  } m_nb;

 public:
  NullStream() : std::ostream(&m_nb) {}
};
}  // namespace

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

  std::ostream& printer = std::cout;
  // auto printer = NullStream();

  auto print_current = [&](size_t num_tabs = 0) {
    std::string tabs(num_tabs, '\t');
    printer << tabs << "expr = (";
    for (size_t i_and = 0; i_and < vec_and.size(); i_and++) {
      const auto& vec_or = vec_and[i_and];

      printer << "\n" << tabs << "\t";
      if (i_and) {
        printer << " and ";
      } else {
        printer << "     ";
      }
      printer << "(";
      for (size_t i_or = 0; i_or < vec_or.size(); i_or++) {
        const PrimExpr& expr = vec_or[i_or];

        if (i_or) {
          printer << " or ";
        }
        printer << expr;
      }
      printer << ")";
    }
    if (vec_and.size()) {
      printer << "\n" << tabs;
    }
    printer << ")" << std::endl;
  };

  print_current();

  // Simplification within each individual expression
  printer << "Starting Step 1 using AND of OR: "
          << "Local simplifications" << std::endl;
  while (true) {
    bool modified_and = false;
    for (size_t i_and = 0; i_and < vec_and.size(); i_and++) {
      // printer << "Visiting i_and = " << i_and << std::endl;
      With<ConstraintContext> context(analyzer, known_from_and(i_and));

      auto& vec_or = vec_and[i_and];

      while (true) {
        bool modified_or = false;
        for (size_t i_or = 0; i_or < vec_or.size(); i_or++) {
          // printer << "Visiting i_and = " << i_and << ", i_or = " << i_or << std::flush
          //         << ", expr = " << vec_or[i_or] << std::endl;
          With<ConstraintContext> context(analyzer, known_from_or(i_and, i_or));
          // printer << "\t"
          //         << "Simplifying vec_and[" << i_and << "][" << i_or
          //         << "] = " << vec_and[i_and][i_or] << std::endl;
          PrimExpr simplified = analyzer->Simplify(vec_or[i_or]);

          // printer << "\t"
          //         << "Simplified vec_and[" << i_and << "][" << i_or
          //         << "] = " << vec_and[i_and][i_or] << " to " << simplified << std::endl;
          if (!expr_equal(simplified, vec_or[i_or])) {
            printer << "\t"
                    << "Replacing vec_and[" << i_and << "][" << i_or
                    << "] = " << vec_and[i_and][i_or] << " with " << simplified << std::endl;
            vec_or[i_or] = simplified;
            modified_or = true;
            modified_and = true;

            print_current();
          }
          // printer << "Finished visiting i_and = " << i_and << ", i_or = " << i_or << std::flush
          //         << ", expr = " << vec_or[i_or] << std::endl;
        }

        if (modified_or) {
          printer << "\t"
                  << "Cleaning up OR-group # " << i_and << std::endl;
          cleanup_or(i_and);
          print_current(1);
        } else {
          break;
        }
      }
    }

    if (modified_and) {
      printer << "\t"
              << "Cleaning up AND-groups" << std::endl;
      cleanup_and();
      print_current(1);
    } else {
      break;
    }
  }
  printer << "Finished Step 1 using AND of OR: "
          << "Local simplifications" << std::endl;

  print_current();

  printer << "Starting Step 2 using AND of OR: "
          << "Simplifications within OR" << std::endl;

  // Simplification within pairs of OR statements
  {
    bool modified_and = false;
    for (size_t i_and = 0; i_and < vec_and.size(); i_and++) {
      auto& vec_or = vec_and[i_and];

      while (true) {
        bool modified_or = false;
        for (size_t i_or = 0; i_or < vec_or.size(); i_or++) {
          for (size_t j_or = i_or + 1; j_or < vec_or.size(); j_or++) {
            auto& expr_i = vec_or[i_or];
            auto& expr_j = vec_or[j_or];

            // printer << "\t"
            //         << "Attempting to simplify (vec_and[" << i_and << "][" << i_or
            //         << "] && vec_and[" << i_and << "][" << j_or << "]) == (" << expr_i << " && "
            //         << expr_j << ")" << std::endl;

            try_merge_or(expr_i, expr_j, [&](PrimExpr new_i, PrimExpr new_j) {
              printer << "\t\t"
                      << "Simplified (vec_and[" << i_and << "][" << i_or << "] && vec_and[" << i_and
                      << "][" << j_or << "]) == (" << expr_i << " && " << expr_j
                      << ") == " << (new_i && new_j) << std::endl;
              expr_i = new_i;
              expr_j = new_j;
              modified_or = true;
              modified_and = true;
              print_current(2);
            });
          }
        }

        if (modified_or) {
          printer << "\t"
                  << "Cleaning up OR-group # " << i_and << std::endl;
          cleanup_or(i_and);
          print_current(1);
        } else {
          break;
        }
      }
    }

    if (modified_and) {
      printer << "\t"
              << "Cleaning up AND-groups" << std::endl;
      cleanup_and();
      print_current(1);
    }
  }

  printer << "Finished Step 2 using AND of OR: "
          << "Simplifications within OR" << std::endl;

  print_current();

  printer << "Starting Step 3 using AND of OR: "
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

          printer << "\t"
                  << "Removing vec_and[" << j_and << "], which is a subset of vec_and[" << i_and
                  << "], then swapping vec_and[" << i_and << "] into location of vec_and[" << j_and
                  << "]" << std::endl;
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

          printer << "\t"
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

          printer << "\t"
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
            printer << "\t"
                    << "vec_and[" << i_and << "] and vec_and[" << j_and << "] differ only by "
                    << expr_i << " in vec_and[" << i_and << "] and " << expr_j << " in vec_and["
                    << j_and << "]" << std::endl;
            try_merge_and(expr_i, expr_j, [&](PrimExpr new_i, PrimExpr new_j) {
              printer << "\t\t"
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

  printer << "Finished Step 3 using AND of OR: "
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

PrimExpr SimplifyUsingOrOfAnds(const PrimExpr& orig, Analyzer* analyzer) {
  // TODO: Swap all usage/semantics of AND/OR
  auto t_vec_or = ExtractOrOfAnds(orig);

  ExprDeepEqual expr_equal;

  auto is_const_true = [](const PrimExpr& expr) {
    const auto* as_int = as_const_int(expr);
    return as_int && *as_int;
  };

  auto is_const_false = [](const PrimExpr& expr) {
    const auto* as_int = as_const_int(expr);
    return as_int && !*as_int;
  };

  auto t_known_from_or = [&](size_t i_and) {
    PrimExpr out = Bool(true);
    for (size_t t_j_or = 0; t_j_or < t_vec_or.size(); t_j_or++) {
      if (i_and != t_j_or && t_vec_or[t_j_or].size() == 1) {
        out = out && RewriteBooleanOperators(tir::Not(t_vec_or[t_j_or][0]));
      }
    }
    return out;
  };

  auto t_known_from_and = [&](size_t t_i_or, size_t t_i_and) {
    PrimExpr out = Bool(true);
    const auto& t_vec_and = t_vec_or[t_i_or];
    for (size_t t_j_and = 0; t_j_and < t_vec_and.size(); t_j_and++) {
      if (t_i_and != t_j_and) {
        out = out && t_vec_and[t_j_and];
      }
    }
    return out;
  };

  auto t_cleanup_and = [&](size_t t_i_or) {
    auto& t_vec_and = t_vec_or[t_i_or];
    if (std::any_of(t_vec_and.begin(), t_vec_and.end(), is_const_false)) {
      t_vec_and = {Bool(false)};
    } else {
      t_vec_and.erase(std::remove_if(t_vec_and.begin(), t_vec_and.end(), is_const_true),
                      t_vec_and.end());
    }
  };

  auto t_cleanup_or = [&]() {
    auto is_vec_false = [](const std::vector<PrimExpr>& t_vec_and) {
      return t_vec_and.size() == 0;
    };
    auto is_vec_true = [is_const_true](const std::vector<PrimExpr>& t_vec_and) {
      return t_vec_and.size() == 1 && is_const_true(t_vec_and[0]);
    };

    if (std::any_of(t_vec_or.begin(), t_vec_or.end(), is_vec_false)) {
      t_vec_or = {{}};
    } else {
      t_vec_or.erase(std::remove_if(t_vec_or.begin(), t_vec_or.end(), is_vec_true), t_vec_or.end());
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

  std::ostream& printer = std::cout;
  // auto printer = NullStream();

  // auto print_current = [&](size_t num_tabs = 0) {
  //   std::string tabs(num_tabs, '\t');
  //   printer << tabs << "expr = (";
  //   for (size_t i_and = 0; i_and < t_vec_or.size(); i_and++) {
  //     const auto& t_vec_and = t_vec_or[i_and];

  //     printer << "\n" << tabs << "\t";
  //     if (i_and) {
  //       printer << " or ";
  //     } else {
  //       printer << "    ";
  //     }
  //     printer << "(";
  //     for (size_t i_or = 0; i_or < t_vec_and.size(); i_or++) {
  //       const PrimExpr& expr = t_vec_and[i_or];

  //       if (i_or) {
  //         printer << " and ";
  //       }
  //       printer << expr;
  //     }
  //     printer << ")";
  //   }
  //   if (t_vec_or.size()) {
  //     printer << "\n" << tabs;
  //   }
  //   printer << ")" << std::endl;
  // };

  auto print_current = [&](size_t num_tabs = 0) {
    std::string tabs(num_tabs, '\t');
    printer << tabs << "expr = any(";
    for (size_t i_and = 0; i_and < t_vec_or.size(); i_and++) {
      const auto& t_vec_and = t_vec_or[i_and];

      printer << "\n" << tabs << "\t";

      if (t_vec_and.size() == 0) {
        printer << "all()";
      } else if (t_vec_and.size() == 1) {
        printer << t_vec_and[0];
      } else {
        printer << "all(";
        for (size_t i_or = 0; i_or < t_vec_and.size(); i_or++) {
          const PrimExpr& expr = t_vec_and[i_or];
          printer << "\n" << tabs << "\t\t" << expr << ",";
        }
        printer << "\n"
                << tabs << "\t"
                << ")";
      }

      printer << ",";
    }
    printer << "\n" << tabs << ")" << std::endl;
  };

  print_current();

  // Simplification within each individual expression
  printer << "Starting Step 1 using OR of AND: "
          << "Local simplifications" << std::endl;
  while (true) {
    bool t_modified_or = false;
    for (size_t t_i_or = 0; t_i_or < t_vec_or.size(); t_i_or++) {
      // printer << "Visiting i_and = " << i_and << std::endl;
      With<ConstraintContext> context(analyzer, t_known_from_or(t_i_or));

      auto& t_vec_and = t_vec_or[t_i_or];

      while (true) {
        bool t_modified_and = false;
        for (size_t t_i_and = 0; t_i_and < t_vec_and.size(); t_i_and++) {
          // printer << "Visiting i_or = " << t_i_or << ", i_and = " << i_and << std::flush
          //         << ", expr = " << t_vec_and[t_i_and] << std::endl;
          With<ConstraintContext> context(analyzer, t_known_from_and(t_i_or, t_i_and));
          // printer << "\t"
          //         << "Simplifying vec_or[" << t_i_or << "][" << t_i_and
          //         << "] = " << t_vec_or[t_i_or][t_i_and] << std::endl;
          PrimExpr simplified = analyzer->Simplify(t_vec_and[t_i_and]);

          // printer << "\t"
          //         << "Simplified vec_or[" << t_i_or << "][" << t_i_and
          //         << "] = " << t_vec_or[t_i_or][t_i_and] << " to " << simplified << std::endl;
          if (!expr_equal(simplified, t_vec_and[t_i_and])) {
            printer << "\t"
                    << "Replacing vec_or[" << t_i_or << "][" << t_i_and
                    << "] = " << t_vec_or[t_i_or][t_i_and] << " with " << simplified << std::endl;
            t_vec_and[t_i_and] = simplified;
            t_modified_and = true;
            t_modified_or = true;

            print_current();
          }
          // printer << "Finished visiting i_and = " << i_and << ", i_or = " << i_or << std::flush
          //         << ", expr = " << vec_or[i_or] << std::endl;
        }

        if (t_modified_and) {
          printer << "\t"
                  << "Cleaning up AND-group # " << t_i_or << std::endl;
          t_cleanup_and(t_i_or);
          print_current(1);
        } else {
          break;
        }
      }
    }

    if (t_modified_or) {
      printer << "\t"
              << "Cleaning up OR-groups" << std::endl;
      t_cleanup_or();
      print_current(1);
    } else {
      break;
    }
  }
  printer << "Finished Step 1 using OR of AND: "
          << "Local simplifications" << std::endl;

  print_current();

  printer << "Starting Step 2 using OR of AND: "
          << "Simplifications within AND" << std::endl;

  // Simplification within pairs of OR statements
  {
    bool t_modified_or = false;
    for (size_t t_i_or = 0; t_i_or < t_vec_or.size(); t_i_or++) {
      auto& t_vec_and = t_vec_or[t_i_or];

      while (true) {
        bool t_modified_and = false;
        for (size_t t_i_and = 0; t_i_and < t_vec_and.size(); t_i_and++) {
          for (size_t t_j_and = t_i_and + 1; t_j_and < t_vec_and.size(); t_j_and++) {
            auto& expr_i = t_vec_and[t_i_and];
            auto& expr_j = t_vec_and[t_j_and];

            // printer << "\t"
            //         << "Attempting to simplify (vec_or[" << t_i_or << "][" << t_i_and
            //         << "] || vec_or[" << t_i_or << "][" << t_j_and << "]) == (" << expr_i << " &&
            //         "
            //         << expr_j << ")" << std::endl;

            try_merge_and(expr_i, expr_j, [&](PrimExpr new_i, PrimExpr new_j) {
              printer << "\t\t"
                      << "Simplified (vec_or[" << t_i_or << "][" << t_i_and << "] && vec_and["
                      << t_i_or << "][" << t_j_and << "]) == (" << expr_i << " && " << expr_j
                      << ") == " << (new_i && new_j) << std::endl;
              expr_i = new_i;
              expr_j = new_j;
              t_modified_and = true;
              t_modified_or = true;
              print_current(2);
            });
          }
        }

        if (t_modified_and) {
          printer << "\t"
                  << "Cleaning up OR-group # " << t_i_or << std::endl;
          t_cleanup_and(t_i_or);
          print_current(1);
        } else {
          break;
        }
      }
    }

    if (t_modified_or) {
      printer << "\t"
              << "Cleaning up AND-groups" << std::endl;
      t_cleanup_or();
      print_current(1);
    }
  }

  printer << "Finished Step 2 using OR of AND: "
          << "Simplifications within AND" << std::endl;

  print_current();

  printer << "Starting Step 3 using OR of AND: "
          << "Simplifications across AND" << std::endl;

  // Simplifications across AND pairs in related OR sets
  //
  // Performs simplifications of the type:
  // (A and B) or (A and C) => A and (B or C),
  // where (B or C) simplifies to a single expression.
  while (true) {
    bool t_modified_or = false;

    for (size_t t_i_or = 0; t_i_or < t_vec_or.size(); t_i_or++) {
      for (size_t t_j_or = t_i_or + 1; t_j_or < t_vec_or.size(); t_j_or++) {
        auto& t_i_vec_and = t_vec_or[t_i_or];
        auto& t_j_vec_and = t_vec_or[t_j_or];

        if (t_i_vec_and.size() == 1 && t_j_vec_and.size() == 1) {
          auto& expr_i = t_i_vec_and[0];
          auto& expr_j = t_j_vec_and[0];
          try_merge_or(expr_i, expr_j, [&](PrimExpr new_i, PrimExpr new_j) {
            expr_i = new_i;
            expr_j = new_j;
            t_modified_or = true;
          });
          continue;
        }
        std::unordered_set<PrimExpr, StructuralHash, StructuralEqual> j_expr_set(
            t_j_vec_and.begin(), t_j_vec_and.end());

        std::optional<size_t> i_distinct_expr;
        for (size_t k = 0; k < t_i_vec_and.size(); k++) {
          if (!j_expr_set.count(t_i_vec_and[k])) {
            i_distinct_expr = k;
            break;
          }
        }

        if (!i_distinct_expr.has_value()) {
          // all(i_vec_and) || (all(i_vec_and) && all(j_but_not_i))
          //
          // i_vec_and is a subset of the conditions in j_vec_and.
          // Anything that passes j_vec_and will also pass i_vec_and,
          // so we can remove j_vec_and.

          t_j_vec_and = {Bool(false)};
          std::swap(t_i_vec_and, t_j_vec_and);
          t_modified_or = true;

          printer << "\t"
                  << "Removing vec_and[" << t_j_or << "], which is a subset of vec_and[" << t_i_or
                  << "], then swapping vec_and[" << t_i_or << "] into location of vec_and["
                  << t_j_or << "]" << std::endl;
          print_current(2);
          continue;
        }

        std::unordered_set<PrimExpr, StructuralHash, StructuralEqual> i_expr_set(
            t_i_vec_and.begin(), t_i_vec_and.end());

        std::optional<size_t> j_distinct_expr;
        for (size_t k = 0; k < t_j_vec_and.size(); k++) {
          if (!i_expr_set.count(t_j_vec_and[k])) {
            j_distinct_expr = k;
            break;
          }
        }

        if (!j_distinct_expr.has_value()) {
          // (all(j_vec_and) && all(i_but_not_j)) || all(j_vec_and)
          //
          // j_vec_and is a subset of i_vec_and.  Anything that passes
          // i_vec_and will also pass j_vec_and, so we can remove
          // i_vec_and.

          t_i_vec_and = {Bool(false)};
          t_modified_or = true;

          printer << "\t"
                  << "Removing vec_or[" << t_i_or << "], which is a subset of vec_and[" << t_j_or
                  << "]" << std::endl;
          print_current(2);

          continue;
        }

        if (t_i_vec_and.size() == t_j_vec_and.size()) {
          size_t num_shared_exprs = 0;
          for (const auto& expr : t_j_vec_and) {
            if (i_expr_set.count(expr)) {
              ++num_shared_exprs;
            }
          }

          printer << "\t"
                  << "vec_or[" << t_i_or << "] and vec_or[" << t_j_or << "] are the same length "
                  << t_i_vec_and.size() << " and have " << num_shared_exprs
                  << " expresssions in common" << std::endl;

          if (num_shared_exprs + 1 == t_i_vec_and.size()) {
            // All but one of the expressions are shared.  If the OR
            // of the distinct expressions can be simplified, we can
            // replace
            //
            // (A and B) or (A and C) => A and (B or C)
            auto& expr_i = t_i_vec_and[i_distinct_expr.value()];
            auto& expr_j = t_j_vec_and[j_distinct_expr.value()];
            bool made_update = false;
            printer << "\t"
                    << "vec_or[" << t_i_or << "] and vec_or[" << t_j_or << "] differ only by "
                    << expr_i << " in vec_or[" << t_i_or << "] and " << expr_j << " in vec_or["
                    << t_j_or << "]" << std::endl;
            try_merge_or(expr_i, expr_j, [&](PrimExpr new_i, PrimExpr new_j) {
              printer << "\t\t"
                      << "Can simplify (" << expr_i << " || " << expr_j << ") to "
                      << (new_i && new_j) << ", "
                      << "replacing " << expr_i << " => " << new_i << " and " << expr_j << " => "
                      << new_j << std::endl;
              expr_i = new_i;
              expr_j = new_j;
              made_update = true;
              print_current(3);
            });
            if (made_update) {
              t_cleanup_and(t_i_or);
              t_cleanup_and(t_j_or);
              t_modified_or = true;
              continue;
            }
          }
        }
      }
    }

    if (t_modified_or) {
      print_current(1);
      printer << "\t"
              << "About to cleanup or" << std::endl;
      t_cleanup_or();
      printer << "\t"
              << "After cleanup of or" << std::endl;
      print_current(1);
    } else {
      break;
    }
  }

  printer << "Finished Step 3 using OR of AND: "
          << "Simplifications across AND" << std::endl;

  print_current();

  // Make the final expression
  PrimExpr output = Bool(false);
  for (const auto& vec_or : t_vec_or) {
    PrimExpr component = Bool(true);
    for (const auto& expr : vec_or) {
      component = component && expr;
    }
    output = output || component;
  }
  return output;
}

PrimExpr SimplifyUsingCNFAndDNF(const PrimExpr& orig, Analyzer* analyzer, int max_rounds) {
  ExprDeepEqual expr_equal;

  PrimExpr lookback = Bool(false);
  PrimExpr expr = orig;

  int temp_total_rounds = 0;

  for (int i = 0; i < max_rounds; i++) {
    temp_total_rounds++;
    std::cout << "\t"
              << "Starting round " << i << ", expr = " << expr << std::endl;

    if (as_const_int(expr)) {
      std::cout << "\t\t"
                << "Round " << i << " started with a constant, breaking" << std::endl;
      break;
    }

    PrimExpr simplified = [&]() {
      if (i % 2 == 0) {
        return SimplifyUsingAndOfOrs(expr, analyzer);
      } else {
        return SimplifyUsingOrOfAnds(expr, analyzer);
      }
    }();

    std::cout << "\t\t"
              << "Round " << i << " simplified from " << expr << " to " << simplified << std::endl;

    bool converged = expr_equal(simplified, lookback);
    lookback = expr;
    expr = simplified;
    if (converged) {
      std::cout << "\t\t"
                << "Round " << i << " is the same as round " << i - 2 << ", breaking" << std::endl;
      break;
    }
  }

  std::cout << "\t"
            << "SimplifyUsingCNFAndDNF, simplified " << orig << " to " << expr << " after "
            << temp_total_rounds << " total rounds" << std::endl;

  return expr;
}

}  // namespace arith
}  // namespace tvm
