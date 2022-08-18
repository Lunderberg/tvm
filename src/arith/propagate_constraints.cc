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
 * \file tvm/arith/propagate_constraints.cc
 */

#include "propagate_constraints.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>

#include "constraint_extract.h"
#include "pattern_match.h"

namespace tvm {
namespace arith {

using namespace tir;

CompareResult operator&(CompareResult lhs, CompareResult rhs) {
  return CompareResult(static_cast<int>(lhs) & static_cast<int>(rhs));
}
CompareResult operator|(CompareResult lhs, CompareResult rhs) {
  return CompareResult(static_cast<int>(lhs) | static_cast<int>(rhs));
}
CompareResult operator~(CompareResult cmp) {
  return CompareResult(~static_cast<int>(cmp) & static_cast<int>(CompareResult::kUnknown));
}
std::ostream& operator<<(std::ostream& os, CompareResult cmp) {
  switch (cmp) {
    case CompareResult::kInconsistent:
      return os << "Inconsistent";
    case CompareResult::kEQ:
      return os << "Equal";
    case CompareResult::kLT:
      return os << "LessThan";
    case CompareResult::kLE:
      return os << "LessThanOrEqual";
    case CompareResult::kGT:
      return os << "GreaterThan";
    case CompareResult::kGE:
      return os << "GreaterThanOrEqual";
    case CompareResult::kNE:
      return os << "NotEqual";
    case CompareResult::kUnknown:
      return os << "Unknown";
    default:
      LOG(FATAL) << "Invalid CompareResult: " << static_cast<int>(cmp);
      return os;
  }
}

namespace {

CompareResult Reverse(CompareResult res) {
  switch (res) {
    case CompareResult::kInconsistent:
      return CompareResult::kInconsistent;
    case CompareResult::kEQ:
      return CompareResult::kEQ;
    case CompareResult::kLT:
      return CompareResult::kGT;
    case CompareResult::kLE:
      return CompareResult::kGE;
    case CompareResult::kGT:
      return CompareResult::kLT;
    case CompareResult::kGE:
      return CompareResult::kLE;
    case CompareResult::kNE:
      return CompareResult::kNE;
    case CompareResult::kUnknown:
      return CompareResult::kUnknown;
    default:
      LOG(FATAL) << "Invalid CompareResult: " << res;
      return CompareResult::kInconsistent;
  }
}

CompareResult Negate(CompareResult res) {
  switch (res) {
    case CompareResult::kInconsistent:
      return CompareResult::kInconsistent;
    case CompareResult::kEQ:
      return CompareResult::kNE;
    case CompareResult::kLT:
      return CompareResult::kGE;
    case CompareResult::kLE:
      return CompareResult::kGT;
    case CompareResult::kGT:
      return CompareResult::kLE;
    case CompareResult::kGE:
      return CompareResult::kLT;
    case CompareResult::kNE:
      return CompareResult::kEQ;
    case CompareResult::kUnknown:
      return CompareResult::kUnknown;
    default:
      LOG(FATAL) << "Invalid CompareResult: " << res;
      return CompareResult::kInconsistent;
  }
}

// // Result of comparing (x RES_A y) and (y RES_B z) into (x OUT z)
// CompareResult Transitive(CompareResult a, CompareResult b) {
//   if (a == CompareResult::kInconsistent || b == CompareResult::kInconsistent) {
//     return CompareResult::kInconsistent;
//   }

//   if (a == CompareResult::kEQ) {
//     return b;
//   }
//   if (b == CompareResult::kEQ) {
//     return a;
//   }

//   if (a == CompareResult::kLE && b == CompareResult::kLE) {
//     return CompareResult::kLE;
//   } else if (a == CompareResult::kLE && b == CompareResult::kLT) {
//     return CompareResult::kLT;
//   } else if (a == CompareResult::kLT && b == CompareResult::kLE) {
//     return CompareResult::kLT;
//   } else if (a == CompareResult::kLT && b == CompareResult::kLT) {
//     return CompareResult::kLT;
//   }

//   if (a == CompareResult::kGE && b == CompareResult::kGE) {
//     return CompareResult::kGE;
//   } else if (a == CompareResult::kGE && b == CompareResult::kGT) {
//     return CompareResult::kGT;
//   } else if (a == CompareResult::kGT && b == CompareResult::kGE) {
//     return CompareResult::kGT;
//   } else if (a == CompareResult::kGT && b == CompareResult::kGT) {
//     return CompareResult::kGT;
//   }

//   return CompareResult::kUnknown;
// }
}  // namespace

Comparison::Comparison(const PrimExpr& expr) : orig_expr_(expr) {
  // std::cout << "Parsing expression " << expr << std::endl;
  PVar<PrimExpr> x, y;
  if ((x <= y).Match(expr)) {
    lhs_ = x.Eval();
    rhs_ = y.Eval();
    result_ = CompareResult::kLE;
  } else if ((x >= y).Match(expr)) {
    lhs_ = x.Eval();
    rhs_ = y.Eval();
    result_ = CompareResult::kGE;
  } else if ((x < y).Match(expr)) {
    lhs_ = x.Eval();
    rhs_ = y.Eval();
    result_ = CompareResult::kLT;
  } else if ((x > y).Match(expr)) {
    lhs_ = x.Eval();
    rhs_ = y.Eval();
    result_ = CompareResult::kGT;
  } else if ((x == y).Match(expr)) {
    lhs_ = x.Eval();
    rhs_ = y.Eval();
    result_ = CompareResult::kEQ;
  } else if ((x != y).Match(expr)) {
    lhs_ = x.Eval();
    rhs_ = y.Eval();
    result_ = CompareResult::kNE;
  }

  // std::cout << "\t"
  //           << "Parsed as lhs = " << lhs_ << ", rhs_ = " << rhs_ << ", comparison = " <<
  //           result_
  //           << std::endl;

  if (lhs_.as<IntImmNode>() && rhs_.as<IntImmNode>()) {
    lhs_ = PrimExpr();
    rhs_ = PrimExpr();
    return;
  }

  Normalize();

  // std::cout << "\t"
  //           << "Normalized to lhs = " << lhs_ << ", rhs_ = " << rhs_ << ", offset = " <<
  //           offset_
  //           << ", comparison = " << result_ << std::endl;
}

Comparison::Comparison(const PrimExpr& lhs, const PrimExpr& rhs, CompareResult result)
    : lhs_(lhs), rhs_(rhs), result_(result) {
  Normalize();
}

Comparison::Comparison(const PrimExpr& lhs, const PrimExpr& rhs, int64_t offset,
                       CompareResult result)
    : lhs_(lhs), rhs_(rhs), offset_(offset), result_(result) {
  Normalize();
}

std::pair<PrimExpr, int64_t> Comparison::RemoveOffset(const PrimExpr& expr) {
  PVar<PrimExpr> x;
  PVar<IntImm> c;
  if ((x + c).Match(expr)) {
    return {x.Eval(), c.Eval()->value};
  } else if ((x - c).Match(expr)) {
    return {x.Eval(), -c.Eval()->value};
  } else if (c.Match(expr)) {
    return {0, c.Eval()->value};
  } else {
    return {expr, 0};
  }
}

bool Comparison::IsValid() const {
  // These < and > should be removed during normalization.
  if (result_ == CompareResult::kLT || result_ == CompareResult::kGT) {
    return false;
  }
  return lhs_.defined() && rhs_.defined();
}

Comparison Comparison::Reversed() const {
  ICHECK(IsValid());

  Comparison output = *this;
  std::swap(output.lhs_, output.rhs_);
  output.offset_ = -output.offset_;
  output.result_ = Reverse(output.result_);
  output.Normalize();
  return output;
}

void Comparison::Normalize() {
  auto lhs_split = RemoveOffset(lhs_);
  auto rhs_split = RemoveOffset(rhs_);
  lhs_ = lhs_split.first;
  rhs_ = rhs_split.first;
  offset_ += (rhs_split.second - lhs_split.second);

  if (result_ == CompareResult::kLT) {
    result_ = CompareResult::kLE;
    offset_ -= 1;
  }
  if (result_ == CompareResult::kGT) {
    result_ = CompareResult::kGE;
    offset_ += 1;
  }
}

Comparison Comparison::NormalizedTo(const PrimExpr& expr) const {
  ExprDeepEqual equal;
  if (equal(expr, lhs_)) {
    return *this;
  } else if (equal(expr, rhs_)) {
    return Reversed();
  } else {
    return Comparison();
  }
}

Comparison Comparison::Negated() const {
  Comparison output = *this;
  output.result_ = Negate(output.result_);
  return output;
}

Optional<PrimExpr> Comparison::debug_as_primexpr() const {
  if (!IsValid()) {
    // return NullOpt;
    if (orig_expr_) {
      return tir::And(Bool(false), orig_expr_.value());
    } else {
      return NullOpt;
    }
  }

  IntImm offset(rhs_.dtype(), offset_);
  switch (result_) {
    case CompareResult::kInconsistent:
      return NullOpt;
    case CompareResult::kEQ:
      return lhs_ == rhs_ + offset;
    case CompareResult::kLT:
      return lhs_ < rhs_ + offset;
    case CompareResult::kLE:
      return lhs_ <= rhs_ + offset;
    case CompareResult::kGT:
      return lhs_ > rhs_ + offset;
    case CompareResult::kGE:
      return lhs_ >= rhs_ + offset;
    case CompareResult::kNE:
      return lhs_ != rhs_ + offset;
    case CompareResult::kUnknown:
      return NullOpt;
    default:
      LOG(FATAL) << "Invalid CompareResult: " << result_;
      return NullOpt;
  }
}

bool Comparison::Implies(const Comparison& other) const {
  // TODO: Remove these checks with appropriate comments
  ExprDeepEqual expr_equal;
  ICHECK(expr_equal(lhs_, other.lhs_) && expr_equal(rhs_, other.rhs_));

  auto any = [](CompareResult cmp) -> bool { return cmp != CompareResult::kInconsistent; };

  // TODO: Simplify these with bitwise operators

  // EQ rules

  if (result_ == CompareResult::kEQ && other.result_ == CompareResult::kEQ &&
      (offset_ == other.offset_)) {
    // if c1 == c2, x == y + c1 => x == y + c2
    return true;
  }

  if (result_ == CompareResult::kEQ && other.result_ == CompareResult::kLT &&
      (offset_ < other.offset_)) {
    // if c1 < c2, x == y + c1 => x < y + c2
    return true;
  }

  if (result_ == CompareResult::kEQ && other.result_ == CompareResult::kLE &&
      (offset_ <= other.offset_)) {
    // if c1 <= c2, x == y + c1 => x <= y + c2
    return true;
  }

  if (result_ == CompareResult::kEQ && other.result_ == CompareResult::kGT &&
      (offset_ > other.offset_)) {
    // if c1 > c2, x == y + c1 => x > y + c2
    return true;
  }

  if (result_ == CompareResult::kEQ && other.result_ == CompareResult::kGE &&
      (offset_ >= other.offset_)) {
    // if c1 >= c2, x == y + c1 => x >= y + c2
    return true;
  }

  if (result_ == CompareResult::kEQ && other.result_ == CompareResult::kNE &&
      (offset_ != other.offset_)) {
    // if c1 != c2, x == y + c1 => x != y + c2
    return true;
  }

  // LT rules

  if (result_ == CompareResult::kLT && other.result_ == CompareResult::kLT &&
      (offset_ <= other.offset_)) {
    // if c1 <= c2, x < y + c1 => x < y + c2
    return true;
  }

  if (result_ == CompareResult::kLT && other.result_ == CompareResult::kLE &&
      (offset_ <= other.offset_)) {
    // if c1 <= c2, x < y + c1 => x <= y + c2
    return true;
  }

  if (result_ == CompareResult::kLT && other.result_ == CompareResult::kNE &&
      (offset_ <= other.offset_)) {
    // if c1 <= c2, x < y + c1 => x != y + c2
    return true;
  }

  // LE rules

  if (result_ == CompareResult::kLE && other.result_ == CompareResult::kLT &&
      (offset_ < other.offset_)) {
    // if c1 < c2, x <= y + c1 => x < y + c2
    return true;
  }

  if (result_ == CompareResult::kLE && other.result_ == CompareResult::kLE &&
      (offset_ <= other.offset_)) {
    // if c1 <= c2, x <= y + c1 => x <= y + c2
    return true;
  }

  if (result_ == CompareResult::kLE && other.result_ == CompareResult::kNE &&
      (offset_ < other.offset_)) {
    // if c1 < c2, x <= y + c1 => x < y + c2 => x != y + c2
    return true;
  }

  // GT rules

  if (result_ == CompareResult::kGT && other.result_ == CompareResult::kGT &&
      (offset_ >= other.offset_)) {
    // if c1 >= c2, x > y + c1 => x > y + c2
    return true;
  }

  if (result_ == CompareResult::kGT && other.result_ == CompareResult::kGE &&
      (offset_ > other.offset_)) {
    // if c1 > c2, x > y + c1 => x >= y + c2
    return true;
  }

  if (result_ == CompareResult::kGT && other.result_ == CompareResult::kNE &&
      (offset_ >= other.offset_)) {
    // if c1 <= c2, x > y + c1 => x > y + c2 => x != y + c2
    return true;
  }

  // GE rules

  if (result_ == CompareResult::kGE && other.result_ == CompareResult::kGT &&
      (offset_ > other.offset_)) {
    // if c1 > c2, x >= y + c1 => x > y + c2
    return true;
  }

  if (result_ == CompareResult::kGE && other.result_ == CompareResult::kGE &&
      (offset_ >= other.offset_)) {
    // if c1 >= c2, x >= y + c1 => x >= y + c2
    return true;
  }

  if (result_ == CompareResult::kGE && other.result_ == CompareResult::kNE &&
      (offset_ > other.offset_)) {
    // if c1 != c2, x >= y + c1 => x > y + c2 => x != y + c2
    return true;
  }

  // NE rules

  if (result_ == CompareResult::kNE && other.result_ == CompareResult::kNE &&
      (offset_ == other.offset_)) {
    // if c1 == c2, x != y + c1 => x != y + c2
    return true;
  }

  return false;
}

Comparison Comparison::IntersectAssumingExpressionsMatch(const Comparison& other) const {
  // TODO: Remove these checks with appropriate comments
  ExprDeepEqual expr_equal;
  ICHECK(expr_equal(lhs_, other.lhs_) && expr_equal(rhs_, other.rhs_));

  if (Implies(other)) {
    return *this;
  } else if (other.Implies(*this)) {
    return other;
  } else if (Implies(other.Negated())) {
    Comparison output = *this;
    output.result_ = CompareResult::kInconsistent;
    return output;
  }

  if (result_ == CompareResult::kGE && other.result_ == CompareResult::kNE &&
      offset_ == other.offset_) {
    // (x >= y + c1) && (x != y + c1) => (x > y + c1) => (x >= y + c1 + 1)
    Comparison output = *this;
    output.offset_ += 1;
    return output;
  }
  if (result_ == CompareResult::kNE && other.result_ == CompareResult::kGE &&
      offset_ == other.offset_) {
    // (x != y + c1) && (x >= y + c1) => (x > y + c1) => (x >= y + c1 + 1)
    Comparison output = other;
    output.offset_ += 1;
    return output;
  }
  if (result_ == CompareResult::kLE && other.result_ == CompareResult::kNE &&
      offset_ == other.offset_) {
    // (x <= y + c1) && (x != y + c1) => (x < y + c1) => (x <= y + c1 + 1)
    Comparison output = *this;
    output.offset_ -= 1;
    return output;
  }
  if (result_ == CompareResult::kNE && other.result_ == CompareResult::kLE &&
      offset_ == other.offset_) {
    // (x != y + c1) && (x <= y + c1) => (x < y + c1) => (x <= y + c1 + 1)
    Comparison output = other;
    output.offset_ -= 1;
    return output;
  }

  // The intersection of these comparisons cannot be expressed as a
  // single comparison.
  return Comparison();
}

ComparisonSet::ComparisonSet(const std::vector<PrimExpr>& knowns) {
  for (const auto& expr : knowns) {
    AddKnown(expr);
  }
}
void ComparisonSet::AddKnown(const PrimExpr& expr) {
  // std::cout << "ComparisonSet, adding known " << expr << std::endl;
  for (const auto& subexpr : ExtractConstraints(expr)) {
    Comparison cmp(subexpr);
    if (cmp.IsValid()) {
      knowns_.push_back(cmp);
    }
  }
}

CompareResult ComparisonSet::TryCompare(const PrimExpr& lhs_input, const PrimExpr& rhs_input,
                                        Analyzer* analyzer) const {
  // std::cout << "Comparing between lhs = " << lhs_input << " and rhs = " << rhs_input <<
  // std::endl; Currently only supports integer checks
  if (!lhs_input.dtype().is_int() || !rhs_input.dtype().is_int()) {
    return CompareResult::kUnknown;
  }

  // Bail out early if possible.  This int check should have been
  // constant-folded earlier, so this check shouldn't occur.
  auto* x_int = lhs_input.as<IntImmNode>();
  auto* y_int = rhs_input.as<IntImmNode>();
  if (x_int && y_int) {
    if (x_int->value < y_int->value) {
      return CompareResult::kLT;
    } else if (x_int->value > y_int->value) {
      return CompareResult::kGT;
    } else {
      return CompareResult::kEQ;
    }
  }

  // Have the integer value on the right, if present.
  if (x_int) {
    // std::cout << "Reversing inequality and running again" << std::endl;
    return Reverse(TryCompare(rhs_input, lhs_input, analyzer));
  }

  auto print_vec_compare = [](const std::vector<Comparison>& vec) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < vec.size(); i++) {
      if (i) {
        ss << ", ";
      }
      ss << vec[i].debug_as_primexpr();
    }
    ss << "]";
    return ss.str();
  };

  // std::cout << "Attempting to compare between " << lhs_input << " and " << rhs_input
  //           << " using transitive knowns" << std::endl;
  // std::cout << "\t"
  //           << "Knowns = " << print_vec_compare(knowns_) << std::endl;

  PrimExpr lhs = lhs_input;
  PrimExpr rhs = rhs_input;
  int64_t offset;

  {
    Comparison output(lhs, rhs, CompareResult::kUnknown);
    lhs = output.lhs_;
    rhs = output.rhs_;
    offset = output.offset_;
  }

  ExprDeepEqual expr_equal;

  // Everything in `to_visit` has lhs as its lhs.
  std::unordered_set<PrimExpr, StructuralHash, StructuralEqual> to_visit;
  std::unordered_map<PrimExpr, std::vector<Comparison>, StructuralHash, StructuralEqual>
      compared_to_x;

  auto x_known_str = [&]() {
    std::stringstream ss;

    ss << "[";

    bool first_set = true;
    for (const auto& pair : compared_to_x) {
      if (first_set) {
        first_set = false;
      } else {
        ss << ", ";
      }

      bool first_expr = true;
      ss << "[";
      for (const auto& comparison : pair.second) {
        if (first_expr) {
          first_expr = false;
        } else {
          ss << ", ";
        }
        ss << comparison.debug_as_primexpr();
      }
      ss << "]";
    }
    ss << "]";
    return ss.str();
  };

  auto declare_known = [&](Comparison cmp) {
    auto& prev_knowns = compared_to_x[cmp.rhs_];

    for (auto& prev_known : prev_knowns) {
      if (prev_known.Implies(cmp)) {
        return;
      }
    }

    to_visit.insert(cmp.rhs_);

    for (auto& prev_known : prev_knowns) {
      Comparison intersect = cmp.IntersectAssumingExpressionsMatch(prev_known);
      if (intersect.IsValid()) {
        prev_known = cmp;
        return;
      }
    }

    prev_knowns.push_back(cmp);
  };

  for (const auto& known : knowns_) {
    Comparison normalized = known.NormalizedTo(lhs);
    if (normalized.IsValid()) {
      declare_known(normalized);
    }
  }

  // std::cout << "\t"
  //           << "After first pass, knowns = " << x_known_str() << std::endl;

  while (to_visit.size()) {
    PrimExpr middle_expr = *to_visit.begin();
    to_visit.erase(to_visit.begin());

    std::vector<Comparison>& prev_knowns_using_middle = compared_to_x.at(middle_expr);
    ICHECK(compared_to_x.count(middle_expr));

    std::vector<Comparison> new_knowns_using_lhs;

    // std::cout << "\t"
    //           << "Checking for transitive comparisons involving " << middle_expr << std::endl;

    auto attempt_transitive = [&](Comparison cmp) {
      if (!cmp.IsValid()) {
        return;
      }

      const PrimExpr& right_expr = cmp.rhs_;
      if (expr_equal(right_expr, lhs)) {
        return;
      }

      // std::cout << "\t\t"
      //           << "Found comparison " << cmp.debug_as_primexpr() << std::endl;

      for (const auto& prev : prev_knowns_using_middle) {
        CompareResult new_result = CompareResult::kUnknown;
        int64_t new_offset = prev.offset_ + cmp.offset_;

        if (prev.result_ == CompareResult::kEQ) {
          // x == y + c1 && y OP z + c2, x OP z + (c1 + c2)
          new_result = cmp.result_;
        } else if (cmp.result_ == CompareResult::kEQ) {
          // x OP y + c1 && y == z + c2, x OP z + (c1 + c2)
          new_result = prev.result_;
        } else if (prev.result_ == CompareResult::kLE && cmp.result_ == CompareResult::kLE) {
          // x <= y + c1 && y <= z + c2, x <= z + (c1 + c2)
          new_result = CompareResult::kLE;
        } else if (prev.result_ == CompareResult::kGE && cmp.result_ == CompareResult::kGE) {
          // x >= y + c1 && y >= z + c2, x >= z + (c1 + c2)
          new_result = CompareResult::kGE;
        }

        if (new_result != CompareResult::kUnknown) {
          Comparison new_known(lhs, right_expr, new_offset, new_result);
          // std::cout << "\t\t\t"
          //           << "Using " << prev.debug_as_primexpr() << " and " << cmp.debug_as_primexpr()
          //           << ", found " << new_known.debug_as_primexpr() << std::endl;
          new_knowns_using_lhs.push_back(new_known);
        }  // else {
        //   std::cout << "\t\t\t"
        //             << "Using " << prev.debug_as_primexpr() << " and " << cmp.debug_as_primexpr()
        //             << ", couldn't find any additional comparisons" << std::endl;
        // }
      }
    };

    for (const auto& known : knowns_) {
      Comparison cmp = known.NormalizedTo(middle_expr);
      attempt_transitive(cmp);
    }

    if (middle_expr->IsInstance<VarNode>()) {
      // std::cout << "\t"
      //           << "Checking for transitive comparisons involving declared bounds of "
      //           << middle_expr << std::endl;
      IntSet int_set = analyzer->int_set(middle_expr);
      // std::cout << "\t\t"
      //           << "Expr " << middle_expr << " has known bounds " << int_set << std::endl;
      if (int_set.HasLowerBound()) {
        PrimExpr expr = middle_expr >= int_set.min();
        Comparison cmp(expr);
        // std::cout << "\t\t"
        //           << "Attempting transitive comparison using expression " << expr
        //           << " (comparison:  " << cmp.debug_as_primexpr() << ")" << std::endl;
        attempt_transitive(cmp);
      }
      if (int_set.HasUpperBound()) {
        Comparison cmp(middle_expr <= int_set.max());
        // std::cout << "\t\t"
        //           << "Attempting transitive comparison using " << cmp.debug_as_primexpr()
        //           << std::endl;
        attempt_transitive(cmp);
      }
    }

    // std::cout << "\t"
    //           << "Found new knowns " << print_vec_compare(new_knowns_using_lhs) << std::endl;

    for (const auto& new_known : new_knowns_using_lhs) {
      declare_known(new_known);
    }

    // std::cout << "\t\t"
    //           << "After applying new knowns, all known comparisons are " << x_known_str()
    //           << std::endl;
  }

  // std::cout << "\t"
  //           << "After propagation, all known comparisons are " << x_known_str() << std::endl;

  auto it = compared_to_x.find(rhs);
  if (it == compared_to_x.end()) {
    // std::cout << "\t"
    //           << "No paths from " << lhs << " to " << rhs << " using known values" << std::endl;
    return CompareResult::kUnknown;
  }

  const std::vector<Comparison>& known_between_lhs_and_rhs = it->second;

  // std::cout << "\t"
  //           << "After propagation, found " << known_between_lhs_and_rhs.size()
  //           << " comparisons between desired expressions, "
  //           << print_vec_compare(known_between_lhs_and_rhs) << std::endl;

  CompareResult result = CompareResult::kUnknown;
  for (const auto& known : known_between_lhs_and_rhs) {
    switch (known.result_) {
      case CompareResult::kInconsistent:
        result = CompareResult::kInconsistent;
        break;

      case CompareResult::kEQ:
        if (offset == known.offset_) {
          result = result & CompareResult::kEQ;
        } else {
          result = result & CompareResult::kNE;
        }
        break;

      case CompareResult::kLE:
        if (known.offset_ < offset) {
          // std::cout << "\t\t"
          //           << "Known value of " << known.debug_as_primexpr()
          //           << " reduced possibilities from " << result;
          result = result & CompareResult::kLT;
          // std::cout << " to " << result << std::endl;
        } else if (known.offset_ <= offset) {
          // std::cout << "\t\t"
          //           << "Known value of " << known.debug_as_primexpr()
          //           << " reduced possibilities from " << result;
          result = result & CompareResult::kLE;
          // std::cout << " to " << result << std::endl;
        }  // else {
        //   std::cout << "\t\t"
        //             << "Known value of " << known.debug_as_primexpr()
        //             << " couldn't be applied to comparison of " << lhs << " and "
        //             << rhs + IntImm(rhs.dtype(), offset) << std::endl;
        // }
        break;

      case CompareResult::kGE:
        if (known.offset_ > offset) {
          // std::cout << "\t\t"
          //           << "Known value of " << known.debug_as_primexpr()
          //           << " reduced possibilities from " << result;
          result = result & CompareResult::kGT;
          // std::cout << " to " << result << std::endl;
        } else if (known.offset_ >= offset) {
          // std::cout << "\t\t"
          //           << "Known value of " << known.debug_as_primexpr()
          //           << " reduced possibilities from " << result;
          result = result & CompareResult::kGE;
          // std::cout << " to " << result << std::endl;
        }  //  else {
        //   std::cout << "\t\t"
        //             << "Known value of " << known.debug_as_primexpr()
        //             << " couldn't be applied to comparison of " << lhs << " and "
        //             << rhs + IntImm(rhs.dtype(), offset) << std::endl;
        // }
        break;

      case CompareResult::kNE:
        if (offset == known.offset_) {
          result = result & CompareResult::kNE;
        }
        break;

      case CompareResult::kUnknown:
        break;

      case CompareResult::kGT:
      case CompareResult::kLT:
        LOG(FATAL) << "Internal error, normalized comparisons should only include <= and <";
        return CompareResult::kInconsistent;

      default:
        LOG(FATAL) << "Invalid CompareResult: " << known.result_;
        return CompareResult::kInconsistent;
    }
  }

  // std::cout << "\t"
  //           << "Final result: " << result << std::endl;

  return result;
}

}  // namespace arith
}  // namespace tvm
