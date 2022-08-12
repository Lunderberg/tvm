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

#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>

#include "pattern_match.h"

namespace tvm {
namespace arith {

using namespace tir;

namespace {

Comparison::CompareResult Reverse(Comparison::CompareResult res) {
  switch (res) {
    case Comparison::kInconsistent:
      return Comparison::kInconsistent;
    case Comparison::kEQ:
      return Comparison::kEQ;
    case Comparison::kLT:
      return Comparison::kGT;
    case Comparison::kLE:
      return Comparison::kGE;
    case Comparison::kGT:
      return Comparison::kLT;
    case Comparison::kGE:
      return Comparison::kLE;
    case Comparison::kNE:
      return Comparison::kNE;
    case Comparison::kUnknown:
      return Comparison::kUnknown;
    default:
      LOG(FATAL) << "Invalid CompareResult: " << res;
      return Comparison::kInconsistent;
  }
}

Comparison::CompareResult Negate(Comparison::CompareResult res) {
  switch (res) {
    case Comparison::kInconsistent:
      return Comparison::kInconsistent;
    case Comparison::kEQ:
      return Comparison::kNE;
    case Comparison::kLT:
      return Comparison::kGE;
    case Comparison::kLE:
      return Comparison::kGT;
    case Comparison::kGT:
      return Comparison::kLE;
    case Comparison::kGE:
      return Comparison::kLT;
    case Comparison::kNE:
      return Comparison::kEQ;
    case Comparison::kUnknown:
      return Comparison::kUnknown;
    default:
      LOG(FATAL) << "Invalid CompareResult: " << res;
      return Comparison::kInconsistent;
  }
}

// Result of comparing (x RES_A y) and (y RES_B z) into (x OUT z)
Comparison::CompareResult Transitive(Comparison::CompareResult a, Comparison::CompareResult b) {
  if (a == Comparison::kInconsistent || b == Comparison::kInconsistent) {
    return Comparison::kInconsistent;
  }

  if (a == Comparison::kEQ) {
    return b;
  }
  if (b == Comparison::kEQ) {
    return a;
  }

  if (a == Comparison::kLE && b == Comparison::kLE) {
    return Comparison::kLE;
  } else if (a == Comparison::kLE && b == Comparison::kLT) {
    return Comparison::kLT;
  } else if (a == Comparison::kLT && b == Comparison::kLE) {
    return Comparison::kLT;
  } else if (a == Comparison::kLT && b == Comparison::kLT) {
    return Comparison::kLT;
  }

  if (a == Comparison::kGE && b == Comparison::kGE) {
    return Comparison::kGE;
  } else if (a == Comparison::kGE && b == Comparison::kGT) {
    return Comparison::kGT;
  } else if (a == Comparison::kGT && b == Comparison::kGE) {
    return Comparison::kGT;
  } else if (a == Comparison::kGT && b == Comparison::kGT) {
    return Comparison::kGT;
  }

  return Comparison::kUnknown;
}
}  // namespace

Comparison::Comparison(const PrimExpr& expr) : orig_expr_(expr) {
  std::cout << "Parsing expression " << expr << std::endl;
  PVar<PrimExpr> x, y;
  if ((x <= y).Match(expr) || (y >= x).Match(expr)) {
    lhs_ = x.Eval();
    rhs_ = y.Eval();
    result_ = kLE;
  } else if ((x < y).Match(expr) || (y < x).Match(expr)) {
    lhs_ = x.Eval();
    rhs_ = y.Eval();
    result_ = kLT;
  } else if ((x == y).Match(expr)) {
    lhs_ = x.Eval();
    rhs_ = y.Eval();
    result_ = kEQ;
  } else if ((x != y).Match(expr)) {
    lhs_ = x.Eval();
    rhs_ = y.Eval();
    result_ = kNE;
  }

  std::cout << "\t"
            << "Parsed as lhs = " << lhs_ << ", rhs_ = " << rhs_ << ", comparison = " << result_
            << std::endl;

  if (lhs_.as<IntImmNode>() && rhs_.as<IntImmNode>()) {
    lhs_ = PrimExpr();
    rhs_ = PrimExpr();
    return;
  }

  Normalize();

  std::cout << "\t"
            << "Normalized to lhs = " << lhs_ << ", rhs_ = " << rhs_ << ", offset = " << offset_
            << ", comparison = " << result_ << std::endl;
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
  // These <= and > should be removed during normalization.
  if (result_ == kLE || result_ == kGT) {
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

  if (result_ == kLE) {
    result_ = kLT;
    offset_ += 1;
  }
  if (result_ == kGT) {
    result_ = kGE;
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
    case Comparison::kInconsistent:
      return NullOpt;
    case Comparison::kEQ:
      return lhs_ == rhs_ + offset;
    case Comparison::kLT:
      return lhs_ < rhs_ + offset;
    case Comparison::kLE:
      return lhs_ <= rhs_ + offset;
    case Comparison::kGT:
      return lhs_ > rhs_ + offset;
    case Comparison::kGE:
      return lhs_ >= rhs_ + offset;
    case Comparison::kNE:
      return lhs_ != rhs_ + offset;
    case Comparison::kUnknown:
      return NullOpt;
    default:
      LOG(FATAL) << "Invalid CompareResult: " << result_;
      return NullOpt;
  }
}

Comparison Comparison::IntersectAssumingExpressionsMatch(const Comparison& other) const {
  Comparison output = *this;

  if (result_ == kEQ && other.result_ == kEQ) {
    output.result_ = (other.offset_ == offset_) ? kEQ : kInconsistent;
    return output;
  }
  if (result_ == kLT && other.result_ == kLT) {
    output.offset_ = std::min(other.offset_, offset_);
    return output;
  }
  if (result_ == kGE && other.result_ == kGE) {
    output.offset_ = std::max(other.offset_, offset_);
    return output;
  }

  return Comparison();
}

Comparison::CompareResult Comparison::TryCompare(const std::vector<Comparison>& knowns,
                                                 const PrimExpr& lhs, const PrimExpr& rhs) {
  std::cout << "Comparing between lhs = " << lhs << " and rhs = " << rhs << std::endl;
  // Currently only supports integer checks
  if (!lhs.dtype().is_int() || !rhs.dtype().is_int()) {
    return Comparison::kUnknown;
  }

  // Bail out early if possible.  This int check should have been
  // constant-folded earlier, so this check shouldn't occur.
  auto* x_int = lhs.as<IntImmNode>();
  auto* y_int = rhs.as<IntImmNode>();
  if (x_int && y_int) {
    if (x_int->value < y_int->value) {
      return Comparison::kLT;
    } else if (x_int->value > y_int->value) {
      return Comparison::kGT;
    } else {
      return Comparison::kEQ;
    }
  }

  // Have the integer value on the right, if present.
  if (x_int) {
    std::cout << "Reversing inequality and running again" << std::endl;
    return Reverse(TryCompare(knowns, rhs, lhs));
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

  std::cout << "Attempting to compare between " << lhs << " and " << rhs
            << " using transitive knowns" << std::endl;
  std::cout << "\t"
            << "Knowns = " << print_vec_compare(knowns) << std::endl;

  Comparison output(lhs, rhs, kUnknown);

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

  for (const auto& known : knowns) {
    Comparison normalized = known.NormalizedTo(output.lhs_);
    if (normalized.IsValid()) {
      compared_to_x[normalized.rhs_].push_back(normalized);
      to_visit.insert(normalized.rhs_);
    }
  }

  std::cout << "\t"
            << "After first pass, knowns = " << x_known_str() << std::endl;

  while (to_visit.size()) {
    PrimExpr middle_expr = *to_visit.begin();
    to_visit.erase(to_visit.begin());

    std::vector<Comparison>& prev_knowns_using_middle = compared_to_x.at(middle_expr);
    ICHECK(compared_to_x.count(middle_expr));

    std::vector<Comparison> new_knowns_using_lhs;

    std::cout << "\t"
              << "Checking for transitive comparisons involving " << middle_expr << std::endl;

    for (const auto& known : knowns) {
      Comparison cmp = known.NormalizedTo(middle_expr);
      if (!cmp.IsValid()) {
        continue;
      }

      const PrimExpr& right_expr = cmp.rhs_;
      if (expr_equal(right_expr, output.lhs_)) {
        continue;
      }

      std::cout << "\t\t"
                << "Found comparison " << cmp.debug_as_primexpr() << std::endl;

      for (const auto& prev : prev_knowns_using_middle) {
        CompareResult new_result = kUnknown;
        int64_t new_offset;

        if (prev.result_ == kEQ) {
          new_result = cmp.result_;
          new_offset = prev.offset_ + cmp.offset_;
        } else if (cmp.result_ == kEQ) {
          new_result = prev.result_;
          new_offset = prev.offset_ + cmp.offset_;
        } else if (prev.result_ == kLT && cmp.result_ == kLT) {
          new_result = kLT;
          // TODO: Normalize to kLE instead of kLT to avoid this -1
          new_offset = prev.offset_ + cmp.offset_ - 1;
        } else if (prev.result_ == kGE && cmp.result_ == kGE) {
          new_result = kGE;
          new_offset = prev.offset_ + cmp.offset_;
        }

        if (new_result != kUnknown) {
          Comparison new_known(output.lhs_, right_expr, new_offset, new_result);
          std::cout << "\t\t\t"
                    << "Using " << prev.debug_as_primexpr() << " and " << cmp.debug_as_primexpr()
                    << ", found " << new_known.debug_as_primexpr() << std::endl;
          new_knowns_using_lhs.push_back(new_known);
        } else {
          std::cout << "\t\t\t"
                    << "Using " << prev.debug_as_primexpr() << " and " << cmp.debug_as_primexpr()
                    << ", couldn't find any additional comparisons" << std::endl;
        }
      }
    }
    std::cout << "\t"
              << "Found new knowns " << print_vec_compare(new_knowns_using_lhs) << std::endl;

    for (const auto& new_known : new_knowns_using_lhs) {
      auto& prev_knowns = compared_to_x[new_known.rhs_];
      bool handled = false;
      for (auto& prev_known : prev_knowns) {
        Comparison intersection = prev_known.IntersectAssumingExpressionsMatch(new_known);
        if (intersection.IsValid()) {
          prev_known = intersection;
          handled = true;
          break;
        }
      }

      if (!handled) {
        prev_knowns.push_back(new_known);
      }
    }

    std::cout << "\t\t"
              << "After applying new knowns, all known comparisons are " << x_known_str()
              << std::endl;
  }

  std::cout << "\t"
            << "After propagation, all known comparisons are " << x_known_str() << std::endl;

  // auto it = compared_to_x.find(rhs);
  // if (it != compared_to_x.end()) {
  //   output = it->second;
  // }

  auto it = compared_to_x.find(output.rhs_);
  if (it == compared_to_x.end()) {
    std::cout << "\t"
              << "No paths from " << output.lhs_ << " to " << output.rhs_ << " using known values"
              << std::endl;
    return kUnknown;
  }

  const std::vector<Comparison>& known_between_lhs_and_rhs = it->second;

  std::cout << "\t"
            << "After propagation, found " << known_between_lhs_and_rhs.size()
            << " comparisons between desired expressions, "
            << print_vec_compare(known_between_lhs_and_rhs) << std::endl;

  CompareResult result = kUnknown;
  for (const auto& known : known_between_lhs_and_rhs) {
    switch (known.result_) {
      case Comparison::kInconsistent:
        result = kInconsistent;
        break;

      case Comparison::kEQ:
        // if (output.offset_ == known.offset_) {
        //   result = CompareResult(result & kEQ);
        // } else {
        //   result = CompareResult(result & kNE);
        // }
        break;

      case Comparison::kLT:
        // if (known.offset_ > output.offset_) {
        //   std::cout << "Known value of " << known.debug_as_primexpr()
        //             << " reduced possibilities from " << result;
        //   result = CompareResult(result & kLT);
        //   std::cout << " to " << result << std::endl;
        //   } else if (known.offset_ <= output.offset_) {
        //     std::cout << "Known value of " << known.debug_as_primexpr()
        //               << " reduced possibilities from " << result;
        //     result = CompareResult(result & kLE);
        //     std::cout << " to " << result << std::endl;
        //     ;
        // } else {
        //   std::cout << "Known value of " << known.debug_as_primexpr()
        //             << " couldn't be applied to comparison of " << output.lhs_ << " and "
        //             << output.rhs_ + IntImm(output.rhs_.dtype(), output.offset_) << std::endl;
        // }
        break;

      case Comparison::kGE:
        if (known.offset_ > output.offset_) {
          std::cout << "Known value of " << known.debug_as_primexpr()
                    << " reduced possibilities from " << result;
          result = CompareResult(result & kGT);
          std::cout << " to " << result << std::endl;
          // } else if (known.offset_ >= output.offset_) {
          //   std::cout << "Known value of " << known.debug_as_primexpr()
          //             << " reduced possibilities from " << result;
          //   result = CompareResult(result & kGE);
          //   std::cout << " to " << result << std::endl;
        } else {
          std::cout << "Known value of " << known.debug_as_primexpr()
                    << " couldn't be applied to comparison of " << output.lhs_ << " and "
                    << output.rhs_ + IntImm(output.rhs_.dtype(), output.offset_) << std::endl;
        }
        break;

      case Comparison::kNE:
        // if (output.offset_ == known.offset_) {
        //   result = CompareResult(result & kNE);
        // }
        break;

      case Comparison::kUnknown:
        break;

      case Comparison::kGT:
      case Comparison::kLE:
        LOG(FATAL) << "Internal error, normalized comparisons should only include <= and <";
        return kInconsistent;

      default:
        LOG(FATAL) << "Invalid CompareResult: " << known.result_;
        return kInconsistent;
    }
  }

  std::cout << "\t"
            << "Final result: " << result << std::endl;

  return result;
}

}  // namespace arith
}  // namespace tvm
