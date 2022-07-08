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

namespace tvm {
namespace arith {

void CollectConstraints(const PrimExpr& expr, Analyzer* analyzer, std::vector<PrimExpr>* collect,
                        bool keep_composite_constraints) {
  if (keep_composite_constraints) {
    collect->push_back(expr);
  }

  PVar<PrimExpr> x, y;
  if ((x && y).Match(expr)) {
    CollectConstraints(x.Eval(), analyzer, collect, keep_composite_constraints);
    CollectConstraints(y.Eval(), analyzer, collect, keep_composite_constraints);
  } else if ((!(x || y)).Match(expr)) {
    CollectConstraints(analyzer->rewrite_simplify(tir::Not(x.Eval())), analyzer, collect,
                       keep_composite_constraints);
    CollectConstraints(analyzer->rewrite_simplify(tir::Not(y.Eval())), analyzer, collect,
                       keep_composite_constraints);
  } else if (!keep_composite_constraints) {
    collect->push_back(expr);
  }
}

std::vector<PrimExpr> ExtractConstraints(const PrimExpr& expr, bool keep_composite_constraints) {
  std::vector<PrimExpr> out;
  Analyzer analyzer;
  CollectConstraints(expr, &analyzer, &out, keep_composite_constraints);
  return out;
}

void CollectComponents(const PrimExpr& expr, Analyzer* analyzer, std::vector<PrimExpr>* collect) {
  PVar<PrimExpr> x, y;
  if ((x || y).Match(expr)) {
    CollectComponents(x.Eval(), analyzer, collect);
    CollectComponents(y.Eval(), analyzer, collect);
  } else if ((!(x && y)).Match(expr)) {
    CollectComponents(analyzer->rewrite_simplify(tir::Not(x.Eval())), analyzer, collect);
    CollectComponents(analyzer->rewrite_simplify(tir::Not(y.Eval())), analyzer, collect);
  } else {
    collect->push_back(expr);
  }
}

std::vector<PrimExpr> ExtractComponents(const PrimExpr& expr) {
  std::vector<PrimExpr> out;
  Analyzer analyzer;
  CollectComponents(expr, &analyzer, &out);
  return out;
}

}  // namespace arith
}  // namespace tvm
