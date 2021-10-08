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
 * \file index_map.cc
 */

#include <tvm/tir/index_map.h>

#include <tvm/arith/analyzer.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace tir {

IndexMap::IndexMap(Array<Var> initial_index, Array<PrimExpr> final_index) {
  // TODO: Apply validity checking.

  auto n = make_object<IndexMapNode>();
  n->initial_index = std::move(initial_index);
  n->final_index = std::move(final_index);
  data_ = std::move(n);
}

Array<PrimExpr> IndexMapNode::map_indices(const Array<PrimExpr>& indices) const {
  ICHECK_EQ(indices.size(), initial_index.size());

  arith::Analyzer analyzer;

  for(size_t i=0; i<initial_index.size(); i++) {
    analyzer.Bind(initial_index[i], indices[i]);
  }

  Array<PrimExpr> output;
  for(const auto& output_dim : final_index) {
    output.push_back(analyzer.Simplify(output_dim));
  }

  return output;
}

Array<Range> IndexMapNode::map_ranges(const Array<Range>& ranges) const {
  ICHECK_EQ(ranges.size(), initial_index.size());

  Map<Var, Range> input_iters;
  for (size_t i = 0; i < initial_index.size(); i++) {
    input_iters.Set(initial_index[i], ranges[i]);
  }

  arith::Analyzer analyzer;
  auto iter_sums = DetectIterMap(final_index, input_iters, 1, true, &analyzer);

  Array<Range> output;
  for (const auto& iter_sum : iter_sums) {
    PrimExpr min = iter_sum->base;
    PrimExpr extent = 0;
    for (const auto& term : iter_sum->args) {
      extent += term->extent * term->scale;
    }
    output.push_back(Range::FromMinExtent(min, extent));
  }

  return output;
}

Array<PrimExpr> IndexMapNode::map_shape(const Array<PrimExpr>& shape) const {
  ICHECK_EQ(shape.size(), initial_index.size());

  Array<Range> ranges;
  for(auto& dim : shape) {
    ranges.push_back(Range(0, dim));
  }
  Array<Range> mapped = map_ranges(std::move(ranges));

  Array<PrimExpr> output;
  for(auto& range : mapped) {
    ICHECK(is_zero(range->min));
    output.push_back(range->extent);
  }

  return output;
}


TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IndexMapNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const IndexMapNode*>(node.get());
      p->stream << "index_map(" << op->initial_index << ", " << op->final_index << ")";
    });

TVM_REGISTER_NODE_TYPE(IndexMapNode);

}  // namespace tir
}  // namespace tvm
