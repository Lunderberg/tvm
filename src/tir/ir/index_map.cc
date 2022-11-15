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

#include "tvm/tir/index_map.h"

#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_set.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <sstream>
#include <unordered_set>

namespace tvm {
namespace tir {

IndexMap::IndexMap(Array<Var> initial_indices, Array<PrimExpr> final_indices,
                   Optional<IndexMap> inverse_index_map) {
  auto n = make_object<IndexMapNode>();
  n->initial_indices = std::move(initial_indices);
  n->final_indices = std::move(final_indices);
  n->inverse_index_map = std::move(inverse_index_map);
  data_ = std::move(n);
}

IndexMap IndexMap::FromFunc(int ndim, runtime::TypedPackedFunc<Array<PrimExpr>(Array<Var>)> func,
                            Optional<IndexMap> inverse_index_map) {
  Array<Var> initial_indices;
  initial_indices.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    initial_indices.push_back(Var("i" + std::to_string(i), DataType::Int(32)));
  }
  return IndexMap(initial_indices, func(initial_indices), std::move(inverse_index_map));
}

std::vector<std::string> GenerateIndexMapNames(const IndexMap& self,
                                               const Array<arith::IterSumExpr>& parsed_indices) {
  const auto& input_vars = self->initial_indices;
  const auto& output_exprs = self->final_indices;

  std::vector<std::string> names(output_exprs.size(), "");
  auto is_done = [&names](size_t i) -> bool { return names[i].size(); };

  std::unordered_set<const VarNode*> input_var_lookup;
  for (const auto& input_var : input_vars) {
    input_var_lookup.insert(input_var.get());
  }
  for (size_t i = 0; i < output_exprs.size(); i++) {
    if (auto* as_var = output_exprs[i].as<VarNode>(); as_var && input_var_lookup.count(as_var)) {
      names[i] = as_var->name_hint;
    }
  }

  auto extract = [&](const arith::IterSumExpr& sum_expr) -> std::optional<std::pair<Var, size_t>> {
    if (sum_expr->args.size() != 1) {
      return std::nullopt;
    }

    const auto& split = sum_expr->args[0];
    PrimExpr source = split->source->source;

    size_t lower_factor = 1;
    if (auto* as_int = split->lower_factor.as<IntImmNode>()) {
      lower_factor = as_int->value;
    } else {
      return std::nullopt;
    }

    size_t iter = 0;
    while (auto* inner_sum_expr = source.as<arith::IterSumExprNode>()) {
      iter++;
      if (inner_sum_expr->args.size() != 1) {
        return std::nullopt;
      }
      const auto& inner_split = inner_sum_expr->args[0];
      source = inner_split->source->source;
    }

    auto* source_var = source.as<VarNode>();
    if (!source_var) {
      return std::nullopt;
    }

    if (input_var_lookup.count(source_var) == 0) {
      return std::nullopt;
    }

    return std::pair{GetRef<Var>(source_var), lower_factor};
  };
  for (size_t i = 0; i < output_exprs.size(); i++) {
    if (is_done(i)) continue;

    auto opt_var = extract(parsed_indices[i]);
    bool is_smallest_split = opt_var.has_value() && opt_var->second == 1;
    if (!is_smallest_split) continue;

    Var split_var = opt_var->first;
    std::vector<size_t> split_ordering = {i};

    while (true) {
      size_t smallest_split = -1;
      std::optional<size_t> next_split_at = std::nullopt;
      for (size_t j = 0; j < output_exprs.size(); j++) {
        if (is_done(j) || std::any_of(split_ordering.begin(), split_ordering.end(),
                                      [&](size_t prev) { return prev == j; }))
          continue;

        auto opt_var = extract(parsed_indices[j]);
        if (opt_var && opt_var->first.same_as(split_var) && opt_var->second < smallest_split) {
          smallest_split = opt_var->second;
          next_split_at = j;
        }
      }
      if (next_split_at.has_value()) {
        split_ordering.push_back(*next_split_at);
      } else {
        break;
      }
    }

    if (split_ordering.size() == 2) {
      // Special casing for the common case of outer/inner splits
      names[split_ordering[0]] = split_var->name_hint + "o";
      names[split_ordering[1]] = split_var->name_hint + "i";
    } else {
      for (size_t split_i = 0; split_i < split_ordering.size(); split_i++) {
        std::stringstream ss;
        ss << split_var->name_hint << "_" << split_i;
        names[split_ordering[split_i]] = ss.str();
      }
    }
  }

  for (size_t i = 0; i < names.size(); i++) {
    if (!is_done(i)) {
      std::stringstream ss;
      ss << "axis" << i;
      names[i] = ss.str();
    }
  }

  return names;
}

std::pair<IndexMap, PrimExpr> IndexMapInverseImpl(const IndexMap& self,
                                                  const Array<Range>& initial_ranges,
                                                  arith::IterMapLevel check_level) {
  if (self->inverse_index_map.defined()) {
    // return the pre-defined inverse index map if exists.  In this
    // case, the user-defined inverse is assumed to be correct and
    // bijective.
    PrimExpr padding_predicate = Bool(false);
    return {Downcast<IndexMap>(self->inverse_index_map.value()), padding_predicate};
  }

  // Dummy ranges for the extent of each input.
  Map<Var, Range> input_iters;
  ICHECK_EQ(self->initial_indices.size(), initial_ranges.size());
  for (size_t i = 0; i < initial_ranges.size(); i++) {
    input_iters.Set(self->initial_indices[i], initial_ranges[i]);
  }

  // Unpack the output indices into linear combinations of the initial
  // indices.
  arith::Analyzer analyzer;
  auto padded_iter_map = DetectIterMap(self->final_indices, input_iters, /* predicate = */ 1,
                                       /*check_level=*/check_level, &analyzer,
                                       /*simplify_trivial_iterators=*/false);
  CHECK(padded_iter_map->errors.empty()) << "Could not parse mapping as sum of iterators.  "
                                         << "Error: " << padded_iter_map->errors[0];

  // Dummy variables to represent the inverse's inputs.
  auto var_names = GenerateIndexMapNames(self, padded_iter_map->indices);
  Array<Var> output_vars;
  ICHECK_EQ(self->final_indices.size(), padded_iter_map->indices.size());
  for (size_t i = 0; i < self->final_indices.size(); i++) {
    const PrimExpr& index = self->final_indices[i];
    Var var_index(var_names[i], index.dtype());
    output_vars.push_back(var_index);
  }

  // Determine expressions for the input variables, in terms of the
  // output variables.
  Map<Var, PrimExpr> inverse_exprs_map = InverseAffineIterMap(
      padded_iter_map->indices, Array<PrimExpr>(output_vars.begin(), output_vars.end()));

  // Unpack the map to an array, maintaining the same parameter order.
  Array<PrimExpr> inverse_exprs;
  for (int i = 0, n = self->initial_indices.size(); i < n; ++i) {
    Var index = self->initial_indices[i];
    PrimExpr expr;
    if (is_one(initial_ranges[i]->extent) && !inverse_exprs_map.count(index)) {
      expr = initial_ranges[i]->min;
    } else {
      expr = inverse_exprs_map.at(index);
    }
    inverse_exprs.push_back(analyzer.Simplify(expr));
  }

  PrimExpr padding_predicate = padded_iter_map->padding_predicate;
  padding_predicate = arith::NormalizeIterMapToExpr(padding_predicate);
  padding_predicate = Substitute(padding_predicate, inverse_exprs_map);

  {
    auto output_ranges = self->MapRanges(initial_ranges);
    ICHECK_EQ(output_ranges.size(), output_vars.size());

    arith::Analyzer analyzer;
    for (size_t i = 0; i < output_vars.size(); ++i) {
      analyzer.Bind(output_vars[i], output_ranges[i]);
    }

    // Additional simplification steps required to unwrap nested floordiv/floormod
    padding_predicate = analyzer.Simplify(padding_predicate, 10);
  }

  return {IndexMap(output_vars, inverse_exprs), padding_predicate};
}

std::pair<IndexMap, PrimExpr> IndexMap::NonSurjectiveInverse(Array<Range> initial_ranges) const {
  return IndexMapInverseImpl(*this, initial_ranges, arith::IterMapLevel::NoCheck);
}

IndexMap IndexMap::Inverse(Array<Range> initial_ranges) const {
  auto [inverse, padding_predicate] =
      IndexMapInverseImpl(*this, initial_ranges, arith::IterMapLevel::Bijective);
  arith::Analyzer analyzer;
  CHECK(analyzer.CanProve(!padding_predicate))
      << "Bijective inverse should not contain padding, but inverse of " << *this << " over range "
      << initial_ranges << " resulted in a padding predicate of " << padding_predicate;
  return inverse;
}

Array<PrimExpr> IndexMapNode::MapIndices(const Array<PrimExpr>& indices,
                                         arith::Analyzer* analyzer) const {
  ICHECK_EQ(indices.size(), initial_indices.size());

  Map<Var, PrimExpr> vmap;

  for (size_t i = 0; i < initial_indices.size(); i++) {
    vmap.Set(initial_indices[i], indices[i]);
  }

  arith::Analyzer local_analyzer;
  if (!analyzer) {
    analyzer = &local_analyzer;
  }

  Array<PrimExpr> output = final_indices.Map([&](PrimExpr index) {
    PrimExpr result = SubstituteWithDataTypeLegalization(
        std::move(index), [&](const Var& var) { return vmap.Get(var); });
    return analyzer->Simplify(result);
  });
  return output;
}

Array<Range> IndexMapNode::MapRanges(const Array<Range>& ranges, arith::Analyzer* analyzer) const {
  ICHECK_EQ(ranges.size(), initial_indices.size());

  Map<Var, Range> input_iters;
  for (size_t i = 0; i < initial_indices.size(); i++) {
    input_iters.Set(initial_indices[i], ranges[i]);
  }

  arith::Analyzer local_analyzer;
  if (!analyzer) {
    analyzer = &local_analyzer;
  }

  auto iter_map = DetectIterMap(final_indices, input_iters, /* predicate = */ 1,
                                /*check_level=*/arith::IterMapLevel::NoCheck, analyzer,
                                /*simplify_trivial_iterators=*/false);
  Array<Range> output;
  if (iter_map->indices.size()) {
    // Preferred route, requires the map to be expressible as an
    // affine sum.  Since the terms are orthogonal, the extent of the
    // sum is the extent of the largest term.
    for (const auto& index : iter_map->indices) {
      Optional<PrimExpr> extent = NullOpt;
      for (const auto& term : index->args) {
        PrimExpr term_extent = term->extent * term->scale;
        if (extent.defined()) {
          extent = tvm::max(extent.value(), term_extent);
        } else {
          extent = term_extent;
        }
      }
      output.push_back(Range::FromMinExtent(index->base, extent.value_or(1)));
    }

  } else {
    // Fall-back method, more general but can ignore intended padding.
    // For example, [N] mapped through i=>[i//4,i%4] should have shape
    // [ceildiv(N,4), 4].  However, for N<4, this method instead
    // results in a shape [1, N].
    std::unordered_map<const VarNode*, arith::IntSet> dom_map;
    for (size_t i = 0; i < initial_indices.size(); i++) {
      dom_map[initial_indices[i].get()] = arith::IntSet::FromRange(ranges[i]);
    }

    for (const auto& final_index : final_indices) {
      auto int_set = arith::EvalSet(final_index, dom_map);
      output.push_back(Range::FromMinExtent(analyzer->Simplify(int_set.min()),
                                            analyzer->Simplify(int_set.max() - int_set.min() + 1)));
    }
  }
  auto output_dtype = [&]() {
    int max_bits = 0;
    for (const auto& range : ranges) {
      max_bits = std::max(max_bits, range->extent.dtype().bits());
    }
    return DataType::Int(max_bits);
  }();
  output.MutateByApply([&](const Range& range) {
    if (range->min.dtype() != output_dtype || range->extent.dtype() != output_dtype) {
      return Range::FromMinExtent(cast(output_dtype, range->min),
                                  cast(output_dtype, range->extent));
    } else {
      return range;
    }
  });
  return output;
}

Array<PrimExpr> IndexMapNode::MapShape(const Array<PrimExpr>& shape,
                                       arith::Analyzer* analyzer) const {
  ICHECK_EQ(shape.size(), initial_indices.size());

  Array<Range> ranges;
  for (auto& dim : shape) {
    ranges.push_back(Range(make_zero(dim.dtype()), dim));
  }
  Array<Range> mapped = MapRanges(std::move(ranges), analyzer);

  Array<PrimExpr> output;
  for (auto& range : mapped) {
    ICHECK(is_zero(range->min));
    output.push_back(range->extent);
  }

  return output;
}

runtime::NDArray IndexMapNode::MapNDArray(runtime::NDArray arr_src) const {
  auto shape = arr_src.Shape();
  ICHECK(shape.size() == initial_indices.size())
      << "The rank of the input array should be " << initial_indices.size() << " but got "
      << shape.size();
  size_t size_1d = 1;
  Array<PrimExpr> orig_shape;
  for (size_t i = 0; i < shape.size(); ++i) {
    size_1d *= shape[i];
    orig_shape.push_back(PrimExpr(static_cast<int>((shape[i]))));
  }
  auto dst_shape = MapShape(orig_shape);

  std::vector<int64_t> dst_shape_int;
  for (size_t i = 0; i < dst_shape.size(); ++i) {
    dst_shape_int.push_back(dst_shape[i].as<IntImmNode>()->value);
  }

  auto elem_bytes = (arr_src->dtype.bits / 8) * arr_src->dtype.lanes;
  std::vector<uint8_t> bytes_src(size_1d * elem_bytes);
  arr_src.CopyToBytes(bytes_src.data(), bytes_src.size());

  std::vector<uint8_t> bytes_dst(bytes_src.size());

  for (size_t i = 0; i < size_1d; ++i) {
    // Convert a linear coordinate to an N-d coordinate tuple
    // z * height * width + y * width + x -> (z, y, x)
    Array<PrimExpr> src_indices;
    auto div_factor = size_1d;
    auto src_linear_index = i;
    for (auto s : shape) {
      div_factor /= s;
      src_indices.push_back(PrimExpr(static_cast<int>((src_linear_index / div_factor))));
      src_linear_index %= div_factor;
    }
    auto dst_indices = MapIndices(src_indices);

    // Convert an N-d coordinate to a linear coordinate
    // (z, y, x) -> z * height * width + y * width + x
    size_t dst_linear_index = 0;
    auto mul_factor = size_1d;
    for (size_t j = 0; j < dst_indices.size(); ++j) {
      mul_factor /= dst_shape_int[j];
      dst_linear_index += dst_indices[j].as<IntImmNode>()->value * mul_factor;
    }
    std::copy(bytes_src.begin() + i * elem_bytes, bytes_src.begin() + (i + 1) * elem_bytes,
              bytes_dst.begin() + dst_linear_index * elem_bytes);
  }

  auto arr_dst = runtime::NDArray::Empty(dst_shape_int, arr_src->dtype, arr_src->device);
  arr_dst.CopyFromBytes(bytes_dst.data(), bytes_dst.size());
  return arr_dst;
}

/*!
 * \brief Auxilarry function to comvert an index map to lambda expression in Python.
 * \param initial_indices The initial indices in the index map.
 * \param final_indices The final indices in the index map.
 * \return The lambda expression string.
 */
std::string IndexMap2PythonLambdaExpr(const Array<Var>& initial_indices,
                                      const Array<PrimExpr>& final_indices) {
  std::unordered_set<std::string> used_names;
  Map<Var, PrimExpr> var_remap;
  for (const Var& initial_index : initial_indices) {
    if (used_names.count(initial_index->name_hint)) {
      std::string new_name = initial_index->name_hint + std::to_string(used_names.size());
      used_names.insert(new_name);
      var_remap.Set(initial_index, Var(new_name));
    } else {
      used_names.insert(initial_index->name_hint);
    }
  }
  std::ostringstream oss;
  oss << "lambda ";
  for (size_t i = 0; i < initial_indices.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    auto it = var_remap.find(initial_indices[i]);
    if (it != var_remap.end()) {
      oss << (*it).second;
    } else {
      oss << initial_indices[i];
    }
  }
  oss << ": (";
  for (size_t i = 0; i < final_indices.size(); ++i) {
    if (i != 0) {
      oss << " ";
    }
    oss << Substitute(final_indices[i], var_remap);
    oss << ",";
  }
  oss << ")";
  return oss.str();
}

String IndexMapNode::ToPythonString() const {
  std::string lambda_expr = IndexMap2PythonLambdaExpr(initial_indices, final_indices);
  if (!inverse_index_map.defined()) {
    return String(lambda_expr);
  }
  // Also convert the inverse index map.
  IndexMap inverse = Downcast<IndexMap>(inverse_index_map.value());
  std::string inverse_lambda_expr =
      IndexMap2PythonLambdaExpr(inverse->initial_indices, inverse->final_indices);
  std::ostringstream oss;
  oss << "tvm.tir.IndexMap.from_func(" << lambda_expr
      << ", inverse_index_map=" << inverse_lambda_expr << ")";
  return String(oss.str());
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IndexMapNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const IndexMapNode*>(node.get());
      p->stream << "index_map(" << op->ToPythonString() << ")";
    });

TVM_REGISTER_NODE_TYPE(IndexMapNode);

TVM_REGISTER_GLOBAL("tir.IndexMap")
    .set_body_typed([](Array<Var> initial_indices, Array<PrimExpr> final_indices,
                       Optional<IndexMap> inverse_index_map) {
      return IndexMap(initial_indices, final_indices, inverse_index_map);
    });

TVM_REGISTER_GLOBAL("tir.IndexMapMapIndices")
    .set_body_typed([](IndexMap map, Array<PrimExpr> indices) { return map->MapIndices(indices); });

TVM_REGISTER_GLOBAL("tir.IndexMapMapShape").set_body_typed([](IndexMap map, Array<PrimExpr> shape) {
  return map->MapShape(shape);
});
TVM_REGISTER_GLOBAL("tir.IndexMapInverse").set_body_method(&IndexMap::Inverse);

TVM_REGISTER_GLOBAL("tir.IndexMapMapNDArray")
    .set_body_typed([](IndexMap map, runtime::NDArray arr) { return map->MapNDArray(arr); });

TVM_REGISTER_GLOBAL("tir.IndexMapNonSurjectiveInverse")
    .set_body_typed([](IndexMap forward, Array<Range> initial_ranges) {
      auto result = forward.NonSurjectiveInverse(initial_ranges);
      return Array<ObjectRef>{result.first, result.second};
    });

}  // namespace tir
}  // namespace tvm
