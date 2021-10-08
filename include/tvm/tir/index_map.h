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
 * \file tvm/tir/index_map.h
 * \brief Defines a remapping of buffer indices
 *
 * For use with tvm::tir::Buffer.
 */
#ifndef TVM_TIR_PHYSICAL_LAYOUT_H_
#define TVM_TIR_PHYSICAL_LAYOUT_H_


#include <tvm/runtime/object.h>
#include <tvm/runtime/container/array.h>
#include <tvm/tir/var.h>
#include <tvm/ir/expr.h>

namespace tvm {
namespace tir {

/*!
 * \brief Defines the mapping from logical layout of a tensor to
 * physical layout of a buffer.
 */
class IndexMapNode : public Object {
 public:
  /*! \brief Variables representing the indices prior to remapping.
   *
   * If initial_index is empty, then final_index should also be
   * empty, and no mapping is applied.
   */
  Array<Var> initial_index;

  /*!
   * \brief Expressions defining the indices after remapping.
   *
   * These expressions should only be in terms of the initial_index,
   * and must be expressible as an IterSumExpr.  The mapping from
   * initial_index to final_index must be injective.
   *
   * If final_index is empty, then initial_index should also be
   * empty, and the map is an identity function.
   */
  Array<PrimExpr> final_index;

  /*!
   * \brief Default constructor
   *
   * Defines the mapping as an identity function, with initial_index
   * equal to the final index.
   */
  IndexMapNode() {}

  /*!
   * \brief Map indices to the output space
   *
   * \param indices The indices in the input space.  Should contain
   * one value for each variable in `initial_index`.
   *
   * \returns The indices in the output space.  Contains one value for
   * each expression in `final_index`.
   */
  Array<PrimExpr> map_indices(const Array<PrimExpr>& indices) const;

  /*! \brief Map a memory range to the output space
   *
   * If contiguous memory locations in the input space are not
   * necessarily contiguous in the output space (e.g. `lambda i:
   * [8*(i%8) + (i//8)]`), then this will return the smallest range
   * such that all valid indices are contained within the given range.
   *
   * \param ranges The ranges in the input space.  Should contain one
   * value for each variable in `initial_index`.
   *
   * \returns The ranges in the output space.  Contains one value for
   * each expression in `final_index`.
   */
  Array<Range> map_ranges(const Array<Range>& ranges) const;

  /*! \brief Map a buffer shape to the output space
   *
   * \param shape The buffer shape in the input space.  Should contain one
   * value for each variable in `initial_index`.
   *
   * \returns The buffer shape in the output space.  Contains one
   * value for each expression in `final_index`.
   */
  Array<PrimExpr> map_shape(const Array<PrimExpr>& shape) const;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("initial_index", &initial_index);
    v->Visit("final_index", &final_index);
  }

  TVM_DECLARE_FINAL_OBJECT_INFO(IndexMapNode, Object);
};

class IndexMap : public ObjectRef {
 public:
  IndexMap(Array<Var> initial_index, Array<PrimExpr> final_index);

  TVM_DEFINE_OBJECT_REF_METHODS(IndexMap, ObjectRef, IndexMapNode);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_PHYSICAL_LAYOUT_H_
