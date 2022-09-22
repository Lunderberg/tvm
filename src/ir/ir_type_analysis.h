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
 * \file src/ir/ir_type_analysis.h
 * \brief Ir_Type_Analysis utilities and passes for TIR.
 */
#ifndef TVM_IR_IR_TYPE_ANALYSIS_H_
#define TVM_IR_IR_TYPE_ANALYSIS_H_

#include <tvm/ir/module.h>
#include <tvm/runtime/object.h>

namespace tvm {
namespace ir {

class AnalysisResultsNode : public Object {
 public:
  // Contains PrimFuncs generated from TE schedules
  bool is_te_derived{false};

  // Contains TE-specific nodes (e.g. ProducerLoadNode, BufferRealize)
  bool contains_te_specific_nodes{false};

  // Contains tir::Block, used by meta-schedule primitives.
  bool contains_tir_blocks{false};

  // Contains tir::Block/tir::BlockRealize with iter_vars/iter_values
  bool contains_nonopaque_tir_blocks{false};

  bool contains_relay_function{false};

  // Contains at least one PrimFunc
  bool contains_tir_primfunc{false};

  // Contains tir::Buffer objects that must be flattened.
  bool requires_buffer_flattening{false};

  // Contains PrimFuncs that perform internal allocations, either
  // through `tir::Allocate` nodes or through
  // `tir::BlockNode::alloc_buffers`.
  bool contains_internal_allocations{false};

  // Contains non-empty `tir::BlockNode::alloc_buffers`.  Should be lowered
  // to Allocate statements as part of `tvm::tir::transform::LowerOpaqueBlock`
  bool contains_block_alloc_buffers{false};

  // Contains a non-empty `BlockNode::match_buffers`, used by TIR
  // schedules to declare views into another buffer.
  bool uses_buffer_views_in_block{false};

  // Contains a "buffer_bind_scope" attribute, used by TE schedules to
  // declare views into another buffer
  bool uses_buffer_views_by_attribute{false};

  // TODO: Buffer aliasing?

  // TODO: Host/device constructs (e.g. call_extern)

  // TODO: Target-specific calls

  // TODO: Target-specific datatypes

  // TODO: Existence of Prefetch

  // TODO: Existence of "warp" memory

  // TODO: Existence of "vthread"

  // TODO: Existence of loop unrolling annotations

  // TODO: Existence of double-buffer attribute

  // TODO: Map all of these to the passes required to remove them

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("is_te_derived", &is_te_derived);
    v->Visit("contains_relay_function", &contains_relay_function);
    v->Visit("contains_tir_primfunc", &contains_tir_primfunc);
    v->Visit("contains_te_specific_nodes", &contains_te_specific_nodes);
    v->Visit("contains_tir_blocks", &contains_tir_blocks);
    v->Visit("contains_nonopaque_tir_blocks", &contains_nonopaque_tir_blocks);
    v->Visit("requires_buffer_flattening", &requires_buffer_flattening);
    v->Visit("contains_internal_allocations", &contains_internal_allocations);
    v->Visit("contains_block_alloc_buffers", &contains_block_alloc_buffers);
    v->Visit("uses_buffer_views_in_block", &uses_buffer_views_in_block);
    v->Visit("uses_buffer_views_by_attribute", &uses_buffer_views_by_attribute);
  }

  static constexpr const char* _type_key = "ir.AnalysisResults";

  TVM_DECLARE_FINAL_OBJECT_INFO(AnalysisResultsNode, Object);
};

class AnalysisResults : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(AnalysisResults, ObjectRef, AnalysisResultsNode);
};

/*! \brief Analyze the contents of an IRModule
 *
 * With several distinct IRs co-existing in a single IRModule type, or
 * even within a single function, it can be difficult to the contents
 * of any given module.  This pass inspects the contents of the
 * module, to determine where in the lowering flow the module may
 * occur.
 */
AnalysisResults AnalyzeModuleIRType(const IRModule& mod);

}  // namespace ir
}  // namespace tvm
#endif  // TVM_IR_IR_TYPE_ANALYSIS_H_
