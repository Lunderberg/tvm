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
  bool is_te_derived{false};
  bool contains_te_specific_nodes{false};
  bool contains_tir_blocks{false};
  bool contains_nonopaque_tir_blocks{false};
  bool contains_relay_function{false};
  bool contains_tir_primfunc{false};
  bool requires_buffer_flattening{false};

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("is_te_derived", &is_te_derived);
    v->Visit("contains_relay_function", &contains_relay_function);
    v->Visit("contains_tir_primfunc", &contains_tir_primfunc);
    v->Visit("contains_te_specific_nodes", &contains_te_specific_nodes);
    v->Visit("contains_tir_blocks", &contains_tir_blocks);
    v->Visit("contains_nonopaque_tir_blocks", &contains_nonopaque_tir_blocks);
    v->Visit("requires_buffer_flattening", &requires_buffer_flattening);
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
