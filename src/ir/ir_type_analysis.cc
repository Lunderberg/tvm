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
 * \file src/ir/analysis.cc
 * \brief Analyze the contents of an IRModule
 */

#include "ir_type_analysis.h"

#include <tvm/relay/function.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace ir {

namespace {

struct Visitor : tir::StmtExprVisitor {
  Visitor(AnalysisResultsNode& output) : output(output) {}
  AnalysisResultsNode& output;

  using Parent = tir::StmtExprVisitor;

  void VisitExpr_(const tir::ProducerLoadNode* op) override {
    output.contains_te_specific_nodes = true;
    Parent::VisitExpr_(op);
  }
  void VisitStmt_(const tir::ProducerStoreNode* op) override {
    output.contains_te_specific_nodes = true;
    Parent::VisitStmt_(op);
  }
  void VisitStmt_(const tir::ProducerRealizeNode* op) override {
    output.contains_te_specific_nodes = true;
    Parent::VisitStmt_(op);
  }
  void VisitStmt_(const tir::BufferRealizeNode* op) override {
    output.contains_te_specific_nodes = true;
    Parent::VisitStmt_(op);
  }

  void VisitStmt_(const tir::BlockNode* block) override {
    output.contains_tir_blocks = true;
    if (block->iter_vars.size()) {
      output.contains_nonopaque_tir_blocks = true;
    }
    Parent::VisitStmt_(block);
  }

  void VisitExpr_(const tir::BufferLoadNode* op) override {
    VisitBuffer(op->buffer);
    Parent::VisitExpr_(op);
  }

  void VisitStmt_(const tir::BufferStoreNode* op) override {
    VisitBuffer(op->buffer);
    Parent::VisitStmt_(op);
  }

  void VisitBuffer(const tir::Buffer& buffer) {
    auto flattened = buffer.GetFlattenedBuffer();
    if (!flattened.same_as(buffer)) {
      output.requires_buffer_flattening = true;
    }
  }
};
}  // namespace

AnalysisResultsNode AnalyzeModuleIRTypeImpl(const IRModule& mod) {
  AnalysisResultsNode output;

  Visitor visitor(output);

  for (const auto& pair : mod->functions) {
    const BaseFunc& base_func = pair.second;
    if (auto* as_relay_func = base_func.as<relay::FunctionNode>()) {
      output.contains_relay_function = true;
    } else if (auto* as_prim_func = base_func.as<tir::PrimFuncNode>()) {
      output.contains_tir_primfunc = true;

      bool from_legacy_te_schedule =
          as_prim_func->GetAttr("from_legacy_te_schedule", Bool(false)).value();
      if (from_legacy_te_schedule) {
        output.is_te_derived = true;
      }

      visitor(as_prim_func->body);
    }
  }

  return output;
}

AnalysisResults AnalyzeModuleIRType(const IRModule& mod) {
  return AnalysisResults(make_object<AnalysisResultsNode>(AnalyzeModuleIRTypeImpl(mod)));
}

TVM_REGISTER_GLOBAL("ir.AnalyzeModuleIRType").set_body_typed(AnalyzeModuleIRType);
TVM_REGISTER_NODE_TYPE(AnalysisResultsNode);

}  // namespace ir
}  // namespace tvm
