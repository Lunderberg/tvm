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
    if (block->alloc_buffers.size()) {
      output.contains_internal_allocations = true;
      output.contains_block_alloc_buffers = true;
    }
    if (block->match_buffers.size()) {
      output.uses_buffer_views_in_block = true;
    }
    Parent::VisitStmt_(block);
  }

  void VisitStmt_(const tir::AttrStmtNode* op) override {
    if (op->attr_key == tir::attr::buffer_bind_scope) {
      output.uses_buffer_views_by_attribute = true;
    }
    Parent::VisitStmt_(op);
  }

  void VisitStmt_(const tir::AllocateNode* op) override {
    output.contains_internal_allocations = true;
    Parent::VisitStmt_(op);
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

bool MatchesPackedAPIParams(const Array<tir::Var>& params) {
  if (params.size() < 6) {
    return false;
  }

  auto is_untyped_handle = [](const tir::Var& var) {
    return var.dtype().is_handle() && !var->type_annotation.as<PointerTypeNode>();
  };
  auto is_pointer_to = [](const tir::Var& var, DataType dtype) {
    return var.dtype().is_handle() && tir::IsPointerType(var->type_annotation, dtype);
  };

  // Var v_packed_args("args", DataType::Handle());
  const auto& v_packed_args = params[0];
  // Buffer buf_packed_arg_type_ids = decl_buffer(
  //     {IntImm(DataType::Int(32), func_ptr->params.size())}, DataType::Int(32), "arg_type_ids");
  const auto& v_packed_args_type_ids = params[1];
  // Var v_num_packed_args("num_args", DataType::Int(32));
  const auto& v_num_packed_args = params[2];

  // TODO: Should it check for any unpacked args between the
  // input/output?  While there is a "num_unpacked_args" argument to
  // MakePackedAPI, it looks like it is only used in a single unit
  // test.

  // Var v_out_ret_value("out_ret_value", PointerType(PrimType(DataType::Void())));
  const auto& v_out_ret_value = params[params.size() - 3];
  // Var v_out_ret_tcode("out_ret_tcode", PointerType(PrimType(DataType::Int(32))));
  const auto& v_out_ret_tcode = params[params.size() - 2];
  // Var v_resource_handle("resource_handle", DataType::Handle());
  const auto& v_resource_handle = params[params.size() - 1];

  return is_untyped_handle(v_packed_args) &&
         is_pointer_to(v_packed_args_type_ids, DataType::Int(32)) &&
         v_num_packed_args.dtype() == DataType::Int(32) &&
         is_pointer_to(v_out_ret_value, DataType::Void()) &&
         is_pointer_to(v_out_ret_tcode, DataType::Int(32)) && is_untyped_handle(v_resource_handle);
}

bool MatchesUnpackedAPIParams(const Array<tir::Var>& params,
                              const Map<tir::Var, tir::Buffer>& buffer_map) {
  for (const auto& param : params) {
    auto it = buffer_map.find(param);
    if (it != buffer_map.end()) {
      const tir::Buffer& buf = (*it).second;
      if (!param.same_as(buf->data)) {
        return false;
      }
    }
  }
  return true;
}

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

      if (as_prim_func->buffer_map.size()) {
        output.has_tir_buffer_arguments = true;
      }

      if (MatchesPackedAPIParams(as_prim_func->params)) {
        output.has_packed_api_buffer_arguments = true;
      } else if (MatchesUnpackedAPIParams(as_prim_func->params, as_prim_func->buffer_map)) {
        output.has_unpacked_api_buffer_arguments = true;
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
