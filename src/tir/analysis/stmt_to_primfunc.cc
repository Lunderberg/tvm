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
 * \file stmt_to_primfunc.cc
 * \brief Implementation of simple passes
 */
#include "./stmt_to_primfunc.h"

#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/topi/detail/extern.h>

#include "../ir/script/script_complete.h"

namespace tvm {
namespace tir {

struct BufferUsage {
  std::unordered_map<Var, std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>, ObjectPtrHash,
                     ObjectPtrEqual>
      aliasing_sets_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> output_buffers_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> internal_buffer_allocation_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> internal_var_allocation_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> internal_var_definition_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> vars_used_;

  StmtToPrimFuncResult Finalize(Stmt body) const;
};

class BufferUsageVisitor : StmtExprVisitor {
 public:
  static StmtToPrimFuncResult Apply(Stmt body) {
    BufferUsageVisitor visitor;
    visitor(body);
    return visitor.usage_.Finalize(std::move(body));
  }

 private:
  using Parent = StmtExprVisitor;
  using Parent::VisitExpr_;
  using Parent::VisitStmt_;

  // Sources of variable definitions
  void VisitStmt_(const LetStmtNode* op) override {
    RecordVarDefinition(op->var);
    Parent::VisitStmt_(op);
  }

  void VisitExpr_(const LetNode* op) override {
    RecordVarDefinition(op->var);
    Parent::VisitExpr_(op);
  }

  void VisitStmt_(const ForNode* op) override {
    RecordVarDefinition(op->loop_var);
    Parent::VisitStmt_(op);
  }

  // Sources of allocation without a corresponding buffer
  void VisitStmt_(const AllocateNode* op) override {
    RecordVarAllocation(op->buffer_var);
    Parent::VisitStmt_(op);
  }

  void VisitStmt_(const AllocateConstNode* op) override {
    RecordVarAllocation(op->buffer_var);
    Parent::VisitStmt_(op);
  }

  // Sources of allocation with a corresponding buffer
  void VisitStmt_(const BlockNode* op) override {
    for (const auto& iter_var : op->iter_vars) {
      RecordVarDefinition(iter_var->var);
    }
    for (const auto& buf : op->alloc_buffers) {
      RecordBufferAllocation(buf);
    }
    for (const auto& match_region : op->match_buffers) {
      RecordBufferUsage(match_region->source->buffer);
      RecordBufferAllocation(match_region->buffer);
    }
    for (const auto& read : op->reads) {
      RecordBufferUsage(read->buffer);
    }
    for (const auto& write : op->writes) {
      RecordOutputBuffer(write->buffer);
    }
    Parent::VisitStmt_(op);
  }

  // Sources of buffer usage
  void VisitStmt_(const BufferStoreNode* op) override {
    RecordOutputBuffer(op->buffer);
    Parent::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode* op) override {
    RecordBufferUsage(op->buffer);
    Parent::VisitExpr_(op);
  }

  // Sources of variable usage
  void VisitExpr_(const VarNode* op) override { RecordVarUsage(GetRef<Var>(op)); }

 private:
  void RecordBufferAllocation(const Buffer& buf) {
    usage_.internal_buffer_allocation_.insert(buf);
    RecordVarAllocation(buf->data);
    RecordBufferUsage(buf);
  }

  void RecordVarAllocation(const Var& buffer_var) {
    usage_.internal_var_allocation_.insert(buffer_var);
  }

  void RecordVarDefinition(const Var& var) { usage_.internal_var_definition_.insert(var); }

  void RecordOutputBuffer(const Buffer& buf) {
    usage_.output_buffers_.insert(buf);
    RecordBufferUsage(buf);
  }

  void RecordBufferUsage(const Buffer& buf) { usage_.aliasing_sets_[buf->data].insert(buf); }

  void RecordVarUsage(const Var& var) { usage_.vars_used_.insert(var); }

  BufferUsage usage_;
};

StmtToPrimFuncResult BufferUsage::Finalize(Stmt body) const {
  // Helper function, returns the implicit variable definitions that
  // could be used if the Buffer is a parameter.  These correspond to
  // dynamic runtime parameters read out from the DLTensor struct.
  auto collect_implicit_var_definitions = [](const Buffer& buf,
                                             std::unordered_set<const VarNode*>& implicit_defs) {
    auto record_var = [&](const PrimExpr& expr) {
      if (auto* var = expr.as<VarNode>()) {
        implicit_defs.insert(var);
      }
    };
    for (const PrimExpr& expr : buf->shape) record_var(expr);
    for (const PrimExpr& expr : buf->strides) record_var(expr);
    record_var(buf->elem_offset);
  };
  auto implicit_var_definitions = [&collect_implicit_var_definitions](const Buffer& buf) {
    std::unordered_set<const VarNode*> implicit_defs;
    collect_implicit_var_definitions(buf, implicit_defs);
    return implicit_defs;
  };

  // Collect the buffers that are needed by the body
  std::vector<Buffer> input_buffer_arguments;
  std::vector<Buffer> output_buffer_arguments;
  for (const auto& [buffer_var, buffers] : aliasing_sets_) {
    if (!internal_var_allocation_.count(buffer_var)) {
      // If more than one buffer uses the same backing allocation,
      // pick the one whose shape/strides/elem_offset would provide
      // implicit definitions for the most variables.

      Buffer buffer_arg = *std::max_element(
          buffers.begin(), buffers.end(),
          [&implicit_var_definitions](const Buffer& a, const Buffer& b) {
            return implicit_var_definitions(a).size() < implicit_var_definitions(b).size();
          });
      if (output_buffers_.count(buffer_arg)) {
        output_buffer_arguments.push_back(buffer_arg);
      } else {
        input_buffer_arguments.push_back(buffer_arg);
      }
    }
  }

  // Sort the buffers by name, to have a consistent ordering for each execution.
  auto sort_func = [](const Buffer& a, const Buffer& b) { return a->name < b->name; };
  std::sort(input_buffer_arguments.begin(), input_buffer_arguments.end(), sort_func);
  std::sort(output_buffer_arguments.begin(), output_buffer_arguments.end(), sort_func);

  // Collect the variables that are implicitly defined by the buffer
  // arguments.
  std::unordered_set<const VarNode*> implicit_defs;
  for (const auto& buf : input_buffer_arguments) {
    collect_implicit_var_definitions(buf, implicit_defs);
  }
  for (const auto& buf : output_buffer_arguments) {
    collect_implicit_var_definitions(buf, implicit_defs);
  }

  // Variables defined in the parent scope must now be defined in the
  // parameters.
  std::vector<Var> var_arguments;
  std::vector<Var> implicit_var_defs;
  for (const Var& var : vars_used_) {
    bool is_defined = internal_var_definition_.count(var) || internal_var_allocation_.count(var);
    ;
    if (!is_defined) {
      if (implicit_defs.count(var.get())) {
        implicit_var_defs.push_back(var);
      } else {
        var_arguments.push_back(var);
      }
    }
  }
  std::sort(var_arguments.begin(), var_arguments.end(),
            [](const Var& a, const Var& b) { return a->name_hint < b->name_hint; });

  return StmtToPrimFuncResult(body, input_buffer_arguments, output_buffer_arguments, var_arguments,
                              implicit_var_defs);
}

StmtToPrimFuncResult StmtToPrimFunc(Stmt body) { return BufferUsageVisitor::Apply(body); }

PrimFunc StmtToPrimFuncResult::ToPrimFunc(BufferParameter conv) const {
  Map<Var, Buffer> buffer_map;
  Array<Var> params;

  auto make_buffer_arg = [&](const Buffer& buffer) {
    switch (conv) {
      case BufferParameter::DLTensor: {
        Var handle(buffer->name + "_handle", DataType::Handle());
        params.push_back(handle);
        buffer_map.Set(handle, buffer);
        break;
      }
    }
  };

  for (const Buffer& buffer : input_buffer_params_) {
    make_buffer_arg(buffer);
  }
  for (const Buffer& buffer : output_buffer_params_) {
    make_buffer_arg(buffer);
  }

  for (const Var& var : var_params_) {
    params.push_back(var);
  }

  PrimFunc ret(params, body_, /* ret_type = */ VoidType(), buffer_map);

  return ScriptComplete(ret, {});
}

Array<PrimExpr> StmtToPrimFuncResult::TIRCallSiteArgs(BufferParameter conv) const {
  Array<PrimExpr> callsite_arguments;

  auto make_buffer_arg = [&](const Buffer& buffer) {
    switch (conv) {
      case BufferParameter::DLTensor:
        callsite_arguments.push_back(topi::detail::pack_buffer(buffer));
        break;
    }
  };

  for (const Buffer& buffer : input_buffer_params_) {
    make_buffer_arg(buffer);
  }
  for (const Buffer& buffer : output_buffer_params_) {
    make_buffer_arg(buffer);
  }

  for (const Var& var : var_params_) {
    callsite_arguments.push_back(var);
  }

  return callsite_arguments;
}

}  // namespace tir
}  // namespace tvm
