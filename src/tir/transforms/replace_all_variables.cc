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
 * \file replace_all_variables.cc
 *
 * \brief Helper functions to de-dup variables in a generated
 * PrimFunc.
 */

#include <tvm/runtime/logging.h>
#include <tvm/tir/stmt_functor.h>

#include "ir_utils.h"

namespace tvm {
namespace tir {

class VariableReplacer : public StmtExprMutator {
 public:
  static PrimFunc Apply(PrimFunc func) {
    VariableReplacer replacer;
    Array<Var> params;
    Map<Var, Buffer> buffer_map;
    for (const Var& param : func->params) {
      auto new_param = replacer.Replace(param);
      params.push_back(new_param);
      if (auto opt = func->buffer_map.Get(param)) {
        auto old_buffer = opt.value();
        auto new_buffer = replacer.Replace(old_buffer);
        buffer_map.Set(new_param, new_buffer);
      }
    }
    auto body = replacer.VisitStmt(func->body);

    return PrimFunc(params, body, func->ret_type, buffer_map, func->attrs, func->span);
  }

 private:
  VariableReplacer() {}

  using Parent = StmtExprMutator;
  using Parent::VisitExpr_;
  using Parent::VisitStmt_;

  Var Replace(Var var) {
    if (auto opt = var_remap_.Get(var)) {
      return opt.value();
    }
    Var output(var->name_hint, var->type_annotation, var->span);
    var_remap_.Set(var, output);
    return output;
  }
  Buffer Replace(Buffer buf) {
    if (auto opt = buffer_remap_.Get(buf)) {
      return opt.value();
    }
    auto visit_expr = [this](const PrimExpr& expr) { return this->VisitExpr(expr); };
    Buffer output(Replace(buf->data), buf->dtype, buf->shape.Map(visit_expr),
                  buf->strides.Map(visit_expr), VisitExpr(buf->elem_offset), buf->name,
                  buf->data_alignment, buf->offset_factor, buf->buffer_type, buf->axis_separators,
                  buf->span);
    buffer_remap_.Set(buf, output);
    return output;
  }

  PrimExpr VisitExpr_(const VarNode* op) override { return Replace(GetRef<Var>(op)); }

  PrimExpr VisitExpr_(const LetNode* op) override {
    auto node = Downcast<Let>(Parent::VisitExpr_(op));
    node.CopyOnWrite()->var = Replace(node->var);
    return std::move(node);
  }
  PrimExpr VisitExpr_(const BufferLoadNode* op) override {
    auto node = Downcast<BufferLoad>(Parent::VisitExpr_(op));
    node.CopyOnWrite()->buffer = Replace(node->buffer);
    return std::move(node);
  }

  Stmt VisitStmt_(const LetStmtNode* op) override {
    auto node = Downcast<LetStmt>(Parent::VisitStmt_(op));
    node.CopyOnWrite()->var = Replace(node->var);
    return std::move(node);
  }
  Stmt VisitStmt_(const ForNode* op) override {
    auto node = Downcast<For>(Parent::VisitStmt_(op));
    node.CopyOnWrite()->loop_var = Replace(node->loop_var);
    return std::move(node);
  }
  Stmt VisitStmt_(const AllocateNode* op) override {
    auto node = Downcast<Allocate>(Parent::VisitStmt_(op));
    node.CopyOnWrite()->buffer_var = Replace(node->buffer_var);
    return std::move(node);
  }
  Stmt VisitStmt_(const AllocateConstNode* op) override {
    auto node = Downcast<AllocateConst>(Parent::VisitStmt_(op));
    node.CopyOnWrite()->buffer_var = Replace(node->buffer_var);
    return std::move(node);
  }
  Stmt VisitStmt_(const DeclBufferNode* op) override {
    auto node = Downcast<DeclBuffer>(Parent::VisitStmt_(op));
    node.CopyOnWrite()->buffer = Replace(node->buffer);
    return std::move(node);
  }
  Stmt VisitStmt_(const BufferStoreNode* op) override {
    auto node = Downcast<BufferStore>(Parent::VisitStmt_(op));
    node.CopyOnWrite()->buffer = Replace(node->buffer);
    return std::move(node);
  }
  Stmt VisitStmt_(const BufferRealizeNode* op) override {
    auto node = Downcast<BufferRealize>(Parent::VisitStmt_(op));
    node.CopyOnWrite()->buffer = Replace(node->buffer);
    return std::move(node);
  }
  Stmt VisitStmt_(const BlockNode* op) override {
    auto node = Downcast<Block>(Parent::VisitStmt_(op));
    auto ptr = node.CopyOnWrite();
    auto visit_region = [&](BufferRegion reg) {
      reg.CopyOnWrite()->buffer = Replace(reg->buffer);
      return reg;
    };
    ptr->iter_vars = ptr->iter_vars.Map([&](IterVar iter_var) {
      iter_var.CopyOnWrite()->var = Replace(iter_var->var);
      return iter_var;
    });
    ptr->reads = ptr->reads.Map(visit_region);
    ptr->writes = ptr->writes.Map(visit_region);
    ptr->match_buffers = ptr->match_buffers.Map([&](MatchBufferRegion match) {
      auto ptr = match.CopyOnWrite();
      ptr->buffer = Replace(ptr->buffer);
      ptr->source = visit_region(ptr->source);
      return match;
    });
    ptr->alloc_buffers = ptr->alloc_buffers.Map([&](const Buffer& buf) { return Replace(buf); });
    return std::move(node);
  }

  Map<Var, Var> var_remap_;
  Map<Buffer, Buffer> buffer_remap_;
};

PrimFunc ReplaceAllVariables(PrimFunc func) { return VariableReplacer::Apply(std::move(func)); }

}  // namespace tir
}  // namespace tvm
