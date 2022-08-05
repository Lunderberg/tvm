/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file convert_block_to_opaque.cc
 * \brief Convert the blocks to opaque blocks which do not have block vars.
 */

#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "ir_utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief Substitute expr via BlockRealize value bindings and convert each block into opaque
 *        blocks.
 */
class SubroutineBlockExtractor : public StmtExprMutator {
 public:
  static Stmt Substitute(const PrimFunc& f) {
    SubroutineBlockExtractor substituter;
    return substituter.VisitStmt(f->body);
  }

  PrimFunc operator()(PrimFunc func) {
    func.CopyOnWrite()->body = this->VisitStmt(std::move(func->body));
    return func;
  }

  Stmt VisitStmt_(const BlockNode* op) override {
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));

    const auto& annot = block->annotations;
    auto it = annot.find(attr::extract_as_subroutine);
    if (it != annot.end()) {
      std::unordered_set<const BufferNode*> buffers;
      for (const auto& region : block->reads) {
        buffers.insert(region->buffer.get());
      }
      for (const auto& region : block->writes) {
        buffers.insert(region->buffer.get());
      }

      std::unordered_set<const VarNode*> buffer_vars;
      Array<Var> function_signature;
      Array<PrimExpr> function_arguments;
      Map<Var, Buffer> buffer_map;
      for (const auto& buf : buffers) {
        buffer_vars.insert(buf->data.get());
        Var handle("handle_" + buf->name, DataType::Handle());
        buffer_map.Set(handle, GetRef<Buffer>(buf));
        function_signature.push_back(handle);
        //
      }

      for (Var var : UndefinedVars(block->body, {})) {
        if (!buffer_vars.count(var.get())) {
          function_arguments.push_back(var);
          function_signature.push_back(var);
        }
      }

      PrimFunc extracted_func(function_signature, op->body);
      GlobalVar gvar(block->name_hint + "_extracted");
      extracted.Set(gvar, extracted_func);

      block.CopyOnWrite()->body = Evaluate(Call(DataType::Void(), gvar, function_arguments));
    }

    return std::move(block);
  }

  Map<GlobalVar, BaseFunc> extracted;
};

namespace transform {

Pass ExtractSubroutineBlocks() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    SubroutineBlockExtractor extractor;

    IRModuleNode* mod_ptr = mod.CopyOnWrite();
    auto* func_dict = mod_ptr->functions.CopyOnWrite();

    for (auto& kv : *func_dict) {
      if (kv.second->IsInstance<PrimFuncNode>()) {
        PrimFunc func = Downcast<PrimFunc>(std::move(kv.second));
        kv.second = extractor(std::move(func));
      }
    }
    mod_ptr->functions = Merge(std::move(mod_ptr->functions), extractor.extracted);

    return mod;
  };

  return tvm::transform::CreateModulePass(pass_func, 0, "tir.ExtractSubroutineBlocks", {});
}

TVM_REGISTER_GLOBAL("tir.transform.ExtractSubroutineBlocks")
    .set_body_typed(ExtractSubroutineBlocks);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
