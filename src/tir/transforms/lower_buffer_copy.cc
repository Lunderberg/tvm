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
 * \file lower_buffer_copy.cc
 * \brief Split device function from host.
 */
#include <tvm/ir/transform.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

class BufferCopyLowerer : public StmtExprMutator {
 public:
  explicit BufferCopyLowerer(DLDeviceType device_type, size_t device_id = 0) {}

  PrimExpr VisitExpr_(const CallNode* op) final {
    auto node = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (node->op.same_as(builtin::buffer_copy())) {
      return MakeBufferCopy(std::move(node));
    } else {
      return std::move(node);
    }
  }

  PrimExpr MakeBufferCopy(Call op) {
    ICHECK_EQ(op->args.size(), 2) << "builtin::buffer_copy() should have two arguments, (dst, src)";

    PrimExpr dst_dltensor = op->args[0];
    PrimExpr src_dltensor = op->args[1];

    return Call(DataType::Int(32), Op::Get("tir.TVMDeviceCopyDataFromTo"),
                {src_dltensor, dst_dltensor,
                 reinterpret(DataType::Handle(), make_const(DataType::Int(64), 0))});
  }

 private:
};

namespace transform {

Pass LowerBufferCopy() {
  auto pass_func = [](PrimFunc func, IRModule mod, PassContext ctx) -> PrimFunc {
    auto opt_target = func->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(opt_target) << "LowerBufferCopy requires the target attribute";
    Target target = opt_target.value();

    if (target->GetHost()) {
      DLDeviceType device_type = static_cast<DLDeviceType>(target->GetTargetDeviceType());
      BufferCopyLowerer mutator(device_type);

      func.CopyOnWrite()->body = mutator(func->body);
    }
    return func;
  };

  return CreatePrimFuncPass(pass_func, 0, "tir.LowerBufferCopy", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerBufferCopy").set_body_typed(LowerBufferCopy);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
