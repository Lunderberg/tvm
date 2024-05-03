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
 * \file ensure_aligned.cc
 * \brief Operator to view an existing tensor.
 */

#include "ensure_aligned.h"

#include <tvm/runtime/device_api.h>

namespace tvm {
namespace relax {

/* relax.op.memory.view */
Expr ensure_aligned(Expr tensor, Optional<Expr> byte_alignment) {
  static const Op& op = Op::Get("relax.memory.ensure_aligned");
  return Call(op, {
                      tensor,
                      byte_alignment.value_or(relax::PrimValue::Int64(runtime::kAllocAlignment)),
                  });
}

TVM_REGISTER_GLOBAL("relax.op.memory.ensure_aligned").set_body_typed(ensure_aligned);

StructInfo InferStructInfoEnsureAligned(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Operator " << call->op << " should receive 2 arguments, "
                     << "but received " << call->args);
  }
  Expr arg_tensor = call->args[0];
  Expr arg_byte_alignment = call->args[1];

  TensorStructInfo tensor_sinfo = [&]() -> TensorStructInfo {
    StructInfo sinfo = GetStructInfo(arg_tensor);
    if (auto opt = sinfo.as<TensorStructInfo>()) {
      return opt.value();
    } else {
      LOG(FATAL) << "TypeError: "
                 << "Operator " << call->op << " expects first argument to be a tensor, "
                 << "but received " << arg_tensor << " with type " << sinfo;
    }
  }();

  auto byte_alignment = [&]() -> Optional<PrimExpr> {
    StructInfo sinfo = GetStructInfo(arg_byte_alignment);

    auto prim_sinfo = sinfo.as<PrimStructInfoNode>();

    CHECK(prim_sinfo) << "TypeError: "
                      << "Operator " << call->op << " expects the byte_alignment argument "
                      << "to be a Relax PrimValue.  "
                      << "However, expression " << call << " provides byte_alignment of "
                      << arg_byte_alignment << ", which has type " << sinfo;

    CHECK_EQ(prim_sinfo->dtype, DataType::Int(64))
        << "TypeError: "
        << "Operator " << call->op
        << " expects the byte_alignment to be a 64-bit integer, but received " << arg_byte_alignment
        << ", which has type " << sinfo;

    return prim_sinfo->value;
  }();

  if (byte_alignment.defined() && ctx->GetAnalyzer()->CanProve(byte_alignment.value() <= 0)) {
    LOG(FATAL) << "ValueError: "
               << "Operator " << call->op
               << " requires the byte_alignment argument to be a positive value.  "
               << "However, the expression " << call << " provides byte_alignment of "
               << byte_alignment;
  }

  return tensor_sinfo;
}

TVM_REGISTER_GLOBAL("tvm.relax.struct_info.infer_ensure_aligned_sinfo")
    .set_body_typed(InferStructInfoEnsureAligned);

Expr LegalizeEnsureAligned(const BlockBuilder& bb, const Call& call) {
  Expr data = call->args[0];
  Expr byte_alignment = call->args[1];

  LOG(FATAL) << "LegalizeEnsureAligned not implemented yet";
}

TVM_REGISTER_OP("relax.memory.ensure_aligned")
    .set_num_inputs(2)
    .add_argument("tensor", "Tensor", "The input tensor.")
    .add_argument("byte_alignment", "Prim(\"int64\")",
                  "The required byte alignment of the returned tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoEnsureAligned)
    .set_attr<FLegalize>("FLegalize", LegalizeEnsureAligned)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace relax
}  // namespace tvm
