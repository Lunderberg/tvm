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

#include <tvm/relax/analysis.h>
#include <tvm/tir/stmt_functor.h>

#include "../../tir/analysis/stmt_to_primfunc.h"

namespace tvm {
namespace relax {

using namespace tir;

namespace {
struct Visitor : StmtExprVisitor {
  static std::unordered_set<const BufferNode*> FindOutputs(const Stmt& stmt) {
    Visitor visitor;
    visitor(stmt);
    return std::move(visitor.outputs);
  }

  std::unordered_set<const BufferNode*> outputs;
  std::unordered_map<const BufferNode*, const BufferNode*> match_buffers;

  using Parent = StmtExprVisitor;
  using Parent::VisitExpr_;
  using Parent::VisitStmt_;

  void VisitStmt_(const BlockNode* op) override {
    for (const auto& match : op->match_buffers) {
      match_buffers[match->buffer.get()] = match->source->buffer.get();
    }

    if (op->reads.size() || op->writes.size()) {
      // If present, block annotations take precedence over
      // BufferStore/BufferLoad.
      for (const auto& reg : op->writes) {
        outputs.insert(UnwrapMatchBuffer(reg->buffer.get()));
      }

    } else {
      Parent::VisitStmt_(op);
    }
  }

  void VisitStmt_(const BufferStoreNode* op) override {
    outputs.insert(UnwrapMatchBuffer(op->buffer.get()));
    Parent::VisitStmt_(op);
  }

  const BufferNode* UnwrapMatchBuffer(const BufferNode* buf) const {
    while (true) {
      if (auto it = match_buffers.find(buf); it != match_buffers.end()) {
        buf = it->second;
      } else {
        break;
      }
    }
    return buf;
  }
};

class InputDimensionExtractor {
 public:
  InputDimensionExtractor(tir::Var var, size_t param_i, size_t dim_i)
      : var(var), param_i(param_i), dim_i(dim_i) {}

  PrimExpr operator()(const relax::Call& call, const BlockBuilder& ctx) const {
    const auto& args = call->args[1];
    Expr arg = [&]() {
      if (auto* tuple = args.as<TupleNode>()) {
        if (param_i >= tuple->fields.size()) {
          ctx->ReportFatal(Diagnostic::Error(call->span)
                           << "Parameter " << param_i << "'s shape[" << dim_i
                           << "] is used to define var " << var << ", but argument has only "
                           << tuple->fields.size() << " param(s)");
        }
        return tuple->fields[param_i];

      } else {
        if (param_i != 0) {
          ctx->ReportFatal(Diagnostic::Error(call->span)
                           << "Parameter " << param_i << "'s shape[" << dim_i
                           << "] is used to define var " << var << ", but call has only 1 param");
        }
        return args;
      }
    }();

    auto arg_sinfo = [&]() {
      auto* ptr = arg->struct_info_.as<StructInfoNode>();
      if (!ptr) {
        ctx->ReportFatal(Diagnostic::Error(arg->span) << "Parameter " << param_i << "'s shape["
                                                      << dim_i << "] is used to define var " << var
                                                      << ", but argument has no shape information");
      }

      return GetRef<StructInfo>(ptr);
    }();

    auto arg_tensor_sinfo = [&]() {
      auto* ptr = arg_sinfo.as<TensorStructInfoNode>();
      if (!ptr) {
        ctx->ReportFatal(Diagnostic::Error(arg->span)
                         << "Parameter " << param_i << "'s shape[" << dim_i
                         << "] is used to define var " << var
                         << ", but the argument doesn't have TensorStructInfo annotation");
      }

      return GetRef<TensorStructInfo>(ptr);
    }();

    ShapeExpr arg_shape_expr = [&]() {
      auto* ptr = arg_tensor_sinfo->shape.as<ShapeExprNode>();
      if (!ptr) {
        ctx->ReportFatal(Diagnostic::Error(arg->span)
                         << "Parameter " << param_i << "'s shape[" << dim_i
                         << "] is used to define var " << var
                         << ", but the TensorStructInfo::shape isn't a ShapeExpr annotation");
      }

      return GetRef<ShapeExpr>(ptr);
    }();

    if (dim_i >= arg_shape_expr->values.size()) {
      ctx->ReportFatal(Diagnostic::Error(arg_shape_expr->span)
                       << "Parameter " << param_i << "'s shape[" << dim_i
                       << "] is used to define var " << var << ", but the argument is only "
                       << arg_shape_expr->values.size() << "-dimensional");
    }

    return arg_shape_expr->values[dim_i];
  }

 private:
  tir::Var var;
  size_t param_i;
  size_t dim_i;
};

}  // namespace

FuncStructInfo PrimFuncSignature(const tir::PrimFunc& func) {
  Array<Buffer> input_buffers;
  Array<Buffer> output_buffers;
  Array<tir::Var> tir_params;

  auto output_buffer_nodes = Visitor::FindOutputs(func->body);

  size_t param_i = 0;
  for (; param_i < func->params.size(); param_i++) {
    if (auto opt = func->buffer_map.Get(func->params[param_i]);
        opt && !output_buffer_nodes.count(opt.value().get())) {
      input_buffers.push_back(opt.value());
    } else {
      break;
    }
  }

  for (; param_i < func->params.size(); param_i++) {
    if (auto opt = func->buffer_map.Get(func->params[param_i])) {
      auto buf = opt.value();
      ICHECK(output_buffer_nodes.count(buf.get()))
          << "Relax requires TIR to use destination-passing convention, "
          << "with output buffers after input buffers, "
          << "but input buffer " << buf << " occurs after output buffers " << output_buffers;
      output_buffers.push_back(buf);
    } else {
      break;
    }
  }

  for (; param_i < func->params.size(); param_i++) {
    if (auto* buf_ptr = func->params[param_i].as<BufferNode>()) {
      LOG(FATAL) << "Relax requires TIR to have PrimExpr arguments to occur "
                 << "after all buffer arguments, "
                 << "but buffer argument " << GetRef<Buffer>(buf_ptr)
                 << " occurs after PrimExpr parameters " << tir_params;
    }
    tir_params.push_back(func->params[param_i]);
  }

  Array<StructInfo> tensor_input;
  std::unordered_map<tir::Var, std::function<PrimExpr(const relax::Call&, const BlockBuilder& ctx)>,
                     ObjectPtrHash, ObjectPtrEqual>
      implicit_primexpr_input;
  Array<tir::Var> explicit_primexpr_input;
  Array<StructInfo> tensor_output;

  std::unordered_set<const tir::VarNode*> defined_vars;
  for (const auto& var : tir_params) {
    defined_vars.insert(var.get());
  }
  for (size_t param_i = 0; param_i < input_buffers.size(); param_i++) {
    const auto& buf = input_buffers[param_i];
    for (size_t dim_i = 0; dim_i < buf->shape.size(); dim_i++) {
      const auto& dim = buf->shape[dim_i];
      if (auto* ptr = dim.as<tir::VarNode>(); ptr && !defined_vars.count(ptr)) {
        defined_vars.insert(ptr);
        auto var = GetRef<tir::Var>(ptr);
        implicit_primexpr_input[var] = InputDimensionExtractor(var, param_i, dim_i);
      }
    }
  }

  for (const auto& buf : output_buffers) {
    for (const auto& dim : buf->shape) {
      if (auto* ptr = dim.as<tir::VarNode>(); ptr && !defined_vars.count(ptr)) {
        LOG(FATAL) << "Output buffer " << buf << " has dynamic shape " << dim
                   << " that wasn't previously defined.  "
                   << "Output buffer shapes must be defined either by explicit tir::Var params, "
                   << "or by the input buffer shapes.";
      }
    }
  }

  // // Static shapes, expose directly as shaped arguments and return values.
  // if (implicit_primexpr_input.empty() && explicit_primexpr_input.empty()) {
  //   Array<StructInfo> params = input_buffers.Map([](const Buffer& buf) -> StructInfo {
  //     return TensorStructInfo(ShapeExpr(buf->shape), buf->dtype);
  //   });
  //   Array<StructInfo> ret = output_buffers.Map([](const Buffer& buf) -> StructInfo {
  //     return TensorStructInfo(ShapeExpr(buf->shape), buf->dtype);
  //   });
  //   if (ret.size() == 0) {
  //     return FuncStructInfo(params, StructInfo());
  //   } else if (ret.size() == 1) {
  //     return FuncStructInfo(params, ret[0]);
  //   } else {
  //     return FuncStructInfo(params, TupleStructInfo(ret));
  //   }
  // }

  // // Static shapes, expose as type inference from the inputs to outputs.

  // // return FuncStructInfo(params, ret);
  // TypedPackedFunc<StructInfo(const Call& call, const BlockBuilder& ctx)> derive_packed_func(
  //     [=](const Call& call, const BlockBuilder& ctx) -> StructInfo {
  //       Array<PrimExpr> dummy = {1, 2, 3, 4};
  //       return TensorStructInfo(ShapeExpr(dummy), DataType::Int(32));
  //     });
  // auto env_func_node = make_object<EnvFuncNode>();
  // env_func_node->name = "type_inference_lambda";
  // env_func_node->func = derive_packed_func;
  // StructInfoDeriveFunc derive_func(env_func_node);

  // return FuncStructInfo::OpaqueFunc(derive_func);

  // Static shapes, expose directly as shaped arguments and return values.
  Array<StructInfo> params = input_buffers.Map([](const Buffer& buf) -> StructInfo {
    return TensorStructInfo(ShapeExpr(buf->shape), buf->dtype);
  });
  Array<StructInfo> ret = output_buffers.Map([](const Buffer& buf) -> StructInfo {
    return TensorStructInfo(ShapeExpr(buf->shape), buf->dtype);
  });
  if (ret.size() == 0) {
    return FuncStructInfo(params, StructInfo());
  } else if (ret.size() == 1) {
    return FuncStructInfo(params, ret[0]);
  } else {
    return FuncStructInfo(params, TupleStructInfo(ret));
  }
}

TVM_REGISTER_GLOBAL("relax.analysis.prim_func_signature").set_body_typed(PrimFuncSignature);

}  // namespace relax
}  // namespace tvm
