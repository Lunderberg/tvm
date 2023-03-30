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
 * \file insert_cross_device_copies.cc
 * \brief Split device function from host.
 */
#include <tvm/arith/bound.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

namespace {
enum class ParameterType {
  ScalarInput,
  BufferInput,
  BufferOutput,
  BufferUnknown,
};
struct Parameter {
  std::string name;
  ParameterType type;
};

std::vector<Parameter> DetectParameterType(const PrimFunc& func) {
  auto buffer_access = arith::DomainTouched(func->body);
  std::unordered_set<const VarNode*> reads;
  std::unordered_set<const VarNode*> writes;

  std::vector<Parameter> params;
  for (const auto& param : func->params) {
    auto param_type = [&]() -> ParameterType {
      if (!param->dtype.is_handle()) {
        return ParameterType::ScalarInput;
      }

      Var buffer_var = param;
      if (auto buf = func->buffer_map.Get(param)) {
        buffer_var = buf.value()->data;
      }
      bool is_read = reads.count(buffer_var.get());
      bool is_write = writes.count(buffer_var.get());
      if (is_read && !is_write) {
        return ParameterType::BufferInput;
      } else if (!is_read && is_write) {
        return ParameterType::BufferOutput;
      } else {
        return ParameterType::BufferUnknown;
      }
    }();
    params.push_back(Parameter{param->name_hint, param_type});
  }

  return params;
}

using ParameterTypeMap =
    std::unordered_map<GlobalVar, std::vector<Parameter>, ObjectPtrHash, ObjectPtrEqual>;
}  // namespace

class CrossDeviceCopyInserter : public StmtMutator {
 public:
  using Parent = StmtMutator;

  explicit CrossDeviceCopyInserter(const Map<GlobalVar, Target>& target_map,
                                   const ParameterTypeMap& param_type_map, Target func_target)
      : target_map_(target_map), param_type_map_(param_type_map), func_target_(func_target) {}

  void DefineBuffer(const Buffer& buf) {
    if (buffer_var_map_.Get(buf->data)) {
      return;
    }
    buffer_var_map_.Set(buf->data, buf);
  }

  Stmt VisitStmt_(const EvaluateNode* op) override {
    auto node = Downcast<Evaluate>(Parent::VisitStmt_(op));

    auto* call_ptr = node->value.as<CallNode>();
    if (!call_ptr) return std::move(node);
    auto call = GetRef<Call>(call_ptr);

    auto* callee_ptr = call->op.as<GlobalVarNode>();
    if (!callee_ptr) return std::move(node);
    auto callee = GetRef<GlobalVar>(callee_ptr);

    Target callee_target = [&]() {
      auto it = target_map_.find(callee);
      CHECK(it != target_map_.end())
          << "Could not find callee " << callee->name_hint << " in IRModule.  "
          << "Known callees/targets = " << target_map_;
      return (*it).second;
    }();

    if (callee_target->GetTargetDeviceType() == func_target_->GetTargetDeviceType()) {
      return std::move(node);
    }

    const auto& parameters = [&]() {
      auto it = param_type_map_.find(callee);
      CHECK(it != param_type_map_.end())
          << "Could not find callee " << callee->name_hint << " as a PrimFunc in IRModule.";
      return it->second;
    }();

    CHECK_EQ(parameters.size(), call->args.size())
        << "Subroutine " << callee << " defined with " << parameters.size()
        << " parameters, but was called with " << call->args.size() << " arguments";

    Array<Buffer> allocations;
    Array<MatchBufferRegion> input_regions;
    Array<MatchBufferRegion> output_regions;

    Array<PrimExpr> args;
    Map<Var, BufferRegion> buffer_map;
    for (size_t i = 0; i < call->args.size(); i++) {
      PrimExpr arg = call->args[i];
      const Parameter& param = parameters[i];

      if (arg->dtype.is_handle()) {
        ICHECK(arg->IsInstance<VarNode>()) << "Buffer argument must be a Buffer's data pointer or "
                                           << "an entry in CallNode::buffer_map";
        auto var = Downcast<Var>(arg);

        MatchBufferRegion match_buffer_region = [&]() {
          if (auto opt = call->buffer_map.Get(var)) {
            auto region = opt.value();
            Buffer exposed_buf =
                decl_buffer(region->region.Map([](const Range& range) { return range->extent; }),
                            region->buffer->dtype, region->buffer->name + "_copy");
            Var handle(region->buffer->name + "_copy_handle", DataType::Handle());
            args.push_back(handle);
            buffer_map.Set(handle, BufferRegion::FullRegion(exposed_buf));

            return MatchBufferRegion(exposed_buf, region);
          } else if (auto opt = buffer_var_map_.Get(var)) {
            auto buffer = opt.value();
            auto region = BufferRegion::FullRegion(buffer);

            Buffer exposed_buf = decl_buffer(buffer->shape, buffer->dtype, buffer->name);
            args.push_back(exposed_buf->data);

            return MatchBufferRegion(exposed_buf, region);
          } else {
            LOG(FATAL) << "Buffer argument must be a Buffer's data pointer or "
                       << "an entry in CallNode::buffer_map";
          }
        }();

        allocations.push_back(match_buffer_region->buffer);

        CHECK(param.type != ParameterType::ScalarInput)
            << "Subroutine " << callee << " was defined with scalar parameter " << param.name
            << ", but was called with buffer/handle argument " << arg;
        if (param.type == ParameterType::BufferInput ||
            param.type == ParameterType::BufferUnknown) {
          input_regions.push_back(match_buffer_region);
        }
        if (param.type == ParameterType::BufferOutput ||
            param.type == ParameterType::BufferUnknown) {
          output_regions.push_back(match_buffer_region);
        }

      } else {
        CHECK(param.type == ParameterType::ScalarInput)
            << "Subroutine " << callee << " was defined with buffer parameter " << param.name
            << ", but was called with scalar argument " << arg;
        args.push_back(arg);
      }
    }

    if (allocations.empty()) {
      return std::move(node);
    }

    {
      auto write_ptr = call.CopyOnWrite();
      write_ptr->args = args;
      write_ptr->buffer_map = buffer_map;
    }

    Array<Stmt> seq;
    for (const auto& match : input_regions) {
      Var src_handle("src", DataType::Handle());
      Var dst_handle("dst", DataType::Handle());
      seq.push_back(Evaluate(Call(
          DataType::Int(32), builtin::buffer_copy(), {dst_handle, src_handle},
          {{src_handle, match->source}, {dst_handle, BufferRegion::FullRegion(match->buffer)}})));
    }
    seq.push_back(Evaluate(call));
    for (const auto& match : output_regions) {
      Var src_handle("src", DataType::Handle());
      Var dst_handle("dst", DataType::Handle());
      seq.push_back(Evaluate(Call(
          DataType::Int(32), builtin::buffer_copy(), {dst_handle, src_handle},
          {{src_handle, BufferRegion::FullRegion(match->buffer)}, {dst_handle, match->source}})));
    }

    Stmt stmt = SeqStmt(seq);

    for (const auto& buf : allocations) {
      // Allocate will be lowered to device-specific
      // TVMBackendAllocWorkspace in LowerTVMBuiltin.
      stmt = Allocate(buf->data, buf->dtype, buf->shape, Bool(true), stmt);
    }

    // TODO(Lunderberg): Integrate with VirtualDevice to allow non-zero device_id.
    int device_id = 0;
    stmt = AttrStmt(StringImm("default"), attr::device_id, device_id, stmt);
    stmt = AttrStmt(StringImm("default"), attr::device_type, callee_target->GetTargetDeviceType(),
                    stmt);

    return std::move(stmt);
  }

 private:
  const Map<GlobalVar, Target>& target_map_;
  const ParameterTypeMap& param_type_map_;

  Target func_target_;
  Map<Var, Buffer> buffer_var_map_;
};

namespace transform {

Pass InsertCrossDeviceCopies() {
  auto pass_func = [](IRModule mod, PassContext ctx) -> IRModule {
    auto target_map = [&]() -> Map<GlobalVar, Target> {
      Map<GlobalVar, Target> target_map;
      for (const auto& [gvar, base_func] : mod->functions) {
        auto target = base_func->GetAttr<Target>(tvm::attr::kTarget).value();
        target_map.Set(gvar, target);
      }
      return target_map;
    }();

    auto param_type_map = [&]() -> ParameterTypeMap {
      ParameterTypeMap param_type_map;
      for (const auto& [gvar, base_func] : mod->functions) {
        if (auto* prim_func = base_func.as<PrimFuncNode>()) {
          param_type_map[gvar] = DetectParameterType(GetRef<PrimFunc>(prim_func));
        }
      }
      return param_type_map;
    }();

    IRModule updates;
    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto* ptr = base_func.as<PrimFuncNode>()) {
        auto func_target = base_func->GetAttr<Target>(tvm::attr::kTarget).value();

        CrossDeviceCopyInserter mutator(target_map, param_type_map, func_target);
        for (const auto& arg : ptr->params) {
          if (auto var = arg.as<VarNode>()) {
            if (auto buf = ptr->buffer_map.Get(GetRef<Var>(var))) {
              mutator.DefineBuffer(buf.value());
            }
          }
        }
        auto body = mutator(ptr->body);

        if (!body.same_as(ptr->body)) {
          auto prim_func = GetRef<PrimFunc>(ptr);
          prim_func.CopyOnWrite()->body = std::move(body);
          updates->Add(gvar, std::move(prim_func));
        }
      }
    }

    if (updates->functions.size()) {
      mod.CopyOnWrite()->Update(updates);
    }
    return mod;
  };

  return tvm::transform::CreateModulePass(pass_func, 0, "tir.InsertCrossDeviceCopies", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InsertCrossDeviceCopies")
    .set_body_typed(InsertCrossDeviceCopies);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
