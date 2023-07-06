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
#include <tvm/topi/detail/extern.h>

#include "ir_utils.h"

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

using topi::detail::pack_buffer;
PrimExpr pack_region(const BufferRegion& region, Optional<PrimExpr> device_type = NullOpt,
                     Optional<PrimExpr> device_id = NullOpt) {
  bool is_full_buffer = [&]() -> bool {
    StructuralEqual struct_equal;
    ICHECK_EQ(region->region.size(), region->buffer->shape.size());
    for (size_t i = 0; i < region->region.size(); i++) {
      const Range& range = region->region[i];
      const PrimExpr& dim = region->buffer->shape[i];
      if (!is_zero(range->min)) {
        return false;
      }
      if (!struct_equal(range->extent, dim)) {
        return false;
      }
    }
    return true;
  }();
  if (is_full_buffer) {
    return pack_buffer(region->buffer, device_type, device_id);
  }

  Array<PrimExpr> begins;
  Array<PrimExpr> extents;
  for (const auto& range : region->region) {
    begins.push_back(range->min);
    extents.push_back(range->extent);
  }
  return pack_buffer(region->buffer.MakeSlice(begins, extents), device_type, device_id);
}

}  // namespace

class CrossDeviceCopyInserter : public StmtExprMutator {
 public:
  using Parent = StmtExprMutator;

  explicit CrossDeviceCopyInserter(const Map<GlobalVar, Target>& target_map,
                                   const ParameterTypeMap& param_type_map, Target func_target)
      : target_map_(target_map), param_type_map_(param_type_map), func_target_(func_target) {}

  void DefineBuffer(const Buffer& buf) {
    if (buffer_var_map_.Get(buf->data)) {
      return;
    }
    buffer_var_map_.Set(buf->data, buf);
  }

  Stmt VisitStmt(const Stmt& stmt) override {
    Stmt copy = stmt;
    postproc_stack_.push_back(nullptr);
    auto node = Parent::VisitStmt(stmt);
    if (postproc_stack_.back()) {
      node = postproc_stack_.back()(node);
    }
    postproc_stack_.pop_back();
    return node;
  }

  Stmt VisitStmt_(const AttrStmtNode* op) override {
    auto cache = current_device_;
    if (op->attr_key == attr::device_type) {
      current_device_.device_type = Downcast<Integer>(op->value);
    } else if (op->attr_key == attr::device_id) {
      current_device_.device_id = op->value;
    }

    auto out = Parent::VisitStmt_(op);
    current_device_ = cache;
    return out;
  }

  PrimExpr VisitExpr_(const CallNode* op) override {
    auto call = Downcast<Call>(Parent::VisitExpr_(op));

    auto* callee_ptr = call->op.as<GlobalVarNode>();
    if (!callee_ptr) return std::move(call);
    auto callee = GetRef<GlobalVar>(callee_ptr);

    Target callee_target = [&]() {
      auto it = target_map_.find(callee);
      CHECK(it != target_map_.end())
          << "Could not find callee " << callee->name_hint << " in IRModule.  "
          << "Known callees/targets = " << target_map_;
      return (*it).second;
    }();

    // Setup prior to the call.  Declared early in the function, as we
    // may need to insert `attr::device_id` and `attr::device_type` if
    // no such annotation already exists.
    std::vector<Stmt> merge_nest;

    Integer callee_device_type(callee_target->GetTargetDeviceType());
    Integer caller_device_type(func_target_->GetTargetDeviceType());
    Integer caller_device_id(0);

    PrimExpr callee_device_id;
    if (current_device_.device_id) {
      ICHECK(current_device_.device_type);
      ICHECK_EQ(current_device_.device_type.value()->value, callee_device_type->value);
      callee_device_id = current_device_.device_id.value();
    } else {
      callee_device_id = Integer(0);
      merge_nest.push_back(
          AttrStmt(StringImm("default"), attr::device_id, callee_device_id, Evaluate(0)));
      merge_nest.push_back(
          AttrStmt(StringImm("default"), attr::device_type, callee_device_type, Evaluate(0)));
    }

    if (callee_device_type->value == caller_device_type->value) {
      return std::move(call);
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
        ICHECK(arg->IsInstance<VarNode>()) << "Buffer argument must be a Buffer's data pointer";
        auto var = Downcast<Var>(arg);

        MatchBufferRegion match_buffer_region = [&]() {
          if (auto opt = buffer_var_map_.Get(var)) {
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
      return std::move(call);
    }

    Array<Stmt> pre_seq = input_regions.Map([&](const auto& match) -> Stmt {
      PrimExpr src_dltensor = pack_region(match->source, caller_device_type, caller_device_id);
      PrimExpr dst_dltensor = pack_buffer(match->buffer, callee_device_type, callee_device_id);
      return Evaluate(
          Call(DataType::Int(32), builtin::buffer_copy(), {dst_dltensor, src_dltensor}));
    });

    Array<Stmt> post_seq = input_regions.Map([&](const auto& match) -> Stmt {
      PrimExpr src_dltensor = pack_buffer(match->buffer, callee_device_type, callee_device_id);
      PrimExpr dst_dltensor = pack_region(match->source, caller_device_type, caller_device_id);
      return Evaluate(
          Call(DataType::Int(32), builtin::buffer_copy(), {dst_dltensor, src_dltensor}));
    });

    for (const auto& buf : allocations) {
      // Allocate will be lowered to device-specific
      // TVMBackendAllocWorkspace in LowerTVMBuiltin.
      merge_nest.push_back(Allocate(buf->data, buf->dtype, buf->shape, Bool(true), Evaluate(0)));
    }

    postproc_stack_.back() = [pre_seq = std::move(pre_seq), post_seq = std::move(post_seq),
                              merge_nest = std::move(merge_nest)](Stmt stmt) -> Stmt {
      return MergeNest(merge_nest, SeqStmt::Flatten(pre_seq, stmt, post_seq));
    };

    auto write_ptr = call.CopyOnWrite();
    write_ptr->args = args;
    return call;
  }

 private:
  const Map<GlobalVar, Target>& target_map_;
  const ParameterTypeMap& param_type_map_;

  Target func_target_;
  Map<Var, Buffer> buffer_var_map_;
  std::vector<std::function<Stmt(Stmt)>> postproc_stack_;

  struct DeviceInfo {
    Optional<Integer> device_type;
    Optional<PrimExpr> device_id;
  };
  DeviceInfo current_device_;
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
