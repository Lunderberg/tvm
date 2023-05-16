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
 * \file split_host_device.cc
 * \brief Split device function from host.
 */
#include <tvm/ir/global_var_supply.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>

#include "../../runtime/thread_storage_scope.h"
#include "../analysis/var_use_def_analysis.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief Visitor class to collect device-side program information.
 */
class LaunchParamsAnnotator : public StmtVisitor {
 public:
  static PrimFunc Apply(PrimFunc func) {
    LaunchParamsAnnotator collector;
    collector(func->body);
    return WithAttr(std::move(func), tir::attr::kKernelLaunchParams, collector.GetLaunchParams());
  }

  Array<String> GetLaunchParams() const {
    Array<String> launch_params = threads_;

    if (uses_dyn_shmem_) {
      launch_params.push_back(tvm::runtime::launch_param::kUseDynamicSharedMemoryTag);
    }
    return launch_params;
  }

 private:
  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      ICHECK_NE(iv->thread_tag.length(), 0U);
      // thread_extent can appear multiple times
      // use the first appearance as def.
      if (!defined_thread.count(iv.get())) {
        defined_thread.insert(iv.get());
        threads_.push_back(iv->thread_tag);
      }
    }

    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocateNode* op) final {
    auto storage_scope = runtime::StorageScope::Create(GetPtrStorageScope(op->buffer_var));
    if (storage_scope.rank == runtime::StorageRank::kShared && storage_scope.tag == ".dyn") {
      ICHECK(!uses_dyn_shmem_) << "Only one dynamic shared memory allocation is allowed.";
      ICHECK_GT(op->extents.size(), 0);

      uses_dyn_shmem_ = true;
    }
    StmtVisitor::VisitStmt_(op);
  }

  Array<String> threads_;
  bool uses_dyn_shmem_{false};
  // recording what thread axis have been visited.
  std::unordered_set<const IterVarNode*> defined_thread;
};

class HostDeviceSplitter : public StmtMutator {
 public:
  explicit HostDeviceSplitter(IRModule* device_mod, std::string name_prefix)
      : device_mod_(device_mod), name_prefix_(name_prefix) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tvm::attr::kTarget) {
      auto device_target = [&]() {
        auto output = make_object<TargetNode>(*op->node.as<TargetNode>());
        output->host = NullOpt;
        return Target(output);
      }();
      return SplitDeviceFunc(op->body, device_target);
    }
    return StmtMutator::VisitStmt_(op);
  }

 private:
  Stmt SplitDeviceFunc(Stmt body, Target device_target) {
    Array<Var> params = [&]() {
      VarUseDefAnalyzer use_def(/*defined_vars=*/{}, /*visit_thread_extent=*/false);
      use_def(body);

      // Sort first by variable typ, then by variable name
      std::vector<Var> params{use_def.undefined_.begin(), use_def.undefined_.end()};
      std::sort(params.begin(), params.end(), [](const Var& a, const Var& b) {
        auto sort_key = [](const Var& var) {
          return std::tuple{
              !var->dtype.is_handle(),
              var->name_hint,
          };
        };
        return sort_key(a) < sort_key(b);
      });
      return params;
    }();

    GlobalVar kernel_symbol_global = [&]() {
      std::stringstream name;
      name << name_prefix_ << "_kernel";
      GlobalVarSupply global_var_supply = GlobalVarSupply(*device_mod_);
      return global_var_supply->FreshGlobal(name.str(), false);
    }();

    PrimFunc device_func(params, body);
    device_func = WithAttrs(std::move(device_func), {{tvm::attr::kTarget, device_target},
                                                     {tir::attr::kNoAlias, Bool(true)}});
    device_func = LaunchParamsAnnotator::Apply(std::move(device_func));

    (*device_mod_)->Add(kernel_symbol_global, device_func);
    Array<PrimExpr> args = params.Map([](const Var& var) -> PrimExpr { return var; });

    return Evaluate(Call(DataType::Int(32), kernel_symbol_global, args));
  }

  // target ir module
  IRModule* device_mod_;
  // function name hint
  std::string name_prefix_;
};

PrimFunc SplitHostDevice(PrimFunc func, IRModule* device_mod, const GlobalVar& gvar) {
  auto opt_target = func->GetAttr<Target>(tvm::attr::kTarget);
  ICHECK(opt_target) << "SplitHostDevice: Require the target attribute";
  Target target = opt_target.value();

  auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
  auto name_prefix = global_symbol.value_or(gvar->name_hint);

  if (target->GetHost()) {
    HostDeviceSplitter splitter(device_mod, name_prefix);

    func.CopyOnWrite()->body = splitter(func->body);
    func = WithAttr(std::move(func), tvm::tir::attr::kIsHostFunc, Bool(true));
  }
  return func;
}

namespace transform {

Pass SplitHostDevice() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    IRModuleNode* mod_ptr = mod.CopyOnWrite();
    auto* func_dict = mod_ptr->functions.CopyOnWrite();
    IRModule device_mod = IRModule(Map<GlobalVar, BaseFunc>({}));

    for (auto& kv : *func_dict) {
      auto gvar = Downcast<GlobalVar>(kv.first);
      auto& base_func = kv.second;
      if (base_func->IsInstance<PrimFuncNode>()) {
        PrimFunc func = Downcast<PrimFunc>(std::move(base_func));
        ICHECK(device_mod.defined()) << "The device module must be defined.";
        base_func = SplitHostDevice(std::move(func), &device_mod, gvar);
      }
    }
    mod->Update(device_mod);
    return ConvertSSA()(mod);
  };

  return tvm::transform::CreateModulePass(pass_func, 0, "tir.SplitHostDevice", {});
}

TVM_REGISTER_GLOBAL("tir.transform.SplitHostDevice").set_body_typed(SplitHostDevice);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
