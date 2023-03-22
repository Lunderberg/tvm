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
 *  Lower intrinsic calls and ops to device specific ir when possible.
 * \file inline_static_arguments.cc
 */
#include <tvm/ir/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <optional>
#include <unordered_set>

namespace tvm {
namespace tir {

namespace {
struct ParamInfo {
  // Variable used for this parameter
  tir::Var var;

  // Collection of all integer arguments passed to this parameter
  std::unordered_set<int64_t> static_arguments;

  // If true, a non-integer parameter was passed at least once.
  bool dynamic_argument_used{false};

  explicit ParamInfo(tir::Var var) : var(var) {}

  Optional<IntImm> UniqueStaticArgument() const {
    if (static_arguments.size() == 1 && !dynamic_argument_used) {
      return IntImm(var->dtype, *static_arguments.begin());
    } else {
      return NullOpt;
    }
  }
};
}  // namespace

class StaticArgumentCollector : public StmtExprVisitor {
 public:
  StaticArgumentCollector(std::unordered_map<GlobalVar, std::vector<ParamInfo>, ObjectPtrHash,
                                             ObjectPtrEqual>& internal_methods)
      : internal_methods_(internal_methods) {}

 private:
  void VisitExpr_(const CallNode* op) override {
    StmtExprVisitor::VisitExpr_(op);

    auto* gvar_ptr = op->op.as<GlobalVarNode>();
    if (!gvar_ptr) return;
    auto gvar = GetRef<GlobalVar>(gvar_ptr);

    auto it = internal_methods_.find(gvar);
    if (it == internal_methods_.end()) return;
    auto& param_info = it->second;

    CHECK_EQ(param_info.size(), op->args.size())
        << "Function " << gvar << " was defined with " << param_info.size()
        << " parameters, but was called with " << op->args.size() << " parameters";
    for (size_t i = 0; i < param_info.size(); i++) {
      auto& param = param_info[i];
      auto arg = op->args[i];
      if (auto* as_int = arg.as<IntImmNode>()) {
        param.static_arguments.insert(as_int->value);
      } else {
        param.dynamic_argument_used = true;
      }
    }
  }

  std::unordered_map<GlobalVar, std::vector<ParamInfo>, ObjectPtrHash, ObjectPtrEqual>&
      internal_methods_;
};

class StaticArgumentRemover : public StmtExprMutator {
 public:
  StaticArgumentRemover(const std::unordered_map<GlobalVar, std::vector<ParamInfo>, ObjectPtrHash,
                                                 ObjectPtrEqual>& internal_methods)
      : internal_methods_(internal_methods) {}

 private:
  PrimExpr VisitExpr_(const CallNode* op) override {
    auto node = Downcast<Call>(StmtExprMutator::VisitExpr_(op));

    if (auto* ptr = node->op.as<GlobalVarNode>()) {
      auto gvar = GetRef<GlobalVar>(ptr);
      if (auto it = internal_methods_.find(gvar); it != internal_methods_.end()) {
        const auto& param_info = it->second;

        ICHECK_EQ(param_info.size(), op->args.size())
            << "Internal error, "
            << "argument/parameter count mismatch "
            << "should be caught during StaticArgumentCollector";

        Array<PrimExpr> args;
        bool made_change = false;
        for (size_t i = 0; i < op->args.size(); i++) {
          if (param_info[i].UniqueStaticArgument()) {
            made_change = true;
          } else {
            args.push_back(op->args[i]);
          }
        }
        if (made_change) {
          node.CopyOnWrite()->args = args;
        }
      }
    }

    return std::move(node);
  }

  const std::unordered_map<GlobalVar, std::vector<ParamInfo>, ObjectPtrHash, ObjectPtrEqual>&
      internal_methods_;
};

namespace transform {

Pass InlineStaticArguments() {
  auto pass_func = [](IRModule mod, PassContext ctx) -> IRModule {
    std::unordered_map<GlobalVar, std::vector<ParamInfo>, ObjectPtrHash, ObjectPtrEqual>
        internal_methods;
    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto* prim_func = base_func.as<PrimFuncNode>()) {
        auto global_symbol = prim_func->GetAttr<String>(tvm::attr::kGlobalSymbol);
        if (!global_symbol) {
          std::vector<ParamInfo> param_info;
          for (const auto& var : prim_func->params) {
            param_info.push_back(ParamInfo(var));
          }
          internal_methods[gvar] = std::move(param_info);
        }
      }
    }

    if (internal_methods.empty()) {
      return mod;
    }

    StaticArgumentCollector collector(internal_methods);
    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto* prim_func = base_func.as<PrimFuncNode>()) {
        collector(prim_func->body);
      }
    }

    Map<GlobalVar, BaseFunc> updated_callee;
    bool made_change = false;
    for (auto [gvar, func] : mod->functions) {
      if (auto* ptr = func.as<PrimFuncNode>()) {
        auto prim_func = GetRef<PrimFunc>(ptr);

        Map<Var, ObjectRef> specialize_map;
        for (const auto& param : internal_methods[gvar]) {
          if (auto unique_arg = param.UniqueStaticArgument()) {
            specialize_map.Set(param.var, unique_arg.value());
          }
        }
        if (specialize_map.size()) {
          func = Specialize(prim_func, specialize_map);
          made_change = true;
        }
      }

      updated_callee.Set(gvar, func);
    }

    if (!made_change) {
      return mod;
    }

    StaticArgumentRemover mutator(internal_methods);
    Map<GlobalVar, BaseFunc> updated_callsite;
    for (auto [gvar, func] : updated_callee) {
      if (auto* ptr = func.as<PrimFuncNode>()) {
        auto prim_func = GetRef<PrimFunc>(ptr);
        prim_func.CopyOnWrite()->body = mutator(prim_func->body);
        func = prim_func;
      }
      updated_callsite.Set(gvar, func);
    }

    mod.CopyOnWrite()->functions = updated_callsite;
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tir.InlineStaticArguments", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InlineStaticArguments").set_body_typed(InlineStaticArguments);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
