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
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/expr_functor.h>

#include <optional>

#include "../../../tir/analysis/stmt_to_primfunc.h"
#include "../../../tir/schedule/utils.h"
#include "../../../tir/transforms/ir_utils.h"
#include "../../op/op_common.h"
#include "../primitive.h"

namespace tvm {
namespace relax {

namespace {
struct VarUseVisitor : public ExprVisitor {
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> vars_used;

  void VisitExpr_(const VarNode* op) final { vars_used.insert(GetRef<Var>(op)); }
};

class BufferParamSubstitute : public tir::StmtExprMutator {
 public:
  static tir::Stmt Apply(tir::Stmt stmt, Map<tir::Buffer, tir::Buffer> buffer_remap,
                         Map<tir::Var, tir::Var> var_remap) {
    BufferParamSubstitute mutator(buffer_remap, var_remap);
    return mutator(stmt);
  }

 private:
  using Parent = tir::StmtExprMutator;

  BufferParamSubstitute(Map<tir::Buffer, tir::Buffer> buffer_remap,
                        Map<tir::Var, tir::Var> var_remap_extern)
      : buffer_remap_(buffer_remap), var_remap_extern_(var_remap_extern) {
    auto define_var_remap = [this](const PrimExpr& expr_before, const PrimExpr& expr_after) {
      auto* ptr = expr_before.as<tir::VarNode>();
      if (!ptr) return;

      tir::Var var_before = GetRef<tir::Var>(ptr);
      tir::Var var_after = Downcast<tir::Var>(expr_after);
      var_remap_.Set(var_before, var_after);
    };

    for (const auto& [buf_before, buf_after] : buffer_remap) {
      ICHECK_EQ(buf_before->shape.size(), buf_after->shape.size());
      for (size_t i = 0; i < buf_before->shape.size(); i++) {
        define_var_remap(buf_before->shape[i], buf_after->shape[i]);
      }
      ICHECK_EQ(buf_before->strides.size(), buf_after->strides.size());
      for (size_t i = 0; i < buf_before->strides.size(); i++) {
        define_var_remap(buf_before->strides[i], buf_after->strides[i]);
      }
      define_var_remap(buf_before->elem_offset, buf_after->elem_offset);
    }
  }

  PrimExpr VisitExpr_(const tir::BufferLoadNode* op) override {
    auto node = Downcast<tir::BufferLoad>(Parent::VisitExpr_(op));
    return ReplaceBuffer(std::move(node));
  }

  tir::Stmt VisitStmt_(const tir::BufferStoreNode* op) override {
    auto node = Downcast<tir::BufferStore>(Parent::VisitStmt_(op));
    return ReplaceBuffer(std::move(node));
  }
  tir::Stmt VisitStmt_(const tir::DeclBufferNode* op) override {
    auto node = Downcast<tir::DeclBuffer>(Parent::VisitStmt_(op));
    return ReplaceBuffer(std::move(node));
  }

  template <typename Node>
  Node ReplaceBuffer(Node node) {
    if (auto opt = buffer_remap_.Get(node->buffer)) {
      node.CopyOnWrite()->buffer = opt.value();
    }
    return node;
  }

  tir::Stmt VisitStmt_(const tir::BlockNode* op) override {
    auto node = Downcast<tir::Block>(Parent::VisitStmt_(op));

    auto visit_region = [this](tir::BufferRegion reg) {
      if (auto opt = buffer_remap_.Get(reg->buffer)) {
        reg.CopyOnWrite()->buffer = opt.value();
      }
      return reg;
    };
    auto visit_match_buffer = [&visit_region](tir::MatchBufferRegion match) {
      auto new_source = visit_region(match->source);
      if (!new_source.same_as(match->source)) {
        match.CopyOnWrite()->source = std::move(new_source);
      }
      return match;
    };

    if (auto reads = node->reads.Map(visit_region); !reads.same_as(node->reads)) {
      node.CopyOnWrite()->reads = std::move(reads);
    }
    if (auto writes = node->writes.Map(visit_region); !writes.same_as(node->writes)) {
      node.CopyOnWrite()->writes = std::move(writes);
    }
    if (auto match_buffers = node->match_buffers.Map(visit_match_buffer);
        !match_buffers.same_as(node->match_buffers)) {
      node.CopyOnWrite()->match_buffers = std::move(match_buffers);
    }

    return std::move(node);
  }

  PrimExpr VisitExpr_(const tir::VarNode* op) override {
    if (auto opt = var_remap_.Get(GetRef<tir::Var>(op))) {
      return opt.value();
    } else {
      return Parent::VisitExpr_(op);
    }
  }

  Map<tir::Buffer, tir::Buffer> buffer_remap_;
  Map<tir::Var, tir::Var> var_remap_;
  Map<tir::Var, tir::Var> var_remap_extern_;
};

/*! \brief Utility class to count the number of usages of each TIR
 *  PrimFunc in a module
 */
class CallTIRUsageCounter : ExprVisitor {
 public:
  static Map<GlobalVar, IntImm> Count(const IRModule& mod) {
    CallTIRUsageCounter counter;
    for (const auto& [global_var, expr] : mod->functions) {
      if (expr.as<relax::FunctionNode>()) {
        counter.VisitExpr(expr);
      }
    }
    return counter.counts_;
  }

 private:
  void VisitExpr_(const CallNode* op) override {
    ExprVisitor::VisitExpr_(op);

    auto gvar_pattern = Wildcard();
    auto pattern = IsOp("relax.call_tir")(gvar_pattern, Wildcard());
    if (auto opt = ExtractMatchedExpr(pattern, GetRef<Call>(op))) {
      GlobalVar gvar = Downcast<GlobalVar>(opt.value()[gvar_pattern]);
      int64_t usage = 1;
      if (auto prev = counts_.Get(gvar)) {
        usage += prev.value()->value;
      }
      counts_.Set(gvar, IntImm(DataType::Int(64), usage));
    }
  }

  Map<GlobalVar, IntImm> counts_;
};

class FuseTIRMutator : public ExprMutator {
 public:
  static Array<GlobalVar> Apply(tir::ScheduleState self, Array<GlobalVar> to_fuse,
                                String fused_primfunc_name) {
    FuseTIRMutator mutator(self->mod, to_fuse, fused_primfunc_name);
    for (const auto& [global_var, expr] : self->mod->functions) {
      if (auto* ptr = expr.as<relax::FunctionNode>()) {
        auto func = GetRef<relax::Function>(ptr);
        func = Downcast<relax::Function>(mutator.VisitExpr(func));
        mutator.builder_->UpdateFunction(global_var, func);
      }
    }
    auto mod = mutator.builder_->GetContextIRModule();

    auto usage_counts = CallTIRUsageCounter::Count(mod);
    for (const auto& old_primfunc_gv : to_fuse) {
      if (!usage_counts.Get(old_primfunc_gv)) {
        mod->Remove(old_primfunc_gv);
      }
    }

    self->mod = mod;
    return mutator.GetFusedFunctions();
  }

  FuseTIRMutator(IRModule mod, Array<GlobalVar> to_fuse, String fused_primfunc_name)
      : ExprMutator(mod), to_fuse_(to_fuse), fused_primfunc_name_(fused_primfunc_name) {
    for (const auto& [global_var, func] : mod->functions) {
      if (auto* ptr = func.as<tir::PrimFuncNode>()) {
        original_primfuncs_.Set(global_var, GetRef<tir::PrimFunc>(ptr));
      }
    }
  }

  Optional<Function> Apply(Function func) {
    made_change_ = false;
    auto ret = Downcast<Function>(VisitExpr(func));
    if (made_change_) {
      return ret;
    } else {
      return NullOpt;
    }
  }

  Array<GlobalVar> GetFusedFunctions() const {
    std::vector<GlobalVar> sorting(generated_fused_primfuncs_.begin(),
                                   generated_fused_primfuncs_.end());
    std::sort(sorting.begin(), sorting.end());
    return Array<GlobalVar>(sorting.begin(), sorting.end());
  }

 private:
  Expr VisitExpr_(const SeqExprNode* op) final {
    auto node = Downcast<SeqExpr>(ExprMutator::VisitExpr_(op));

    Array<BindingBlock> blocks = node->blocks;
    for (size_t i = 0; i < blocks.size(); i++) {
      const Array<Binding>& old_bindings = blocks[i]->bindings;
      Array<Binding> new_bindings;

      size_t j = 0;
      while (j < old_bindings.size()) {
        bool is_fuseable = [&]() -> bool {
          // Ensure that we have enough remaining functions to fuse.
          if (old_bindings.size() - j < to_fuse_.size()) {
            return false;
          }

          for (size_t k = 0; k < to_fuse_.size(); k++) {
            const Binding& old_binding = old_bindings[j + k];
            auto* var_binding = old_binding.as<VarBindingNode>();
            if (!var_binding) {
              return false;
            }
            auto pattern = IsCallTIR(to_fuse_[k]->name_hint);
            if (!MatchExpr(pattern, var_binding->value)) {
              return false;
            }
          }

          return true;
        }();

        if (is_fuseable) {
          // Collect variable usage that occurs after the fused
          // call_tir.  This will be used to determine whether a
          // binding should be internal to the fused PrimFunc.
          VarUseVisitor used_after_binding;
          for (size_t k = j + to_fuse_.size(); k < old_bindings.size(); k++) {
            used_after_binding.VisitBinding(old_bindings[k]);
          }
          for (size_t k = i + 1; k < blocks.size(); k++) {
            used_after_binding.VisitBindingBlock(blocks[k]);
          }
          used_after_binding.VisitExpr(node->body);

          Array<VarBinding> fuseable;
          Array<Var> output_vars;
          // Collect the bindings to be fused together
          for (size_t k = 0; k < to_fuse_.size(); k++) {
            auto binding = Downcast<VarBinding>(old_bindings[j + k]);
            fuseable.push_back(binding);
            if (used_after_binding.vars_used.count(binding->var)) {
              output_vars.push_back(binding->var);
            }
          }

          for (const auto& new_binding : FuseBindings(fuseable, output_vars)) {
            new_bindings.push_back(new_binding);
          }
          j += to_fuse_.size();
        } else {
          new_bindings.push_back(old_bindings[j]);
          j++;
        }
      }

      if (old_bindings.size() > new_bindings.size()) {
        auto block = blocks[i];
        block.CopyOnWrite()->bindings = new_bindings;
        blocks.Set(i, block);
      }
    }

    if (!blocks.same_as(node->blocks)) {
      made_change_ = true;
      node.CopyOnWrite()->blocks = std::move(blocks);
    }
    return std::move(node);
  }

  Array<VarBinding> FuseBindings(Array<VarBinding> bindings, Array<Var> output_vars) {
    Array<Expr> fused_args;
    Array<tir::Buffer> fused_buffer_params;
    Array<tir::Buffer> fused_output_buffer_params;
    Array<tir::Stmt> primfunc_bodies;
    Array<tir::Buffer> root_alloc_buffers;

    std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> output_var_lookup(output_vars.begin(),
                                                                              output_vars.end());
    Map<tir::Var, tir::Var> previous_shape_var;
    Map<Expr, tir::Buffer> previous_buffer_representation;
    Array<tir::Buffer> internal_allocations;

    auto get_relax_shape = [](const relax::Expr& expr) {
      auto sinfo = GetStructInfo(expr);
      auto tensor_sinfo = Downcast<TensorStructInfo>(sinfo);
      auto shape_expr = Downcast<ShapeExpr>(tensor_sinfo->shape);
      return shape_expr->values;
    };

    for (const auto& binding : bindings) {
      auto call = Downcast<Call>(binding->value);
      auto primfunc_gv = Downcast<GlobalVar>(call->args[0]);
      auto primfunc = [&]() {
        auto opt = original_primfuncs_.Get(primfunc_gv);
        CHECK(opt) << "GlobalVar " << primfunc_gv
                   << " should refer to a PrimFunc within the current IRModule, "
                   << "but no such PrimFunc found";
        // In case the sequence to be fused contains repeated entires
        // (e.g. Fusing adjacent calls to "add"), ensure that the
        // resulting bodies do not share variable/buffer definitions.
        return ReplaceAllVariables(opt.value());
      }();

      Array<Expr> args = Downcast<Tuple>(call->args[1])->fields;
      Map<tir::Buffer, tir::Buffer> buffer_remap;
      Map<tir::Var, tir::Var> var_remap;

      auto define_var_remap = [&](const PrimExpr& expr_before, const PrimExpr& expr_after) {
        if (expr_before.same_as(expr_after)) return;

        auto* ptr = expr_before.as<tir::VarNode>();
        if (!ptr) return;

        tir::Var var_before = GetRef<tir::Var>(ptr);
        tir::Var var_after = Downcast<tir::Var>(expr_after);
        var_remap.Set(var_before, var_after);
      };

      auto define_buffer_remap = [&](const tir::Buffer& buf_before, const tir::Buffer& buf_after) {
        if (buf_before.same_as(buf_after)) return;

        ICHECK_EQ(buf_before->shape.size(), buf_after->shape.size());
        for (size_t i = 0; i < buf_before->shape.size(); i++) {
          define_var_remap(buf_before->shape[i], buf_after->shape[i]);
        }
        ICHECK_EQ(buf_before->strides.size(), buf_after->strides.size());
        for (size_t i = 0; i < buf_before->strides.size(); i++) {
          define_var_remap(buf_before->strides[i], buf_after->strides[i]);
        }
        define_var_remap(buf_before->elem_offset, buf_after->elem_offset);

        buffer_remap.Set(buf_before, buf_after);
      };

      auto get_param_buf = [&](size_t i_arg, const auto& arg) -> tir::Buffer {
        tir::Var param_var = primfunc->params[i_arg];
        auto opt = primfunc->buffer_map.Get(param_var);
        CHECK(opt) << "Relax expr " << arg << " should be an input parameter to PrimFunc "
                   << primfunc_gv << ", but tir::Var parameter " << param_var
                   << " did not appear in the PrimFunc's buffer_map";
        auto param_buf = opt.value();

        Map<tir::Var, PrimExpr> var_to_expr;
        for (const auto& [old_var, new_var] : var_remap) {
          var_to_expr.Set(old_var, new_var);
        }

        auto updated_shape = param_buf->shape.Map(
            [&](const PrimExpr& dim) { return tir::Substitute(dim, var_to_expr); });

        if (!updated_shape.same_as(param_buf->shape)) {
          auto updated_buf = param_buf;
          updated_buf.CopyOnWrite()->shape = updated_shape;
          define_buffer_remap(param_buf, updated_buf);
          param_buf = updated_buf;
        }
        return param_buf;
      };

      // The fused PrimFunc should be passed arguments that went to
      // any one of the input functions.
      for (size_t i_arg = 0; i_arg < args.size(); i_arg++) {
        const auto& arg = args[i_arg];
        tir::Buffer param_buf = get_param_buf(i_arg, arg);

        if (auto opt = previous_buffer_representation.Get(arg)) {
          // If the same argument is passed to two functions being
          // fused together, then the fused function should only
          // require the argument to be passed once.  In addition, the
          // fused body should only contain a single tir::Buffer
          // object, even though each of the function bodies initially
          // defined their own independent tir::Buffer object.
          define_buffer_remap(param_buf, opt.value());
          continue;
        }

        // A relax-level dynamic shape argument may appear in the
        // shape of arguments being passed into multiple functions
        // that are being fused together.  In the fused function,
        // these should be treated as a single dynamic shape
        // parameter, rather than two independent shape parameters.
        const Array<PrimExpr>& relax_shape = get_relax_shape(arg);
        ICHECK_EQ(relax_shape.size(), param_buf->shape.size());
        for (size_t i = 0; i < relax_shape.size(); i++) {
          auto* relax_ptr = relax_shape[i].as<tir::VarNode>();
          auto* buffer_ptr = param_buf->shape[i].as<tir::VarNode>();
          if (relax_ptr && buffer_ptr) {
            auto relax_var = GetRef<tir::Var>(relax_ptr);
            auto buffer_var = GetRef<tir::Var>(buffer_ptr);
            if (auto opt = previous_shape_var.Get(relax_var)) {
              if (!buffer_var.same_as(opt.value())) {
                var_remap.Set(buffer_var, opt.value());
              }
            } else {
              previous_shape_var.Set(relax_var, buffer_var);
            }
          }
        }

        fused_args.push_back(arg);
        fused_buffer_params.push_back(param_buf);
        previous_buffer_representation.Set(arg, param_buf);
      }

      if (call->sinfo_args.size() == 1) {
        // Later use of this output buffer should read from the buffer
        // that is generated here.
        ICHECK(!previous_buffer_representation.Get(binding->var))
            << "Variable binding of " << binding->var << " occurred as an input "
            << "before the call to " << primfunc_gv << " that produces " << binding->var
            << " as an output";
        auto output_sinfo = Downcast<TensorStructInfo>(call->sinfo_args[0]);
        auto i_arg = args.size();
        auto param_buf = get_param_buf(i_arg, binding->var);
        previous_buffer_representation.Set(binding->var, param_buf);

        if (output_var_lookup.count(binding->var)) {
          fused_output_buffer_params.push_back(param_buf);
        } else {
          // Outputs of the fused PrimFunc are still implicitly
          // allocated by the relax language definition.  If a
          // producer and all consumers are fused together, then the
          // producer's output is no longer exposed into relax, and
          // must have an explicit TIR allocation to replace the
          // previous implicit relax allocation.
          internal_allocations.push_back(param_buf);
        }
      } else {
        // To support multiple outputs, will need to change the
        // previous_representation Map to use structural hash/equality
        // instead of ptr hash/equality.  Otherwise, the previous
        // representation of `TupleGetItem(binding->var, i)` would
        // fail to be de-duped.
        LOG(FATAL) << "Fusing of TIR PrimFuncs with multiple outputs not yet supported";
      }

      auto body = primfunc->body;
      if (auto* ptr = body.as<tir::BlockRealizeNode>()) {
        // Root block may not have iterators, reads, writes, or
        // match_buffer, so we only need to collect allocations owned
        // by each root block.
        for (const auto& buf : ptr->block->alloc_buffers) {
          root_alloc_buffers.push_back(buf);
        }
        body = ptr->block->body;
      }
      if (buffer_remap.size()) {
        body = BufferParamSubstitute::Apply(body, buffer_remap, var_remap);
      }
      primfunc_bodies.push_back(body);
    }

    for (const auto& buf : fused_output_buffer_params) {
      fused_buffer_params.push_back(buf);
    }
    for (const auto& buf : internal_allocations) {
      root_alloc_buffers.push_back(buf);
    }

    tir::Stmt fused_body = tir::BlockRealize(
        /* iter_values = */ {}, /* predicate = */ Bool(true),
        tir::Block(/* iter_vars = */ {}, /* reads = */ {}, /* writes = */ {}, "root",
                   tir::SeqStmt(primfunc_bodies), /* init = */ NullOpt, root_alloc_buffers));

    auto fused_params = fused_buffer_params.Map(
        [](const auto& buf) { return tir::Var(buf->name + "_handle", DataType::Handle()); });
    Map<tir::Var, tir::Buffer> fused_buffer_map;
    for (size_t i = 0; i < fused_buffer_params.size(); i++) {
      fused_buffer_map.Set(fused_params[i], fused_buffer_params[i]);
    }

    tir::PrimFunc fused_primfunc(fused_params, fused_body, VoidType(), fused_buffer_map);
    fused_primfunc->struct_info_ = PrimFuncSignature(fused_primfunc);

    auto fused_id = builder_->AddFunction(fused_primfunc, fused_primfunc_name_);
    generated_fused_primfuncs_.insert(fused_id);

    Array<TensorStructInfo> output_sinfo;
    for (const auto& var : output_vars) {
      output_sinfo.push_back(Downcast<TensorStructInfo>(GetStructInfo(var)));
    }
    auto call_tir = MakeCallTIR(fused_id, Tuple(fused_args), output_sinfo);

    if (output_vars.size() == 1) {
      return {VarBinding(output_vars[0], call_tir)};
    } else {
      TupleStructInfo tuple_sinfo(
          output_sinfo.Map([](const auto& info) -> StructInfo { return info; }));
      Var tuple_var(fused_id->name_hint + "_output", tuple_sinfo);

      Array<VarBinding> output;
      output.push_back(VarBinding(tuple_var, call_tir));
      for (size_t i = 0; i < output_vars.size(); i++) {
        output.push_back(VarBinding(output_vars[i], TupleGetItem(tuple_var, i)));
      }
      return output;
    }
  }

  Array<GlobalVar> to_fuse_;
  String fused_primfunc_name_;
  bool made_change_{false};
  Map<GlobalVar, tir::PrimFunc> original_primfuncs_;
  std::unordered_set<GlobalVar, ObjectPtrHash, ObjectPtrEqual> generated_fused_primfuncs_;
};
}  // namespace

Array<GlobalVar> FuseTIR(tir::ScheduleState self, Array<GlobalVar> to_fuse,
                         String fused_primfunc_name) {
  return FuseTIRMutator::Apply(self, to_fuse, fused_primfunc_name);
}

}  // namespace relax
}  // namespace tvm
