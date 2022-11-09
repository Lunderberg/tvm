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
 * \file tvm/tir/universal_visitor.h
 */

#include <tvm/ir/module.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>

#include <type_traits>

#ifndef TVM_TIR_UNIVERSAL_VISITOR_H_
#define TVM_TIR_UNIVERSAL_VISITOR_H_

namespace tvm {
namespace tir {

enum class VisitorFlag : int {
  None = 0,
  Mutate = 1,
};

constexpr VisitorFlag operator&(VisitorFlag lhs, VisitorFlag rhs) {
  return static_cast<VisitorFlag>(static_cast<int>(lhs) & static_cast<int>(rhs));
}
constexpr VisitorFlag operator|(VisitorFlag lhs, VisitorFlag rhs) {
  return static_cast<VisitorFlag>(static_cast<int>(lhs) | static_cast<int>(rhs));
}

template <VisitorFlag flags>
class UniversalVisitor {
  static constexpr bool is_mutator = static_cast<bool>(flags & VisitorFlag::Mutate);

  template <typename T>
  using RetType = std::conditional_t<is_mutator, T, void>;

  template <typename T>
  using ArgType = std::conditional_t<is_mutator, T, const T&>;

  template <typename T>
  auto forward(T& obj) {
    if constexpr (is_mutator) {
      return std::move(obj);
    } else {
      return obj;
    }
  }

 public:
  // Exposing scope-dependent
  virtual std::function<void()> EnterConstraint(const PrimExpr& expr) { return nullptr; }
  virtual std::function<void()> EnterDefinition(const Var& var, const Optional<Range>&) {
    return nullptr;
  }
  virtual std::function<void()> EnterDefinition(const Buffer& buf) { return nullptr; }

  // TODO: Rename these all to "Visit" instead of "VisitExpr",
  // "VisitExpr_", "VisitStmt", and "VisitStmt_".

  // PrimExpr mutators.  In any case where the static type is one of
  // the following, it can be replaced with a PrimExpr.
  virtual RetType<PrimExpr> Visit(ArgType<PrimExpr>);
  virtual RetType<PrimExpr> Visit(ArgType<Var>);
  virtual RetType<PrimExpr> Visit(ArgType<SizeVar>);
  virtual RetType<PrimExpr> Visit(ArgType<BufferLoad>);
  virtual RetType<PrimExpr> Visit(ArgType<ProducerLoad>);
  virtual RetType<PrimExpr> Visit(ArgType<Load>);
  virtual RetType<PrimExpr> Visit(ArgType<Let>);
  virtual RetType<PrimExpr> Visit(ArgType<Call>);
  virtual RetType<PrimExpr> Visit(ArgType<Add>);
  virtual RetType<PrimExpr> Visit(ArgType<Sub>);
  virtual RetType<PrimExpr> Visit(ArgType<Mul>);
  virtual RetType<PrimExpr> Visit(ArgType<Div>);
  virtual RetType<PrimExpr> Visit(ArgType<Mod>);
  virtual RetType<PrimExpr> Visit(ArgType<FloorDiv>);
  virtual RetType<PrimExpr> Visit(ArgType<FloorMod>);
  virtual RetType<PrimExpr> Visit(ArgType<Min>);
  virtual RetType<PrimExpr> Visit(ArgType<Max>);
  virtual RetType<PrimExpr> Visit(ArgType<EQ>);
  virtual RetType<PrimExpr> Visit(ArgType<NE>);
  virtual RetType<PrimExpr> Visit(ArgType<LT>);
  virtual RetType<PrimExpr> Visit(ArgType<LE>);
  virtual RetType<PrimExpr> Visit(ArgType<GT>);
  virtual RetType<PrimExpr> Visit(ArgType<GE>);
  virtual RetType<PrimExpr> Visit(ArgType<And>);
  virtual RetType<PrimExpr> Visit(ArgType<Or>);
  virtual RetType<PrimExpr> Visit(ArgType<Reduce>);
  virtual RetType<PrimExpr> Visit(ArgType<Cast>);
  virtual RetType<PrimExpr> Visit(ArgType<Not>);
  virtual RetType<PrimExpr> Visit(ArgType<Select>);
  virtual RetType<PrimExpr> Visit(ArgType<Ramp>);
  virtual RetType<PrimExpr> Visit(ArgType<Broadcast>);
  virtual RetType<PrimExpr> Visit(ArgType<Shuffle>);
  virtual RetType<PrimExpr> Visit(ArgType<IntImm>);
  virtual RetType<PrimExpr> Visit(ArgType<FloatImm>);
  virtual RetType<PrimExpr> Visit(ArgType<StringImm>);
  virtual RetType<PrimExpr> Visit(ArgType<Any>);

  // Stmt mutators.  In any case where the static type is one of
  // the following, it can be replaced with a Stmt.
  virtual RetType<Stmt> Visit(ArgType<Stmt>);
  virtual RetType<Stmt> Visit(ArgType<LetStmt>);
  virtual RetType<Stmt> Visit(ArgType<AttrStmt>);
  virtual RetType<Stmt> Visit(ArgType<IfThenElse>);
  virtual RetType<Stmt> Visit(ArgType<For>);
  virtual RetType<Stmt> Visit(ArgType<While>);
  virtual RetType<Stmt> Visit(ArgType<Allocate>);
  virtual RetType<Stmt> Visit(ArgType<AllocateConst>);
  virtual RetType<Stmt> Visit(ArgType<DeclBuffer>);
  virtual RetType<Stmt> Visit(ArgType<Store>);
  virtual RetType<Stmt> Visit(ArgType<BufferStore>);
  virtual RetType<Stmt> Visit(ArgType<BufferRealize>);
  virtual RetType<Stmt> Visit(ArgType<AssertStmt>);
  virtual RetType<Stmt> Visit(ArgType<ProducerStore>);
  virtual RetType<Stmt> Visit(ArgType<ProducerRealize>);
  virtual RetType<Stmt> Visit(ArgType<Prefetch>);
  virtual RetType<Stmt> Visit(ArgType<SeqStmt>);
  virtual RetType<Stmt> Visit(ArgType<Evaluate>);
  virtual RetType<Stmt> Visit(ArgType<BlockRealize>);

  // Type-specific mutators.  In cases where the return type must be
  // the same as the mutated type.  For example, every BufferLoad must
  // contain a Buffer, and every BlockRealize must contain a Block.
  // It would not be valid to replace
  virtual RetType<IRModule> Visit(ArgType<IRModule>);
  virtual RetType<PrimFunc> Visit(ArgType<PrimFunc>);
  virtual RetType<Block> Visit(ArgType<Block>);
  virtual RetType<Buffer> Visit(ArgType<Buffer>);
  virtual RetType<BufferRegion> Visit(ArgType<BufferRegion>);
  virtual RetType<MatchBufferRegion> Visit(ArgType<MatchBufferRegion>);
  virtual RetType<IterVar> Visit(ArgType<IterVar>);
  virtual RetType<Range> Visit(ArgType<Range>);

  // VarDef and Integer have type-specific mutators, but deserve
  // special mention.  They are subtypes of Var/IntImm, respectively,
  // but represent cases where the more specific type is required.  In
  // cases where Var/IntImm are used as PrimExprs, they may be
  // replaced with any other PrimExpr.  However, there are cases where
  // a variable *must* be present to define the values, or where a
  // value *must* be a literal integer.  In these cases, the more
  // specific child class may be used.

  /* virtual RetType<VarDef> Visit(ArgType<VarDef>); */
  /* virtual RetType<Integer> Visit(ArgType<Integer>); */

  // Containers
  template <typename T, typename = std::enable_if_t<std::is_same_v<ArgType<Array<T>>, Array<T>>>>
  Array<T> Visit(Array<T> arr) {
    auto mutate_func = [this](T t) { return Visit(std::move(t)); };
    if (arr.unique()) {
      arr.MutateByApply(mutate_func);
      return std::move(arr);
    } else {
      return arr.Map(mutate_func);
    }
  }

  template <typename T,
            typename = std::enable_if_t<std::is_same_v<ArgType<Array<T>>, const Array<T>&>>>
  void Visit(const Array<T>& arr) {
    for (const auto& element : arr) {
      Visit(element);
    }
  }

  template <typename T>
  RetType<Optional<T>> Visit(ArgType<Optional<T>> opt) {
    if constexpr (is_mutator) {
      if (!opt.has_value()) {
        return NullOpt;
      } else if (opt.unique()) {
        T val = opt.value();
        opt = NullOpt;
        return Visit(std::move(val));
      } else {
        return Visit(opt.value());
      }
    } else {
      if (opt.has_value()) {
        Visit(opt.value());
      }
    }
  }

 private:
#define TVM_UNIVERSAL_VISITOR_DISPATCH(ObjType, RefType)                                 \
  vtable.template set_dispatch<ObjType>([](const ObjectRef& node, decltype(this) self) { \
    const Object* ptr = node.get();                                                      \
    const ObjType* add_ptr = static_cast<const ObjType*>(ptr);                           \
    RefType add_ref = GetRef<RefType>(add_ptr);                                          \
    return self->Visit(add_ref);                                                         \
  })

  RetType<PrimExpr> Dispatch(ArgType<PrimExpr> expr) {
    using FType = NodeFunctor<RetType<PrimExpr>(const ObjectRef&, decltype(this))>;
    static FType vtable = []() -> FType {
      FType vtable;
      TVM_UNIVERSAL_VISITOR_DISPATCH(VarNode, Var);
      TVM_UNIVERSAL_VISITOR_DISPATCH(SizeVarNode, SizeVar);
      TVM_UNIVERSAL_VISITOR_DISPATCH(BufferLoadNode, BufferLoad);
      TVM_UNIVERSAL_VISITOR_DISPATCH(ProducerLoadNode, ProducerLoad);
      TVM_UNIVERSAL_VISITOR_DISPATCH(LoadNode, Load);
      TVM_UNIVERSAL_VISITOR_DISPATCH(LetNode, Let);
      TVM_UNIVERSAL_VISITOR_DISPATCH(CallNode, Call);
      TVM_UNIVERSAL_VISITOR_DISPATCH(AddNode, Add);
      TVM_UNIVERSAL_VISITOR_DISPATCH(SubNode, Sub);
      TVM_UNIVERSAL_VISITOR_DISPATCH(MulNode, Mul);
      TVM_UNIVERSAL_VISITOR_DISPATCH(DivNode, Div);
      TVM_UNIVERSAL_VISITOR_DISPATCH(ModNode, Mod);
      TVM_UNIVERSAL_VISITOR_DISPATCH(FloorDivNode, FloorDiv);
      TVM_UNIVERSAL_VISITOR_DISPATCH(FloorModNode, FloorMod);
      TVM_UNIVERSAL_VISITOR_DISPATCH(MinNode, Min);
      TVM_UNIVERSAL_VISITOR_DISPATCH(MaxNode, Max);
      TVM_UNIVERSAL_VISITOR_DISPATCH(EQNode, EQ);
      TVM_UNIVERSAL_VISITOR_DISPATCH(NENode, NE);
      TVM_UNIVERSAL_VISITOR_DISPATCH(LTNode, LT);
      TVM_UNIVERSAL_VISITOR_DISPATCH(LENode, LE);
      TVM_UNIVERSAL_VISITOR_DISPATCH(GTNode, GT);
      TVM_UNIVERSAL_VISITOR_DISPATCH(GENode, GE);
      TVM_UNIVERSAL_VISITOR_DISPATCH(AndNode, And);
      TVM_UNIVERSAL_VISITOR_DISPATCH(OrNode, Or);
      TVM_UNIVERSAL_VISITOR_DISPATCH(ReduceNode, Reduce);
      TVM_UNIVERSAL_VISITOR_DISPATCH(CastNode, Cast);
      TVM_UNIVERSAL_VISITOR_DISPATCH(NotNode, Not);
      TVM_UNIVERSAL_VISITOR_DISPATCH(SelectNode, Select);
      TVM_UNIVERSAL_VISITOR_DISPATCH(RampNode, Ramp);
      TVM_UNIVERSAL_VISITOR_DISPATCH(BroadcastNode, Broadcast);
      TVM_UNIVERSAL_VISITOR_DISPATCH(ShuffleNode, Shuffle);
      TVM_UNIVERSAL_VISITOR_DISPATCH(IntImmNode, IntImm);
      TVM_UNIVERSAL_VISITOR_DISPATCH(FloatImmNode, FloatImm);
      TVM_UNIVERSAL_VISITOR_DISPATCH(StringImmNode, StringImm);
      TVM_UNIVERSAL_VISITOR_DISPATCH(AnyNode, Any);
      return vtable;
    }();
    return vtable(expr, this);
  }

  RetType<Stmt> Dispatch(ArgType<Stmt> expr) {
    using FType = NodeFunctor<RetType<Stmt>(const ObjectRef&, decltype(this))>;
    static FType vtable = []() -> FType {
      FType vtable;
      TVM_UNIVERSAL_VISITOR_DISPATCH(LetStmtNode, LetStmt);
      TVM_UNIVERSAL_VISITOR_DISPATCH(AttrStmtNode, AttrStmt);
      TVM_UNIVERSAL_VISITOR_DISPATCH(IfThenElseNode, IfThenElse);
      TVM_UNIVERSAL_VISITOR_DISPATCH(ForNode, For);
      TVM_UNIVERSAL_VISITOR_DISPATCH(WhileNode, While);
      TVM_UNIVERSAL_VISITOR_DISPATCH(AllocateNode, Allocate);
      TVM_UNIVERSAL_VISITOR_DISPATCH(AllocateConstNode, AllocateConst);
      TVM_UNIVERSAL_VISITOR_DISPATCH(DeclBufferNode, DeclBuffer);
      TVM_UNIVERSAL_VISITOR_DISPATCH(StoreNode, Store);
      TVM_UNIVERSAL_VISITOR_DISPATCH(BufferStoreNode, BufferStore);
      TVM_UNIVERSAL_VISITOR_DISPATCH(BufferRealizeNode, BufferRealize);
      TVM_UNIVERSAL_VISITOR_DISPATCH(AssertStmtNode, AssertStmt);
      TVM_UNIVERSAL_VISITOR_DISPATCH(ProducerStoreNode, ProducerStore);
      TVM_UNIVERSAL_VISITOR_DISPATCH(ProducerRealizeNode, ProducerRealize);
      TVM_UNIVERSAL_VISITOR_DISPATCH(PrefetchNode, Prefetch);
      TVM_UNIVERSAL_VISITOR_DISPATCH(SeqStmtNode, SeqStmt);
      TVM_UNIVERSAL_VISITOR_DISPATCH(EvaluateNode, Evaluate);
      TVM_UNIVERSAL_VISITOR_DISPATCH(BlockRealizeNode, BlockRealize);
      return vtable;
    }();
    return vtable(expr, this);
  }

#undef TVM_UNIVERSAL_VISITOR_DISPATCH
};

template <VisitorFlag flags>
typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
    ArgType<PrimExpr> expr) {
  return Dispatch(std::move(expr));
}

template <VisitorFlag flags>
typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
    ArgType<Var> var) {
  return static_cast<RetType<PrimExpr>>(std::move(var));
}

template <VisitorFlag flags>
typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
    ArgType<SizeVar> size_var) {
  ArgType<Var> var = std::move(size_var);
  return Visit(std::move(var));
}

template <VisitorFlag flags>
typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
    ArgType<BufferLoad> buffer_load) {
  if constexpr (!is_mutator) {
    Visit(buffer_load->buffer);
    Visit(buffer_load->indices);
  } else if (buffer_load.unique()) {
    auto* node = buffer_load.CopyOnWrite();
    node->buffer = Visit(std::move(node->buffer));
    node->indices = Visit(std::move(node->indices));
    return std::move(buffer_load);
  } else {
    auto buf = Visit(buffer_load->buffer);
    auto indices = Visit(buffer_load->indices);
    if (buf.same_as(buffer_load->buffer) && indices.same_as(buffer_load->indices)) {
      return std::move(buffer_load);
    } else {
      return BufferLoad(buf, indices);
    }
  }
}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<ProducerLoad> producer_load) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<Let> let) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<Call> call) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<Add> add) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<Sub> sub) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<Mul> mul) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<Div> div) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<Mod> mod) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<FloorDiv> floor_div) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<FloorMod> floor_mod) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<Min> min_node) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<Max> max_node) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<EQ> eq) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<NE> ne) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<LT> lt) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<LE> le) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<GT> gt) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<GE> ge) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<And> and_node) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<Or> or_node) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<Reduce> reduce) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<Cast> cast) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<Not> not_node) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<Select> select) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<Ramp> ramp) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<Broadcast> broadcast) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<Shuffle> shuffle) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<IntImm> int_node) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<FloatImm> float_node) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<StringImm> string) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimExpr> UniversalVisitor<flags>::Visit(
//     ArgType<Any> any) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<IRModule> UniversalVisitor<flags>::Visit(
//     ArgType<IRModule> ir_module) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<PrimFunc> UniversalVisitor<flags>::Visit(
//     ArgType<PrimFunc> prim_func) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<Stmt> UniversalVisitor<flags>::Visit(
//     ArgType<Stmt> stmt) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<Stmt> UniversalVisitor<flags>::Visit(
//     ArgType<LetStmt> let_stmt) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<Stmt> UniversalVisitor<flags>::Visit(
//     ArgType<AttrStmt> attr_stmt) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<Stmt> UniversalVisitor<flags>::Visit(
//     ArgType<IfThenElse> if_then_else) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<Stmt> UniversalVisitor<flags>::Visit(
//     ArgType<For> for_node) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<Stmt> UniversalVisitor<flags>::Visit(
//     ArgType<While> while_node) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<Stmt> UniversalVisitor<flags>::Visit(
//     ArgType<Allocate> alloc) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<Stmt> UniversalVisitor<flags>::Visit(
//     ArgType<AllocateConst> alloc_const) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<Stmt> UniversalVisitor<flags>::Visit(
//     ArgType<DeclBuffer> decl_buffer) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<Stmt> UniversalVisitor<flags>::Visit(
//     ArgType<BufferStore> buffer_store) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<Stmt> UniversalVisitor<flags>::Visit(
//     ArgType<BufferRealize> buffer_realize) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<Stmt> UniversalVisitor<flags>::Visit(
//     ArgType<AssertStmt> assert_stmt) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<Stmt> UniversalVisitor<flags>::Visit(
//     ArgType<ProducerStore> producer_store) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<Stmt> UniversalVisitor<flags>::Visit(
//     ArgType<ProducerRealize> producer_realize) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<Stmt> UniversalVisitor<flags>::Visit(
//     ArgType<Prefetch> prefetch) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<Stmt> UniversalVisitor<flags>::Visit(
//     ArgType<SeqStmt> seq_stmt) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<Stmt> UniversalVisitor<flags>::Visit(
//     ArgType<Evaluate> eval) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<Stmt> UniversalVisitor<flags>::Visit(
//     ArgType<BlockRealize> block_realize) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<Block> UniversalVisitor<flags>::Visit(
//     ArgType<Block> block) {}

template <VisitorFlag flags>
typename UniversalVisitor<flags>::template RetType<Buffer> UniversalVisitor<flags>::Visit(
    ArgType<Buffer> buffer) {
  if constexpr (!is_mutator) {
    Visit(buffer->shape);
    Visit(buffer->axis_separators);
    Visit(buffer->strides);
    Visit(buffer->elem_offset);
  } else if (buffer.unique()) {
    auto* node = buffer.CopyOnWrite();
    node->shape = Visit(std::move(node->shape));
    node->axis_separators = Visit(std::move(node->axis_separators));
    node->strides = Visit(std::move(node->strides));
    node->elem_offset = Visit(std::move(node->elem_offset));
    return std::move(buffer);
  } else {
    auto data = Visit(buffer->data);
    auto shape = Visit(buffer->shape);
    auto axis_separators = Visit(buffer->axis_separators);
    auto strides = Visit(buffer->strides);
    auto elem_offset = Visit(buffer->elem_offset);
    // TODO: Enforce uniqueness
    if (data.same_as(buffer->data) && shape.same_as(buffer->shape) &&
        axis_separators.same_as(buffer->axis_separators) && strides.same_as(buffer->strides) &&
        elem_offset.same_as(buffer->elem_offset)) {
      return std::move(buffer);
    } else {
      return Buffer(Downcast<Var>(data), buffer->dtype, shape, strides, elem_offset, buffer->name,
                    buffer->data_alignment, buffer->offset_factor, buffer->buffer_type,
                    axis_separators);
    }
  }
}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<BufferRegion> UniversalVisitor<flags>::Visit(
//     ArgType<BufferRegion> buffer_region) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<MatchBufferRegion>
// UniversalVisitor<flags>::Visit(ArgType<MatchBufferRegion> match_buffer_region) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<IterVar> UniversalVisitor<flags>::Visit(
//     ArgType<IterVar> iter_var) {}

// // template <VisitorFlag flags>
// // typename UniversalVisitor<flags>::template RetType<VarDef>
// // UniversalVisitor<flags>::Visit(ArgType<VarDef> var_def) {}

// // template <VisitorFlag flags>
// // typename UniversalVisitor<flags>::template RetType<Integer>
// // UniversalVisitor<flags>::Visit(ArgType<Integer> int_obj) {}

// template <VisitorFlag flags>
// typename UniversalVisitor<flags>::template RetType<Range> UniversalVisitor<flags>::Visit(
//     ArgType<Range> range) {}

}  // namespace tir
}  // namespace tvm

#endif /* TVM_TIR_UNIVERSAL_VISITOR_H_ */
