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
  using ArgType = std::conditional_t<is_mutator, const T&, T>;

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
  virtual std::function<void()> EnterConstraint(const PrimExpr& expr);
  virtual std::function<void()> EnterDefinition(const Var& var, const Optional<Range>&);
  virtual std::function<void()> EnterDefinition(const Buffer& buf);

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
  /* virtual RetType<BufferLocation> Visit( ArgType<BufferLocation>); */
  virtual RetType<Range> Visit(ArgType<Range>);

  // VarDef and Integer have type-specific mutators, but deserve
  // special mention.  They are subtypes of Var/IntImm, respectively,
  // but represent cases where the more specific type is required.  In
  // cases where Var/IntImm are used as PrimExprs, they may be
  // replaced with any other PrimExpr.  However, there are cases where
  // a variable *must* be present to define the values, or where a
  // value *must* be an integer.  In these cases, the more specific
  // child class may be used.

  /* virtual RetType<VarDef> Visit(ArgType<VarDef>); */
  /* virtual RetType<Integer> Visit(ArgType<Integer>); */

  // Containers
  template <typename T>
  auto Visit(const Array<T>& arr) {
    if constexpr (is_mutator) {
      return arr.Map([this](T t) { return Visit(std::move(t)); });
    } else {
      for (const auto& element : arr) {
        Visit(arr);
      }
    }
  }

  template <typename T>
  auto Visit(const Optional<T>& opt) {
    if constexpr (is_mutator) {
      return opt.has_value() ? Visit(opt.value()) : NullOpt;
    } else {
      if (opt.has_value()) {
        Visit(opt.value());
      }
    }
  }

  // TODO: Remove backwards-compatibility wrappers
  virtual PrimExpr VisitExpr(const PrimExpr& expr);
  virtual PrimExpr VisitExpr_(const VarNode* op);
  virtual PrimExpr VisitExpr_(const SizeVarNode* op);
  virtual PrimExpr VisitExpr_(const BufferLoadNode* op);
  virtual PrimExpr VisitExpr_(const ProducerLoadNode* op);
  virtual PrimExpr VisitExpr_(const LoadNode* op);
  virtual PrimExpr VisitExpr_(const LetNode* op);
  virtual PrimExpr VisitExpr_(const CallNode* op);
  virtual PrimExpr VisitExpr_(const AddNode* op);
  virtual PrimExpr VisitExpr_(const SubNode* op);
  virtual PrimExpr VisitExpr_(const MulNode* op);
  virtual PrimExpr VisitExpr_(const DivNode* op);
  virtual PrimExpr VisitExpr_(const ModNode* op);
  virtual PrimExpr VisitExpr_(const FloorDivNode* op);
  virtual PrimExpr VisitExpr_(const FloorModNode* op);
  virtual PrimExpr VisitExpr_(const MinNode* op);
  virtual PrimExpr VisitExpr_(const MaxNode* op);
  virtual PrimExpr VisitExpr_(const EQNode* op);
  virtual PrimExpr VisitExpr_(const NENode* op);
  virtual PrimExpr VisitExpr_(const LTNode* op);
  virtual PrimExpr VisitExpr_(const LENode* op);
  virtual PrimExpr VisitExpr_(const GTNode* op);
  virtual PrimExpr VisitExpr_(const GENode* op);
  virtual PrimExpr VisitExpr_(const AndNode* op);
  virtual PrimExpr VisitExpr_(const OrNode* op);
  virtual PrimExpr VisitExpr_(const ReduceNode* op);
  virtual PrimExpr VisitExpr_(const CastNode* op);
  virtual PrimExpr VisitExpr_(const NotNode* op);
  virtual PrimExpr VisitExpr_(const SelectNode* op);
  virtual PrimExpr VisitExpr_(const RampNode* op);
  virtual PrimExpr VisitExpr_(const BroadcastNode* op);
  virtual PrimExpr VisitExpr_(const ShuffleNode* op);
  virtual PrimExpr VisitExpr_(const IntImmNode* op);
  virtual PrimExpr VisitExpr_(const FloatImmNode* op);
  virtual PrimExpr VisitExpr_(const StringImmNode* op);
  virtual PrimExpr VisitExpr_(const AnyNode* op);

  virtual Stmt VisitStmt(const Stmt& stmt);
  virtual Stmt VisitStmt_(const LetStmtNode* op);
  virtual Stmt VisitStmt_(const AttrStmtNode* op);
  virtual Stmt VisitStmt_(const IfThenElseNode* op);
  virtual Stmt VisitStmt_(const ForNode* op);
  virtual Stmt VisitStmt_(const WhileNode* op);
  virtual Stmt VisitStmt_(const AllocateNode* op);
  virtual Stmt VisitStmt_(const AllocateConstNode* op);
  virtual Stmt VisitStmt_(const DeclBufferNode* op);
  virtual Stmt VisitStmt_(const StoreNode* op);
  virtual Stmt VisitStmt_(const BufferStoreNode* op);
  virtual Stmt VisitStmt_(const BufferRealizeNode* op);
  virtual Stmt VisitStmt_(const AssertStmtNode* op);
  virtual Stmt VisitStmt_(const ProducerStoreNode* op);
  virtual Stmt VisitStmt_(const ProducerRealizeNode* op);
  virtual Stmt VisitStmt_(const PrefetchNode* op);
  virtual Stmt VisitStmt_(const SeqStmtNode* op);
  virtual Stmt VisitStmt_(const EvaluateNode* op);
  virtual Stmt VisitStmt_(const BlockRealizeNode* op);
};

#define TVM_UNIVERSAL_VISITOR_VISIT_METHOD(Ret, Arg, param)                               \
  template <VisitorFlag flags>                                                            \
  typename UniversalVisitor<flags>::template RetType<Ret> UniversalVisitor<flags>::Visit( \
      UniversalVisitor<flags>::ArgType<Arg> param)

TVM_UNIVERSAL_VISITOR_VISIT_METHOD(PrimExpr, PrimExpr, expr) {
  switch (expr->virtual_type_id()) {
    case VarNode::constexpr_type_id:
      return Visit(Var(forward(expr)));
    case SizeVarNode::constexpr_type_id:
      return Visit(SizeVar(forward(expr)));
    case BufferLoadNode::constexpr_type_id:
      return Visit(BufferLoad(forward(expr)));
    case ProducerLoadNode::constexpr_type_id:
      return Visit(ProducerLoad(forward(expr)));
    case LoadNode::constexpr_type_id:
      return Visit(Load(forward(expr)));
    case LetNode::constexpr_type_id:
      return Visit(Let(forward(expr)));
    case CallNode::constexpr_type_id:
      return Visit(Call(forward(expr)));
    case AddNode::constexpr_type_id:
      return Visit(Add(forward(expr)));
    case SubNode::constexpr_type_id:
      return Visit(Sub(forward(expr)));
    case MulNode::constexpr_type_id:
      return Visit(Mul(forward(expr)));
    case DivNode::constexpr_type_id:
      return Visit(Div(forward(expr)));
    case ModNode::constexpr_type_id:
      return Visit(Mod(forward(expr)));
    case FloorDivNode::constexpr_type_id:
      return Visit(FloorDiv(forward(expr)));
    case FloorModNode::constexpr_type_id:
      return Visit(FloorMod(forward(expr)));
    case MinNode::constexpr_type_id:
      return Visit(Min(forward(expr)));
    case MaxNode::constexpr_type_id:
      return Visit(Max(forward(expr)));
    case EQNode::constexpr_type_id:
      return Visit(EQ(forward(expr)));
    case NENode::constexpr_type_id:
      return Visit(NE(forward(expr)));
    case LTNode::constexpr_type_id:
      return Visit(LT(forward(expr)));
    case LENode::constexpr_type_id:
      return Visit(LE(forward(expr)));
    case GTNode::constexpr_type_id:
      return Visit(GT(forward(expr)));
    case GENode::constexpr_type_id:
      return Visit(GE(forward(expr)));
    case AndNode::constexpr_type_id:
      return Visit(And(forward(expr)));
    case OrNode::constexpr_type_id:
      return Visit(Or(forward(expr)));
    case ReduceNode::constexpr_type_id:
      return Visit(Reduce(forward(expr)));
    case CastNode::constexpr_type_id:
      return Visit(Cast(forward(expr)));
    case NotNode::constexpr_type_id:
      return Visit(Not(forward(expr)));
    case SelectNode::constexpr_type_id:
      return Visit(Select(forward(expr)));
    case RampNode::constexpr_type_id:
      return Visit(Ramp(forward(expr)));
    case BroadcastNode::constexpr_type_id:
      return Visit(Broadcast(forward(expr)));
    case ShuffleNode::constexpr_type_id:
      return Visit(Shuffle(forward(expr)));
    case IntImmNode::constexpr_type_id:
      return Visit(IntImm(forward(expr)));
    case FloatImmNode::constexpr_type_id:
      return Visit(FloatImm(forward(expr)));
    case StringImmNode::constexpr_type_id:
      return Visit(StringImm(forward(expr)));
    case AnyNode::constexpr_type_id:
      return Visit(Any(forward(expr)));
    case PrimExprNode::constexpr_type_id:
      LOG(FATAL) << "Untyped PrimExprNode";
    default:
      LOG(FATAL) << "Unsupported type: " << expr->GetTypeKey();
  }
}

TVM_UNIVERSAL_VISITOR_VISIT_METHOD(PrimExpr, Var, var) { return var; }

#undef TVM_UNIVERSAL_VISITOR_METHOD

}  // namespace tir
}  // namespace tvm

#endif /* TVM_TIR_UNIVERSAL_VISITOR_H_ */
