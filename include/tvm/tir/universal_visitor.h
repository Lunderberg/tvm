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

#ifndef TVM_TIR_UNIVERSAL_VISITOR_H_
#define TVM_TIR_UNIVERSAL_VISITOR_H_

namespace tvm {
namespace tir {

class UniversalMutator {
 public:
  // Exposing scope-dependent
  virtual std::function<void()> EnterConstraint(const PrimExpr& expr);
  virtual std::function<void()> EnterDefinition(const Var& var, const Optional<Range>&);
  virtual std::function<void()> EnterDefinition(const Buffer& buf);

  // PrimExpr mutators.  In any case where the static type is one of
  // the following, it can be replaced with a PrimExpr.
  virtual Optional<PrimExpr> Visit(const PrimExpr&);
  virtual Optional<PrimExpr> Visit(const Var&);
  virtual Optional<PrimExpr> Visit(const SizeVar&);
  virtual Optional<PrimExpr> Visit(const BufferLoad&);
  virtual Optional<PrimExpr> Visit(const ProducerLoad&);
  virtual Optional<PrimExpr> Visit(const Load&);
  virtual Optional<PrimExpr> Visit(const Let&);
  virtual Optional<PrimExpr> Visit(const Call&);
  virtual Optional<PrimExpr> Visit(const Add&);
  virtual Optional<PrimExpr> Visit(const Sub&);
  virtual Optional<PrimExpr> Visit(const Mul&);
  virtual Optional<PrimExpr> Visit(const Div&);
  virtual Optional<PrimExpr> Visit(const Mod&);
  virtual Optional<PrimExpr> Visit(const FloorDiv&);
  virtual Optional<PrimExpr> Visit(const FloorMod&);
  virtual Optional<PrimExpr> Visit(const Min&);
  virtual Optional<PrimExpr> Visit(const Max&);
  virtual Optional<PrimExpr> Visit(const EQ&);
  virtual Optional<PrimExpr> Visit(const NE&);
  virtual Optional<PrimExpr> Visit(const LT&);
  virtual Optional<PrimExpr> Visit(const LE&);
  virtual Optional<PrimExpr> Visit(const GT&);
  virtual Optional<PrimExpr> Visit(const GE&);
  virtual Optional<PrimExpr> Visit(const And&);
  virtual Optional<PrimExpr> Visit(const Or&);
  virtual Optional<PrimExpr> Visit(const Reduce&);
  virtual Optional<PrimExpr> Visit(const Cast&);
  virtual Optional<PrimExpr> Visit(const Not&);
  virtual Optional<PrimExpr> Visit(const Select&);
  virtual Optional<PrimExpr> Visit(const Ramp&);
  virtual Optional<PrimExpr> Visit(const Broadcast&);
  virtual Optional<PrimExpr> Visit(const Shuffle&);
  virtual Optional<PrimExpr> Visit(const IntImm&);
  virtual Optional<PrimExpr> Visit(const FloatImm&);
  virtual Optional<PrimExpr> Visit(const StringImm&);
  virtual Optional<PrimExpr> Visit(const Any&);

  // Stmt mutators.  In any case where the static type is one of
  // the following, it can be replaced with a Stmt.
  virtual Optional<Stmt> Visit(const Stmt&);
  virtual Optional<Stmt> Visit(const LetStmt&);
  virtual Optional<Stmt> Visit(const AttrStmt&);
  virtual Optional<Stmt> Visit(const IfThenElse&);
  virtual Optional<Stmt> Visit(const For&);
  virtual Optional<Stmt> Visit(const While&);
  virtual Optional<Stmt> Visit(const Allocate&);
  virtual Optional<Stmt> Visit(const AllocateConst&);
  virtual Optional<Stmt> Visit(const DeclBuffer&);
  virtual Optional<Stmt> Visit(const Store&);
  virtual Optional<Stmt> Visit(const BufferStore&);
  virtual Optional<Stmt> Visit(const BufferRealize&);
  virtual Optional<Stmt> Visit(const AssertStmt&);
  virtual Optional<Stmt> Visit(const ProducerStore&);
  virtual Optional<Stmt> Visit(const ProducerRealize&);
  virtual Optional<Stmt> Visit(const Prefetch&);
  virtual Optional<Stmt> Visit(const SeqStmt&);
  virtual Optional<Stmt> Visit(const Evaluate&);
  virtual Optional<Stmt> Visit(const BlockRealize&);

  // Type-specific mutators.  In cases where the return type must be
  // the same as the mutated type.  For example, every BufferLoad must
  // contain a Buffer, and every BlockRealize must contain a Block.
  // It would not be valid to replace
  virtual Optional<IRModule> Visit(const IRModule&);
  virtual Optional<PrimFunc> Visit(const PrimFunc&);
  virtual Optional<Block> Visit(const Block&);
  virtual Optional<Buffer> Visit(const Buffer&);
  virtual Optional<BufferRegion> Visit(const BufferRegion&);
  virtual Optional<MatchBufferRegion> Visit(const MatchBufferRegion&);
  virtual Optional<IterVar> Visit(const IterVar&);
  /* virtual BufferLocation Visit(const BufferLocation&); */
  virtual Optional<Range> Visit(const Range&);

  // VarDef and Integer have type-specific mutators, but deserve
  // special mention.  They are subtypes of Var/IntImm, respectively,
  // but represent cases where the more specific type is required.  In
  // cases where Var/IntImm are used as PrimExprs, they may be
  // replaced with any other PrimExpr.  However, there are cases where
  // a variable *must* be present to define the values, or where a
  // value *must* be an integer.  In these cases, the more specific
  // child class may be used.

  /* virtual Optional<VarDef> Visit(const VarDef&); */
  virtual Optional<Integer> Visit(const Integer&);

  // Containers
  template <typename T>
  auto Visit(const Array<T>& arr) {
    return arr.Map([this](T t) { return Visit(t); });
  }

  template <typename T>
  auto Visit(const Optional<T>& opt) {
    return opt.has_value() ? Visit(opt.value()) : NullOpt;
  }
};

}  // namespace tir
}  // namespace tvm

#endif /* TVM_TIR_UNIVERSAL_VISITOR_H_ */
