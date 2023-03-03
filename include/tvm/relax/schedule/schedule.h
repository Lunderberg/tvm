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
#ifndef TVM_RELAX_SCHEDULE_SCHEDULE_H_
#define TVM_RELAX_SCHEDULE_SCHEDULE_H_

#include <tvm/tir/schedule/schedule.h>

namespace tvm {
namespace relax {

using tir::BlockRV;
using tir::ExprRV;
using tir::LoopRV;
using tir::ScheduleErrorRenderLevel;

class ScheduleNode : public tir::ScheduleNode {
 public:
  /* \brief Split a TIR stage out into an independent PrimFunc and call_tir
   *
   * \param block_rv The block to extract out into a new PrimFunc
   *
   * \param tir_primfunc The name of the PrimFunc from which the block
   *     should be extracted.  If not specified, will use the PrimFunc
   *     previously specified by `ScheduleNode::WorkOn`.
   *
   * \param new_primfunc_names Optional parameter to specify the names
   *     of the PrimFuncs resulting from the split.
   *
   *     The names in the array are applied first to the extracted
   *     PrimFunc, then to any stages before the extracted stage (if
   *     they exist), then to any stages after the extracted stage (if
   *     they exist).  Stages without a provided name will be
   *     automatically named.
   *
   *     These names may be modified in order to produce names that
   *     are unique across the scheduled module.
   *
   * \returns The GlobalVar representing each PrimFunc generated from
   *     the original split
   */
  virtual Array<GlobalVar> SplitTIR(const BlockRV& block_rv, Optional<String> tir_primfunc,
                                    Array<String> new_primfunc_names = {}) = 0;

  static constexpr const char* _type_key = "relax.Schedule";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleNode, tir::ScheduleNode);
};

/*! \brief Utility class for implementing relax::ScheduleNode
 *
 * The majority of scheduling operations just need to be delegated to
 * the corresponding TIR schedule.  However, this would cause a
 * diamond dependency.
 *
 *     tir::ScheduleNode         <--     relax::ScheduleNode
 *
 *             ^                            ^
 *             |                            |
 *             |                            |
 *
 *     tir::ConcreteScheduleNode  <--  relax::ConcreteScheduleNode
 *
 * Ideally, this would be handled through virtual inheritance of
 * `tir::ScheduleNode`.  However, the `tvm::Object` API only handles
 * static inheritance, so that solution doesn't apply here.
 *
 * Instead, DelegateTIRScheduleToTIR allows subclasses to only
 * implement the relax-specific scheduling methods.
 */
class ScheduleDelegatingTIRPrimitivesToTIRSchedule : public ScheduleNode {
 protected:
  virtual tir::ScheduleNode* GetInnerSchedule() = 0;
  virtual const tir::ScheduleNode* GetInnerSchedule() const = 0;

  tir::ScheduleState state() const override { return GetInnerSchedule()->state(); }
  Optional<tir::Trace> trace() const override { return GetInnerSchedule()->trace(); }
  void WorkOn(const String& func_name) override { return GetInnerSchedule()->WorkOn(func_name); }
  tir::Schedule Copy() override { return GetInnerSchedule()->Copy(); }
  void Seed(support::LinearCongruentialEngine::TRandState seed) override {
    return GetInnerSchedule()->Seed(seed);
  }
  support::LinearCongruentialEngine::TRandState ForkSeed() override {
    return GetInnerSchedule()->ForkSeed();
  }
  tir::Block Get(const BlockRV& block_rv) const override {
    return GetInnerSchedule()->Get(block_rv);
  }
  tir::For Get(const LoopRV& loop_rv) const override { return GetInnerSchedule()->Get(loop_rv); }
  PrimExpr Get(const ExprRV& expr_rv) const override { return GetInnerSchedule()->Get(expr_rv); }
  tir::StmtSRef GetSRef(const BlockRV& block_rv) const override {
    return GetInnerSchedule()->GetSRef(block_rv);
  }
  tir::StmtSRef GetSRef(const LoopRV& loop_rv) const override {
    return GetInnerSchedule()->GetSRef(loop_rv);
  }
  bool HasBlock(const BlockRV& block_rv) const override {
    return GetInnerSchedule()->HasBlock(block_rv);
  }
  tir::StmtSRef GetSRef(const tir::StmtNode* stmt) const override {
    return GetInnerSchedule()->GetSRef(stmt);
  }
  void RemoveRV(const BlockRV& block_rv) override { return GetInnerSchedule()->RemoveRV(block_rv); }
  void RemoveRV(const LoopRV& loop_rv) override { return GetInnerSchedule()->RemoveRV(loop_rv); }
  void RemoveRV(const ExprRV& expr_rv) override { return GetInnerSchedule()->RemoveRV(expr_rv); }
  ExprRV SampleCategorical(const Array<Integer>& candidates, const Array<FloatImm>& probs,
                           Optional<Integer> decision = NullOpt) override {
    return GetInnerSchedule()->SampleCategorical(candidates, probs, decision);
  }
  Array<ExprRV> SamplePerfectTile(const LoopRV& loop_rv, int n, int max_innermost_factor,
                                  Optional<Array<Integer>> decision = NullOpt) override {
    return GetInnerSchedule()->SamplePerfectTile(loop_rv, n, max_innermost_factor, decision);
  }
  LoopRV SampleComputeLocation(const BlockRV& block_rv,
                               Optional<Integer> decision = NullOpt) override {
    return GetInnerSchedule()->SampleComputeLocation(block_rv, decision);
  }
  BlockRV GetBlock(const String& name, const Optional<String>& func_name = NullOpt) override {
    return GetInnerSchedule()->GetBlock(name, func_name);
  }
  Array<LoopRV> GetLoops(const BlockRV& block_rv) override {
    return GetInnerSchedule()->GetLoops(block_rv);
  }
  Array<BlockRV> GetChildBlocks(const BlockRV& block_rv) override {
    return GetInnerSchedule()->GetChildBlocks(block_rv);
  }
  Array<BlockRV> GetChildBlocks(const LoopRV& loop_rv) override {
    return GetInnerSchedule()->GetChildBlocks(loop_rv);
  }
  Array<BlockRV> GetProducers(const BlockRV& block_rv) override {
    return GetInnerSchedule()->GetProducers(block_rv);
  }
  Array<BlockRV> GetConsumers(const BlockRV& block_rv) override {
    return GetInnerSchedule()->GetConsumers(block_rv);
  }
  LoopRV Fuse(const Array<LoopRV>& loop_rvs, bool preserve_unit_iters = true) override {
    return GetInnerSchedule()->Fuse(loop_rvs, preserve_unit_iters);
  }
  Array<LoopRV> Split(const LoopRV& loop_rv, const Array<Optional<ExprRV>>& factors,
                      bool preserve_unit_iters = true) override {
    return GetInnerSchedule()->Split(loop_rv, factors, preserve_unit_iters);
  }
  void Reorder(const Array<LoopRV>& ordered_loop_rvs) override {
    return GetInnerSchedule()->Reorder(ordered_loop_rvs);
  }
  LoopRV AddUnitLoop(const BlockRV& block_rv) override {
    return GetInnerSchedule()->AddUnitLoop(block_rv);
  }
  LoopRV AddUnitLoop(const LoopRV& loop_rv) override {
    return GetInnerSchedule()->AddUnitLoop(loop_rv);
  }
  void Parallel(const LoopRV& loop_rv) override { return GetInnerSchedule()->Parallel(loop_rv); }
  void Vectorize(const LoopRV& loop_rv) override { return GetInnerSchedule()->Vectorize(loop_rv); }
  void Bind(const LoopRV& loop_rv, const String& thread_axis) override {
    return GetInnerSchedule()->Bind(loop_rv, thread_axis);
  }
  void Unroll(const LoopRV& loop_rv) override { return GetInnerSchedule()->Unroll(loop_rv); }
  BlockRV CacheRead(const BlockRV& block_rv, int read_buffer_index, const String& storage_scope,
                    const Array<BlockRV> consumer_blocks = {}) override {
    return GetInnerSchedule()->CacheRead(block_rv, read_buffer_index, storage_scope,
                                         consumer_blocks);
  }
  BlockRV CacheWrite(const BlockRV& block_rv, int write_buffer_index, const String& storage_scope,
                     const Array<BlockRV> consumer_blocks = {}) override {
    return GetInnerSchedule()->CacheWrite(block_rv, write_buffer_index, storage_scope,
                                          consumer_blocks);
  }
  Array<BlockRV> CacheInplace(const BlockRV& block_rv, int read_buffer_index,
                              const String& storage_scope) override {
    return GetInnerSchedule()->CacheInplace(block_rv, read_buffer_index, storage_scope);
  }
  Array<BlockRV> CacheIndex(const BlockRV& block_rv, const String& storage_scope,
                            int cse_thresh) override {
    return GetInnerSchedule()->CacheIndex(block_rv, storage_scope, cse_thresh);
  }
  BlockRV ReIndex(const BlockRV& block_rv, int buffer_index,
                  tir::BufferIndexType buffer_index_type) override {
    return GetInnerSchedule()->ReIndex(block_rv, buffer_index, buffer_index_type);
  }
  void ComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv, bool preserve_unit_loops,
                 int index = -1) override {
    return GetInnerSchedule()->ComputeAt(block_rv, loop_rv, preserve_unit_loops);
  }
  void ReverseComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv, bool preserve_unit_loops,
                        int index = -1) override {
    return GetInnerSchedule()->ReverseComputeAt(block_rv, loop_rv, preserve_unit_loops);
  }
  void ComputeInline(const BlockRV& block) override {
    return GetInnerSchedule()->ComputeInline(block);
  }
  void ReverseComputeInline(const BlockRV& block) override {
    return GetInnerSchedule()->ReverseComputeInline(block);
  }
  BlockRV DecomposeReduction(const BlockRV& block_rv, const LoopRV& loop_rv) override {
    return GetInnerSchedule()->DecomposeReduction(block_rv, loop_rv);
  }
  BlockRV RFactor(const LoopRV& loop_rv, int factor_axis) override {
    return GetInnerSchedule()->RFactor(loop_rv, factor_axis);
  }
  void StorageAlign(const BlockRV& block_rv, int buffer_index, int axis, int factor,
                    int offset) override {
    return GetInnerSchedule()->StorageAlign(block_rv, buffer_index, axis, factor, offset);
  }
  void SetScope(const BlockRV& block_rv, int buffer_index, const String& storage_scope) override {
    return GetInnerSchedule()->SetScope(block_rv, buffer_index, storage_scope);
  }
  BlockRV Blockize(const LoopRV& loop_rv, bool preserve_unit_iters = true) override {
    return GetInnerSchedule()->Blockize(loop_rv, preserve_unit_iters);
  }
  void Tensorize(const LoopRV& loop_rv, const String& intrin,
                 bool preserve_unit_iters = true) override {
    return GetInnerSchedule()->Tensorize(loop_rv, intrin, preserve_unit_iters);
  }
  void Tensorize(const BlockRV& block_rv, const String& intrin,
                 bool preserve_unit_iters = true) override {
    return GetInnerSchedule()->Tensorize(block_rv, intrin, preserve_unit_iters);
  }
  void Annotate(const LoopRV& loop_rv, const String& ann_key, const ObjectRef& ann_val) override {
    return GetInnerSchedule()->Annotate(loop_rv, ann_key, ann_val);
  }
  void Annotate(const BlockRV& block_rv, const String& ann_key, const ObjectRef& ann_val) override {
    return GetInnerSchedule()->Annotate(block_rv, ann_key, ann_val);
  }
  void Unannotate(const LoopRV& loop_rv, const String& ann_key) override {
    return GetInnerSchedule()->Unannotate(loop_rv, ann_key);
  }
  void Unannotate(const BlockRV& block_rv, const String& ann_key) override {
    return GetInnerSchedule()->Unannotate(block_rv, ann_key);
  }
  void TransformLayout(const BlockRV& block_rv, int buffer_index,
                       tir::BufferIndexType buffer_index_type, const tir::IndexMap& index_map,
                       const Optional<tir::IndexMap>& pad_value = NullOpt) override {
    return GetInnerSchedule()->TransformLayout(block_rv, buffer_index, buffer_index_type, index_map,
                                               pad_value);
  }
  void TransformBlockLayout(const BlockRV& block_rv, const tir::IndexMap& index_map) override {
    return GetInnerSchedule()->TransformBlockLayout(block_rv, index_map);
  }
  void SetAxisSeparator(const BlockRV& block_rv, int buffer_index,
                        tir::BufferIndexType buffer_index_type,
                        const Array<IntImm>& axis_separators) override {
    return GetInnerSchedule()->SetAxisSeparator(block_rv, buffer_index, buffer_index_type,
                                                axis_separators);
  }
  BlockRV DecomposePadding(const BlockRV& block_rv, const LoopRV& loop_rv) override {
    return GetInnerSchedule()->DecomposePadding(block_rv, loop_rv);
  }
  void PadEinsum(const BlockRV& block_rv, const Array<Integer>& padding) override {
    return GetInnerSchedule()->PadEinsum(block_rv, padding);
  }
  void RollingBuffer(const BlockRV& block_rv, int write_buffer_index) override {
    return GetInnerSchedule()->RollingBuffer(block_rv, write_buffer_index);
  }
  void EnterPostproc() override { return GetInnerSchedule()->EnterPostproc(); }
};

class Schedule : public tir::Schedule {
 public:
  /*!
   * \brief Construct a concrete TensorIR schedule from an IRModule
   * \param mod The IRModule to be scheduled
   * \param seed The seed value for schedule's random state
   * \param debug_mask Do extra correctness checking after the class creation
   * and each time after calling the Replace method.
   * \param error_render_level The level of error rendering
   * \return The concrete schedule created
   * \sa ScheduleDebugMask
   * \note The checks performed includes:
   * 1) VerifySRefTree
   * 2) VerifyCachedFlags
   */
  TVM_DLL static Schedule Concrete(IRModule mod, support::LinearCongruentialEngine::TRandState seed,
                                   int debug_mask, ScheduleErrorRenderLevel error_render_level);

  TVM_DLL static Schedule Traced(IRModule mod, support::LinearCongruentialEngine::TRandState seed,
                                 int debug_mask, ScheduleErrorRenderLevel error_render_level) {
    LOG(FATAL) << "Not implemented yet";
  }
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Schedule, tir::Schedule, ScheduleNode);
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_SCHEDULE_SCHEDULE_H_
