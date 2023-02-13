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
#ifndef TVM_RELAX_SCHEDULE_CONCRETE_SCHEDULE_H_
#define TVM_RELAX_SCHEDULE_CONCRETE_SCHEDULE_H_

#include <tvm/relax/schedule/schedule.h>

#include "../../tir/schedule/concrete_schedule.h"

namespace tvm {
namespace relax {

class ConcreteScheduleNode : public ScheduleDelegatingTIRPrimitivesToTIRSchedule {
 public:
  tir::ConcreteScheduleNode inner;

  ConcreteScheduleNode() = default;
  ConcreteScheduleNode(IRModule mod, support::LinearCongruentialEngine::TRandState seed,
                       int debug_mask, ScheduleErrorRenderLevel error_render_level)
      : inner(mod, seed, debug_mask, error_render_level) {}

  void SplitTIR(const BlockRV& block_rv, Optional<String> tir_primfunc,
                Optional<String> extracted_primfunc_name,
                Optional<String> remainder_primfunc_name) override;

 protected:
  tir::ScheduleNode* GetInnerSchedule() override { return &inner; }
  const tir::ScheduleNode* GetInnerSchedule() const override { return &inner; }
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_SCHEDULE_CONCRETE_SCHEDULE_H
