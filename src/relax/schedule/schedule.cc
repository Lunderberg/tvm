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

#include <tvm/relax/schedule/schedule.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace relax {

TVM_REGISTER_OBJECT_TYPE(ScheduleNode);

TVM_REGISTER_GLOBAL("relax.schedule.ConcreteSchedule")
    .set_body_typed([](IRModule mod, support::LinearCongruentialEngine::TRandState seed,
                       int debug_mask, int error_render_level) -> Schedule {
      return Schedule::Concrete(mod, debug_mask, seed,
                                static_cast<ScheduleErrorRenderLevel>(error_render_level));
    });
TVM_REGISTER_GLOBAL("relax.schedule.TracedSchedule")
    .set_body_typed([](IRModule mod, support::LinearCongruentialEngine::TRandState seed,
                       int debug_mask, int error_render_level) -> Schedule {
      return Schedule::Traced(mod, seed, debug_mask,
                              static_cast<ScheduleErrorRenderLevel>(error_render_level));
    });

TVM_REGISTER_GLOBAL("relax.schedule.ScheduleSplitTIR")
    .set_body_method<Schedule>(&ScheduleNode::SplitTIR);

}  // namespace relax
}  // namespace tvm
