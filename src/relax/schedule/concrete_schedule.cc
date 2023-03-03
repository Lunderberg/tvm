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

#include "./concrete_schedule.h"

#include "./primitive.h"

namespace tvm {
namespace relax {

Schedule Schedule::Concrete(IRModule mod, support::LinearCongruentialEngine::TRandState seed,
                            int debug_mask, ScheduleErrorRenderLevel error_render_level) {
  auto node = make_object<ConcreteScheduleNode>(mod, seed, debug_mask, error_render_level);
  return Schedule(node);
}

Array<GlobalVar> ConcreteScheduleNode::SplitTIR(const BlockRV& block_rv,
                                                Optional<String> tir_primfunc,
                                                Array<String> new_primfunc_names) {
  GlobalVar primfunc = [&]() {
    if (tir_primfunc) {
      return inner.mod()->GetGlobalVar(tir_primfunc.value());
    } else if (inner.func_working_on_) {
      return inner.func_working_on_.value();
    } else {
      LOG(FATAL)
          << "Must specify tir_primfunc either in function parameter or with Schedule::WorkOn";
    }
  }();

  return relax::SplitTIR(inner.state_, this->GetSRef(block_rv), primfunc, new_primfunc_names);
}

}  // namespace relax
}  // namespace tvm
