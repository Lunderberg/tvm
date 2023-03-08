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
#ifndef TVM_RELAX_SCHEDULE_PRIMITIVE_H_
#define TVM_RELAX_SCHEDULE_PRIMITIVE_H_

#include <tvm/tir/schedule/schedule.h>
#include <tvm/tir/schedule/state.h>

namespace tvm {
namespace relax {

Array<GlobalVar> SplitTIR(tir::ScheduleState self, const tir::StmtSRef& block_sref,
                          GlobalVar tir_primfunc, Array<String> new_primfunc_names);

Array<GlobalVar> FuseTIR(tir::ScheduleState self, Array<GlobalVar> to_fuse,
                         String fused_primfunc_name);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_SCHEDULE_PRIMITIVE_H_
