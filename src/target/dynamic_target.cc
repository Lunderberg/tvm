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
 *  Represent dispatch to a runtime-dependent target.
 * \file src/target/dynamic_target.cc
 */

#include <tvm/target/dynamic_target.h>

#include <utility>

namespace tvm {

DynamicTarget::DynamicTarget(Target target, PrimExpr device_id) {
  auto node = make_object<DynamicTargetNode>();
  node->target = std::move(target);
  node->device_id = std::move(device_id);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(DynamicTargetNode);

TVM_REGISTER_GLOBAL("target.DynamicTarget").set_body_typed([](Target target, PrimExpr device_id) {
  return DynamicTarget(std::move(target), std::move(device_id));
});

}  // namespace tvm
