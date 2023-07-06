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
 * \file tvm/target/dynamic_target.h
 * \brief Compilation target object.
 */
#ifndef TVM_TARGET_DYNAMIC_TARGET_H_
#define TVM_TARGET_DYNAMIC_TARGET_H_

#include <tvm/ir/expr.h>
#include <tvm/target/target.h>

namespace tvm {

class DynamicTargetNode : public Object {
 public:
  Target target;
  PrimExpr device_id;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("target", &target);
    v->Visit("device_id", &device_id);
  }

  static constexpr const char* _type_key = "DynamicTarget";
  TVM_DECLARE_BASE_OBJECT_INFO(DynamicTargetNode, Object);
};

class DynamicTarget : public ObjectRef {
 public:
  explicit DynamicTarget(Target target, PrimExpr device_id = Integer(0));

  TVM_DEFINE_OBJECT_REF_METHODS(DynamicTarget, ObjectRef, DynamicTargetNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DynamicTargetNode);
};

}  // namespace tvm
#endif  // TVM_TARGET_TARGET_H_
