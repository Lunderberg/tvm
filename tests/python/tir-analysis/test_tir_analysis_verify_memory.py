# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import tvm
import tvm.testing
from tvm import te
from tvm.script import ir as I, tir as T

import pytest

# The following DLDeviceType/TVMDeviceExtType values
# are originally defined in dlpack.h and c_runtime_api.h.
gpu_devices = ["cuda", "opencl", "metal", "vulkan"]
other_devices = ["llvm", "ext_dev"]


# All computations are bound.
# So VerifyMemory pass is expected to succeed.
#
@tvm.testing.uses_gpu
def test_verify_memory_all_bind():
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda i: A[i] + 1.0, name="B")

    # B is bound to threads.
    s = te.create_schedule(B.op)
    bx, tx = s[B].split(B.op.axis[0], factor=64)
    s[B].bind(bx, te.thread_axis("blockIdx.x"))
    s[B].bind(tx, te.thread_axis("threadIdx.x"))

    mod = tvm.lower(s, [A, B])

    for dev_type in gpu_devices + other_devices:
        if tvm.testing.device_enabled(dev_type):
            binded_mod = tvm.tir.transform.Apply(
                lambda f: f.with_attr("target", tvm.target.Target(dev_type))
            )(mod)
            tvm.tir.transform.VerifyMemory()(binded_mod)


# Computations are not bound.
# So VerifyMemory pass fails when device type is GPU.
#
@tvm.testing.uses_gpu
def test_verify_memory_not_bind():
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda i: A[i] + 1.0, name="B")

    # B is not bound to threads.
    s = te.create_schedule(B.op)

    mod = tvm.lower(s, [A, B])

    for dev_type in gpu_devices:
        if tvm.testing.device_enabled(dev_type):
            binded_mod = tvm.tir.transform.Apply(
                lambda f: f.with_attr("target", tvm.target.Target(dev_type))
            )(mod)
            with pytest.raises(RuntimeError):
                tvm.tir.transform.VerifyMemory()(binded_mod)

    for dev_type in other_devices:
        if tvm.testing.device_enabled(dev_type):
            binded_mod = tvm.tir.transform.Apply(
                lambda f: f.with_attr("target", tvm.target.Target(dev_type))
            )(mod)
            tvm.tir.transform.VerifyMemory()(binded_mod)


# Computations are partially bound.
# So VerifyMemory pass fails when device type is GPU.
#
@tvm.testing.uses_gpu
def test_verify_memory_partially_bind():
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda i: A[i] + 1.0, name="B")
    C = te.compute(B.shape, lambda i: B[i] + 2.0, name="C")
    D = te.compute(C.shape, lambda i: C[i] + 2.0, name="D")

    # C is bound to threads, but B and D are not.
    s = te.create_schedule([B.op, C.op, D.op])
    bx, tx = s[C].split(C.op.axis[0], factor=64)
    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))

    mod = tvm.lower(s, [A, B, C, D])

    for dev_type in gpu_devices:
        if tvm.testing.device_enabled(dev_type):
            binded_mod = tvm.tir.transform.Apply(
                lambda f: f.with_attr("target", tvm.target.Target(dev_type))
            )(mod)
            with pytest.raises(RuntimeError):
                tvm.tir.transform.VerifyMemory()(binded_mod)

    for dev_type in other_devices:
        if tvm.testing.device_enabled(dev_type):
            binded_mod = tvm.tir.transform.Apply(
                lambda f: f.with_attr("target", tvm.target.Target(dev_type))
            )(mod)
            tvm.tir.transform.VerifyMemory()(binded_mod)


@tvm.testing.parametrize_targets("llvm", "cuda")
def test_host_may_compute_address_on_specific_targets(target):
    """Some targets support T.address_of on host

    Some targets have the `tvm::attr::kAllowPointerArithmeticOnHost`
    attribute.  For these targets, the `T.address_of` built-in may
    appear in TIR regions that will be executed on the host.  The host
    will use pointer arithmetic to determine the address of the
    requested element.

    """

    @I.ir_module
    class Module:
        @T.prim_func
        def is_128bit_aligned(A_handle: T.handle) -> T.bool:
            T.func_attr({"target": T.target(target)})
            elem_offset = T.int64()
            A = T.match_buffer(A_handle, 4096, "float16", elem_offset=elem_offset)

            is_offset_zero = elem_offset == 0
            is_aligned = T.reinterpret("int64", T.address_of(A[0])) % 128 == 0
            return is_offset_zero and is_aligned

    tvm.tir.transform.VerifyMemory()(Module)


@tvm.testing.parametrize_targets("vulkan", "opencl")
def test_error_to_compute_address_on_host_for_most_targets(target):
    """Some targets do not support T.address_of on host

    By default, the `DLTensor::data` pointer must be treated as
    opaque.  For any target without the
    `tvm::attr::kAllowPointerArithmeticOnHost` attribute, the host may
    not use pointer arithmetic to determine the address of a value
    within a buffer.

    """

    @I.ir_module
    class Module:
        @T.prim_func
        def is_128bit_aligned(A_handle: T.handle) -> T.bool:
            T.func_attr({"target": T.target(target)})
            elem_offset = T.int64()
            A = T.match_buffer(A_handle, 4096, "float16", elem_offset=elem_offset)

            is_offset_zero = elem_offset == 0
            is_aligned = T.reinterpret("int64", T.address_of(A[0])) % 128 == 0
            return is_offset_zero and is_aligned

    with pytest.raises(RuntimeError):
        tvm.tir.transform.VerifyMemory()(Module)


if __name__ == "__main__":
    tvm.testing.main()
