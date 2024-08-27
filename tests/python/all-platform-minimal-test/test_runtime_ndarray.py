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
"""Basic runtime enablement test."""

import math

import pytest
import numpy as np

import tvm
import tvm.testing
from tvm import te

dtype = tvm.testing.parameter("uint8", "int8", "uint16", "int16", "uint32", "int32", "float32")


def test_nd_create(target, dev, dtype):
    x = np.random.randint(0, 10, size=(3, 4))
    x = np.array(x, dtype=dtype)
    y = tvm.nd.array(x, device=dev)
    z = y.copyto(dev)
    assert y.dtype == x.dtype
    assert y.shape == x.shape
    assert isinstance(y, tvm.nd.NDArray)
    np.testing.assert_equal(x, y.numpy())
    np.testing.assert_equal(x, z.numpy())

    # no need here, just to test usablity
    dev.sync()


def test_memory_usage(target, dev, dtype):
    available_memory_before = dev.available_global_memory
    if available_memory_before is None:
        pytest.skip(reason=f"Target '{target}' does not support queries of available memory")

    arr = tvm.nd.empty([1024, 1024], dtype=dtype, device=dev)
    available_memory_after = dev.available_global_memory

    num_elements = math.prod(arr.shape)
    element_nbytes = tvm.runtime.DataType(dtype).itemsize()
    expected_memory_after = available_memory_before - num_elements * element_nbytes

    # Allocations may be padded out to provide alignment, to match a
    # page boundary, due to additional device-side bookkeeping
    # required by the TVM backend or the driver, etc.  Therefore, the
    # available memory may decrease by more than the requested amount.
    assert available_memory_after <= expected_memory_after

    # TVM's NDArray type is a reference-counted handle to the
    # underlying reference.  After the last reference to an NDArray is
    # cleared, the backing allocation will be freed.
    del arr

    assert dev.available_global_memory == available_memory_before


@pytest.mark.parametrize(
    "src_dst",
    [
        ("float32", "float32"),
        # ("float32", "float16"),
        # ("float16", "float32"),
    ],
    ids=[
        "f32-to-f32",
        # "f32-to-f16",
        # "f16-to-f32",
    ],
)
@tvm.testing.parametrize_targets(
    # "llvm",
    "llvm -opt-level=0",
)
def test_fp16_conversion(src_dst, target, dev):
    # DEBUG PRINT, REMOVE BEFORE MERGE
    print("LLVM version:", tvm.support.libinfo()["LLVM_VERSION"])

    src, dst = src_dst
    n = 100

    from tvm.script import ir as I, tir as T

    @I.ir_module
    class Module:
        @T.prim_func
        def main(
            output_handle: T.handle,
            input_handle: T.handle,
        ):
            Input = T.match_buffer(output_handle, 100, src)
            Output = T.match_buffer(input_handle, 100, dst)

            T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})

            T.call_extern("void", "printf", "Start of function\n")
            T.call_extern("void", "fflush", T.int64(0))

            for i in range(100):
                T.call_extern("void", "printf", "Start of iteration %d\n", i)
                T.call_extern("void", "fflush", T.int64(0))

                input_value = Input[i]

                T.call_extern("void", "printf", "Read value in iteration %d\n", i)
                T.call_extern("void", "fflush", T.int64(0))

                output_value = T.Cast(dst, input_value)

                T.call_extern(
                    "void", "printf", "Converted value to output type in iteration %d\n", i
                )
                T.call_extern("void", "fflush", T.int64(0))

                Output[i] = output_value

                T.call_extern("void", "printf", "Wrote value to output array in iteration %d\n", i)
                T.call_extern("void", "fflush", T.int64(0))

                T.call_extern("void", "printf", "End of iteration %d\n", i)
                T.call_extern("void", "fflush", T.int64(0))

            T.call_extern("void", "printf", "End of function\n")
            T.call_extern("void", "fflush", T.int64(0))

    # A = te.placeholder((n,), dtype=src)
    # B = te.compute((n,), lambda i: A[i].astype(dst))

    # s = te.create_schedule([B.op])

    # # DEBUG PRINT, REMOVE BEFORE MERGE
    # tvm.lower(s, [A, B]).show()

    # func = tvm.build(s, [A, B], target)

    func = tvm.build(Module, target=target)

    # DEBUG PRINT, REMOVE BEFORE MERGE
    print(func.get_source(), flush=True)

    x_tvm = tvm.nd.array(100 * np.random.randn(n).astype(src) - 50, dev)
    y_tvm = tvm.nd.array(100 * np.random.randn(n).astype(dst) - 50, dev)

    print(f"Input shape: {x_tvm.shape}, input dtype: {x_tvm.dtype}", flush=True)
    print(f"Output shape: {y_tvm.shape}, output dtype: {y_tvm.dtype}", flush=True)

    func(x_tvm, y_tvm)

    expected = x_tvm.numpy().astype(dst)
    real = y_tvm.numpy()

    tvm.testing.assert_allclose(expected, real)


def test_dtype():
    dtype = tvm.DataType("handle")
    assert dtype.type_code == tvm.DataTypeCode.HANDLE


if __name__ == "__main__":
    tvm.testing.main()
