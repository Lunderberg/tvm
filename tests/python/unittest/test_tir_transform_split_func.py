#!/usr/bin/env python3

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
# pylint: disable=missing-function-docstring,missing-module-docstring

import sys

import pytest
import numpy as np

import tvm
import tvm.testing

from tvm import tir
from tvm.script import tir as T, ir as I, relax as R


def dummy_call_tir():
    @I.ir_module
    class mod:
        @T.prim_func
        def main(
            A: T.Buffer((16, 16), "float32"),
            B: T.Buffer((16, 16), "float32"),
        ):
            mod.subroutine(A[2:6, 0:8], B[4:7, 8:16])

        @T.prim_func
        def subroutine(a: T.handle, b: T.handle):
            ah = T.int32()
            aw = T.int32()
            bh = T.int32()
            bw = T.int32()
            A = T.match_buffer(a, [ah, aw], dtype="float32")
            B = T.match_buffer(b, [bh, bw], dtype="float32")
            T.evaluate(0)

    return mod


def test_parsing_buffer_region():
    mod = dummy_call_tir()

    call_node = mod["main"].body.value
    assert len(call_node.args) == 2
    assert len(call_node.buffer_map) == 2

    A_handle = call_node.args[0]
    assert isinstance(A_handle, tir.Var)
    assert A_handle in call_node.buffer_map
    A_region = call_node.buffer_map[A_handle]
    assert isinstance(A_region, tir.BufferRegion)
    assert A_region.buffer.name == "A"
    assert A_region.region[0].min == 2
    assert A_region.region[0].extent == 4
    assert A_region.region[1].min == 0
    assert A_region.region[1].extent == 8

    B_handle = call_node.args[1]
    assert isinstance(B_handle, tir.Var)
    assert B_handle in call_node.buffer_map
    B_region = call_node.buffer_map[B_handle]
    assert isinstance(B_region, tir.BufferRegion)
    assert B_region.buffer.name == "B"
    assert B_region.region[0].min == 4
    assert B_region.region[0].extent == 3
    assert B_region.region[1].min == 8
    assert B_region.region[1].extent == 8


def test_roundtrip():
    original = dummy_call_tir()
    tvmscript = original.script(show_meta=True)
    after_roundtrip = tvm.script.from_source(tvmscript)
    tvm.ir.assert_structural_equal(original, after_roundtrip, True)


@tvm.testing.parametrize_targets("llvm")
def test_run_static_shape_calls(target, dev):
    @I.ir_module
    class mod:
        @T.prim_func
        def main(A: T.Buffer((32, 8), "float32")):
            T.func_attr({"global_symbol": "main", "tir.is_entry_func": True})
            mod.subroutine(A[0:8, 0:8], 42.0)
            mod.subroutine(A[8:16, 0:8], 12345.0)
            mod.subroutine(A[16:24, 0:8], -5.0)
            mod.subroutine(A[24:32, 0:8], 24601.0)

        @T.prim_func
        def subroutine(a: T.handle, val: T.float32):
            elem_offset = T.int32()
            A = T.match_buffer(a, (8, 8), "float32", elem_offset=elem_offset)
            for i, j in T.grid(8, 8):
                A[i, j] = val

    built = tvm.build(mod)

    expected = np.ndarray(shape=(32, 8), dtype="float32")
    expected[0:8, 0:8] = 42.0
    expected[8:16, 0:8] = 12345.0
    expected[16:24, 0:8] = -5.0
    expected[24:32, 0:8] = 24601.0

    a = tvm.nd.empty(shape=expected.shape, dtype=expected.dtype, device=dev)
    built(a)
    actual = a.numpy()

    np.testing.assert_array_equal(expected, actual)


@tvm.testing.parametrize_targets("llvm")
def test_run_static_strided_shape_calls(target, dev):
    @I.ir_module
    class mod:
        @T.prim_func
        def main(A: T.Buffer((16, 16), "float32")):
            T.func_attr({"global_symbol": "main", "tir.is_entry_func": True})
            mod.subroutine(A[0:8, 0:8], 42.0)
            mod.subroutine(A[0:8, 8:16], 12345.0)
            mod.subroutine(A[8:16, 0:8], -5.0)
            mod.subroutine(A[8:16, 8:16], 24601.0)

        @T.prim_func
        def subroutine(a: T.handle, val: T.float32):
            elem_offset = T.int32()
            stride_i = T.int32()
            stride_j = T.int32()
            A = T.match_buffer(
                a, (8, 8), "float32", strides=[stride_i, stride_j], elem_offset=elem_offset
            )
            for i, j in T.grid(8, 8):
                A[i, j] = val

    built = tvm.build(mod)

    expected = np.ndarray(shape=(16, 16), dtype="float32")
    expected[0:8, 0:8] = 42.0
    expected[0:8, 8:16] = 12345.0
    expected[8:16, 0:8] = -5.0
    expected[8:16, 8:16] = 24601.0

    a = tvm.nd.empty(shape=expected.shape, dtype=expected.dtype, device=dev)
    built(a)
    actual = a.numpy()

    np.testing.assert_array_equal(expected, actual)


@tvm.testing.parametrize_targets("llvm")
def test_run_dynamic_param_calls(target, dev):
    @I.ir_module
    class mod:
        @T.prim_func
        def main(A: T.Buffer((16, 16), "int32")):
            T.func_attr({"global_symbol": "main", "tir.is_entry_func": True})
            mod.subroutine(A[0:12, 0:12], 42)
            mod.subroutine(A[0:12, 12:16], 12345)
            mod.subroutine(A[12:16, 0:12], -5)
            mod.subroutine(A[12:16, 12:16], 24601)

        @T.prim_func
        def subroutine(a: T.handle, val: T.int32):
            elem_offset = T.int32()
            ah = T.int32()
            aw = T.int32()
            stride_i = T.int32()
            stride_j = T.int32()
            A = T.match_buffer(
                a,
                [ah, aw],
                "int32",
                strides=[stride_i, stride_j],
                elem_offset=elem_offset,
            )
            for i, j in T.grid(ah, aw):
                A[i, j] = val

    built = tvm.build(mod)

    expected = np.ndarray(shape=(16, 16), dtype="int32")
    expected[0:12, 0:12] = 42
    expected[0:12, 12:16] = 12345
    expected[12:16, 0:12] = -5
    expected[12:16, 12:16] = 24601

    a = tvm.nd.empty(shape=expected.shape, dtype=expected.dtype, device=dev)
    built(a)
    actual = a.numpy()

    np.testing.assert_array_equal(expected, actual)


@tvm.testing.parametrize_targets("llvm")
def test_run_dynamic_dltensor_calls(target, dev):
    """Like test_run_dynamic_dltensor_calls, but with exposed subroutine

    Changing the signature of a subroutine is only allowed for
    internal subroutines.  For external subroutines, indicated by the
    presence of the "global_symbol" attribute, the user may call the
    subroutine directly using the PackedFunc interface.  The main
    function should call the subroutine using the same PackedFunc
    interface.

    Note that this requires the caller to be on a target that can
    execute PackedFunc calls (e.g. LLVM on the CPU).

    """

    @I.ir_module
    class mod:
        @T.prim_func
        def main(A: T.Buffer((16, 16), "int32")):
            T.func_attr({"global_symbol": "main", "tir.is_entry_func": True})
            mod.subroutine(A[0:12, 0:12], 42)
            mod.subroutine(A[0:12, 12:16], 12345)
            mod.subroutine(A[12:16, 0:12], -5)
            mod.subroutine(A[12:16, 12:16], 24601)

        @T.prim_func
        def subroutine(a: T.handle, val: T.int32):
            T.func_attr({"global_symbol": "subroutine"})
            ah = T.int32()
            aw = T.int32()
            stride_i = T.int32()
            stride_j = T.int32()
            elem_offset = T.int32()
            A = T.match_buffer(
                a, [ah, aw], "int32", strides=[stride_i, stride_j], elem_offset=elem_offset
            )
            for i, j in T.grid(ah, aw):
                A[i, j] = val

    built = tvm.build(mod, target=target)

    expected = np.ndarray(shape=(16, 16), dtype="int32")
    expected[0:12, 0:12] = 42
    expected[0:12, 12:16] = 12345
    expected[12:16, 0:12] = -5
    expected[12:16, 12:16] = 24601

    a = tvm.nd.empty(shape=expected.shape, dtype=expected.dtype, device=dev)
    built(a)
    actual = a.numpy()

    np.testing.assert_array_equal(expected, actual)


def test_run_subroutine_on_other_target():
    @I.ir_module
    class mod:
        @T.prim_func
        def main(A: T.Buffer((32, 8), "float32")):
            T.func_attr(
                {
                    "global_symbol": "main",
                    "tir.is_entry_func": True,
                    # "tir.is_host_func": True,
                }
            )
            mod.subroutine(A[0:8, 0:8], 42.0)
            mod.subroutine(A[8:16, 0:8], 12345.0)
            mod.subroutine(A[16:24, 0:8], -5.0)
            mod.subroutine(A[24:32, 0:8], 24601.0)

        @T.prim_func
        def subroutine(
            a: T.handle,
            val: T.float32,
        ):
            # TODO: Pull in https://github.com/apache/tvm/pull/14495,
            # which should allow the kernel launch params to be
            # expressed here.
            elem_offset = T.int32()
            stride_i = T.int32()
            stride_j = T.int32()
            A = T.match_buffer(
                a,
                (8, 8),
                "float32",
                elem_offset=elem_offset,
                strides=[stride_i, stride_j],
            )
            T.func_attr({"target": T.target("cuda"), "tir.kernel_launch_params": ["threadIdx.x"]})
            i = T.launch_thread("threadIdx.x", 8)
            for j in range(8):
                A[i, j] = val

    built = tvm.build(mod)

    expected = np.ndarray(shape=(32, 8), dtype="float32")
    expected[0:8, 0:8] = 42.0
    expected[8:16, 0:8] = 12345.0
    expected[16:24, 0:8] = -5.0
    expected[24:32, 0:8] = 24601.0

    a = tvm.nd.empty(shape=expected.shape, dtype=expected.dtype, device=tvm.cpu())
    built(a)
    actual = a.numpy()

    np.testing.assert_array_equal(expected, actual)


def test_run_segment_on_other_target():
    # TODO: Experiment with the value of the "target" AttrStmt being
    # the device_id instead of just zero.

    @I.ir_module
    class mod:
        @T.prim_func
        def main(A: T.Buffer((32, 8), "float32")):
            T.func_attr({"global_symbol": "main", "tir.is_entry_func": True})
            with T.attr(T.target("cuda"), "target", 0):
                io = T.launch_thread("threadIdx.x", 4)
                for ii, j in T.grid(8, 8):
                    A[8 * io + ii, j] = 42.0

    built = tvm.build(mod)

    expected = np.ndarray(shape=(32, 8), dtype="float32")
    expected[0:32, 0:8] = 42.0

    a = tvm.nd.empty(shape=expected.shape, dtype=expected.dtype, device=tvm.cpu())
    built(a)
    actual = a.numpy()

    np.testing.assert_array_equal(expected, actual)


if __name__ == "__main__":
    tvm.testing.main()
