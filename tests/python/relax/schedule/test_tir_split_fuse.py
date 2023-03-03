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

from typing import Union, Sequence, Tuple

import pytest
import numpy as np

import tvm
import tvm.testing

from tvm import tir, relax
from tvm.script import tir as T, ir as I, relax as R

exec_mode = tvm.testing.parameter("bytecode", "compiled")
dtype = tvm.testing.parameter("float32")


@tvm.testing.fixture
def fused_module(dtype):
    @I.ir_module
    class fused_module:
        @R.function
        def main(A: R.Tensor(("m", "n"), dtype)) -> R.Tensor:
            m, n = T.var("int64"), T.var("int64")
            C = R.call_tir(two_stage, (A,), R.Tensor((m, n), dtype=dtype))
            return C

        @T.prim_func
        def two_stage(a: T.handle, c: T.handle):
            m = T.var("int64")
            n = T.var("int64")
            A = T.match_buffer(a, (m, n), dtype=dtype)
            C = T.match_buffer(c, (m, n), dtype=dtype)

            B = T.alloc_buffer([m, n], dtype=dtype)
            for i, j in T.grid(m, n):
                with T.block("step1_multiply"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] * 2.0

            for i, j in T.grid(m, n):
                with T.block("step2_add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    C[vi, vj] = B[vi, vj] + 42.0

    return fused_module


@tvm.testing.fixture
def split_module(dtype):
    @I.ir_module
    class split_module:
        @R.function
        def main(A: R.Tensor(("m", "n"), dtype)) -> R.Tensor:
            m, n = T.var("int64"), T.var("int64")
            B = R.call_tir(step1_multiply, (A,), R.Tensor((m, n), dtype=dtype))
            C = R.call_tir(step2_add, (B,), R.Tensor((m, n), dtype=dtype))
            return C

        @T.prim_func
        def step1_multiply(a: T.handle, b: T.handle):
            m = T.var("int64")
            n = T.var("int64")
            A = T.match_buffer(a, (m, n), dtype=dtype)
            B = T.match_buffer(b, (m, n), dtype=dtype)

            for i, j in T.grid(m, n):
                with T.block("step1_multiply"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] * 2.0

        @T.prim_func
        def step2_add(b: T.handle, c: T.handle):
            m = T.var("int64")
            n = T.var("int64")
            B = T.match_buffer(b, (m, n), dtype=dtype)
            C = T.match_buffer(c, (m, n), dtype=dtype)

            for i, j in T.grid(m, n):
                with T.block("step2_add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    C[vi, vj] = B[vi, vj] + 42.0

    return split_module


@tvm.testing.fixture
def single_stage_module(dtype):
    @I.ir_module
    class fused_module:
        @R.function
        def main(A: R.Tensor(("m", "n"), dtype)) -> R.Tensor:
            m, n = T.var("int64"), T.var("int64")
            with R.dataflow():
                C = R.call_tir(single_stage, (A,), R.Tensor((m, n), dtype=dtype))
                R.output(C)
            return C

        @T.prim_func
        def single_stage(a: T.handle, c: T.handle):
            m = T.var("int64")
            n = T.var("int64")
            A = T.match_buffer(a, (m, n), dtype=dtype)
            C = T.match_buffer(c, (m, n), dtype=dtype)

            for i, j in T.grid(m, n):
                with T.block("step1_multiply_step2_add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    C[vi, vj] = A[vi, vj] * 2.0 + 42.0

    return fused_module


def verify_module(
    module: tvm.IRModule,
    target: Union[str, tvm.target.Target],
    dev: tvm.runtime.Device,
    exec_mode: str,
    np_input_output: Tuple[Union[np.ndarray, Sequence[np.ndarray]], np.ndarray],
):
    before, expected = np_input_output
    if isinstance(before, np.ndarray):
        before = [before]

    executor = relax.vm.build(module, target, exec_mode=exec_mode)
    vm = relax.VirtualMachine(executor, dev)

    nd_input = [tvm.nd.array(arr, device=dev) for arr in before]
    nd_output = vm["main"](*nd_input)

    after = nd_output.numpy()

    tvm.testing.assert_allclose(after, expected)


@tvm.testing.fixture
def np_input_output(dtype):
    before = np.random.uniform(size=(16, 64)).astype(dtype)
    expected = before * 2.0 + 42.0
    return before, expected


@tvm.testing.parametrize_targets("llvm")
@pytest.mark.parametrize("which_module", ["split", "fused", "single_stage"])
def test_execute_module(
    target,
    dev,
    exec_mode,
    which_module,
    dtype,
    np_input_output,
    request,
):
    if which_module == "split":
        module = request.getfixturevalue("split_module")
    elif which_module == "fused":
        module = request.getfixturevalue("fused_module")
    elif which_module == "single_stage":
        module = request.getfixturevalue("single_stage_module")
    else:
        raise ValueError(f"Unknown module: {which_module}")
    assert module is not None
    verify_module(module, target, dev, exec_mode, np_input_output)


def test_split_module(fused_module, split_module):
    assert relax.analysis.well_formed(fused_module)
    sch = relax.Schedule(fused_module)
    sch.split_tir(block="step1_multiply", tir_primfunc="two_stage")
    assert relax.analysis.well_formed(sch.mod)
    tvm.ir.assert_structural_equal(sch.mod, split_module)


def test_split_of_single_stage_is_no_op(single_stage_module):
    sch = relax.Schedule(single_stage_module)
    sch.split_tir(block="step1_multiply_step2_add", tir_primfunc="single_stage")
    tvm.ir.assert_structural_equal(sch.mod, single_stage_module)


if __name__ == "__main__":
    tvm.testing.main()
