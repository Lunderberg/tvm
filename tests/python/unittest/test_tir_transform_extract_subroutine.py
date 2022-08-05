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


@pytest.mark.skip(reason="Not fully implemented yet")
class TestSplitFunctionToSubroutine(tvm.testing.CompareBeforeAfter):
    nthreads = tvm.testing.parameter(4)
    shape = tvm.testing.parameter(
        (1024, 64, 8),
    )
    dtype = tvm.testing.parameter("int8")
    vector_size = tvm.testing.parameter(128)

    transform = tvm.tir.transform.ExtractSubroutineBlocks()

    @tvm.testing.fixture
    def before(self, shape, dtype):
        nthreads, scalar_size, vector_size = shape

        @I.ir_module
        class module:
            @T.prim_func
            def main(X: T.Buffer(shape, dtype), Y: T.Buffer(shape, dtype)):
                B = T.alloc_buffer(shape, dtype)
                for n, i, j in T.grid(nthreads, scalar_size, vector_size):
                    with T.block("dma_copy"):
                        vn, vi, vj = T.axis.remap("SSS", [n, i, j])
                        B[vn, vi, vj] = X[vn, vi, vj]

                with T.block("block_to_extract"):
                    T.block_attr({"extract_as_subroutine": 1})
                    for n in T.parallel(nthreads):
                        for i in range(scalar_size):
                            for j in T.vectorized(vector_size):
                                with T.block("vector_compute"):
                                    vn_i, vi_i, vj_i = T.axis.remap("SSS", [n, i, j])
                                    T.reads(B[vn_i, vi_i, vj_i])
                                    T.writes(Y[vn_i, vi_i, vj_i])
                                    Y[vn_i, vi_i, vj_i] = B[vn_i, vi_i, vj_i]

        return module

    @tvm.testing.fixture
    def expected(self, shape, dtype):
        nthreads, scalar_size, vector_size = shape

        @I.ir_module
        class module:
            @T.prim_func
            def main(X: T.Buffer[shape, dtype], Y: T.Buffer[shape, dtype]):

                B = T.alloc_buffer(shape, dtype=dtype)
                for n, i, j in T.grid(nthreads, scalar_size, vector_size):
                    with T.block("dma_copy"):
                        vn, vi, vj = T.axis.remap("SSS", [n, i, j])
                        B[vn, vi, vj] = X[vn, vi, vj]

                module.extracted_subroutine(
                    B[0:nthreads, 0:scalar_size, 0:vector_size],
                    Y[0:nthreads, 0:scalar_size, 0:vector_size],
                )

            @T.prim_func
            def extracted_subroutine(B: T.Buffer[shape, dtype], Y: T.Buffer[shape, dtype]):
                for n in T.parallel(nthreads):
                    for i in T.serial(scalar_size):
                        for j in T.vectorized(vector_size):
                            with T.block("vector_compute"):
                                vn, vi, vj = T.axis.remap("SSS", [n, i, j])
                                Y[vn, vi, vj] = B[vn, vi, vj]

        return module


@pytest.mark.skip(reason="Not fully implemented yet")
class TestSplitFunctionCorrectValues:
    dtype = tvm.testing.parameter("float32")
    shape = tvm.testing.parameter((128, 32, 1024))

    @tvm.testing.fixture
    def np_data(self, shape, dtype):
        N, K, M = shape
        a_np = np.random.uniform(size=(N, K)).astype(dtype)
        b_np = np.random.uniform(size=(K, M)).astype(dtype)
        c_np = a_np @ b_np
        return (a_np, b_np, c_np)

    def test_split_func(self, shape, dtype, target, dev, np_data):
        N, K, M = shape

        @T.prim_func
        def func(
            A: T.Buffer((N, K), dtype), B: T.Buffer((K, M), dtype), C: T.Buffer((N, M), dtype)
        ):
            for i, j, k in T.grid(N, M, K):
                with T.block("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.cast(0.0, dtype)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        sch = tvm.tir.Schedule(func)

        block = sch.get_block("matmul")
        i, j, k = sch.get_loops(block)

        block_to_extract = sch.blockize(j)
        sch.annotate(block_to_extract, "extract_as_subroutine", True)

        built = tvm.build(sch.mod, target)

        (a_np, b_np, c_np) = np_data
        a_ndarray = tvm.nd.array(a_np, dev)
        b_ndarray = tvm.nd.array(b_np, dev)
        c_ndarray = tvm.nd.empty(c_np.shape, dtype=dtype, device=dev)
        built(a_ndarray, b_ndarray, c_ndarray)

        c_tvm = c_ndarray.numpy()
        tvm.testing.assert_allclose(c_np, c_tvm)


if __name__ == "__main__":
    tvm.testing.main()
