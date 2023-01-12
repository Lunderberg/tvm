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
import tvm
import tvm.testing
from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import verify_trace_roundtrip
import pytest

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks

@T.prim_func
def single_elementwise(A: T.Buffer[(128, 128), "float32"], B: T.Buffer[(128, 128), "float32"]):
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0

# fmt: on
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks


def test_blockize_outer():
    @T.prim_func
    def after_blockize_outer(
        A: T.Buffer[(128, 128), "float32"], B: T.Buffer[(128, 128), "float32"]
    ):
        with T.block("root"):
            T.reads()
            T.writes()
            with T.block("blockized_B"):
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

    func = single_elementwise
    s = tir.Schedule(func, debug_mask="all")
    x, _ = s.get_loops(s.get_block("B"))
    s.blockize(x)
    tvm.ir.assert_structural_equal(s.mod["main"], after_blockize_outer)
    verify_trace_roundtrip(sch=s, mod=func)


def test_blockize_inner():
    @T.prim_func
    def after_blockize_inner(
        A: T.Buffer[(128, 128), "float32"],
        B: T.Buffer[(128, 128), "float32"],
    ) -> None:
        for i in T.serial(128):
            with T.block("blockized_B"):
                vi = T.axis.spatial(128, i)
                for j in T.serial(128):
                    with T.block("B"):
                        vj = T.axis.remap("S", [j])
                        B[vi, vj] = A[vi, vj] * 2.0

    func = single_elementwise
    s = tir.Schedule(func, debug_mask="all")
    _, y = s.get_loops(s.get_block("B"))
    s.blockize(y)
    tvm.ir.assert_structural_equal(s.mod["main"], after_blockize_inner)
    verify_trace_roundtrip(sch=s, mod=func)


def test_two_elementwise_blockize_reverse_compute_at():
    @T.prim_func
    def before_blockize_rca(
        A: T.Buffer[(128, 128), "float32"],
        C: T.Buffer[(128, 128), "float32"],
    ) -> None:
        B = T.alloc_buffer([128, 128], dtype="float32")
        for i, j in T.grid(8, 8):
            with T.block("B_o"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                T.writes(B[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                for i_1, j_1 in T.grid(16, 16):
                    with T.block("B"):
                        vi_i, vj_i = T.axis.remap("SS", [i_1, j_1])
                        T.reads(A[vi * 16 + vi_i, vj * 16 + vj_i])
                        T.writes(B[vi * 16 + vi_i, vj * 16 + vj_i])
                        B[vi * 16 + vi_i, vj * 16 + vj_i] = A[vi * 16 + vi_i, vj * 16 + vj_i] * 2.0
            for ax0, ax1 in T.grid(16, 16):
                with T.block("C"):
                    vi = T.axis.spatial(128, i * 16 + ax0)
                    vj = T.axis.spatial(128, j * 16 + ax1)
                    T.reads(B[vi, vj])
                    T.writes(C[vi, vj])
                    C[vi, vj] = B[vi, vj] + 1.0

    @T.prim_func
    def after_blockize_rca(
        A: T.Buffer[(128, 128), "float32"],
        C: T.Buffer[(128, 128), "float32"],
    ) -> None:
        B = T.alloc_buffer([128, 128], dtype="float32")
        for i, j in T.grid(8, 8):
            with T.block("B_o"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                T.writes(B[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                for i_1, j_1 in T.grid(16, 16):
                    with T.block("B"):
                        vi_i, vj_i = T.axis.remap("SS", [i_1, j_1])
                        T.reads(A[vi * 16 + vi_i, vj * 16 + vj_i])
                        T.writes(B[vi * 16 + vi_i, vj * 16 + vj_i])
                        B[vi * 16 + vi_i, vj * 16 + vj_i] = A[vi * 16 + vi_i, vj * 16 + vj_i] * 2.0
            with T.block("C_o"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(B[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                T.writes(C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                for ax0, ax1 in T.grid(16, 16):
                    with T.block("C"):
                        vi_i, vj_i = T.axis.remap("SS", [ax0, ax1])
                        T.reads(B[vi * 16 + vi_i, vj * 16 + vj_i])
                        T.writes(C[vi * 16 + vi_i, vj * 16 + vj_i])
                        C[vi * 16 + vi_i, vj * 16 + vj_i] = B[vi * 16 + vi_i, vj * 16 + vj_i] + 1.0

    func = before_blockize_rca
    s = tir.Schedule(func, debug_mask="all")
    _, _, x, _ = s.get_loops(s.get_block("C"))
    s.blockize(x)
    tvm.ir.assert_structural_equal(s.mod["main"], after_blockize_rca)
    verify_trace_roundtrip(sch=s, mod=func)


def test_two_elementwise_blockize_compute_at():
    @T.prim_func
    def before_blockize_compute_at(
        A: T.Buffer[(128, 128), "float32"],
        C: T.Buffer[(128, 128), "float32"],
    ) -> None:
        # body
        # with T.block("root")
        B = T.alloc_buffer([128, 128], dtype="float32")
        for i_0, j_0 in T.grid(8, 8):
            for ax0, ax1 in T.grid(16, 16):
                with T.block("B"):
                    vi = T.axis.spatial(128, i_0 * 16 + ax0)
                    vj = T.axis.spatial(128, j_0 * 16 + ax1)
                    T.reads(A[vi, vj])
                    T.writes(B[vi, vj])
                    B[vi, vj] = A[vi, vj] * 2.0
            with T.block("C_o"):
                vi_o, vj_o = T.axis.remap("SS", [i_0, j_0])
                T.reads(B[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16])
                T.writes(C[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16])
                for i_1, j_1 in T.grid(16, 16):
                    with T.block("C"):
                        vi_i, vj_i = T.axis.remap("SS", [i_1, j_1])
                        T.reads(B[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                        T.writes(C[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                        C[vi_o * 16 + vi_i, vj_o * 16 + vj_i] = (
                            B[vi_o * 16 + vi_i, vj_o * 16 + vj_i] + 1.0
                        )

    @T.prim_func
    def after_blockize_compute_at(
        A: T.Buffer[(128, 128), "float32"],
        C: T.Buffer[(128, 128), "float32"],
    ) -> None:
        B = T.alloc_buffer([128, 128], dtype="float32")
        for i_0, j_0 in T.grid(8, 8):
            with T.block("B_o"):
                vi_o, vj_o = T.axis.remap("SS", [i_0, j_0])
                T.reads(A[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16])
                T.writes(B[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16])
                for ax0, ax1 in T.grid(16, 16):
                    with T.block("B"):
                        vi_i, vj_i = T.axis.remap("SS", [ax0, ax1])
                        T.reads(A[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                        T.writes(B[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                        B[vi_o * 16 + vi_i, vj_o * 16 + vj_i] = (
                            A[vi_o * 16 + vi_i, vj_o * 16 + vj_i] * 2.0
                        )
            with T.block("C_o"):
                vi_o, vj_o = T.axis.remap("SS", [i_0, j_0])
                T.reads(B[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16])
                T.writes(C[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16])
                for i_1, j_1 in T.grid(16, 16):
                    with T.block("C"):
                        vi_i, vj_i = T.axis.remap("SS", [i_1, j_1])
                        T.reads(B[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                        T.writes(C[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                        C[vi_o * 16 + vi_i, vj_o * 16 + vj_i] = (
                            B[vi_o * 16 + vi_i, vj_o * 16 + vj_i] + 1.0
                        )

    func = before_blockize_compute_at
    s = tir.Schedule(func, debug_mask="all")
    _, _, x, _ = s.get_loops(s.get_block("B"))
    s.blockize(x)
    tvm.ir.assert_structural_equal(s.mod["main"], after_blockize_compute_at)
    verify_trace_roundtrip(sch=s, mod=func)


@pytest.mark.parametrize("blockize_loop", [0, 1])
def test_blockize_init_loops(blockize_loop):
    @T.prim_func
    def rowsum(A: T.Buffer[(128, 128), "float32"], B: T.Buffer[(128,), "float32"]):
        for k, i in T.grid(128, 128):
            with T.block("B"):
                vk, vi = T.axis.remap("RS", [k, i])
                with T.init():
                    B[vi] = 0.0
                B[vi] = B[vi] + A[vi, vk]

    if blockize_loop == 0:
        # If the reduction axes remain with the inner block, the init
        # block remains where it is.
        @T.prim_func
        def after_rowsum_blockize(
            A: T.Buffer[(128, 128), "float32"], B: T.Buffer[(128,), "float32"]
        ):
            with T.block("root"):
                T.reads()
                T.writes()
                with T.block("blockized_B"):
                    for i0, i1_1 in T.grid(128, 128):
                        with T.block("B"):
                            vk, vi = T.axis.remap("RS", [i0, i1_1])
                            with T.init():
                                B[vi] = 0.0
                            B[vi] = B[vi] + A[vi, vk]

    elif blockize_loop == 1:
        # If one or more of the reduction axes belong to the outer
        # block, then the init must be hoisted to the outer block.
        @T.prim_func
        def after_rowsum_blockize(
            A: T.Buffer[(128, 128), "float32"], B: T.Buffer[(128,), "float32"]
        ):
            for i0 in T.grid(128):
                with T.block("blockized_B"):
                    vk = T.axis.remap("R", [i0])
                    with T.init():
                        for i1 in T.serial(0, 128):
                            with T.block("B_init"):
                                vi_init = T.axis.S(128, i1)
                                B[vi_init] = T.float32(0)
                    for i1_1 in T.grid(128):
                        with T.block("B"):
                            vi = T.axis.remap("S", [i1_1])
                            B[vi] = B[vi] + A[vi, vk]

    s = tir.Schedule(rowsum, debug_mask="all")
    k = s.get_loops(s.get_block("B"))[blockize_loop]
    s.blockize(k)
    tvm.ir.assert_structural_equal(s.mod["main"], after_rowsum_blockize)
    verify_trace_roundtrip(sch=s, mod=rowsum)


@pytest.mark.parametrize("preserve_unit_iters", [True, False])
def test_blockize_outer_int64_shape(preserve_unit_iters):
    @T.prim_func
    def single_elementwise_int64(
        A: T.Buffer[(T.int64(16), T.int64(128)), "float32"],
        B: T.Buffer[(T.int64(16), T.int64(128)), "float32"],
    ) -> None:
        for i0, j0, i1, j1 in T.grid(T.int64(1), T.int64(8), T.int64(16), T.int64(16)):
            with T.block("B"):
                vi = T.axis.S(T.int64(16), i0 * T.int64(16) + i1)
                vj = T.axis.S(T.int64(128), j0 * T.int64(16) + j1)
                B[vi, vj] = A[vi, vj] + 1.0

    @T.prim_func
    def after_single_elementwise_int64_blockize(
        A: T.Buffer[(T.int64(16), T.int64(128)), "float32"],
        B: T.Buffer[(T.int64(16), T.int64(128)), "float32"],
    ) -> None:
        for i0, j0 in T.grid(T.int64(1), T.int64(8)):
            with T.block("B_o"):
                vi_o = T.axis.spatial(T.int64(1), T.int64(0))
                vj_o = T.axis.spatial(T.int64(8), j0)
                for i1, j1 in T.grid(T.int64(16), T.int64(16)):
                    with T.block("B"):
                        vi_i, vj_i = T.axis.remap("SS", [i1, j1])
                        B[vi_i, vj_o * T.int64(16) + vj_i] = A[
                            vi_i, vj_o * T.int64(16) + vj_i
                        ] + T.float32(1)

    @T.prim_func
    def after_single_elementwise_int64_blockize_preserve_unit_iters(
        A: T.Buffer[(T.int64(16), T.int64(128)), "float32"],
        B: T.Buffer[(T.int64(16), T.int64(128)), "float32"],
    ) -> None:
        for i0, j0 in T.grid(T.int64(1), T.int64(8)):
            with T.block("B_o"):
                vi_o = T.axis.spatial(T.int64(1), i0)
                vj_o = T.axis.spatial(T.int64(8), j0)
                for i1, j1 in T.grid(T.int64(16), T.int64(16)):
                    with T.block("B"):
                        vi_i, vj_i = T.axis.remap("SS", [i1, j1])
                        B[vi_i, vj_o * T.int64(16) + vj_i] = A[
                            vi_i, vj_o * T.int64(16) + vj_i
                        ] + T.float32(1)

    s = tir.Schedule(single_elementwise_int64, debug_mask="all")
    _, _, i1, _ = s.get_loops(s.get_block("B"))
    s.blockize(i1, preserve_unit_iters=preserve_unit_iters)
    expected = (
        after_single_elementwise_int64_blockize_preserve_unit_iters
        if preserve_unit_iters
        else after_single_elementwise_int64_blockize
    )
    tvm.ir.assert_structural_equal(s.mod["main"], expected)
    verify_trace_roundtrip(sch=s, mod=single_elementwise_int64)


class TestBlockizeWithReduction(tvm.testing.CompareBeforeAfter):
    # preserve_unit_iters has an effect when falling through from
    # TrivialSubspaceDivision to arith::SubspaceDivision.  This test
    # has only trivial subspaces, so this parameter shouldn't have an
    # effect.
    preserve_unit_iters = tvm.testing.parameter(
        by_dict={
            "keep_unit_iters": True,
            "remove_unit_iters": False,
        }
    )

    @pytest.fixture
    def transform(self, preserve_unit_iters):
        def transform_pass(mod):
            sch = tir.Schedule(mod, debug_mask="all")
            i0, i1, i2, i3 = sch.get_loops(sch.get_block("B"))
            sch.blockize(i2, preserve_unit_iters=preserve_unit_iters)
            return sch.mod

        return transform_pass

    def before(A: T.Buffer[(1, 8, 1, 8), "float32"], B: T.Buffer[(1, 8), "float32"]):
        for i0, i1, i2, i3 in T.grid(1, 8, 1, 8):
            with T.block("B"):
                v0, v1, v2, v3 = T.axis.remap("SSRR", [i0, i1, i2, i3])

                with T.init():
                    B[v0, v1] = 0.0

                B[v0, v1] = B[v0, v1] + A[v0, v1, v2, v3]

    def expected(A: T.Buffer[(1, 8, 1, 8), "float32"], B: T.Buffer[(1, 8), "float32"]):
        for i0, i1 in T.grid(1, 8):
            with T.block("B_o"):
                v0, v1 = T.axis.remap("SS", [i0, i1])
                for i2, i3 in T.grid(1, 8):
                    with T.block("B"):
                        with T.init():
                            B[v0, v1] = 0.0
                        v2, v3 = T.axis.remap("RR", [i2, i3])
                        B[v0, v1] = B[v0, v1] + A[v0, v1, v2, v3]


if __name__ == "__main__":
    tvm.testing.main()
