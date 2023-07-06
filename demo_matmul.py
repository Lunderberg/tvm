#!/usr/bin/env python3

import numpy as np

import tvm
from tvm import tir
import tvm.testing

from tvm.script import tir as T, ir as I


dtype = tvm.testing.parameter("float32")
use_dynamic_shape = tvm.testing.parameter(
    by_dict={
        "static": False,
        # Neither StorageRewrite nor CompactBufferAllocation
        # recognizes that each split is accessing distinct regions of
        # the cache read/write buffers.
        #
        # "dynamic": True,
    }
)


@tvm.testing.fixture
def matmul_module(use_dynamic_shape, dtype):
    if use_dynamic_shape:
        M = tir.Var("M", "int64")
        K = tir.Var("K", "int64")
        N = tir.Var("N", "int64")
    else:
        M, K, N = 128, 64, 128

    @I.ir_module
    class mod:
        @T.prim_func
        def func(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "func"})
            A = T.match_buffer(a, [M, K], dtype=dtype)
            B = T.match_buffer(b, [K, N], dtype=dtype)
            C = T.match_buffer(c, [M, N], dtype=dtype)

            for m, n, k in T.grid(M, N, K):
                with T.block("matmul"):
                    vm, vn, vk = T.axis.remap("SSR", [m, n, k])
                    with T.init():
                        C[vm, vn] = 0.0

                    C[vm, vn] = C[vm, vn] + A[vm, vk] * B[vk, vn]

    sch = tvm.tir.Schedule(mod)
    sch.work_on("func")

    matmul = sch.get_block("matmul")
    m, n, k = sch.get_loops(matmul)
    m_outer, m_inner = sch.split(m, [8, None])

    # sch.annotate(m_outer, "target", tvm.target.Target("cuda"))
    # sch.annotate(m_outer, "target", tvm.target.DynamicTarget("cuda"))
    sch.annotate(
        m_outer,
        "target",
        tvm.target.DynamicTarget(
            "cuda",
            device_id=sch.get(m_outer).loop_var,
        ),
    )

    return sch.mod


@tvm.testing.fixture
def np_data(dtype):
    m, k, n = 128, 64, 128
    a_np = np.random.random((m, k)).astype(dtype)
    b_np = np.random.random((k, n)).astype(dtype)
    c_np = a_np @ b_np
    return a_np, b_np, c_np


@tvm.testing.parametrize_targets("llvm")
def test_matmul(matmul_module, np_data, target, dev):
    from lunderberg_tvm_instrument import PrintTransformSequence

    with tvm.transform.PassContext(instruments=[PrintTransformSequence()]):
        built = tvm.build(matmul_module, target=target)

    a_np, b_np, c_np = np_data
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.array(np.zeros_like(c_np), device=dev)
    built(a_tvm, b_tvm, c_tvm)

    c_actual = c_tvm.numpy()

    tvm.testing.assert_allclose(c_np, c_actual, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    tvm.testing.main()
