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
from tvm import te
from tvm.driver.build_module import schedule_to_module
from tvm.script import tir as T
from tvm.relay import GlobalVar


def test_flatten2():
    m = te.size_var("m")
    l = te.size_var("l")
    A = te.placeholder((m, l), name="A")
    A1 = te.compute((m, l), lambda i, j: A[i, j], name="A1")
    A2 = te.compute((m, l), lambda i, j: A1[i, j] + 3, name="A2")

    s = te.create_schedule(A2.op)
    xo, xi = s[A2].split(A2.op.axis[0], 8)
    s[A1].compute_at(s[A2], xo)
    Ab = tvm.tir.decl_buffer(A.shape, A.dtype, name="A")
    A2b = tvm.tir.decl_buffer(A2.shape, A2.dtype, name="A2")

    mod = schedule_to_module(s, [Ab, A2b], binds={A: Ab, A2: A2b})
    mod = tvm.tir.transform.StorageFlatten(64)(mod)


def test_flatten_prefetch():
    A = te.placeholder((25, 100, 4), name="A")
    _A = tvm.tir.decl_buffer(A.shape, A.dtype, name="A")
    i = te.size_var("i")
    j = te.size_var("j")
    region = [tvm.ir.Range.from_min_extent(i[0], i[1]) for i in [(i, 2), (j, 8), (0, 4)]]
    stmt = tvm.tir.Prefetch(_A, region)

    func = tvm.te.schedule.SchedulePostProcToPrimFunc([_A], stmt, {A: _A})

    mod = tvm.IRModule.from_expr(func)
    mod = tvm.transform.Sequential(
        [tvm.tir.transform.StorageFlatten(64), tvm.tir.transform.Simplify()]
    )(mod)
    stmt = mod["main"].body
    assert stmt.extent.value == 2
    assert isinstance(stmt.body, tvm.tir.For)
    assert stmt.body.extent.value == 2


def test_flatten_storage_align():
    m = 8
    l = 16
    A = te.placeholder((m, l), name="A")
    A1 = te.compute((m, l), lambda i, j: A[i, j], name="A1")
    A2 = te.compute((m, l), lambda i, j: A1[i, j] + 3, name="A2")

    s = te.create_schedule(A2.op)
    s[A1].storage_align(A1.op.axis[0], 2, 1)

    mod = schedule_to_module(s, [A, A2])
    mod = tvm.transform.Sequential(
        [tvm.tir.transform.StorageFlatten(64), tvm.tir.transform.Simplify()]
    )(mod)

    stmt = mod["main"].body
    assert stmt.extent.value == 17 * 8


def test_flatten_double_buffer():
    dtype = "int64"
    n = 100
    buffer_size = 4
    tx = te.thread_axis("threadIdx.x")
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    C = ib.pointer("float32", name="C")
    ib.scope_attr(tx, "thread_extent", 1)
    with ib.for_range(0, n) as i:
        B = ib.allocate("float32", buffer_size, name="B", scope="shared")
        with ib.new_scope():
            ib.scope_attr(B.asobject(), "double_buffer_scope", 1)
            with ib.for_range(0, buffer_size) as j:
                B[j] = A[i * 4 + j]
        with ib.for_range(0, buffer_size) as j:
            C[j] = B[j] + 1

    stmt = ib.get()

    mod = tvm.IRModule.from_expr(
        tvm.tir.PrimFunc([A, C], stmt).with_attr("from_legacy_te_schedule", True)
    )

    with tvm.transform.PassContext(config={"tir.InjectDoubleBuffer": {"split_loop": 2}}):
        mod = tvm.transform.Sequential(
            [
                tvm.tir.transform.StorageFlatten(64),
                tvm.tir.transform.InjectDoubleBuffer(),
                tvm.tir.transform.Simplify(),
            ]
        )(mod)

    stmt = mod["main"].body
    assert isinstance(stmt.body, tvm.tir.Allocate)
    assert stmt.body.extent.value == 2 * buffer_size

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A, C], stmt).with_attr("global_symbol", "db"))
    f = tvm.tir.transform.ThreadSync("shared")(mod)["db"]

    count = [0]

    def count_sync(op):
        if isinstance(op, tvm.tir.Call) and op.op.same_as(tvm.ir.Op.get("tir.tvm_storage_sync")):
            count[0] += 1

    tvm.tir.stmt_functor.post_order_visit(f.body, count_sync)
    assert count[0] == 4


@T.prim_func
def tir_func(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [2, 2])
    B = T.match_buffer(a, [2, 2])
    A[0, 1] = B[1, 1]


def test_flatten_tir():
    orig_mod = tvm.IRModule({GlobalVar("main"): tir_func})
    mod = tvm.tir.transform.StorageFlatten(64)(orig_mod)
    tvm.ir.assert_structural_equal(
        orig_mod, mod
    )  # StorageFlatten should do nothing to TIR functions


if __name__ == "__main__":
    test_flatten2()
    test_flatten_storage_align()
    test_flatten_double_buffer()
    test_flatten_prefetch()
    test_flatten_tir()
