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

import numpy as np

import tvm
import tvm.testing

from tvm import te
from tvm.script import tir as T


@tvm.register_func("tvm.test_matmul")
def my_matmul(a, b, c):
    c.copyfrom(np.dot(a.numpy(), b.numpy()))


class TestLowerPackedFunc(tvm.testing.CompareBeforeAfter):
    transform = tvm.tir.transform.LowerTVMStackParameters()

    def before(a: T.handle, b: T.handle, c: T.handle):
        s1 = T.var("int32")
        s2 = T.var("int32")
        s3 = T.var("int32")
        A = T.match_buffer(a, [16, 16], dtype="float64", strides=[s1, 1], offset_factor=1)
        B = T.match_buffer(b, [16, 16], dtype="float64", strides=[s2, 1], offset_factor=1)
        C = T.match_buffer(c, [16, 16], dtype="float64", strides=[s3, 1], offset_factor=1)
        # body
        T.attr("default", "device_id", 0)
        T.attr("default", "device_type", 1)
        for i in T.parallel(10):
            T.evaluate(
                T.tvm_call_packed(
                    "tvm.test_matmul",
                    T.tvm_stack_make_array(
                        A.data,
                        T.tvm_stack_make_shape(16, 16, dtype="handle"),
                        T.tvm_stack_make_shape(s1, 1, dtype="handle"),
                        2,
                        T.float64(0),
                        A.elem_offset,
                        dtype="handle",
                    ),
                    T.tvm_stack_make_array(
                        B.data,
                        T.tvm_stack_make_shape(16, 16, dtype="handle"),
                        T.tvm_stack_make_shape(s2, 1, dtype="handle"),
                        2,
                        T.float64(0),
                        B.elem_offset,
                        dtype="handle",
                    ),
                    T.tvm_stack_make_array(
                        C.data,
                        T.tvm_stack_make_shape(16, 16, dtype="handle"),
                        T.tvm_stack_make_shape(s3, 1, dtype="handle"),
                        2,
                        T.float64(0),
                        C.elem_offset,
                        dtype="handle",
                    ),
                    dtype="int32",
                )
            )

    def expected(a: T.handle, b: T.handle, c: T.handle):
        # function attr dict
        s1 = T.var("int32")
        s2 = T.var("int32")
        s3 = T.var("int32")
        A = T.match_buffer(a, [16, 16], dtype="float64", strides=[s1, 1], offset_factor=1)
        B = T.match_buffer(b, [16, 16], dtype="float64", strides=[s2, 1], offset_factor=1)
        C = T.match_buffer(c, [16, 16], dtype="float64", strides=[s3, 1], offset_factor=1)
        # body
        for i in T.parallel(10):
            stack_tcode: T.Ptr[T.int32] = T.tvm_stack_alloca("arg_tcode", 4, dtype="handle")
            stack_tcode_1 = T.buffer_decl([T.uint64(4)], dtype="int32", data=stack_tcode)
            stack_value: T.handle = T.tvm_stack_alloca("arg_value", 4, dtype="handle")
            stack_array: T.handle = T.tvm_stack_alloca("array", 3, dtype="handle")
            stack_shape: T.Ptr[T.int64] = T.tvm_stack_alloca("shape", 12, dtype="handle")
            stack_shape_1 = T.buffer_decl([T.int64(12)], dtype="int64", data=stack_shape)
            stack_shape_2 = T.buffer_decl([1], dtype="int64", data=stack_shape)
            stack_shape_3 = T.buffer_decl([3], dtype="int64", data=stack_shape)
            stack_shape_4 = T.buffer_decl([5], dtype="int64", data=stack_shape)
            stack_shape_5 = T.buffer_decl([7], dtype="int64", data=stack_shape)
            stack_shape_6 = T.buffer_decl([9], dtype="int64", data=stack_shape)
            stack_shape_7 = T.buffer_decl([11], dtype="int64", data=stack_shape)
            stack_shape_1[0] = T.int64(16)
            stack_shape_1[1] = T.int64(16)
            stack_shape_1[2] = T.cast(s1, "int64")
            stack_shape_1[3] = T.int64(1)
            T.evaluate(T.tvm_struct_set(stack_array, 0, 1, A.data, dtype="int32"))
            T.evaluate(
                T.tvm_struct_set(
                    stack_array, 0, 2, T.address_of(stack_shape_2[0], dtype="handle"), dtype="int32"
                )
            )
            T.evaluate(
                T.tvm_struct_set(
                    stack_array, 0, 3, T.address_of(stack_shape_3[2], dtype="handle"), dtype="int32"
                )
            )
            T.evaluate(T.tvm_struct_set(stack_array, 0, 4, 2, dtype="int32"))
            T.evaluate(T.tvm_struct_set(stack_array, 0, 5, T.uint8(2), dtype="int32"))
            T.evaluate(T.tvm_struct_set(stack_array, 0, 6, T.uint8(64), dtype="int32"))
            T.evaluate(T.tvm_struct_set(stack_array, 0, 7, T.uint16(1), dtype="int32"))
            T.evaluate(
                T.tvm_struct_set(
                    stack_array, 0, 8, T.cast(A.elem_offset * 8, "uint64"), dtype="int32"
                )
            )
            T.evaluate(T.tvm_struct_set(stack_array, 0, 9, 0, dtype="int32"))
            T.evaluate(T.tvm_struct_set(stack_array, 0, 10, 1, dtype="int32"))
            stack_shape_1[4] = T.int64(16)
            stack_shape_1[5] = T.int64(16)
            stack_shape_1[6] = T.cast(s2, "int64")
            stack_shape_1[7] = T.int64(1)
            T.evaluate(T.tvm_struct_set(stack_array, 1, 1, B.data, dtype="int32"))
            T.evaluate(
                T.tvm_struct_set(
                    stack_array, 1, 2, T.address_of(stack_shape_4[4], dtype="handle"), dtype="int32"
                )
            )
            T.evaluate(
                T.tvm_struct_set(
                    stack_array, 1, 3, T.address_of(stack_shape_5[6], dtype="handle"), dtype="int32"
                )
            )
            T.evaluate(T.tvm_struct_set(stack_array, 1, 4, 2, dtype="int32"))
            T.evaluate(T.tvm_struct_set(stack_array, 1, 5, T.uint8(2), dtype="int32"))
            T.evaluate(T.tvm_struct_set(stack_array, 1, 6, T.uint8(64), dtype="int32"))
            T.evaluate(T.tvm_struct_set(stack_array, 1, 7, T.uint16(1), dtype="int32"))
            T.evaluate(
                T.tvm_struct_set(
                    stack_array, 1, 8, T.cast(B.elem_offset * 8, "uint64"), dtype="int32"
                )
            )
            T.evaluate(T.tvm_struct_set(stack_array, 1, 9, 0, dtype="int32"))
            T.evaluate(T.tvm_struct_set(stack_array, 1, 10, 1, dtype="int32"))
            stack_shape_1[8] = T.int64(16)
            stack_shape_1[9] = T.int64(16)
            stack_shape_1[10] = T.cast(s3, "int64")
            stack_shape_1[11] = T.int64(1)
            T.evaluate(T.tvm_struct_set(stack_array, 2, 1, C.data, dtype="int32"))
            T.evaluate(
                T.tvm_struct_set(
                    stack_array, 2, 2, T.address_of(stack_shape_6[8], dtype="handle"), dtype="int32"
                )
            )
            T.evaluate(
                T.tvm_struct_set(
                    stack_array,
                    2,
                    3,
                    T.address_of(stack_shape_7[10], dtype="handle"),
                    dtype="int32",
                )
            )
            T.evaluate(T.tvm_struct_set(stack_array, 2, 4, 2, dtype="int32"))
            T.evaluate(T.tvm_struct_set(stack_array, 2, 5, T.uint8(2), dtype="int32"))
            T.evaluate(T.tvm_struct_set(stack_array, 2, 6, T.uint8(64), dtype="int32"))
            T.evaluate(T.tvm_struct_set(stack_array, 2, 7, T.uint16(1), dtype="int32"))
            T.evaluate(
                T.tvm_struct_set(
                    stack_array, 2, 8, T.cast(C.elem_offset * 8, "uint64"), dtype="int32"
                )
            )
            T.evaluate(T.tvm_struct_set(stack_array, 2, 9, 0, dtype="int32"))
            T.evaluate(T.tvm_struct_set(stack_array, 2, 10, 1, dtype="int32"))
            T.evaluate(
                T.tvm_struct_set(
                    stack_value,
                    0,
                    12,
                    T.tvm_struct_get(stack_array, 0, 0, dtype="handle"),
                    dtype="int32",
                )
            )
            stack_tcode_1[0] = 7
            T.evaluate(
                T.tvm_struct_set(
                    stack_value,
                    1,
                    12,
                    T.tvm_struct_get(stack_array, 1, 0, dtype="handle"),
                    dtype="int32",
                )
            )
            stack_tcode_1[1] = 7
            T.evaluate(
                T.tvm_struct_set(
                    stack_value,
                    2,
                    12,
                    T.tvm_struct_get(stack_array, 2, 0, dtype="handle"),
                    dtype="int32",
                )
            )
            stack_tcode_1[2] = 7
            T.evaluate(
                T.tvm_call_packed_lowered(
                    "tvm.test_matmul", stack_value, stack_tcode, 0, 3, dtype="int32"
                )
            )


@tvm.testing.parametrize_targets("llvm", "stackvm")
def test_lower_packed_func(target):
    target = tvm.target.Target(target)

    ib = tvm.tir.ir_builder.create()

    m = n = k = 16

    #
    # Prepare buffer for a, b and c:
    #
    a = te.placeholder((m, k), name="a", dtype="float64")
    b = te.placeholder((k, n), name="b", dtype="float64")
    k = te.reduce_axis((0, k), name="k")
    c = te.compute((m, n), lambda i, j: te.sum(a[i, k] * b[k, j], axis=k), name="c")

    a_buffer = tvm.tir.decl_buffer(
        a.shape, a.dtype, name="a_buffer", offset_factor=1, strides=[te.var("s1"), 1]
    )
    b_buffer = tvm.tir.decl_buffer(
        b.shape, b.dtype, name="b_buffer", offset_factor=1, strides=[te.var("s2"), 1]
    )
    c_buffer = tvm.tir.decl_buffer(
        c.shape, c.dtype, name="c_buffer", offset_factor=1, strides=[te.var("s3"), 1]
    )

    ib.scope_attr("default", "device_id", 0)
    ib.scope_attr("default", "device_type", target.kind.device_type)

    with ib.for_range(0, 10, "i", kind="parallel"):
        ib.emit(tvm.tir.call_packed("tvm.test_matmul", a_buffer, b_buffer, c_buffer))

    stmt = ib.get()

    # Construct a valid IRModule to be lowered:
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([a_buffer, b_buffer, c_buffer], stmt))

    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("target", target))(mod)
    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("global_symbol", "main"))(mod)
    # mod = tvm.tir.transform.MakePackedAPI()(mod)

    # Do the lowering:
    mod = tvm.tir.transform.LowerTVMStackParameters()(mod)

    # Get the PrimFunc from module:
    prim_func = mod.functions.items()[0][1]

    node = prim_func.body

    # Recursively visit PrimFunc until we meet the for-loop:
    while isinstance(node, (tvm.tir.AssertStmt, tvm.tir.LetStmt, tvm.tir.AttrStmt)):
        node = node.body

    # For-loop:
    assert isinstance(node, tvm.tir.stmt.For)

    #
    # let stack_tcode = tir.tvm_stack_alloca("arg_tcode", 4)
    #
    alloca_tcode = node.body
    assert isinstance(alloca_tcode, tvm.tir.LetStmt)

    expected_value = tvm.tir.call_intrin(
        "handle", tvm.ir.Op.get("tir.tvm_stack_alloca"), "arg_tcode", 4
    )
    expected_var = alloca_tcode.var
    expected_stmt = tvm.tir.LetStmt(expected_var, expected_value, alloca_tcode.body)

    tvm.ir.assert_structural_equal(alloca_tcode, expected_stmt, map_free_vars=True)

    #
    # let stack_value = tir.tvm_stack_alloca("arg_value", 4)
    #
    alloca_value = alloca_tcode.body
    assert isinstance(alloca_value, tvm.tir.LetStmt)

    expected_value = tvm.tir.call_intrin(
        "handle", tvm.ir.Op.get("tir.tvm_stack_alloca"), "arg_value", 4
    )
    expected_var = alloca_value.var
    expected_stmt = tvm.tir.LetStmt(expected_var, expected_value, alloca_value.body)

    tvm.ir.assert_structural_equal(alloca_value, expected_stmt, map_free_vars=True)

    #
    # let stack_array = tir.tvm_stack_alloca("array", 3)
    #
    alloca_array = alloca_value.body
    assert isinstance(alloca_array, tvm.tir.LetStmt)

    expected_value = tvm.tir.call_intrin(
        "handle", tvm.ir.Op.get("tir.tvm_stack_alloca"), "array", 3
    )
    expected_var = alloca_array.var
    expected_stmt = tvm.tir.LetStmt(expected_var, expected_value, alloca_array.body)

    tvm.ir.assert_structural_equal(alloca_array, expected_stmt, map_free_vars=True)

    #
    # let stack_shape = tir.tvm_stack_alloca("shape", 12)
    #
    alloca_shape = alloca_array.body
    assert isinstance(alloca_shape, tvm.tir.LetStmt)

    expected_value = tvm.tir.call_intrin(
        "handle", tvm.ir.Op.get("tir.tvm_stack_alloca"), "shape", 12
    )
    expected_var = alloca_shape.var
    expected_stmt = tvm.tir.LetStmt(expected_var, expected_value, alloca_shape.body)

    tvm.ir.assert_structural_equal(alloca_shape, expected_stmt, map_free_vars=True)


@tvm.testing.parametrize_targets("llvm")
def test_call_packed_return_non_i32(target):
    # This call packed that return non i32 types
    expected_value = np.array([1.2, 1.4], dtype="float32")

    @T.prim_func
    def func(buffer: T.Buffer[2, "float32"]):
        T.func_attr({"global_symbol": "packed_test"})
        buffer[0] = T.tvm_call_packed("testing.echo", T.float32(1.2), dtype="float32")
        Aptr_dup: T.Ptr[T.float32] = T.tvm_call_packed("testing.echo", buffer.data, dtype="handle")
        buffer_alias = T.buffer_decl(2, "float32", data=Aptr_dup)
        buffer_alias[1] = T.float32(1.4)

    mod = tvm.IRModule.from_expr(func)

    f = tvm.build(mod, None, target)
    a = tvm.nd.array(np.zeros(2, dtype="float32"))
    f(a)
    tvm.testing.assert_allclose(a.numpy(), expected_value)


if __name__ == "__main__":
    tvm.testing.main()
