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

import tvm
import tvm.testing

from tvm.script import tir as T, ir as I


class BaseCompare(tvm.testing.CompareBeforeAfter):
    transform = tvm.tir.transform.InlineStaticArguments()


class TestInlineStaticArguments(BaseCompare):
    """Subroutine accepts strided buffers, which are passed as parameter"""

    def before(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer((16, 16), "int32")):
                T.func_attr({"global_symbol": "main"})
                mod.subroutine(T.address_of(A[0, 0]), 16, 1)
                mod.subroutine(T.address_of(A[0, 8]), 16, 1)
                mod.subroutine(T.address_of(A[8, 0]), 16, 1)
                mod.subroutine(T.address_of(A[8, 8]), 16, 1)

            @T.prim_func
            def subroutine(A_data: T.handle("int32"), stride_i: T.int32, stride_j: T.int32):
                A = T.decl_buffer((8, 8), "int32", strides=[stride_i, stride_j], data=A_data)
                for i, j in T.grid(8, 8):
                    A[i, j] = 0

        return mod

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer((16, 16), "int32")):
                T.func_attr({"global_symbol": "main"})
                mod.subroutine(T.address_of(A[0, 0]))
                mod.subroutine(T.address_of(A[0, 8]))
                mod.subroutine(T.address_of(A[8, 0]))
                mod.subroutine(T.address_of(A[8, 8]))

            @T.prim_func
            def subroutine(A_data: T.handle("int32")):
                A = T.decl_buffer((8, 8), "int32", strides=[16, 1], data=A_data)
                for i, j in T.grid(8, 8):
                    A[i, j] = 0

        return mod


class TestKeepStaticArgumentsToExternalFunctions(BaseCompare):
    """Like TestInlineStaticArguments, but the subroutine is exposed externally

    Only signatures of internal subroutines may be rewritten.
    """

    def before(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer((16, 16), "int32")):
                T.func_attr({"global_symbol": "main"})
                mod.subroutine(T.address_of(A[0, 0]), 16, 1)
                mod.subroutine(T.address_of(A[0, 8]), 16, 1)
                mod.subroutine(T.address_of(A[8, 0]), 16, 1)
                mod.subroutine(T.address_of(A[8, 8]), 16, 1)

            @T.prim_func
            def subroutine(A_data: T.handle("int32"), stride_i: T.int64, stride_j: T.int64):
                T.func_attr({"global_symbol": "subroutine"})
                A = T.decl_buffer((8, 8), "int32", strides=[stride_i, stride_j], data=A_data)
                for i, j in T.grid(8, 8):
                    A[i, j] = 0

        return mod

    expected = before


class TestInlineStaticBufferShape(BaseCompare):
    """Static shape arguments can be inlined"""

    def before(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer((16, 16), "int32")):
                T.func_attr({"global_symbol": "main"})
                mod.subroutine(
                    T.tvm_stack_make_array(
                        A.data,
                        T.tvm_stack_make_shape(16, 16, dtype="handle"),
                        0,
                        2,
                        A.dtype,
                        0,
                        dtype="handle",
                    )
                )

            @T.prim_func
            def subroutine(a_handle: T.handle):
                M = T.int32()
                N = T.int32()
                A = T.match_buffer(a_handle, (M, N), "int32", strides=[])
                for i, j in T.grid(M, N):
                    A[i, j] = 0

        return mod

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer((16, 16), "int32")):
                T.func_attr({"global_symbol": "main"})
                mod.subroutine(
                    T.tvm_stack_make_array(
                        A.data,
                        T.tvm_stack_make_shape(16, 16, dtype="handle"),
                        0,
                        2,
                        A.dtype,
                        0,
                        dtype="handle",
                    )
                )

            @T.prim_func
            def subroutine(a_handle: T.handle):
                A = T.match_buffer(a_handle, (16, 16), "int32", strides=[])
                for i, j in T.grid(16, 16):
                    A[i, j] = 0

        return mod


class TestInlineStaticBufferStride(BaseCompare):
    """Static stride arguments can be inlined"""

    def before(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer((16, 16), "int32")):
                T.func_attr({"global_symbol": "main"})
                mod.subroutine(
                    T.tvm_stack_make_array(
                        A.data,
                        T.tvm_stack_make_shape(16, 16, dtype="handle"),
                        T.tvm_stack_make_shape(1, 16, dtype="handle"),
                        2,
                        A.dtype,
                        0,
                        dtype="handle",
                    )
                )

            @T.prim_func
            def subroutine(a_handle: T.handle):
                M = T.int32()
                N = T.int32()
                A = T.match_buffer(a_handle, (16, 16), "int32", strides=[M, N])
                for i, j in T.grid(16, 16):
                    A[i, j] = 0

        return mod

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer((16, 16), "int32")):
                T.func_attr({"global_symbol": "main"})
                mod.subroutine(
                    T.tvm_stack_make_array(
                        A.data,
                        T.tvm_stack_make_shape(16, 16, dtype="handle"),
                        T.tvm_stack_make_shape(1, 16, dtype="handle"),
                        2,
                        A.dtype,
                        0,
                        dtype="handle",
                    )
                )

            @T.prim_func
            def subroutine(a_handle: T.handle):
                A = T.match_buffer(a_handle, (16, 16), "int32", strides=[1, 16])
                for i, j in T.grid(16, 16):
                    A[i, j] = 0

        return mod


class TestInlineStaticBufferElementOffset(BaseCompare):
    """Static elem_offset arguments can be inlined"""

    def before(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer((2, 16, 16), "int32")):
                T.func_attr({"global_symbol": "main"})
                mod.subroutine(
                    T.tvm_stack_make_array(
                        T.address_of(A[1, 16, 16]),
                        T.tvm_stack_make_shape(16, 16, dtype="handle"),
                        0,
                        2,
                        A.dtype,
                        256,
                        dtype="handle",
                    )
                )

            @T.prim_func
            def subroutine(a_handle: T.handle):
                elem_offset = T.int32()
                A = T.match_buffer(a_handle, (16, 16), "int32", elem_offset=elem_offset)
                for i, j in T.grid(16, 16):
                    A[i, j] = 0

        return mod

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer((2, 16, 16), "int32")):
                T.func_attr({"global_symbol": "main"})
                mod.subroutine(
                    T.tvm_stack_make_array(
                        T.address_of(A[1, 16, 16]),
                        T.tvm_stack_make_shape(16, 16, dtype="handle"),
                        0,
                        2,
                        A.dtype,
                        256,
                        dtype="handle",
                    )
                )

            @T.prim_func
            def subroutine(a_handle: T.handle):
                A = T.match_buffer(a_handle, (16, 16), "int32", elem_offset=256)
                for i, j in T.grid(16, 16):
                    A[i, j] = 0

        return mod


class TestInlineFullyDynamicBuffer(BaseCompare):
    """All dynamic buffer parameters can be inlined"""

    def before(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer((2, 16, 16), "int32")):
                T.func_attr({"global_symbol": "main"})
                mod.subroutine(
                    T.tvm_stack_make_array(
                        T.address_of(A[1, 16, 16]),
                        T.tvm_stack_make_shape(16, 16, dtype="handle"),
                        T.tvm_stack_make_shape(1, 16, dtype="handle"),
                        2,
                        A.dtype,
                        256,
                        dtype="handle",
                    )
                )

            @T.prim_func
            def subroutine(a_handle: T.handle):
                M = T.int32()
                N = T.int32()
                stride_i = T.int32()
                stride_j = T.int32()
                elem_offset = T.int32()
                A = T.match_buffer(
                    a_handle, (M, N), "int32", strides=[stride_i, stride_j], elem_offset=elem_offset
                )
                for i, j in T.grid(M, N):
                    A[i, j] = 0

        return mod

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer((2, 16, 16), "int32")):
                T.func_attr({"global_symbol": "main"})
                mod.subroutine(
                    T.tvm_stack_make_array(
                        T.address_of(A[1, 16, 16]),
                        T.tvm_stack_make_shape(16, 16, dtype="handle"),
                        T.tvm_stack_make_shape(1, 16, dtype="handle"),
                        2,
                        A.dtype,
                        256,
                        dtype="handle",
                    )
                )

            @T.prim_func
            def subroutine(a_handle: T.handle):
                A = T.match_buffer(a_handle, (16, 16), "int32", strides=[1, 16], elem_offset=256)
                for i, j in T.grid(16, 16):
                    A[i, j] = 0

        return mod


class TestInlineTransitively(BaseCompare):
    """Inlining a static shape param may allow additional inlining"""

    def before(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A_data: T.handle("float32")):
                T.func_attr({"global_symbol": "main"})
                mod.subroutine(A_data, 16, 4)

            @T.prim_func
            def subroutine(A_data: T.handle("float32"), M: T.int32, N: T.int32):
                A = T.decl_buffer(shape=[M, N], dtype="float32", data=A_data)
                for i in range(M):
                    mod.subsubroutine(T.address_of(A[i, 0]), N)

            @T.prim_func
            def subsubroutine(A_data: T.handle("float32"), N: T.int32):
                A = T.decl_buffer(shape=[N], dtype="float32", data=A_data)
                for i in range(N):
                    A[i] = 0.0

        return mod

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A_data: T.handle("float32")):
                T.func_attr({"global_symbol": "main"})
                mod.subroutine(A_data)

            @T.prim_func
            def subroutine(A_data: T.handle("float32")):
                A = T.decl_buffer(shape=[16, 4], dtype="float32", data=A_data)
                for i in range(16):
                    mod.subsubroutine(T.address_of(A[i, 0]))

            @T.prim_func
            def subsubroutine(A_data: T.handle("float32")):
                A = T.decl_buffer(shape=[4], dtype="float32", data=A_data)
                for i in range(4):
                    A[i] = 0.0

        return mod


if __name__ == "__main__":
    tvm.testing.main()
