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

from tvm import tir
from tvm.script import tir as T, ir as I


class TestLowerToPointer(tvm.testing.CompareBeforeAfter):
    transform = tvm.tir.transform.LowerBufferArguments()

    def before(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer(16, "int32")):
                T.func_attr({"global_symbol": "main"})
                mod.subroutine(A[0:8])

            @T.prim_func
            def subroutine(A: T.Buffer(8, "int32")):
                for i in T.serial(8):
                    A[i] = 0

        return mod

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer(16, "int32")):
                T.func_attr({"global_symbol": "main"})
                mod.subroutine(A.data)

            @T.prim_func
            def subroutine(A_data: T.handle("int32")):
                A = T.buffer_decl(8, "int32", data=A_data)
                for i in T.serial(8):
                    A[i] = 0

        return mod


class TestErrorForNonContiguousParameter(tvm.testing.CompareBeforeAfter):
    """Subroutine requires contiguous buffers, but is provided strided buffers"""

    transform = tvm.tir.transform.LowerBufferArguments()

    def before(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer((16, 16), "int32")):
                T.func_attr({"global_symbol": "main"})
                mod.subroutine(A[0:8, 0:8])
                mod.subroutine(A[0:8, 8:16])
                mod.subroutine(A[8:16, 0:8])
                mod.subroutine(A[8:16, 8:16])

            @T.prim_func
            def subroutine(A: T.Buffer((8, 8), "int32")):
                for i, j in T.grid(8, 8):
                    A[i, j] = 0

        return mod

    expected = tvm.TVMError


class TestLowerToPointerAndStride(tvm.testing.CompareBeforeAfter):
    """Subroutine accepts strided buffers, which are passed as parameter"""

    transform = tvm.tir.transform.LowerBufferArguments()

    def before(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer((16, 16), "int32")):
                T.func_attr({"global_symbol": "main"})
                mod.subroutine(A[0:8, 0:8])
                mod.subroutine(A[0:8, 8:16])
                mod.subroutine(A[8:16, 0:8])
                mod.subroutine(A[8:16, 8:16])

            @T.prim_func
            def subroutine(a: T.handle):
                elem_offset = T.int32()
                sh = T.int32()
                sw = T.int32()
                A = T.match_buffer(
                    a,
                    (8, 8),
                    "int32",
                    elem_offset=elem_offset,
                    strides=[sh, sw],
                )
                for i, j in T.grid(8, 8):
                    A[i, j] = 0

        return mod

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer((16, 16), "int32")):
                T.func_attr({"global_symbol": "main"})
                mod.subroutine(A.data, 0, 16, 1)
                mod.subroutine(A.data, 8, 16, 1)
                mod.subroutine(A.data, 128, 16, 1)
                mod.subroutine(A.data, 136, 16, 1)

            @T.prim_func
            def subroutine(
                A_data: T.handle("int32"),
                elem_offset: T.int32,
                stride_i: T.int32,
                stride_j: T.int32,
            ):
                A = T.buffer_decl(
                    (8, 8),
                    "int32",
                    elem_offset=elem_offset,
                    strides=[stride_i, stride_j],
                    data=A_data,
                )
                for i, j in T.grid(8, 8):
                    A[i, j] = 0

        return mod


if __name__ == "__main__":
    tvm.testing.main()
