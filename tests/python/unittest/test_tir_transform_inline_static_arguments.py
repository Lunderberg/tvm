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


class TestInlineStaticArguments(tvm.testing.CompareBeforeAfter):
    """Subroutine accepts strided buffers, which are passed as parameter"""

    transform = tvm.tir.transform.InlineStaticArguments()

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
                A = T.buffer_decl((8, 8), "int32", strides=[stride_i, stride_j], data=A_data)
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
                A = T.buffer_decl((8, 8), "int32", strides=[T.int64(16), T.int64(1)], data=A_data)
                for i, j in T.grid(8, 8):
                    A[i, j] = 0

        return mod


if __name__ == "__main__":
    tvm.testing.main()
