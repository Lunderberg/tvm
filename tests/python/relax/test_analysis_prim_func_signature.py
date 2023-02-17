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

from typing import List, Set, Union

import pytest

import tvm
import tvm.testing
from tvm import tir
from tvm import relax as rx

from tvm.script import relax as R, tir as T

from tvm.relax.analysis import prim_func_signature
from tvm.relax.struct_info import FuncStructInfo, TensorStructInfo


class Base:
    def __init_subclass__(cls):
        cls.prim_func = tvm.testing.CompareBeforeAfter._normalize_before(cls.prim_func)

        expected = cls.expected
        if not hasattr(expected, "_pytestfixturefunction"):

            def inner(self):
                return expected

            cls.expected = pytest.fixture(inner)

    def test_signature(self, prim_func, expected):
        output = prim_func_signature(prim_func)
        tvm.ir.assert_structural_equal(output, expected)


class TestOutput(Base):
    def prim_func(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            A[i] = 0

    expected = FuncStructInfo(
        params=[],
        ret=TensorStructInfo((16,), "int32"),
    )


class TestInput(Base):
    def prim_func(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            T.evaluate(A[i])

    expected = FuncStructInfo(
        params=[TensorStructInfo((16,), "int32")],
        ret=None,
    )


class TestCopy(Base):
    def prim_func(A: T.Buffer[16, "int32"], B: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            B[i] = A[i]

    expected = FuncStructInfo(
        params=[TensorStructInfo((16,), "int32")],
        ret=TensorStructInfo((16,), "int32"),
    )


class TestDynamic(Base):
    def prim_func(a: T.handle, b: T.handle):
        N = T.var("int64")
        A = T.match_buffer(a, N, "int32")
        B = T.match_buffer(b, N, "int32")
        for i in T.serial(N):
            B[i] = A[i]

    @pytest.fixture
    def expected(self):
        N = tir.Var("N", "int64")
        return FuncStructInfo(
            params=[TensorStructInfo((N,), "int32")],
            ret=TensorStructInfo((N,), "int32"),
        )


if __name__ == "__main__":
    tvm.testing.main()
