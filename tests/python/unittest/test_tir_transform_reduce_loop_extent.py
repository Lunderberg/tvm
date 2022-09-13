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
import tvm.testing
from tvm.script import tir as T

import pytest

# Tests for ReduceLoopExtents.  This is a weaker form of
# LoopPartition, which only applies if an else case is empty.


@pytest.mark.xfail(reason="Not implemented yet")
class BaseBeforeAfter(tvm.testing.CompareBeforeAfter):
    @tvm.testing.fixture
    def transform(self):
        return tvm.tir.transform.ReduceLoopExtents()


class TestRequireNoopElse(BaseBeforeAfter):
    """Do not modify loop bounds if an else condition is present."""

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            if i < 8:
                A[i] = 0
            else:
                A[i] = 1

    expected = before


class TestReduceUpperBound(BaseBeforeAfter):
    """Only evaluate the loop through the upper bound."""

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            if i < 8:
                A[i] = 0

    def expected(A: T.Buffer[16, "int32"]):
        for i in T.serial(8):
            A[i] = 0


class TestReduceLowerBound(BaseBeforeAfter):
    """Only evaluate the loop starting at the lower bound."""

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            if i >= 8:
                A[i] = 0

    def expected(A: T.Buffer[16, "int32"]):
        for i in T.serial(8, 16):
            A[i] = 0


class TestReduceEquality(BaseBeforeAfter):
    """An extent of one can be reduced to a Let binding."""

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            if i == 8:
                A[i] = 0

    def expected(A: T.Buffer[16, "int32"]):
        i = 8
        A[i] = 0


class TestReduceUpperAndLowerBound(BaseBeforeAfter):
    """Only evaluate the loop between the upper and lower bounds."""

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            if i >= 4 and i < 12:
                A[i] = 0

    def expected(A: T.Buffer[16, "int32"]):
        for i in T.serial(4, 12):
            A[i] = 0


class TestReduceDynamicBound(BaseBeforeAfter):
    """Dynamic bounds can be reduced."""

    def before(a: T.handle, n: T.int32):
        A = T.match_buffer(a, [n], "int32")
        for i in T.serial(n):
            if i >= n // 4 and i < n // 2:
                A[i] = 0

    def expected(a: T.handle, n: T.int32):
        A = T.match_buffer(a, [n], "int32")
        for i in T.serial(n // 4, n // 2):
            A[i] = 0


if __name__ == "__main__":
    tvm.testing.main()
