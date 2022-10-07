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
from tvm import te
from tvm.script import tir as T


def nop():
    return tvm.tir.Evaluate(0)


def test_remove_no_op():
    i = te.var("i")
    j = te.var("j")
    k = te.var("k")
    m = te.var("m")
    n = te.var("n")
    dtype = "int64"
    Ab = tvm.tir.decl_buffer((n,), dtype)
    stmt = tvm.tir.For(
        i,
        0,
        4,
        tvm.tir.ForKind.SERIAL,
        tvm.tir.For(
            j,
            0,
            n,
            tvm.tir.ForKind.SERIAL,
            tvm.tir.For(
                k,
                0,
                m,
                tvm.tir.ForKind.SERIAL,
                tvm.tir.IfThenElse((i * m + j + k < n), tvm.tir.Evaluate(m), tvm.tir.Evaluate(n)),
            ),
        ),
    )

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([Ab], stmt))
    ret = tvm.tir.transform.RemoveNoOp()(mod)["main"].body

    assert isinstance(ret, tvm.tir.Evaluate)
    store = tvm.tir.BufferStore(Ab, tvm.tir.BufferLoad(Ab, [i]) + 1, [i + 1])
    stmt2 = tvm.tir.SeqStmt([nop(), tvm.tir.SeqStmt([store, nop()])])

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([Ab], stmt2))
    ret = tvm.tir.transform.RemoveNoOp()(mod)["main"].body
    assert ret == store

    # remove zero extent loop
    stmt3 = tvm.tir.For(i, 0, 0, tvm.tir.ForKind.SERIAL, store)
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([Ab], stmt3))
    ret = tvm.tir.transform.RemoveNoOp()(mod)["main"].body
    assert isinstance(ret, tvm.tir.Evaluate)


def test_remove_no_op_with_invalid_extent():
    @T.prim_func
    def main(A: T.Buffer[(16), "int32"], B: T.Buffer[(16), "int32"]) -> None:
        for i in T.serial(16):
            for j in T.serial(i - 20):
                B[i] = A[i] + j

    mod = tvm.ir.module.IRModule.from_expr(main)
    ret = tvm.tir.transform.RemoveNoOp()(mod)["main"].body
    assert isinstance(ret, tvm.tir.Evaluate)


class BaseBeforeAfter(tvm.testing.CompareBeforeAfter):
    transform = tvm.tir.transform.RemoveNoOp()


class TestRemoveEmptyForLoop(BaseBeforeAfter):
    """A for-loop whose body is a no-op is itself a no-op."""

    def before():
        for i in T.serial(16):
            T.evaluate(0)

    def expected():
        T.evaluate(0)


class TestRemoveZeroExtentLoop(BaseBeforeAfter):
    """A for-loop with no extent is a no-op."""

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(0):
            A[i] = 42

    def expected(A: T.Buffer[16, "int32"]):
        T.evaluate(0)


class TestRemoveUnusedLet(BaseBeforeAfter):
    """A let statement that is never used is a no-op."""

    def before(A: T.Buffer[16, "int32"]):
        x = 5
        for i in T.serial(16):
            A[i] = 0

    def expected(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            A[i] = 0


class TestRemoveLetUsedOnlyInNoOp(BaseBeforeAfter):
    """A let statement that is never used is a no-op.

    Similar to TestRemoveUnusedLet, but the usage of the let binding
    may have been removed by an earlier removal of another no-op.
    """

    def before(A: T.Buffer[16, "int32"]):
        x = 5
        for i in T.serial(0):
            A[i] = x

    def expected(A: T.Buffer[16, "int32"]):
        T.evaluate(0)


class TestKeepSideEffectsOfLet(BaseBeforeAfter):
    """The side effects of a no-op let must be kept."""

    def before():
        x = T.call_extern("extern_func", dtype="int32")
        T.evaluate(0)

    def expected():
        T.evaluate(T.call_extern("extern_func", dtype="int32"))


class TestRemoveEmptyThenCase(BaseBeforeAfter):
    """A no-op then_case can be removed."""

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            if i < 8:
                T.evaluate(0)
            else:
                A[i] = 42

    def expected(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            if not (i < 8):
                A[i] = 42


class TestRemoveEmptyElseCase(BaseBeforeAfter):
    """A no-op else_case can be removed."""

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            if i < 8:
                A[i] = 42
            else:
                T.evaluate(0)

    def expected(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            if i < 8:
                A[i] = 42


class TestRemoveUnusedWrite(BaseBeforeAfter):
    """For two sequential writes, the first is a no-op"""

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            A[i] = 100
            A[i] = 42

    def expected(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            A[i] = 42


class TestKeepSideEffectsOfUnusedWrite(BaseBeforeAfter):
    """For two sequential writes, the first value may have side effects"""

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            A[i] = T.call_extern("extern_func", dtype="int32")
            A[i] = 42

    def expected(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            T.evaluate(T.call_extern("extern_func", dtype="int32"))
            A[i] = 42


class TestKeepFirstWriteWhenUsed(BaseBeforeAfter):
    """For two sequential writes, keep the first if it is used"""

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            A[i] = 100
            A[i] = A[i] + 1

    expected = before


class TestRemoveOverwrittenLoop(BaseBeforeAfter):
    """Remove repeated writes to the same region

    If two loops write to the same region, the first is a no-op.
    """

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            A[i] = 100

        for i in T.serial(16):
            A[i] = 42

    def expected(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            A[i] = 42


class TestRemoveOverwrittenSubloop(BaseBeforeAfter):
    """Remove repeated writes to the same region

    If the first loop writes to a subset of the region, the first loop
    is a no-op.  Similar to TestRemoveOverwrittenLoop, but the first
    loop's extents are a subset of the second loop.
    """

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(4, 12):
            A[i] = 100

        for i in T.serial(16):
            A[i] = 42

    def expected(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            A[i] = 42


class TestKeepPartiallyOverwrittenLoop(BaseBeforeAfter):
    """Keep partially overwritten regions

    If the second loop doesn't entirely overwrite the first, the first
    may not be removed be kept.
    """

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            A[i] = 100

        for i in T.serial(16):
            if i < 12:
                A[i] = 42

    expected = before


class TestRemoveOverwrittenPredicatedLoop(BaseBeforeAfter):
    """Remove repeated writes to the same predicated region.

    Similar to TestKeepPartiallyOverwrittenLoop, except the first loop
    has the same predicate as the second, and can therefore be
    removed.
    """

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            if i < 12:
                A[i] = 100

        for i in T.serial(16):
            if i < 12:
                A[i] = 42

    def expected(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            if i < 12:
                A[i] = 100


class TestRemoveOverwrittenPredicatedLoop(BaseBeforeAfter):
    """Remove repeated writes to the same predicated region.

    Similar to TestRemoveOverwrittenPredicatedLoop, except the first
    loop's predicate is not a precise match for the second loop's
    predicate.  So long as the regions written in the first loop are a
    subset of those written in the second loop, they can be removed.
    """

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            if i < 10:
                A[i] = 100

        for i in T.serial(16):
            if i // 4 < 3:
                A[i] = 42

    def expected(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            if i // 4 < 3:
                A[i] = 42


class TestRemoveSeparatedOverwrites(BaseBeforeAfter):
    """Remove repeated writes to the same predicated region.

    Similar to TestRemoveOverwrittenLoopRegion, but with an
    independent loop between the first and second write of the buffer.
    """

    def before(A: T.Buffer[16, "int32"], B: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            A[i] = 100

        for i in T.serial(16):
            B[i] = 0

        for i in T.serial(16):
            A[i] = 42

    def expected(A: T.Buffer[16, "int32"], B: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            B[i] = 0

        for i in T.serial(16):
            A[i] = 42


class TestRemoveSeparatedOverwriteOfPredicatedLoop(BaseBeforeAfter):
    """Remove repeated writes to the same predicated region.

    Similar to TestRemoveSeparatedOverwrites, but the independent loop
    between the first and second writes writes to a different subset
    of the same buffer.
    """

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            if i < 12:
                A[i] = 100

        for i in T.serial(16):
            if i > 12:
                A[i] = 15

        for i in T.serial(16):
            if i < 12:
                A[i] = 42

    def expected(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            if i > 12:
                A[i] = 15

        for i in T.serial(16):
            if i < 12:
                A[i] = 42


class TestRemoveReadWrite(BaseBeforeAfter):
    """Writing a value to the same location as was just read is a no-op."""

    def before(A: T.Buffer[1, "int32"]):
        A[0] = A[0]

    def expected(A: T.Buffer[1, "int32"]):
        T.evaluate(0)


class TestKeepReadWriteToDifferentIndices(BaseBeforeAfter):
    """Writing a value to a different index should not be removed"""

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(15):
            A[i] = A[i + 1]

    expected = before


class TestRemoveReadWriteSameIndexDifferentExpression(BaseBeforeAfter):
    """Writing a value to the same location as the read is a no-op.

    If the value of the index can be proven to be the same, then the
    no-op can be removed, even if they have different forms of the
    expression.
    """

    def before(A: T.Buffer[16, "int32"]):
        for io, ii in T.grid(4, 4):
            i = 4 * io + ii
            A[4 * io + ii] = A[i]

    def expected(A: T.Buffer[16, "int32"]):
        T.evaluate(0)


class TestRemoveReadWriteSameIndexUsingConstraint(BaseBeforeAfter):
    """Writing a value to the same location as the read is a no-op.

    If the value of the index can be proven to be the same, then the
    no-op can be removed.  This may require using the a constraint
    that is known from a conditional containing the read/write.
    """

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            if i != 0:
                A[i] = A[i - 1]
            else:
                A[i] = A[0]

    def expected(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            if i != 0:
                A[i] = A[i - 1]


class TestRemoveWritingOfKnownValue(BaseBeforeAfter):
    """Writing a value that already exists at that index is a no-op"""

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            A[i] = i

        A[4] = 4

    def expected(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            A[i] = i


class TestKeepOneOfDuplicateLoops(BaseBeforeAfter):
    """Must not reason based on a touch point after removing it.

    If the first loop is removed because it is overwritten by the
    second loop, and the second loop is removed because it writes the
    same value as the first loop, the overall transformation is no
    longer valid.  In this case, only one of the two should be
    removed.
    """

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            A[i] = i

        for i in T.serial(16):
            A[i] = i

    def expected(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            A[i] = i


if __name__ == "__main__":
    tvm.testing.main()
