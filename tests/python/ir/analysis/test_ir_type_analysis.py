#!/usr/bin/env python3

import tvm
import tvm.testing

from tvm import te, tir
from tvm.script import tir as T

from tvm.driver.build_module import schedule_to_module
from tvm.ir.ir_type_analysis import analyze_module_ir


@tvm.testing.fixture
def te_module():
    shape = [1024]
    A = te.placeholder(shape, name="A")
    B = te.compute(shape, lambda i: A[i] + tvm.tir.const(1, A.dtype), name="B")
    s = te.create_schedule(B.op)
    return schedule_to_module(s, [A, B])


def test_te_module(te_module):
    details = analyze_module_ir(te_module)
    assert details.is_te_derived


if __name__ == "__main__":
    tvm.testing.main()
