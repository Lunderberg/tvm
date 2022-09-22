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


@tvm.testing.fixture
def schedulable_tir_module():
    shape = [1024]
    A = te.placeholder(shape, name="A")
    B = te.compute(shape, lambda i: A[i] + tvm.tir.const(1, A.dtype), name="B")
    func = te.create_prim_func([A, B])
    return tvm.IRModule.from_expr(func)


@tvm.testing.fixture
def logical_2d_physical_1d():
    @T.prim_func
    def func(a: T.handle):
        A = T.match_buffer(a, [16, 64], dtype="float32")
        T.evaluate(A[0, 0])

    return tvm.IRModule.from_expr(func)


@tvm.testing.fixture
def logical_2d_physical_2d():
    @T.prim_func
    def func(a: T.handle):
        A = T.match_buffer(a, [16, 64], dtype="float32", axis_separators=[1])
        T.evaluate(A[0, 0])

    return tvm.IRModule.from_expr(func)


@tvm.testing.fixture
def internal_allocation():
    @T.prim_func
    def func():
        A = T.alloc_buffer([16, 16], "float32")
        T.evaluate(A[0, 0])

    return tvm.IRModule.from_expr(func)


def test_te_module(te_module):
    mod = te_module
    details = analyze_module_ir(mod)
    assert details.is_te_derived
    assert details.contains_te_specific_nodes


def test_lowered_te_module(te_module):
    mod = te_module
    mod = tvm.lower(mod)

    details = analyze_module_ir(mod)
    assert details.is_te_derived
    assert not details.contains_te_specific_nodes


def test_schedulable_tir_blocks(schedulable_tir_module):
    mod = schedulable_tir_module
    details = analyze_module_ir(mod)
    assert not details.is_te_derived
    assert not details.contains_te_specific_nodes
    assert details.contains_tir_blocks
    assert details.contains_nonopaque_tir_blocks


def test_opaque_tir_blocks(schedulable_tir_module):
    mod = schedulable_tir_module
    mod = tvm.tir.transform.ConvertBlocksToOpaque()(mod)
    details = analyze_module_ir(mod)
    assert not details.is_te_derived
    assert not details.contains_te_specific_nodes
    assert details.contains_tir_blocks
    assert not details.contains_nonopaque_tir_blocks


def test_flatten_to_1d(logical_2d_physical_1d):
    mod = logical_2d_physical_1d
    details = analyze_module_ir(mod)
    assert details.requires_buffer_flattening


def test_flatten_to_2d(logical_2d_physical_2d):
    mod = logical_2d_physical_2d
    details = analyze_module_ir(mod)
    assert not details.requires_buffer_flattening


def test_internal_alloc_buffer(internal_allocation):
    mod = internal_allocation
    details = analyze_module_ir(mod)
    assert details.contains_internal_allocations
    assert details.contains_block_alloc_buffers


def test_internal_allocate_node(internal_allocation):
    mod = internal_allocation
    mod = tvm.tir.transform.LowerOpaqueBlock()(mod)
    details = analyze_module_ir(mod)
    assert details.contains_internal_allocations
    assert not details.contains_block_alloc_buffers


if __name__ == "__main__":
    tvm.testing.main()
