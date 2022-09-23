#!/usr/bin/env python3

import inspect
import textwrap

import pytest

import tvm
import tvm.testing

from tvm import te, tir, relay
from tvm.script import tir as T

from tvm.driver.build_module import schedule_to_module
from tvm.ir.ir_type_analysis import analyze_module_ir


@tvm.testing.fixture
def relay_module():
    dtype = "float32"
    arg = relay.var("arg", shape=(relay.Any(),), dtype=dtype)
    func = relay.Function([arg], relay.sqrt(arg))
    mod = tvm.IRModule.from_expr(func)
    return mod


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


@tvm.testing.fixture
def block_buffer_view():
    @T.prim_func
    def func(A: T.Buffer[(16, 64), "float32"]):
        with T.block():
            B = T.match_buffer(A[8:16, 32:64], [8, 32], offset_factor=1)
            T.evaluate(B[0, 0])

    return tvm.IRModule.from_expr(func)


@tvm.testing.fixture
def attribute_buffer_view():
    @T.prim_func
    def func(A: T.Buffer[(16, 64), "float32"]):
        T.func_attr({"from_legacy_te_schedule": True})
        stride_i = T.var("int32")
        stride_j = T.var("int32")
        B = T.buffer_decl([8, 32], dtype="float32", offset_factor=1, strides=[stride_i, stride_j])
        T.attr(
            [B, A],
            "buffer_bind_scope",
            T.tvm_tuple(
                # min/extent on dim1
                8,
                8,
                # min/extent on dim2
                32,
                32,
                dtype="handle",
            ),
        )
        T.evaluate(B[0, 0])

    return tvm.IRModule.from_expr(func)


@tvm.testing.fixture
def buffer_argument():
    @T.prim_func
    def func(A: T.Buffer[1, "float32"]):
        T.func_attr({"global_symbol": "func", "target": T.target("nvidia/nvidia-a100")})
        T.evaluate(A[0])

    return tvm.IRModule.from_expr(func)


def test_relay_module(relay_module):
    mod = relay_module
    details = analyze_module_ir(mod)
    assert details.contains_relay_function


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


def test_buffer_view_in_block(block_buffer_view):
    mod = block_buffer_view
    details = analyze_module_ir(mod)
    assert details.uses_buffer_views_in_block
    assert not details.uses_buffer_views_by_attribute


def test_removed_buffer_view_in_block(block_buffer_view):
    mod = block_buffer_view
    mod = tvm.tir.transform.LowerMatchBuffer()(mod)
    details = analyze_module_ir(mod)
    assert not details.uses_buffer_views_in_block
    assert not details.uses_buffer_views_by_attribute


def test_buffer_view_by_attribute(attribute_buffer_view):
    mod = attribute_buffer_view
    details = analyze_module_ir(mod)
    assert not details.uses_buffer_views_in_block
    assert details.uses_buffer_views_by_attribute


def test_removed_buffer_view_by_attribute(attribute_buffer_view):
    mod = attribute_buffer_view
    mod = tvm.tir.transform.StorageFlatten(64)(mod)
    details = analyze_module_ir(mod)
    assert not details.uses_buffer_views_in_block
    assert not details.uses_buffer_views_by_attribute


def test_tir_buffer_argument(buffer_argument):
    mod = buffer_argument
    details = analyze_module_ir(mod)
    assert details.has_tir_buffer_arguments
    assert not details.has_packed_api_buffer_arguments
    assert not details.has_unpacked_api_buffer_arguments


def test_packed_api_buffer_argument(buffer_argument):
    mod = buffer_argument
    mod = tvm.tir.transform.MakePackedAPI()(mod)
    details = analyze_module_ir(mod)
    assert not details.has_tir_buffer_arguments
    assert details.has_packed_api_buffer_arguments
    assert not details.has_unpacked_api_buffer_arguments


def test_unpacked_api_buffer_argument(buffer_argument):
    mod = buffer_argument
    mod = tvm.tir.transform.MakeUnpackedAPI()(mod)
    details = analyze_module_ir(mod)
    # Probably should remove the buffer_map arguments.  No longer
    # needed, now that the buffer itself is kept around in
    # BufferLoad/BufferStore/DeclBuffer.  That would improve API
    # consistency, that a non-empty buffer_map implies that lowering
    # is required.
    assert details.has_tir_buffer_arguments
    assert not details.has_packed_api_buffer_arguments
    assert details.has_unpacked_api_buffer_arguments


class CompileOnlyCases:
    def tvm_call_packed(A: T.Buffer[(128, 128), "float32"]):
        T.func_attr({"global_symbol": "func", "target": T.target("nvidia/nvidia-a100")})
        T.attr("default", "device_id", 0)
        T.attr("default", "device_type", 1)
        T.evaluate(
            T.tvm_call_packed(
                "packed_func_name",
                T.tvm_stack_make_array(
                    A.data,
                    T.tvm_stack_make_shape(128, 128, dtype="handle"),
                    0,
                    2,
                    0.0,
                    0,
                    dtype="handle",
                ),
                dtype="int32",
            )
        )


class HostSideCases:
    def tvm_call_packed_lowered(A: T.Buffer[(128, 128), "float32"]):
        # body
        stack_tcode: T.Ptr[T.int32] = T.tvm_stack_alloca("arg_tcode", 2, dtype="handle")
        stack_tcode_1 = T.buffer_decl([T.uint64(2)], dtype="int32", data=stack_tcode)
        stack_value: T.handle = T.tvm_stack_alloca("arg_value", 2, dtype="handle")
        stack_array: T.handle = T.tvm_stack_alloca("array", 1, dtype="handle")
        stack_shape: T.Ptr[T.int64] = T.tvm_stack_alloca("shape", 2, dtype="handle")
        stack_shape_1 = T.buffer_decl([T.int64(2)], dtype="int64", data=stack_shape)
        stack_shape_2 = T.buffer_decl([1], dtype="int64", data=stack_shape)
        stack_shape_1[0] = T.int64(128)
        stack_shape_1[1] = T.int64(128)
        T.evaluate(T.tvm_struct_set(stack_array, 0, 1, A.data, dtype="int32"))
        T.evaluate(
            T.tvm_struct_set(
                stack_array, 0, 2, T.address_of(stack_shape_2[0], dtype="handle"), dtype="int32"
            )
        )
        T.evaluate(
            T.tvm_struct_set(
                stack_array, 0, 3, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"
            )
        )
        T.evaluate(T.tvm_struct_set(stack_array, 0, 4, 2, dtype="int32"))
        T.evaluate(T.tvm_struct_set(stack_array, 0, 5, T.uint8(2), dtype="int32"))
        T.evaluate(T.tvm_struct_set(stack_array, 0, 6, T.uint8(32), dtype="int32"))
        T.evaluate(T.tvm_struct_set(stack_array, 0, 7, T.uint16(1), dtype="int32"))
        T.evaluate(T.tvm_struct_set(stack_array, 0, 8, T.uint64(0), dtype="int32"))
        T.evaluate(T.tvm_struct_set(stack_array, 0, 9, 0, dtype="int32"))
        T.evaluate(T.tvm_struct_set(stack_array, 0, 10, 1, dtype="int32"))
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
            T.tvm_call_packed_lowered(
                "packed_func_name", stack_value, stack_tcode, 0, 1, dtype="int32"
            )
        )


class DeviceSideCases:
    def tvm_fill_fragment(A: T.Buffer[2048, "float32"]):
        T.evaluate(T.tvm_fill_fragment(A.data, 16, 16, 16, 0, T.float32(0), dtype="handle"))


@pytest.fixture(
    params=[value for key, value in HostSideCases.__dict__.items() if not key.startswith("_")]
)
def host_side_mod(request):
    source_code = "@T.prim_func\n" + textwrap.dedent(inspect.getsource(request.param))
    func = tvm.script.from_source(source_code)
    return tvm.IRModule.from_expr(func)


@pytest.fixture(
    params=[value for key, value in DeviceSideCases.__dict__.items() if not key.startswith("_")]
)
def device_side_mod(request):
    source_code = "@T.prim_func\n" + textwrap.dedent(inspect.getsource(request.param))
    func = tvm.script.from_source(source_code)
    return tvm.IRModule.from_expr(func)


@pytest.fixture(
    params=[value for key, value in CompileOnlyCases.__dict__.items() if not key.startswith("_")]
)
def compile_time_only_mod(request):
    source_code = "@T.prim_func\n" + textwrap.dedent(inspect.getsource(request.param))
    func = tvm.script.from_source(source_code)
    return tvm.IRModule.from_expr(func)


def test_host_side_only(host_side_mod):
    mod = host_side_mod
    details = analyze_module_ir(mod)
    assert details.is_host_only
    assert not details.is_device_only


def test_device_side_only(device_side_mod):
    mod = device_side_mod
    details = analyze_module_ir(mod)
    assert not details.is_host_only
    assert details.is_device_only


def test_compile_time_only(compile_time_only_mod):
    mod = compile_time_only_mod
    details = analyze_module_ir(mod)
    assert details.is_compile_time_only


if __name__ == "__main__":
    tvm.testing.main()
