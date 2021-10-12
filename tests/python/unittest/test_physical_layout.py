#!/usr/bin/env python3

import functools
import sys
import pytest

import numpy as np

import tvm
import tvm.testing
from tvm import te
from tvm.tir.stmt_functor import post_order_visit
from tvm.driver.build_module import schedule_to_module

dtype = tvm.testing.parameter("int32")

A_definition_style = tvm.testing.parameter(
    "explicit_reshape",
    "set_physical_layout",
)
B_definition_style = tvm.testing.parameter(
    "explicit_reshape",
    "set_physical_layout",
)

reordered_shape = tvm.testing.parameter((2, 3, 4))


@tvm.testing.fixture
def n_items(reordered_shape):
    return functools.reduce(lambda x, y: x * y, reordered_shape, 1)


@tvm.testing.fixture
def fphysical_layout(reordered_shape):
    def mapping(i):
        output = []
        for dim in reversed(reordered_shape):
            output.append(i % dim)
            i //= dim
        return output[::-1]

    return mapping


@tvm.testing.fixture
def fcompute(A_definition_style, B_definition_style, reordered_shape, n_items, dtype):
    assert A_definition_style in ["explicit_reshape", "set_physical_layout"]
    assert B_definition_style in ["explicit_reshape", "set_physical_layout"]

    def func():
        if A_definition_style == "explicit_reshape":
            A_input = te.placeholder(shape=reordered_shape, name="A_input", dtype=dtype)
            A = te.compute(
                shape=(n_items,),
                fcompute=lambda i: A_input[
                    i // (reordered_shape[1] * reordered_shape[2]),
                    (i // reordered_shape[2]) % reordered_shape[1],
                    i % reordered_shape[2],
                ],
                name="A",
            )

        elif A_definition_style == "set_physical_layout":
            A = te.placeholder(shape=(n_items,), name="A", dtype=dtype)
            A_input = A

        B = te.compute(shape=A.shape, fcompute=lambda i: A[i], name="B")

        if B_definition_style == "explicit_reshape":
            B_output = te.compute(
                shape=reordered_shape,
                fcompute=lambda i, j, k: B[
                    i * reordered_shape[1] * reordered_shape[2] + j * reordered_shape[2] + k
                ],
                name="B_output",
            )
        elif B_definition_style == "set_physical_layout":
            B_output = B

        return A_input, B_output

    return func


def traverse(s, op, callback):
    visited = set()

    def _traverse(op):
        if op in visited:
            return
        visited.add(op)
        for tensor in op.input_tensors:
            _traverse(tensor.op)
        callback(op)

    _traverse(op)


@tvm.testing.fixture
def fschedule(A_definition_style, B_definition_style, fphysical_layout):
    def func(outs):
        outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
        s = te.create_schedule([x.op for x in outs])

        def callback(op):
            if (op.name == "A" and A_definition_style == "set_physical_layout") or (
                op.name == "B" and B_definition_style == "set_physical_layout"
            ):
                s[op].set_physical_layout(fphysical_layout)

        traverse(s, outs[0].op, callback)
        return s

    return func


@tvm.testing.parametrize_targets("llvm")
def test_external_reshape(target, dev, fcompute, fschedule, n_items, reordered_shape, dtype):
    A, B = fcompute()
    s = fschedule(B)

    func = tvm.build(s, [A, B], target=target, name="copy_reshape")

    a_np = np.arange(n_items).reshape(reordered_shape).astype(dtype)
    b_np = np.arange(n_items).reshape(reordered_shape).astype(dtype)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.empty(b_np.shape, dtype=dtype, device=dev)

    func(a, b)

    tvm.testing.assert_allclose(b.numpy(), b_np)


@tvm.testing.parametrize_targets("llvm")
def test_internal_reshape(target, dev, n_items, reordered_shape, dtype, fphysical_layout):
    # The reshaping of the buffer gets flattened away in
    # StorageFlatten.  Therefore, testing the behavior by running only
    # ApplyPhysicalLayout.
    logical_shape = (n_items,)
    A = te.placeholder(logical_shape, name="A", dtype=dtype)
    B = te.compute(shape=logical_shape, fcompute=lambda i: A[i], name="B")
    C = te.compute(shape=logical_shape, fcompute=lambda i: B[i], name="C")

    s = te.create_schedule(C.op)
    s[B].set_physical_layout(fphysical_layout)

    mod = schedule_to_module(s, [A, C])
    body = mod["main"].body

    def walk_buffer_interactions(stmt, callback):
        buffer_classes = [
            tvm.tir.BufferLoad,
            tvm.tir.BufferStore,
            tvm.tir.BufferRealize,
        ]

        def inner(node):
            if (type(node) in buffer_classes) and node.buffer.name == "B":
                callback(node)

        post_order_visit(stmt, inner)

    # All references to the buffer are the same object
    def check_references():
        buffer_object = None

        def inner(node):
            nonlocal buffer_object
            if buffer_object is None:
                buffer_object = node.buffer
            else:
                assert node.buffer.same_as(buffer_object)

        return inner

    # The buffer has the expected shape.
    def check_shape(expected_shape):
        def inner(node):
            assert tuple(node.buffer.shape) == expected_shape

        return inner

    # Before the transform, the buffer should be in the logical shape.
    walk_buffer_interactions(body, check_references())
    walk_buffer_interactions(body, check_shape(logical_shape))

    mod = tvm.tir.transform.ApplyPhysicalLayout()(mod)
    body = mod["main"].body

    # After the transform, the buffer should be in the physical shape.
    walk_buffer_interactions(body, check_references())
    walk_buffer_interactions(body, check_shape(reordered_shape))


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
