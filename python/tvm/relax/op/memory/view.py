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

"""Operations that act on the DLTensor container

While most operations require inspecting the values stored within the
allocated buffers, some operations only require updating the fields in
a `DLTensor`, without touching the values that are stored within it.
For example, given an array of shape `[16,16]`, the slice at
`[0:8,0:16]` can be generated by changing the `DLTensor::shape` field,
while keeping the same underlying data.

"""
from typing import Optional, Sequence, Union

from tvm.tir import PrimExpr
from tvm.relax import Expr, ShapeExpr, DataTypeImm, PrimValue

from . import _ffi_api


PrimExprLike = Union[int, PrimExpr]


def _normalize(expr, relax_cls):
    if expr is None or isinstance(expr, Expr):
        return expr
    else:
        return relax_cls(expr)


def view(
    data: Expr,
    shape: Optional[Union[Sequence[PrimExprLike], Expr]] = None,
    dtype: Optional[Expr] = None,
    relative_byte_offset: Optional[Expr] = None,
) -> Expr:
    """Provide a view into an existing tensor

    The view may have a different shape, may be a different datatype,
    and may start at an offset relative to the source array.

    Regardless of which combination of these options are used, the
    view may never access memory that was not accessible through the
    input `data` array.  This restriction applies even if the `data`
    array is itself a view into a shared backing array.

    Parameters
    ----------
    data : relax.Expr

        The input data to the operator.

    shape : Optional[Union[Sequence[PrimExprLike], Expr]]

        The target shape.  Should be a `relax.ShapeExpr`, or a
        collection that can be converted to a `relax.ShapeExpr`.

    dtype : Optional[Expr]

        The target datatype.  Should be a `relax.ShapeExpr`, or a
        collection that can be converted to a `relax.ShapeExpr`.

    relative_byte_offset: Optional[Expr]

        The offset of the output NDArray, relative to the byte offset
        of `data`.  If `None`, the offset of the view is the same as
        the offset of `data`.

    Returns
    -------
    result : relax.Expr
        The tensor view

    """

    def _normalize(expr, relax_cls):
        if expr is None or isinstance(expr, Expr):
            return expr
        else:
            return relax_cls(expr)

    shape = _normalize(shape, ShapeExpr)
    dtype = _normalize(dtype, DataTypeImm)
    relative_byte_offset = _normalize(relative_byte_offset, PrimValue)

    return _ffi_api.view(data, shape, dtype, relative_byte_offset)  # type: ignore


def ensure_aligned(
    tensor: Expr,
    byte_alignment: Optional[Expr] = None,
) -> Expr:
    """Ensures that a tensor satisfies a minimum alignment

    While compute kernels are frequently written to accept NDArray
    arguments, a compute kernel may have stronger requirements than
    the NDArray format imposes.  For example, a compute kernel may
    require that the `byte_offset` field is zero.  In general, this
    assumption is neither upheld by outputs from Relax operations, nor
    is this assumption checked for arguments passed to Relax
    functions.  If a compute kernels requires additional constraints
    on its NDArray arguments, these extra constraints must be
    explicitly provided.

    The `R.memory.ensure_aligned` operator can be used to produce a
    `R.Tensor` that provides stronger guarantees than Relax provides
    by default.  The `NDArray` generated as output from
    `R.memory.ensure_aligned` will have a `DLTensor::byte_offset` of
    zero, and the `DLTensor::data` pointer will be evenly divisible by
    `byte_alignment`.

    The legalization of this operator depends on the target being used.

    - For any target, the result of `R.memory.ensure_aligned` may
      alias the `tensor` argument, if the `tensor` argument already
      satisfies the conditions for `DLTensor::byte_offset` and
      `DLTensor::data`.

    - For a target that supports host-side pointer arithmetic on the
      `DLTensor::data` pointer (e.g. cuda), the result of
      `R.memory.ensure_aligned` may alias the `tensor` argument, even
      if the `tensor` argument does not satisfy the conditions for
      `DLTensor::byte_offset` and `DLTensor::data`.

    - For a target that does not support host-side pointer arithmetic
      on the `DLTensor::data` pointer (e.g. Vulkan and OpenCL),
      `R.memory.ensure_aligned` may perform a memory allocation.



    Parameters
    ----------
    tensor : relax.Expr

        The input tensor to the operator.

    byte_alignment : Optional[Expr]

        The alignment that should be provided.  Should be a
        `relax.PrimValue`, or convertible to a `relax.PrimValue`.

        If None, will default to `tvm::runtime::kAllocAlignment`,
        providing the same alignment guarantees as are provided by
        TVM allocations.

    Returns
    -------
    result : relax.Expr
        The tensor view

    """

    byte_alignment = _normalize(byte_alignment, PrimValue)

    return _ffi_api.ensure_aligned(tensor, byte_alignment)  # type: ignore
