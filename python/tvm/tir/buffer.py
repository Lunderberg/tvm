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
"""Abstraction for array data structures."""
from numbers import Integral
from typing import Optional

import tvm._ffi
from tvm._ffi.base import string_types
from tvm.runtime import Object, convert
from tvm.ir import PrimExpr, PointerType, PrimType
from . import _ffi_api
from .expr import Var


@tvm._ffi.register_object("tir.Buffer")
class Buffer(Object):
    """Symbolic data buffer in TVM.

    Buffer provide a way to represent data layout
    specialization of data structure in TVM.

    Do not construct directly, use :py:func:`~decl_buffer` instead.
    See the documentation of :py:func:`decl_buffer` for more details.

    See Also
    --------
    decl_buffer : Declare a buffer
    """

    READ = 1
    WRITE = 2

    def access_ptr(self, access_mask, ptr_type="handle", content_lanes=1, offset=0):
        """Get an access pointer to the head of buffer.

        This is the recommended method to get buffer data
        ptress when interacting with external functions.

        Parameters
        ----------
        access_mask : int
            The access pattern MASK. Indicate whether the
            access will read or write to the data content.

        ptr_type : str, optional
            The data type of the result pointer. Do not specify
            unless we want to cast pointer to specific type.

        content_lanes: int, optional
            The number of lanes for the data type. This value
            is greater than one for vector types.

        offset: Expr, optional
            The offset of pointer. We can use it to offset by
            the number of elements from the address of ptr.

        Examples
        --------
        .. code-block:: python

          # Get access ptr for read
          buffer.access_ptr("r")
          # Get access ptr for read/write with bitmask
          buffer.access_ptr(Buffer.READ | Buffer.WRITE)
          # Get access ptr for read/write with str flag
          buffer.access_ptr("rw")
          # Get access ptr for read with offset
          buffer.access_ptr("r", offset = 100)
        """
        if isinstance(access_mask, string_types):
            mask = 0
            for value in access_mask:
                if value == "r":
                    mask = mask | Buffer.READ
                elif value == "w":
                    mask = mask | Buffer.WRITE
                else:
                    raise ValueError("Unknown access_mask %s" % access_mask)
            access_mask = mask
        offset = convert(offset)
        return _ffi_api.BufferAccessPtr(
            self, access_mask, ptr_type, content_lanes, offset  # type: ignore
        )

    def vload(self, begin, dtype=None):
        """Generate an Expr that loads dtype from begin index.

        Parameters
        ----------
        begin : Array of Expr
            The beginning index in unit of Buffer.dtype

        dtype : str
            The data type to be loaded,
            can be vector type which have lanes that is multiple of Buffer.dtype

        Returns
        -------
        load : Expr
            The corresponding load expression.
        """
        begin = (begin,) if isinstance(begin, (int, PrimExpr)) else begin
        dtype = dtype if dtype else self.dtype
        return _ffi_api.BufferVLoad(self, begin, dtype)  # type: ignore

    def vstore(self, begin, value):
        """Generate a Stmt that store value into begin index.

        Parameters
        ----------
        begin : Array of Expr
            The beginning index in unit of Buffer.dtype

        value : Expr
            The value to be stored.

        Returns
        -------
        store : Stmt
            The corresponding store stmt.
        """
        begin = (begin,) if isinstance(begin, (int, PrimExpr)) else begin
        return _ffi_api.BufferVStore(self, begin, value)  # type: ignore

    def scope(self):
        """Return the storage scope associated with this buffer.
        Returns
        -------
        scope : str
            The storage scope associated with this buffer.
        """
        return _ffi_api.BufferStorageScope(self)  # type: ignore

    @property
    def elem_offset(self):
        """Returns the elem_offset of a flat array

        Backwards compatibility function for interaction with 1-d
        memory, prior to refactoring parameters for each physical axis
        into `physical_axes`.
        """
        physical_axes = self.physical_axes
        if len(physical_axes) > 1:
            raise RuntimeError("Can only use Buffer.elem_offset on flat 1-d memory")
        else:
            return physical_axes[0].elem_offset

    @elem_offset.setter
    def elem_offset(self, value):
        """Sets the elem_offset of a flat array

        Backwards compatibility function for interaction with 1-d
        memory, prior to refactoring parameters for each physical axis
        into `physical_axes`.
        """
        physical_axes = self.physical_axes
        if len(physical_axes) > 1:
            raise RuntimeError("Can only use Buffer.elem_offset on flat 1-d memory")
        else:
            physical_axes[0].elem_offset = value

    @property
    def offset_factor(self):
        """Returns the offset_factor of a flat array

        Backwards compatibility function for interaction with 1-d
        memory, prior to refactoring parameters for each physical axis
        into `physical_axes`.
        """
        physical_axes = self.physical_axes
        if len(physical_axes) > 1:
            raise RuntimeError("Can only use Buffer.offset_factor on flat 1-d memory")
        else:
            return physical_axes[0].offset_factor

    @offset_factor.setter
    def offset_factor(self, value):
        """Sets the offset_factor of a flat array

        Backwards compatibility function for interaction with 1-d
        memory, prior to refactoring parameters for each physical axis
        into `physical_axes`.
        """
        physical_axes = self.physical_axes
        if len(physical_axes) > 1:
            raise RuntimeError("Can only use Buffer.offset_factor on flat 1-d memory")
        else:
            physical_axes[0].offset_factor = value


def decl_buffer(
    shape,
    dtype=None,
    name="buffer",
    data=None,
    strides=None,
    elem_offset=None,
    scope="",
    data_alignment=-1,
    offset_factor=0,
    buffer_type="",
    span=None,
    physical_axes=None,
):
    """Declare a new symbolic buffer.

    Normally buffer is created automatically during lower and build.
    This is only needed if user want to specify their own buffer layout.

    See the note below for detailed discussion on usage of buffer.

    Parameters
    ----------
    shape : Union[tvm.tir.PrimExpr, Tuple[tvm.tir.PrimExpr]]
        The logical shape of the buffer.

    dtype : Optional[str]
        The data type of the buffer.

    name : Optional[str]
        The name of the buffer.

    data : Optional[tvm.tir.Var]
        The data pointer in the buffer.

    strides: Optional[List[tvm.tir.PrimExpr]]
        The stride of the buffer.

    elem_offset: Optional[PrimExpr]
        The beginning offset of the array to data.
        In terms of number of elements of dtype.

        If `elem_offset` is specified, `physical_axes` may not be
        specified.

    scope: Optional[str]

        The storage scope of the buffer.  If unset, or set as the
        empty string, the scope defaults to "global".

    data_alignment: Optional[int]

        The alignment of data pointer in bytes.  If -1 is passed, the
        alignment will be set to TVM's internal default,
        `kAllocAlignment`.

    offset_factor: int

        The factor of elem_offset field.  When set, `elem_offset` is
        required to be multiple of `offset_factor`.  If 0 is passed,
        the alignment will be set to 1.  if non-zero is passed, we
        will created a Var for elem_offset if elem_offset is not None.

        If `offset_factor` is specified as non-zero, `physical_axes` may not
        be specified.

    buffer_type: Optional[str], optional, {"", "auto_broadcast"}

        Must be either empty string or "auto_broadcast".  An
        auto_broadcast buffer can be used in a broadcast computation
        without considering whether dimension size equals to one.
        That is, for an auto_broadcast buffer, TVM maps `buffer[i, j,
        k]` to `buffer[i, 0, k]` if dimension `j`'s shape equals 1.

    span: Optional[Span]
        The location of the decl_buffer creation in the source.

    physical_axes : Optional[List[BufferParamsPerPhysicalAxis]]

        A list of tuples describing the N-d physical axes of the
        allocation that will back this buffer.  If passed, neither
        `elem_offset` nor `offset_factor` may be specified.

    Returns
    -------
    buffer : tvm.tir.Buffer
        The created buffer

    Example
    -------
    Here's an example of how broadcast buffer can be used to define a symbolic broadcast operation,

    .. code-block:: python

        m0, m1, m2 = te.var("m0"), te.var("m1"), te.var("m2")
        n0, n1, n2 = te.var("n0"), te.var("n1"), te.var("n2")
        o0, o1, o2 = te.var("o0"), te.var("o1"), te.var("o2")
        A = te.placeholder((m0, m1, m2), name='A')
        B = te.placeholder((n0, n1, n2), name='B')
        C = te.compute((o0, o1, o2), lambda i, j, k: A[i, j, k] + B[i, j, k], name='C')
        Ab = tvm.tir.decl_buffer(A.shape, A.dtype, name="Ab", buffer_type="auto_broadcast")
        Bb = tvm.tir.decl_buffer(B.shape, B.dtype, name="Bb", buffer_type="auto_broadcast")
        s = te.create_schedule(C.op)
        fadd = tvm.build(s, [A, B, C], target='llvm', name='bcast_add', binds={A:Ab, B:Bb})
        dev = tvm.cpu(0)
        a = tvm.nd.array(np.random.uniform(size=(2, 4, 3)).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=(2, 1, 3)).astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros((2, 4, 3), dtype=C.dtype), dev)
        fadd(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

    Note
    ----
    Buffer data structure reflects the DLTensor structure in dlpack.
    While DLTensor data structure is very general, it is usually helpful
    to create function that only handles specific case of data structure
    and make compiled function benefit from it.

    If user pass strides and elem_offset is passed as None
    when constructing the function, then the function will be specialized
    for the DLTensor that is compact and aligned.
    If user pass a fully generic symbolic array to the strides,
    then the resulting function becomes fully generic.

    """

    shape = (shape,) if isinstance(shape, (PrimExpr, Integral)) else shape
    dtype = "float32" if dtype is None else dtype
    strides = () if strides is None else strides

    shape_dtype = shape[0].dtype if shape and hasattr(shape[0], "dtype") else "int32"

    if physical_axes is None:
        physical_axes = [BufferParamsPerPhysicalAxis(0, elem_offset, offset_factor)]
    elif elem_offset is not None:
        raise TypeError("elem_offset may not be specified if physical_axes is specified")
    elif offset_factor != 0:
        raise TypeError("offset_factor may not be specified if physical_axes is specified")

    if data is None:
        # Bool is represented as uint1 in the IR, but stored as int8
        storage_type = PrimType(dtype)
        storage_type = PrimType("int8") if storage_type.dtype == "bool" else storage_type
        data = Var(name, PointerType(storage_type, scope), span)

    return _ffi_api.Buffer(  # type: ignore
        data,
        dtype,
        shape,
        strides,
        name,
        data_alignment,
        physical_axes,
        buffer_type,
        span,
    )


@tvm._ffi.register_object("tir.BufferParamsPerPhysicalAxis")
class BufferParamsPerPhysicalAxis(Object):
    def __init__(
        self,
        first_tensor_axis: int = 0,
        elem_offset: Optional[PrimExpr] = None,
        offset_factor: int = 0,
    ):
        """Parameters
        ----------
        first_tensor_axis: int

            The ``first_tensor_axis`` to be used to generate a physical axis.
            That is, the ``i``-th physical axis is generated using a row-major
            traversal of all tensor axes from ``physical_axes[i][0]``
            (inclusive) to ``physical_axis[i+1][0]`` (exclusive), or to the
            last tensor axis for the last physical axis.
            ``physical_axes[0][0]`` must equal zero.  `physical_axes` must be
            sorted in increasing order of the ``first_tensor_axis``.

        elem_offset: Optional[PrimExpr]

            Interpreted identically to `elem_offset` argument to
            `decl_buffer`.

        offset_factor: Optional[PrimExpr]

            Interpreted identically to `offset_factor` argument to
            `decl_buffer`.

        """
        # shape_dtype = shape[0].dtype if shape and hasattr(shape[0], "dtype") else "int32"
        shape_dtype = "int32"

        if offset_factor != 0 and elem_offset is None:
            elem_offset = Var("elem_offset", shape_dtype)
        self.__init_handle_by_constructor__(
            _ffi_api.BufferParamsPerPhysicalAxis, first_tensor_axis, elem_offset, offset_factor
        )


@tvm._ffi.register_object("tir.DataProducer")
class DataProducer(Object):
    pass
