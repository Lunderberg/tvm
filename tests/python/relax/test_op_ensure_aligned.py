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
from tvm.script import ir as I, relax as R, tir as T

import numpy as np
import pytest

TVM_ALLOC_ALIGNMENT: int = tvm.tir.decl_buffer([]).data_alignment


def test_infer_sinfo_ensure_alignment():
    """R.memory.ensure_aligned does not change the argument's type

    The Relax IR does not represent a buffer's offset.  The type
    returned by `R.memory.ensure_aligned` is identical to the input
    type.

    """

    @R.function(private=True)
    def explicit_sinfo(A: R.Tensor([4096], "float16")) -> R.Tensor([4096], "float16"):
        B: R.Tensor([4096], "float16") = R.memory.ensure_aligned(
            A, byte_alignment=R.prim_value(1024)
        )
        return B

    @R.function(private=True)
    def inferred_sinfo(A: R.Tensor([4096], "float16")):
        B = R.memory.ensure_aligned(A, 1024)
        return B

    tvm.ir.assert_structural_equal(explicit_sinfo, inferred_sinfo)


def test_infer_sinfo_with_default_alignment():
    """If unspecified, uses tvm::runtime::kAllocAlignment

    This is applied on conversion to Relax IR.  Within the Relax IR,
    the byte alignment provided by `ensure_aligned` is always
    present.

    """

    @R.function(private=True)
    def explicit_sinfo(A: R.Tensor([4096], "float16")) -> R.Tensor([4096], "float16"):
        B: R.Tensor([4096], "float16") = R.memory.ensure_aligned(
            A, R.prim_value(TVM_ALLOC_ALIGNMENT)
        )
        return B

    @R.function(private=True)
    def inferred_sinfo(A: R.Tensor([4096], "float16")):
        B = R.memory.ensure_aligned(A)
        return B

    tvm.ir.assert_structural_equal(explicit_sinfo, inferred_sinfo)


def test_error_for_empty_tuple_as_alignment():
    """The byte_alignment must be specified in Relax IR

    Even though the Python API and TVMScript accept `None`, the Relax
    representation of `None` is not supported.  The translation to use
    `tvm::runtime::kAllocAlignment` is only done when performed Relax
    IR.

    """

    with pytest.raises(tvm.TVMError):

        @R.function
        def func(A: R.Tensor([4096], "float16")):
            B = R.memory.ensure_aligned(A, R.tuple())
            return B


def test_error_for_negative_byte_alignment():
    """The byte_alignment must be a positive value"""

    with pytest.raises(tvm.TVMError):

        @R.function
        def func(A: R.Tensor([4096], "float16")):
            B = R.memory.ensure_aligned(A, R.prim_value(-1))
            return B


def test_error_for_zero_byte_alignment():
    """The byte_alignment must be a positive value"""

    with pytest.raises(tvm.TVMError):

        @R.function
        def func(A: R.Tensor([4096], "float16")):
            B = R.memory.ensure_aligned(A, R.prim_value(0))
            return B


def test_infer_sinfo_with_python_integer_alignment():
    """Syntactic sugar, `ensure_aligned` accepts python integers"""

    @R.function(private=True)
    def using_relax_prim_value(A: R.Tensor([4096], "float16")) -> R.Tensor([4096], "float16"):
        B: R.Tensor([4096], "float16") = R.memory.ensure_aligned(A, R.prim_value(128))
        return B

    @R.function(private=True)
    def using_python_int(A: R.Tensor([4096], "float16")):
        B = R.memory.ensure_aligned(A, 128)
        return B

    tvm.ir.assert_structural_equal(using_relax_prim_value, using_python_int)


def test_infer_sinfo_with_dynamic_alignment():
    """The byte_alignment may be a dynamic expression"""

    @R.function(private=True)
    def explicit_sinfo(
        A: R.Tensor([4096], "float16"),
        _: R.Prim(value="elem_alignment"),
    ) -> R.Tensor([4096], "float16"):
        elem_alignment = T.int64()
        B: R.Tensor([4096], "float16") = R.memory.ensure_aligned(
            A,
            byte_alignment=elem_alignment * 2,
        )
        return B

    @R.function(private=True)
    def inferred_sinfo(A: R.Tensor([4096], "float16"), _: R.Prim(value="elem_alignment")):
        elem_alignment = T.int64()
        B = R.memory.ensure_aligned(
            A,
            byte_alignment=elem_alignment * 2,
        )
        return B

    tvm.ir.assert_structural_equal(explicit_sinfo, inferred_sinfo)


def test_legalize_ensure_aligned():
    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([4096], "float16")):
            B = R.memory.ensure_aligned(A, 128)
            return B

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor([4096], "float16")):
            cls = Expected
            B = cls.ensure_aligned(A)
            return B

        @R.function(private=True)
        def ensure_aligned(A: R.Tensor([4096], "float16")) -> R.Tensor([4096], "float16"):
            cls = Expected

            is_aligned = cls.is_aligned(A)
            if is_aligned:
                output = A
            else:
                output = R.call_tir(cls.copy_to_aligned, [A], out_sinfo=R.Tensor([4096], "float16"))
            return output

        @T.prim_func(private=True)
        def is_aligned(A_handle: T.handle) -> T.bool:
            elem_offset = T.int64()
            A = T.match_buffer(A_handle, 4096, "float16", elem_offset=elem_offset)

            is_offset_zero = elem_offset == 0
            is_aligned = T.reinterpret("int64", T.address_of(A[0])) % 128 == 0
            return is_offset_zero and is_aligned

        @T.prim_func(private=True)
        def copy_to_aligned(
            A: T.Buffer(4096, "float16", offset_factor=1), B: T.Buffer(4096, "float16")
        ):
            for i in range(4096):
                with T.block("copy"):
                    B[i] = A[i]

    After = tvm.relax.transform.LegalizeOps()(Before)
    tvm.ir.assert_structural_equal(Expected, After)


# @tvm.testing.parametrize_targets("llvm", "cuda")
# def test_execute_ensure_aligned(target, dev):
#     target = tvm.target.Target(target)

#     # @I.ir_module
#     # class Module:
#     #     @R.function
#     #     def main(A: R.Tensor([4096], "float16")):
#     #         B = R.memory.ensure_aligned(A, 128)
#     #         return B

#     @I.ir_module
#     class Module:
#         @R.function
#         def main(A: R.Tensor([4096], "float16")):
#             cls = Module
#             B = cls.ensure_aligned(A)
#             return B

#         @R.function(private=True)
#         def ensure_aligned(A: R.Tensor([4096], "float16")) -> R.Tensor([4096], "float16"):
#             cls = Module

#             is_aligned = cls.is_aligned(A)
#             if is_aligned:
#                 output = A
#             else:
#                 output = R.call_tir(cls.copy_to_aligned, [A], out_sinfo=R.Tensor([4096], "float16"))
#             return output

#         @T.prim_func(private=True)
#         def is_aligned(A_handle: T.handle) -> T.bool:
#             elem_offset = T.int64()
#             A = T.match_buffer(A_handle, 4096, "float16", elem_offset=elem_offset)

#             is_offset_zero = elem_offset == 0
#             is_aligned = T.reinterpret("int64", T.address_of(A[0])) % 128 == 0
#             return is_offset_zero and is_aligned

#         @T.prim_func(private=True)
#         def copy_to_aligned(
#             A: T.Buffer(4096, "float16", offset_factor=1), B: T.Buffer(4096, "float16")
#         ):
#             for i in range(4096):
#                 with T.block("copy"):
#                     vi = T.axis.remap("S", [i])
#                     B[vi] = A[vi]

#     seq = [tvm.relax.transform.LegalizeOps()]
#     if "gpu" in target.keys:
#         seq.append(tvm.dlight.ApplyDefaultSchedule(tvm.dlight.gpu.Fallback()))

#     with target:
#         built = tvm.relax.build(tvm.ir.transform.Sequential(seq)(Module), target=target)
#     vm = tvm.relax.VirtualMachine(built, device=dev)

#     np_A = np.random.random([4096]).astype("float16")
#     A = tvm.nd.array(np_A, dev)
#     B = vm["main"](A)

#     tvm.testing.assert_allclose(B.numpy(), np_A)
#     assert A.handle.contents.data == B.handle.contents.data
#     assert A.handle.contents.byte_offset == 0
#     assert B.handle.contents.byte_offset == 0


@tvm.testing.parametrize_targets("llvm", "cuda")
def test_execute_ensure_aligned(target, dev):
    target = tvm.target.Target(target)

    # @I.ir_module
    # class Module:
    #     @R.function
    #     def main(A: R.Tensor([4096], "float16")):
    #         B = R.memory.ensure_aligned(A, 128)
    #         return B

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([4096], "float16")):
            cls = Module
            B = cls.ensure_aligned(A)
            return B

        @R.function(private=True)
        def ensure_aligned(A: R.Tensor([4096], "float16")) -> R.Tensor([4096], "float16"):
            cls = Module

            is_aligned = cls.is_aligned(A)
            if is_aligned:
                output = A
            else:
                output = R.call_tir(cls.copy_to_aligned, [A], out_sinfo=R.Tensor([4096], "float16"))
            return output

        @T.prim_func(private=True)
        def is_aligned(A_handle: T.handle) -> T.bool:
            elem_offset = T.int64()
            A = T.match_buffer(A_handle, 4096, "float16", elem_offset=elem_offset)

            is_offset_zero = elem_offset == 0
            is_aligned = T.reinterpret("int64", T.address_of(A[0])) % 128 == 0
            return is_offset_zero and is_aligned

        @T.prim_func(private=True)
        def copy_to_aligned(
            A: T.Buffer(4096, "float16", offset_factor=1), B: T.Buffer(4096, "float16")
        ):
            for i in range(4096):
                with T.block("copy"):
                    vi = T.axis.remap("S", [i])
                    B[vi] = A[vi]

        @T.prim_func(private=True)
        def view_as_aligned(tensor: T.handle):
            T.func_attr({"tir.is_scheduled": True, "tir.is_host_func": True})

            # From #include <tvm/tir/builtin.h>
            kArrTypeBits = T.meta_var(6)
            kArrTypeLanes = T.meta_var(7)

            type_bits = T.tvm_struct_get(tensor, 0, kArrTypeBits, dtype="uint8")
            type_lanes = T.tvm_struct_get(tensor, 0, kArrTypeLanes, dtype="uint16")

        # @T.prim_func(private=True)
        # def is_bfloat16_dtype(tensor: T.handle) -> T.bool:
        #     T.func_attr({"tir.is_scheduled": True, "tir.is_host_func": True})

        #     # From #include <tvm/tir/builtin.h>
        #     kArrTypeCode = T.meta_var(5)
        #     kArrTypeBits = T.meta_var(6)
        #     kArrTypeLanes = T.meta_var(7)

        #     is_bfloat16: T.bool = (
        #         (type_code == kDLBfloat) and (type_bits == 16) and (type_lanes == 1)
        #     )
        #     T.ret(is_bfloat16)

    seq = [tvm.relax.transform.LegalizeOps()]
    if "gpu" in target.keys:
        seq.append(tvm.dlight.ApplyDefaultSchedule(tvm.dlight.gpu.Fallback()))

    with target:
        built = tvm.relax.build(tvm.ir.transform.Sequential(seq)(Module), target=target)
    vm = tvm.relax.VirtualMachine(built, device=dev)

    np_A = np.random.random([8192]).astype("float16")
    A = tvm.nd.array(np_A, dev)
    A_view = A._create_view([4096], "float16", 1024)
    B = vm["main"](A_view)

    tvm.testing.assert_allclose(B.numpy(), np_A[512 : 4096 + 512])
    assert A.handle.contents.byte_offset == 0
    assert A_view.handle.contents.byte_offset == 1024
    assert B.handle.contents.byte_offset == 0

    assert A.handle.contents.data + 1024 == B.handle.contents.data


if __name__ == "__main__":
    tvm.testing.main()
