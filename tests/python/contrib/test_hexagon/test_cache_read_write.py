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

import pytest
import numpy as np

import tvm.testing
from tvm import te
from tvm.contrib import utils
from tvm.contrib.hexagon.build import HexagonLauncher
import tvm.contrib.hexagon as hexagon

from .conftest import requires_hexagon_toolchain


def intrin_mem_copy(shape, dtype, dst_scope, src_scope):
    assert len(shape) == 1
    src = te.placeholder(shape=shape, dtype=dtype, name="src")
    dst = te.compute(shape, lambda i: src[i], name="dst")
    size = shape[0] * np.dtype(dtype).itemsize

    src_buffer = tvm.tir.decl_buffer(
        shape,
        dtype,
        scope=src_scope,
        offset_factor=1,
    )

    dst_buffer = tvm.tir.decl_buffer(
        shape,
        dtype,
        scope=dst_scope,
        offset_factor=1,
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        _src = ins[0]
        _dst = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle", "tir.mem_copy", _dst.access_ptr("w"), _src.access_ptr("r"), size
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(dst.op, intrin_func, binds={src: src_buffer, dst: dst_buffer})


def layout_transform_2d(n):
    return [n // 16, te.AXIS_SEPARATOR, n % 16]


@requires_hexagon_toolchain
def test_cache_read_write(hexagon_session):
    size = 128
    outer_shape = (size,)
    factor = 16
    inner_shape = (factor,)
    dtype = "int8"

    x = te.placeholder(shape=outer_shape, dtype=dtype, name="x")
    y = te.placeholder(shape=outer_shape, dtype=dtype, name="y")
    z = te.compute(outer_shape, lambda i: x[i] + y[i], name="z")
    s = te.create_schedule(z.op)

    x_global = s.cache_read(x, "global", [z])
    y_global = s.cache_read(y, "global", [z])
    z_global = s.cache_write(z, "global")

    s[x_global].transform_layout(layout_transform_2d)
    s[y_global].transform_layout(layout_transform_2d)
    s[z_global].transform_layout(layout_transform_2d)

    # zouter, zinner = s[z_global].split(z_global.op.axis[0], factor=factor)

    # s[x_global].compute_at(s[z_global], zouter)
    # s[y_global].compute_at(s[z_global], zouter)

    mem_copy_read = intrin_mem_copy(inner_shape, dtype, "global", "global")

    # (cache_read_x,) = s[x_global].op.axis
    # s[x_global].tensorize(cache_read_x, mem_copy_read)

    # (cache_read_y,) = s[y_global].op.axis
    # s[y_global].tensorize(cache_read_y, mem_copy_read)

    # mem_copy_write = intrin_mem_copy(outer_shape, dtype, "global", "global.vtcm")

    # (cache_write_z,) = s[z].op.axis
    # s[z].tensorize(cache_write_z, mem_copy_write)

    print(tvm.lower(s, [x, y, z]))

    target_hexagon = tvm.target.hexagon("v68", link_params=True)
    func = tvm.build(
        s, [x, y, z], tvm.target.Target(target_hexagon, host=target_hexagon), name="dmacpy"
    )

    if hexagon_session is None:
        pytest.skip("Skip hardware test since ANDROID_SERIAL_NUMBER is not set.")

    mod = hexagon_session.load_module(func)
    xt = tvm.nd.array(
        np.random.randint(low=-128, high=127, size=size, dtype=x.dtype),
        device=hexagon_session.device,
    )
    yt = tvm.nd.array(
        np.random.randint(low=-128, high=127, size=size, dtype=y.dtype),
        device=hexagon_session.device,
    )
    zt = tvm.nd.array(
        np.random.randint(low=-128, high=127, size=size, dtype=z.dtype),
        device=hexagon_session.device,
    )
    mod["dmacpy"](xt, yt, zt)

    ref = xt.numpy() + yt.numpy()
    np.testing.assert_equal(zt.numpy(), ref)
