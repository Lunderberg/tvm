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
"""Target data structure, with runtime-dependent device id."""

import tvm

from tvm.runtime import Object

from . import _ffi_api


@tvm._ffi.register_object
class DynamicTarget(Object):
    """Dynamic target, used to represent dispatch to multiple devices"""

    def __init__(self, target, device_id=0):
        """Construct a DynamicTarget

        Parameters
        ----------

        target: tvm.target.Target

            The TVM target

        device_id: Union[int, PrimExpr]

            The device id.

        Returns
        -------
        res: DynamicTarget

            The dynamic target
        """

        if not isinstance(target, tvm.target.Target):
            target = tvm.target.Target(target)

        self.__init_handle_by_constructor__(_ffi_api.DynamicTarget, target, device_id)
