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
"""The TensorIR schedule class"""

from typing import Union, Optional

import tvm
from tvm import tir, IRModule
from tvm.tir import PrimFunc

from tvm.tir.schedule._type_checker import type_checked
from tvm.tir.schedule.schedule import BlockRV, _parse_seed, _parse_error_render_level
from tvm.tir.schedule.state import _parse_debug_mask, _parse_mod

from . import _ffi_api


@tvm._ffi.register_object("relax.Schedule")
class Schedule(tir.Schedule):
    """The user-facing schedule class

    Where a tir.Schedule may only perform transformations that are
    valid to a PrimFunc in isolation (e.g. fusing loops), a
    relax.Schedule may perform transformations that updating the
    end-to-end graph (e.g. fusing two PrimFuncs, splitting a stage
    into an independent PrimFunc) or that require simultaneous updates
    of multiple PrimFuncs (e.g. changing a buffer layout by updating
    both the producer and consumer of that buffer).
    """

    @type_checked
    def __init__(
        self,
        mod: Union[PrimFunc, IRModule],
        *,
        seed: Optional[int] = None,
        debug_mask: Union[str, int] = "none",
        error_render_level: str = "detail",
    ) -> None:
        """Construct a TensorIR schedule class from an IRModule

        Parameters
        ----------
        mod : Union[PrimFunc, IRModule]
            The IRModule or PrimFunc to be scheduled
        seed: Optional[int]
            The seed value for schedule's random state
            Note that None and -1 means use device random, otherwise only integer between 1 and
            2147483647 is allowed.
        debug_mask : Union[str, int]
            Do extra correctness checking after the class creation and each time
            after calling the Replace method.
            Possible choices of `debug_mask`:
            1) "all" - Turn on all the checks
            2) "none" - Turn off all the checks
            3) An integer - Turn on checks according to the bitmasks provided in ScheduleDebugMask
        error_render_level : str = "detail"
            The level of error rendering. Choices: "detail", "fast", "none".
            - "detail": Render a detailed error message, with the TIR and error locations printed
            - "fast: Show a simple error message without rendering or string manipulation
            - "none": Do not show any error message.

        Note
        ----
        The checks performed includes:
        1) VerifySRefTree
        2) VerifyCachedFlags
        """
        # call the constructor
        self.__init_handle_by_constructor__(
            _ffi_api.ConcreteSchedule,  # type: ignore # pylint: disable=no-member
            _parse_mod(mod),
            _parse_seed(seed),
            _parse_debug_mask(debug_mask),
            _parse_error_render_level(error_render_level),
        )

    def split_tir(
        self,
        block: Union[BlockRV, str],
        tir_primfunc: Optional[str],
        extracted_primfunc_name: Optional[str] = None,
        remainder_primfunc_name: Optional[str] = None,
    ):
        """Split a stage from a TIR function into a sibling function

        Parameters
        ----------
        block: Union[BlockRV, str]

            The block to be extracted into a separate PrimFunc.

        tir_primfunc: Optional[str],

            The function that contains the stage that should be
            extracted.  If None, operates on the function specified by
            `Schedule.work_on`.

        extracted_primfunc_name: Optional[str]

            The name of the PrimFunc made from the extracted TIR
            block.  If None, will auto-generate a name based on the
            name of the block.

        remainder_primfunc_name: Optional[str]

            The name of the PrimFunc remaining after the TIR block has
            been extracted.  If None, will auto-generate a name from
            the name of a remaining block within the PrimFunc, or from
            the PrimFunc's original name.
        """

        block = self._normalize_block_arg(block, tir_primfunc)

        _ffi_api.ScheduleSplitTIR(  # type: ignore # pylint: disable=no-member
            self, block, tir_primfunc, extracted_primfunc_name, remainder_primfunc_name
        )
