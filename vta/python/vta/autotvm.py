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

"""Defines AutoTVM components used with VTA."""

from . import rpc_client


def pre_load_function(bitstream=None):
    """Construct a pre-load function specialized for VTA.

    Parameters
    ----------
    bitsream : Optional[str]

        Path to the bitstream to write prior to uploading code.

    Returns
    -------
    function : (RPCSession, BuildResult) -> void

        The function to be executed on the remote RPC session.
    """

    def reprogram_fpga(remote, _build_result):
        """pre_load_function callback which reprograms the FPGA.

        Parameters
        ----------
        remote : tvm.rpc.RPCSession
            RPC session established to the remote device.

        _build_result : tvm.autotvm.measure.measure_methods.BuildResult
            Artifact from the build phase, unused here.
        """
        rpc_client.program_fpga(remote, bitstream)
        rpc_client.reconfig_runtime(remote)

    return reprogram_fpga
