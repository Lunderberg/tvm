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
# pylint: disable=invalid-name,too-many-function-args,too-many-nested-blocks
"""
Functions that run on executor for measurement.

These functions are responsible for building the tvm module, uploading it to
remote devices, recording the running time costs, and checking the correctness of the output.
"""

import contextlib
import logging
import os
import tempfile
import threading
import time
import typing
from collections import namedtuple
from random import getrandbits
import warnings

import tvm._ffi
import tvm.ir.transform
from tvm import nd
from tvm import rpc as _rpc
from tvm.contrib import ndk, nvcc, stackvm, tar
from tvm.driver import build
from tvm.error import TVMError
from tvm.target import Target

from ..env import AutotvmGlobalScope
from ..task.space import InstantiationError
from ..utils import get_const_tuple, grouper
from .executor import TimeoutError, ExecutionError
from .local_executor import LocalExecutor
from .measure import Builder, MeasureErrorNo, MeasureResult, Runner

logger = logging.getLogger("autotvm")


class BuildResult(namedtuple("BuildResult", ("filename", "arg_info", "error", "time_cost"))):
    """
    Stores all the necessary inputs for a measurement.

    Parameters
    ----------
    filename : str
        The filename of generated library
    arg_info : Tuple
        The shape and dtype information of tvm tensor arguments
    error : Exception
        The error happens during compilation.
    time_cost : float
        The time cost of building
    """


class LocalBuilder(Builder):
    """Run compilation on local machine

    Parameters
    ----------
    timeout: float
        The timeout of a compilation
    n_parallel: int
        The number of tasks run in parallel. "None" will use all cpu cores
    build_func: callable or str
        If is 'default', use default build function
        If is 'ndk', use function for android ndk
        If id 'stackvm', use function for stackvm
        If is callable, use it as custom build function, expect lib_format field.
    """

    def __init__(self, timeout=10, n_parallel=None, build_func="default"):
        super(LocalBuilder, self).__init__(timeout, n_parallel)

        if isinstance(build_func, str):
            if build_func == "default":
                build_func = tar.tar
            elif build_func == "ndk":
                build_func = ndk.create_shared
            elif build_func == "stackvm":
                build_func = stackvm.build
            else:
                raise ValueError("Invalid build_func" + build_func)
        self.build_func = _WrappedBuildFunc(build_func)
        self.executor = LocalExecutor(timeout=timeout)

    @contextlib.contextmanager
    def build(self, measure_inputs):
        results = []

        with tempfile.TemporaryDirectory(prefix="tvm_build") as tmp_dir:
            for i in range(0, len(measure_inputs), self.n_parallel):
                with contextlib.ExitStack() as futures_cleanup:
                    futures = []
                    for inp in measure_inputs[i : i + self.n_parallel]:
                        ret = self.executor.submit(
                            self.build_func, inp, tmp_dir, **self.build_kwargs
                        )
                        futures_cleanup.enter_context(ret)
                        futures.append(ret)

                    for future in futures:
                        results.append(self._unpack_future(future))

            # contextlib.contextmanager will clean up the temporary
            # directory when exiting the scope.
            yield results

    def _unpack_future(self, future):
        # WrappedBuildFunc returns exceptions for the
        # Exceptions could either be returned as the
        # result of a remote call, or thrown from
        # LocalFuture.get.
        try:
            res = future.get()
        except (TimeoutError, ExecutionError) as err:
            res = err

        if isinstance(res, TimeoutError):
            return MeasureResult((res,), MeasureErrorNo.BUILD_TIMEOUT, self.timeout, time.time())

        elif isinstance(res, ExecutionError):
            return MeasureResult((res,), MeasureErrorNo.BUILD_TERMINATED, self.timeout, time.time())

        elif res.error is None:
            return res

        elif isinstance(res.error, InstantiationError):
            return MeasureResult(
                (res.error,),
                MeasureErrorNo.INSTANTIATION_ERROR,
                res.time_cost,
                time.time(),
            )

        elif "InstantiationError" in str(res.error):
            msg = str(res.error)
            try:
                msg = msg.split("\n")[-2].split(": ")[1]
            except IndexError:
                pass
            return MeasureResult(
                (InstantiationError(msg),),
                MeasureErrorNo.INSTANTIATION_ERROR,
                res.time_cost,
                time.time(),
            )

        else:
            # tvm error
            return MeasureResult(
                (res.error,),
                MeasureErrorNo.COMPILE_HOST,
                res.time_cost,
                time.time(),
            )


class DeviceRunner(Runner):
    """Run generated code on a device.

    Parameters
    ----------
    timeout: float

        The timeout when running the measurement.

    n_parallel: Union[int, None]

        The number of tasks to be run in parallel. If "None", will run
        one task for each CPU core on the local machine.

    number: int

        The number of times to run the generated code when taking a
        single measurement of the average runtime.

    repeat : int, optional

        The number of times to repeat the measurement.  In total, the
        generated code will be run ``(1 + number x repeat)`` times,
        where the first iteration is a warm up and will be discarded.
        The returned result contains `repeat` costs, each of which is
        an average of `number` costs.

    min_repeat_ms: int, optional

        The minimum duration of one `repeat` in milliseconds.  By
        default, one `repeat` contains `number` runs. If this
        parameter is set, the parameters `number` will be dynamically
        adjusted to meet the minimum duration requirement of one
        `repeat`.  i.e., When the run time of one `repeat` falls below
        this time, the `number` parameter will be automatically
        increased.

    cooldown_interval: float, optional

        The cool down interval between two measurements, in seconds.

    enable_cpu_cache_flush: bool

        Whether to flush cache on CPU between repeated measurements.
        Flushing cache can make the measured latency of one operator
        closer to its actual latency during end-to-end inference.  To
        make this option effective, the argument `number` should also
        be set to 1.  This is only has effect on CPU task.

    """

    def __init__(
        self,
        timeout=10,
        n_parallel=None,
        number=4,
        repeat=3,
        min_repeat_ms=0,
        cooldown_interval=0.1,
        enable_cpu_cache_flush=False,
    ):
        super().__init__(timeout, n_parallel)

        self.number = number
        self.repeat = repeat
        self.min_repeat_ms = min_repeat_ms
        self._ref_input = None

        self.enable_cpu_cache_flush = enable_cpu_cache_flush
        self.cooldown_interval = cooldown_interval

        self.executor = LocalExecutor(timeout=timeout * (self.n_parallel + 1))

    def get_device(self, target):
        """Returns context manager that returns device"""
        raise NotImplementedError()

    def get_module(self, build_result):
        """Returns context manager that returns module"""
        raise NotImplementedError()

    def get_packed_function(self, name):
        """Returns context manager that returns packed function from registry"""
        raise NotImplementedError()

    @property
    def ref_input(self):
        """
        Fixed input for tuning special operators, e.g., sparse operators
        requiring indices as input.
        """
        return self._ref_input

    @ref_input.setter
    def ref_input(self, val):
        warnings.warn(
            "You are specifying fixed input for tuning the operator. "
            "Be sure your input always fits the operator. Some "
            "operators may conduct layout transformation during tuning, "
            "thus can lead to unexpected behaviors. ",
            RuntimeWarning,
        )
        self._ref_input = val

    def get_build_kwargs(self):
        """Implementation of Runner.get_build_kwargs"""
        kwargs = {}
        if (
            "cuda" in self.task.target.keys
            or "opencl" in self.task.target.keys
            or "rocm" in self.task.target.keys
            or "vulkan" in self.task.target.keys
        ):
            with self.get_device(self.task.target) as dev:
                max_dims = dev.max_thread_dimensions
                kwargs["check_gpu"] = {
                    "max_shared_memory_per_block": dev.max_shared_memory_per_block,
                    "max_threads_per_block": dev.max_threads_per_block,
                    "max_thread_x": max_dims[0],
                    "max_thread_y": max_dims[1],
                    "max_thread_z": max_dims[2],
                }

                if "cuda" in self.task.target.keys:
                    kwargs["cuda_arch"] = "sm_" + "".join(dev.compute_version.split("."))

        return kwargs

    def run(self, measure_inputs, build_results):
        """Implementation of Runner.run"""
        assert len(measure_inputs) == len(build_results)

        results = []

        batches = grouper(zip(measure_inputs, build_results), self.n_parallel)
        for batch in batches:
            with contextlib.ExitStack() as stack:
                futures = []
                for measure_input, build_result in batch:
                    ret = self.executor.submit(self._run_measurement, measure_input, build_result)
                    stack.enter_context(ret)
                    futures.append(ret)

                for future in futures:
                    results.append(self._unpack_future(future))

        return results

    def _run_measurement(self, measure_input, build_result):
        if isinstance(build_result, MeasureResult):
            return build_result

        time_initial = time.perf_counter()

        errno = 0
        try:
            costs = self._get_costs(measure_input, build_result)
        except TVMError as exc:
            msg = str(exc)
            if "Stack trace returned" in msg:
                msg = msg[: msg.index("Stack trace returned")]
            if "CUDA Source" in msg:
                msg = msg[: msg.index("CUDA Source")]
            costs = (RuntimeError(msg[:1024]),)
            errno = MeasureErrorNo.RUNTIME_DEVICE

        time_final = time.perf_counter()
        unix_timestamp = time.time()
        return MeasureResult(
            costs, errno, time_final - time_initial + build_result.time_cost, unix_timestamp
        )

    def _get_costs(self, measure_input, build_result):
        with self.get_device(measure_input.target) as dev, self.get_module(build_result) as mod:
            # Limitation:
            # We can not get PackFunction directly in the remote mode as it is wrapped
            # under the std::function. We could lift the restriction later once we fold
            # the PackedFunc as an object. Currently, we pass function name to work
            # around it.
            f_prepare = "cache_flush_cpu_non_first_arg" if self.enable_cpu_cache_flush else ""

            time_f = mod.time_evaluator(
                mod.entry_name,
                dev,
                number=self.number,
                repeat=self.repeat,
                min_repeat_ms=self.min_repeat_ms,
                f_preproc=f_prepare,
            )

            args = self._get_function_inputs(measure_input, build_result, dev)

            costs = time_f(*args).results

        if len(costs) > 2:
            # remove largest and smallest value to reduce variance
            costs = list(costs)
            costs.sort()
            costs = tuple(costs[1:-1])

        return costs

    def _get_function_inputs(self, measure_input, build_result, dev):
        if self.ref_input:
            args = [nd.array(x, device=dev) for x in ref_input]
        else:
            try:
                random_fill = self.get_packed_function("tvm.contrib.random.random_fill")
            except AttributeError:
                raise AttributeError(
                    "Please make sure USE_RANDOM is ON in the config.cmake.  "
                    "If running over RPC, this must be set on the remote devices."
                )

            args = [nd.empty(x[0], x[1], dev) for x in build_result.arg_info]
            if "scatter" not in measure_input.task.name:
                # the index tensor of scatter op cannot be randomly initialized
                for arg in args:
                    random_fill(arg)
            dev.sync()

        return args

    def _unpack_future(self, future):
        try:
            res = future.get()
        except (TimeoutError, ExecutionError) as err:
            res = err

        if isinstance(res, TimeoutError):
            return MeasureResult((str(res),), MeasureErrorNo.RUN_TIMEOUT, self.timeout, time.time())

        elif isinstance(res, ExecutionError):
            return MeasureResult(
                (str(res),), MeasureErrorNo.RUN_TERMINATED, self.timeout, time.time()
            )

        else:
            return res


class LocalRunner(DeviceRunner):
    """Run generated code on local devices.

    Parameters
    ----------
    timeout: float

        The timeout when running the measurement.

    number: int

        The number of times to run the generated code when taking a
        single measurement of the average runtime.

    repeat : int, optional

        The number of times to repeat the measurement.  In total, the
        generated code will be run ``(1 + number x repeat)`` times,
        where the first iteration is a warm up and will be discarded.
        The returned result contains `repeat` costs, each of which is
        an average of `number` costs.

    min_repeat_ms: int, optional

        The minimum duration of one `repeat` in milliseconds.  By
        default, one `repeat` contains `number` runs. If this
        parameter is set, the parameters `number` will be dynamically
        adjusted to meet the minimum duration requirement of one
        `repeat`.  i.e., When the run time of one `repeat` falls below
        this time, the `number` parameter will be automatically
        increased.

    cooldown_interval: float, optional

        The cool down interval between two measurements, in seconds.

    enable_cpu_cache_flush: bool

        Whether to flush cache on CPU between repeated measurements.
        Flushing cache can make the measured latency of one operator
        closer to its actual latency during end-to-end inference.  To
        make this option effective, the argument `number` should also
        be set to 1.  This is only has effect on CPU task.

    """

    def __init__(
        self,
        timeout=10,
        number=4,
        repeat=3,
        min_repeat_ms=0,
        cooldown_interval=0.1,
        enable_cpu_cache_flush=False,
    ):
        super().__init__(
            timeout=timeout,
            n_parallel=1,
            number=number,
            repeat=repeat,
            min_repeat_ms=min_repeat_ms,
            cooldown_interval=cooldown_interval,
            enable_cpu_cache_flush=enable_cpu_cache_flush,
        )

    @contextlib.contextmanager
    def get_device(self, target):
        yield tvm.device(str(target))

    @contextlib.contextmanager
    def get_module(self, build_result):
        yield tvm.runtime.module.load_module(build_result.filename)

    @contextlib.contextmanager
    def get_packed_function(self, name):
        yield tvm.get_global_func(name)


class RPCRunner(DeviceRunner):
    """Run generated code on remote devices.

    This function will ask a RPC Tracker to get device for measurement.

    Parameters
    ----------
    key: str

        The key of the device registered in the tracker

    host: str

        The host address of RPC Tracker

    port: int

        The port of RPC Tracker

    priority : int

        The job priority

    pre_load_function : Callable[ [RPCSession, BuildResult], None ]

        If given, an additional function to be called when loading the module.

    args, kwargs: List, Dict

        Additional arguments, passed to `DeviceRunner` superclass.

    """

    def __init__(self, key, host, port, priority=1, pre_load_function=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.key = key
        self.host = host
        self.port = port
        self.priority = priority
        self.pre_load_function = pre_load_function

    def set_task(self, task):
        super().set_task(task)

        if check_remote(task.target, self.key, self.host, self.port):
            logger.info("Get devices for measurement successfully!")
        else:
            raise RuntimeError(
                "Cannot get remote devices from the tracker. "
                "Please check the status of tracker by "
                "'python -m tvm.exec.query_rpc_tracker --port [THE PORT YOU USE]' "
                "and make sure you have free devices on the queue status."
            )

    @contextlib.contextmanager
    def get_remote(self):
        """Returns a remote.

        This is re-entrant, and if called multiple times in nested
        context managers will return the same remote.  This ensures
        that the a call to self.get_device() and self.get_module()
        both refer to the same remote.

        """
        if self._remote is None:
            try:
                self._remote = request_remote(
                    device_key=self.key,
                    host=self.host,
                    port=self.port,
                    priority=self.priority,
                    timeout=self.timeout,
                )
                yield self._remote
            finally:
                self._remote = None

        else:
            yield self._remote

    @contextlib.contextmanager
    def get_device(self, target):
        """Implementation of DeviceRunner.get_device"""
        with self.get_remote() as remote:
            yield remote.device(str(target), 0)

    @contextlib.contextmanager
    def get_module(self, build_result):
        with self.get_remote() as remote:
            if self.pre_load_function is not None:
                self.pre_load_function(remote, build_result)

            remote.upload(build_result.filename)
            try:
                yield remote.load_module(os.path.split(build_result.filename)[1])
            finally:
                remote.remove(build_result.filename)
                remote.remove(os.path.splitext(build_result.filename)[0] + ".so")
                remote.remove("")

    @contextlib.contextmanager
    def get_packed_function(self, name):
        with self.get_remote() as remote:
            yield remote.get_function(name)


def _build_func_common(measure_input, check_gpu=None, cuda_arch=None, build_option=None):
    """Common part for building a configuration"""
    target, task, config = measure_input
    target, task.target_host = Target.check_and_update_host_consist(target, task.target_host)

    with target:
        s, args = task.instantiate(config)

        # check invalidity of template and code hash consistency
        if not config.valid():
            raise InstantiationError(config.errors)

        opts = build_option or {}
        if check_gpu:  # Add verify pass to filter out invalid configs in advance.
            opts["tir.add_lower_pass"] = [(2, gpu_verify_pass(**check_gpu))]
        if cuda_arch:
            set_cuda_target_arch(cuda_arch)

        # if target is vta, we need to use vta build
        if (
            hasattr(measure_input.target, "device_name")
            and measure_input.target.device_name == "vta"
        ):
            # pylint: disable=import-outside-toplevel
            import vta

            func = vta.build(s, args, target_host=task.target_host)
        else:
            with tvm.ir.transform.PassContext(config=opts):
                func = build(s, args, target_host=task.target_host)
    return func, tuple((get_const_tuple(x.shape), x.dtype) for x in args)


class _WrappedBuildFunc:
    """
    Wrap build_func to a function that can be used in measure.

    Note: this is a class instead of a closure so that it can be pickled when
    using multiprocessing.

    Parameters
    ----------
    build_func : The compilation function
        We expect fcompile to contain an attr "output_format".

    Returns
    -------
    wrapped_build_func : callable
        The wrapped build function
    """

    def __init__(self, build_func):
        if not hasattr(build_func, "output_format"):
            raise AttributeError("Expect build_func to have the attribute output_format.")
        self.build_func = build_func

    def __call__(self, measure_input, tmp_dir, **kwargs):
        """
        Wrapped build func.

        Parameters
        ----------
        measure_input: MeasureInput
            The input of measurement

        tmp_dir: str
            The path of temporary directory to export generated library
        """
        tic = time.time()
        try:
            filename = os.path.join(
                tmp_dir, "tmp_func_%0x.%s" % (getrandbits(64), self.build_func.output_format)
            )
            # TODO(tvm-team) consider linline _build_func_common
            func, arg_info = _build_func_common(measure_input, **kwargs)
            func.export_library(filename, self.build_func)
        except Exception as e:  # pylint: disable=broad-except
            return BuildResult(None, None, e, time.time() - tic)
        return BuildResult(filename, arg_info, None, time.time() - tic)


def request_remote(device_key, host=None, port=None, priority=1, timeout=60):
    """Request a remote session

    Parameters
    ----------
    device_key: string
        The device key of registered device in tracker
    host: host, optional
        The host address of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_HOST"
    port: int, optional
        The port of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_PORT"
    priority: int, optional
        The priority of this request, larger is more prior
    timeout: float, optional
        The timeout of this session (units: second)

    Returns
    ------
    session: RPCSession
    """
    # connect to the tracker
    host = host or os.environ["TVM_TRACKER_HOST"]
    port = port or int(os.environ["TVM_TRACKER_PORT"])

    tracker = _rpc.connect_tracker(host, port)
    remote = tracker.request(device_key, priority=priority, session_timeout=timeout)
    return remote


def check_remote(target, device_key, host=None, port=None, priority=100, timeout=10):
    """
    Check the availability of a remote device

    Parameters
    ----------
    target: Target
        The wanted compilation target
    device_key: string
        device key of registered device in tracker
    host: host, optional
        The host address of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_HOST"
    port: int, optional
        The port address of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_PORT"
    priority: int, optional
        The priority of this request, larger is more prior
    timeout: float, optional
        The timeout of this check (units: seconds).

    Returns
    -------
    available: bool
        True if can find available device
    """

    def _check():
        remote = request_remote(device_key, host, port, priority)
        dev = remote.device(str(target))
        while not dev.exist:  # wait until we get an available device
            pass

    t = threading.Thread(
        target=_check,
    )
    t.start()
    t.join(timeout)
    return not t.is_alive()


@tvm._ffi.register_func
def tvm_callback_cuda_compile(code):
    """use nvcc to generate ptx code for better optimization"""
    curr_cuda_target_arch = AutotvmGlobalScope.current.cuda_target_arch
    # e.g., target arch could be [
    #   "-gencode", "arch=compute_52,code=sm_52",
    #   "-gencode", "arch=compute_70,code=sm_70"
    # ]
    target = "fatbin" if isinstance(curr_cuda_target_arch, list) else "ptx"
    ptx = nvcc.compile_cuda(code, target=target, arch=AutotvmGlobalScope.current.cuda_target_arch)
    return ptx


def set_cuda_target_arch(arch):
    """set target architecture of nvcc compiler

    Parameters
    ----------
    arch: str or list
        The argument of nvcc -arch. (e.g. "sm_51", "sm_62")
        it can also be a count of gencode arguments pass to nvcc command line,
        e.g., ["-gencode", "arch=compute_52,code=sm_52", "-gencode", "arch=compute_70,code=sm_70"]
    """
    AutotvmGlobalScope.current.cuda_target_arch = arch


def gpu_verify_pass(**kwargs):
    """Verify the validity of a gpu kernel.
    This pass will check memory usage and number of threads per block.
    """

    def verify_pass(f, *_):
        valid = tvm.tir.analysis.verify_gpu_code(f, kwargs)
        if not valid:
            raise InstantiationError("Skipped because of invalid gpu kernel")
        return f

    return tvm.tir.transform.prim_func_pass(verify_pass, opt_level=0)
