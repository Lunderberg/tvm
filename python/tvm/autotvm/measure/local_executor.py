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
"""Local based implementation of the executor using multiprocessing"""

import multiprocessing
import signal

from multiprocessing import Process, Queue

try:
    from queue import Empty
except ImportError:
    from Queue import Empty

from tvm.contrib.popen_pool import kill_gracefully

from . import executor


def _execute_func(func, queue, args, kwargs):
    """execute function and return the result or exception to a queue"""
    try:
        res = func(*args, **kwargs)
    except Exception as exc:  # pylint: disable=broad-except
        res = exc
    except KeyboardInterrupt:
        return
    queue.put(res)


def call_with_timeout(queue, timeout, func, args, kwargs):
    """A wrapper to support timeout of a function call"""

    # Start a new process to handle the timeout.  Cannot use a python
    # thread, because we may need to kill a subprocess that is
    # currently in a C function.
    try:
        p = Process(target=_execute_func, args=(func, queue, args, kwargs))
        p.start()
        p.join(timeout=timeout)

        queue.put(executor.TimeoutError())

    except KeyboardInterrupt:
        pass

    finally:
        try:
            kill_gracefully(p.pid)
        except KeyboardInterrupt:
            pass


class LocalFuture(executor.Future):
    """Local wrapper for the future

    Parameters
    ----------
    process: multiprocessing.Process
        process for running this task
    queue: multiprocessing.Queue
        queue for receiving the result of this task
    """

    def __init__(self, process, queue):
        self._done = False
        self._process = process
        self._queue = queue

    def done(self):
        self._done = self._done or not self._queue.empty()
        return self._done

    def get(self, timeout=None):
        # If no timeout is specified, we still want to periodically
        # check if the process is alive.  Otherwise, we could wait
        # forever on a zombie process.
        if timeout is None:
            timeout = 0.1
            check_until_process_ends = True
        else:
            check_until_process_ends = False

        while True:
            process_is_alive = self._process.is_alive()

            try:
                res = self._queue.get(block=True, timeout=timeout)
                break
            except Empty:
                if check_until_process_ends and process_is_alive:
                    continue
                elif check_until_process_ends:
                    raise executor.ExecutionError(
                        f"Process {self._process.pid} ended without pushing to the result queue"
                    )
                else:
                    raise executor.TimeoutError()

        self.stop()
        return res

    def stop(self):
        if self._process is None:
            return

        if self._process.is_alive():
            kill_gracefully(self._process.pid)
        self._process.join()
        self._queue.close()
        self._queue.join_thread()

        self._done = True
        self._process = None
        self._queue = None


class LocalFutureNoFork(executor.Future):
    """Local wrapper for the future.
    This is a none-fork version of LocalFuture.
    Use this for the runtime that does not support fork (like cudnn)
    """

    def __init__(self, result):
        self._result = result

    def done(self):
        return True

    def get(self, timeout=None):
        return self._result

    def stop(self):
        pass


class LocalExecutor(executor.Executor):
    """Local executor that runs workers on the same machine with multiprocessing.

    Parameters
    ----------
    timeout: float, optional

        timeout of a job. If time is out. A TimeoutError will be returned (not raised)

    start_method: str, optional

        Allowed options are "immediate", "spawn", and "fork".
        "immediate" will run the function immediately within the
        current process.  "spawn" and "fork" will start
        multiprocessing.Process instances using the specified method
        of starting.

        Some runtime systems that do not support fork after
        initialization (e.g. cuda runtime, cudnn).  Set this to
        something other than "fork" if you have used these runtime
        before submitting jobs.

    """

    def __init__(self, timeout=None, start_method="spawn"):
        assert start_method in ["spawn", "fork", "immediate"]

        self.timeout = timeout or executor.Executor.DEFAULT_TIMEOUT
        self.start_method = start_method

    def submit(self, func, *args, **kwargs):
        if self.start_method == "immediate":
            return LocalFutureNoFork(func(*args, **kwargs))

        context = multiprocessing.get_context(self.start_method)

        # Queue may contain the function output, a TimeoutError, or in
        # rare race cases, both.  Therefore, size must be 2.
        queue = context.Queue(2)
        process = context.Process(
            target=call_with_timeout, args=(queue, self.timeout, func, args, kwargs)
        )
        process.start()
        return LocalFuture(process, queue)
