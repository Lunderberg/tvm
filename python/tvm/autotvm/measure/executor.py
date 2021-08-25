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
""" Abstraction for asynchronous job execution """


class Executor(object):
    """
    Base abstract executor interface for asynchronous job submission.
    Allows submit asynchronous jobs and returns the Future object.
    """

    # timeout for jobs that may hang
    DEFAULT_TIMEOUT = 120

    def submit(self, func, *args, **kwargs):
        """
        Pass task (function, arguments) to the Executor.

        Parameters
        ----------
        func : callable
            function to be run by a worker
        args : list or tuple, optional
            arguments passed to the function
        kwargs : dict, optional
            The keyword arguments

        Returns
        -------
        future : Future
            Future object wrapping the task which can be used to
            collect the task's result.
        """
        raise NotImplementedError()


class Future(object):
    """Base class of the future object.

    Submissions to an executor object will return a subclass of
    Future.  These objects encapsulate the asynchronous execution of
    task submitted to another thread, or another worker for execution.

    Future objects store the state of tasks.  The result can be polled
    with non-blocking calls to `get()`, or can be waited on with a blocking call
    to `get()`.

    If used as a context manager, all in-progress computations will be
    halted when exiting the "with" block.

    """

    def done(self):
        """
        Return True if job was successfully cancelled or finished running.
        """
        raise NotImplementedError()

    def get(self, timeout=None):
        """Get the result.

        If a result is available, will return immediately.  If not,
        will block until a result is available, until the specified
        timeout is reached, or until the call computing the result has
        terminated without producing output.

        Parameters
        ----------
        timeout : int or float, optional

            Maximum number of seconds to wait before it timeouts.  If
            not specified, the call will block until either the result
            is available or the result call has terminated without
            producing output.

        Returns
        -------
        result : Any

            The result returned by the submitted function.

        Raises
        ------
        TimeoutError

            If the result call times out.  If TimeoutError is raised,
            a subsequent call to .get() may return a result.

        ExecutionError

            If the result call terminated without producing output.
            If ExecutionError is raised, no subsequent calls to .get()
            will return a result.

        """
        raise NotImplementedError()

    def stop(self):
        """Stop the in-progress computation.

        Terminates and cleans up any in-progress computation.  Can be
        called on its own, or can be called on exiting a block when
        used as a context manager.
        """
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


class FutureError(RuntimeError):
    """Base error class of all future events"""


# pylint:disable=redefined-builtin
class TimeoutError(FutureError):
    """Error raised when a task is timeout."""


class ExecutionError(FutureError):
    """
    Error raised when future execution crashes or failed.
    """
