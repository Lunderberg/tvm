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

import inspect

import pytest

import tvm.testing
from tvm import relax
from tvm.script import ir as I, relax as R


class Base:
    routed_param = "weight"
    routing_table_size = None

    def test_compare(self):
        transform = relax.transform.InjectRoutingTable(self.routed_param, self.routing_table_size)

        if inspect.isclass(self.Expected) and issubclass(self.Expected, Exception):
            with pytest.raises(self.Expected):
                transform(self.Before)
        else:
            after = transform(self.Before)
            tvm.ir.assert_structural_equal(self.Expected, after)


class TestSimple(Base):
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 1, 16], "float32"),
            weight: R.Tensor(["batch_size", 16, 32], "float32"),
        ) -> R.Tensor(["batch_size", 1, 32], "float32"):
            out = R.matmul(x, weight)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 1, 16], "float32"),
            weight_table: R.Tensor(["routing_table_size", 16, 32], "float32"),
            routing_table: R.Tensor(["batch_size"], "int64"),
        ) -> R.Tensor(["batch_size", 1, 32], "float32"):
            with R.dataflow():
                weight = R.take(weight_table, routing_table, axis=0)
                R.output(weight)
            out = R.matmul(x, weight)
            return out


class TestRuntime(Base):
    """The routing table should be a runtime input"""

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 1, 16], "float32"),
            weight: R.Tensor(["batch_size", 16, 32], "float32"),
        ) -> R.Tensor(["batch_size", 1, 32], "float32"):
            R.func_attr({"num_input": 1})
            out = R.matmul(x, weight)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 1, 16], "float32"),
            routing_table: R.Tensor(["batch_size"], "int64"),
            weight_table: R.Tensor(["routing_table_size", 16, 32], "float32"),
        ) -> R.Tensor(["batch_size", 1, 32], "float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                weight = R.take(weight_table, routing_table, axis=0)
                R.output(weight)
            out = R.matmul(x, weight)
            return out


if __name__ == "__main__":
    tvm.testing.main()
