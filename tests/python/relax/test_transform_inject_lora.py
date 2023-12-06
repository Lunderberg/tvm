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
    lora_param = "weight"
    lora_r = None
    lora_param_order = "after_corresponding_base_weight"

    def test_compare(self):
        transform = relax.transform.InjectLora(self.lora_param, self.lora_r, self.lora_param_order)

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
            x: R.Tensor([16], "float32"),
            weight: R.Tensor([32, 16], "float32"),
        ) -> R.Tensor([32], "float32"):
            out = R.matmul(weight, x)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            weight_base: R.Tensor([32, 16], "float32"),
            weight_LA: R.Tensor(["weight_lora_r", 16], "float32"),
            weight_LB: R.Tensor([32, "weight_lora_r"], "float32"),
        ) -> R.Tensor([32], "float32"):
            R.func_attr({"tir_var_upper_bound": {"weight_lora_r": 16}})
            with R.dataflow():
                lora_contrib = R.matmul(weight_LB, weight_LA)
                weight = R.add(weight_base, lora_contrib)
                R.output(weight)
            out = R.matmul(weight, x)
            return out


class TestExplicitLoraR(Base):
    """If provided, a static LoRA dimension may be used"""

    lora_r = 2

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            weight: R.Tensor([32, 16], "float32"),
        ) -> R.Tensor([32], "float32"):
            out = R.matmul(weight, x)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            weight_base: R.Tensor([32, 16], "float32"),
            weight_LA: R.Tensor([2, 16], "float32"),
            weight_LB: R.Tensor([32, 2], "float32"),
        ) -> R.Tensor([32], "float32"):
            with R.dataflow():
                lora_contrib = R.matmul(weight_LB, weight_LA)
                weight = R.add(weight_base, lora_contrib)
                R.output(weight)
            out = R.matmul(weight, x)
            return out


class TestTransformByRelaxVar(TestSimple):
    lora_param = TestSimple.Before["main"].params[1]


class TestErrorWhenWeightIsNotTensor(Base):
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            weight: R.Tuple([R.Tensor([32, 16], "float32")]),
        ) -> R.Tensor([32], "float32"):
            out = R.matmul(weight[0], x)
            return out

    Expected = TypeError


class TestErrorWhenWeightIsRankOne(Base):
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            weight: R.Tensor([16], "float32"),
        ) -> R.Tensor([32], "float32"):
            out = R.add(x, weight)
            return out

    Expected = TypeError


class TestNoChangeIfParamNotFound(Base):
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            weight_with_other_name: R.Tensor([32, 16], "float32"),
        ) -> R.Tensor([32], "float32"):
            out = R.matmul(weight_with_other_name, x)
            return out

    Expected = Before


class TestUpdateOneWeight(Base):
    lora_param = "weight_2"

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            weight_1: R.Tensor([32, 16], "float32"),
            bias_1: R.Tensor([32], "float32"),
            weight_2: R.Tensor([64, 32], "float32"),
            bias_2: R.Tensor([64], "float32"),
        ) -> R.Tensor([64], "float32"):
            x = R.matmul(weight_1, x)
            x = R.add(x, bias_1)
            x = R.matmul(weight_2, x)
            x = R.add(x, bias_2)
            return x

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            weight_1: R.Tensor([32, 16], "float32"),
            bias_1: R.Tensor([32], "float32"),
            weight_2_base: R.Tensor([64, 32], "float32"),
            weight_2_LA: R.Tensor(["lora_r", 32], "float32"),
            weight_2_LB: R.Tensor([64, "lora_r"], "float32"),
            bias_2: R.Tensor([64], "float32"),
        ) -> R.Tensor([64], "float32"):
            R.func_attr({"tir_var_upper_bound": {"weight_2_lora_r": 32}})
            with R.dataflow():
                weight_2_lora_contrib = R.matmul(weight_2_LB, weight_2_LA)
                weight_2 = R.add(weight_2_base, weight_2_lora_contrib)
                R.output(weight_2)

            x = R.matmul(weight_1, x)
            x = R.add(x, bias_1)
            x = R.matmul(weight_2, x)
            x = R.add(x, bias_2)
            return x


class TestMultipleWeightsByRegex(Base):
    lora_param = r"weight_\d"

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            weight_1: R.Tensor([32, 16], "float32"),
            bias_1: R.Tensor([32], "float32"),
            weight_2: R.Tensor([64, 32], "float32"),
            bias_2: R.Tensor([64], "float32"),
        ) -> R.Tensor([64], "float32"):
            x = R.matmul(weight_1, x)
            x = R.add(x, bias_1)
            x = R.matmul(weight_2, x)
            x = R.add(x, bias_2)
            return x

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            weight_1_base: R.Tensor([32, 16], "float32"),
            weight_1_LA: R.Tensor(["weight_1_lora_r", 16], "float32"),
            weight_1_LB: R.Tensor([32, "weight_1_lora_r"], "float32"),
            bias_1: R.Tensor([32], "float32"),
            weight_2_base: R.Tensor([64, 32], "float32"),
            weight_2_LA: R.Tensor(["weight_2_lora_r", 32], "float32"),
            weight_2_LB: R.Tensor([64, "weight_2_lora_r"], "float32"),
            bias_2: R.Tensor([64], "float32"),
        ) -> R.Tensor([64], "float32"):
            R.func_attr(
                {
                    "tir_var_upper_bound": {
                        "weight_1_lora_r": 16,
                        "weight_2_lora_r": 32,
                    }
                }
            )
            with R.dataflow():
                weight_1_lora_contrib = R.matmul(weight_1_LB, weight_1_LA)
                weight_1 = R.add(weight_1_base, weight_1_lora_contrib)
                weight_2_lora_contrib = R.matmul(weight_2_LB, weight_2_LA)
                weight_2 = R.add(weight_2_base, weight_2_lora_contrib)
                R.output(weight_1, weight_2)

            x = R.matmul(weight_1, x)
            x = R.add(x, bias_1)
            x = R.matmul(weight_2, x)
            x = R.add(x, bias_2)
            return x


class TestTransposed(Base):
    """The LoRA can be applied to a weight used by R.linear"""

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            weight: R.Tensor([32, 16], "float32"),
        ) -> R.Tensor([32], "float32"):
            out = R.linear(x, weight)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            weight_base: R.Tensor([32, 16], "float32"),
            weight_LA: R.Tensor(["weight_lora_r", 16], "float32"),
            weight_LB: R.Tensor([32, "weight_lora_r"], "float32"),
        ) -> R.Tensor([32], "float32"):
            R.func_attr({"tir_var_upper_bound": {"weight_lora_r": 16}})
            with R.dataflow():
                lora_contrib = R.matmul(weight_LB, weight_LA)
                weight = R.add(weight_base, lora_contrib)
                R.output(weight)
            out = R.linear(x, weight)
            return out


class TestBatchDimensionOnActivations(Base):
    """The LoRA-mutated weights may be applied to batched activations"""

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 16, 1], "float32"),
            weight: R.Tensor([32, 16], "float32"),
        ) -> R.Tensor(["batch_size", 32, 1], "float32"):
            out = R.matmul(weight, x)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 16, 1], "float32"),
            weight_base: R.Tensor([32, 16], "float32"),
            weight_LA: R.Tensor(["weight_lora_r", 16], "float32"),
            weight_LB: R.Tensor([32, "weight_lora_r"], "float32"),
        ) -> R.Tensor(["batch_size", 32, 1], "float32"):
            R.func_attr({"tir_var_upper_bound": {"weight_lora_r": 16}})
            with R.dataflow():
                lora_contrib = R.matmul(weight_LB, weight_LA)
                weight = R.add(weight_base, lora_contrib)
                R.output(weight)
            out = R.matmul(weight, x)
            return out


class TestBatchDimensionOnActivationsAndWeight(Base):
    """The LoRA-mutated weights themselves be batched"""

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 16, 1], "float32"),
            weight: R.Tensor(["batch_size", 32, 16], "float32"),
        ) -> R.Tensor(["batch_size", 32, 1], "float32"):
            out = R.matmul(weight, x)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 16, 1], "float32"),
            weight_base: R.Tensor(["batch_size", 32, 16], "float32"),
            weight_LA: R.Tensor(["batch_size", "weight_lora_r", 16], "float32"),
            weight_LB: R.Tensor(["batch_size", 32, "weight_lora_r"], "float32"),
        ) -> R.Tensor(["batch_size", 32, 1], "float32"):
            R.func_attr({"tir_var_upper_bound": {"weight_lora_r": 16}})
            with R.dataflow():
                lora_contrib = R.matmul(weight_LB, weight_LA)
                weight = R.add(weight_base, lora_contrib)
                R.output(weight)
            out = R.matmul(weight, x)
            return out


class TestInsertParamsAfterCorrespondingBaseWeight(Base):
    """Place LoRA weights after the updated base weight"""

    lora_param_order = "after_corresponding_base_weight"

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            weight: R.Tensor([32, 16], "float32"),
            bias: R.Tensor([32], "float32"),
        ):
            x = R.matmul(weight, x)
            x = R.add(x, bias)
            return x

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            weight_base: R.Tensor([32, 16], "float32"),
            weight_LA: R.Tensor(["weight_lora_r", 16], "float32"),
            weight_LB: R.Tensor([32, "weight_lora_r"], "float32"),
            bias: R.Tensor([32], "float32"),
        ):
            R.func_attr({"tir_var_upper_bound": {"weight_lora_r": 16}})
            with R.dataflow():
                weight_lora_contrib = R.matmul(weight_LB, weight_LA)
                weight = R.add(weight_base, weight_lora_contrib)
                R.output(weight)

            x = R.matmul(weight, x)
            x = R.add(x, bias)
            return x


class TestAppendParamsToEnd(Base):
    """Place LoRA weights after all base weights"""

    lora_param_order = "after_all_params"

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            weight: R.Tensor([32, 16], "float32"),
            bias: R.Tensor([32], "float32"),
        ):
            x = R.matmul(weight, x)
            x = R.add(x, bias)
            return x

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            weight_base: R.Tensor([32, 16], "float32"),
            bias: R.Tensor([32], "float32"),
            weight_LA: R.Tensor(["weight_lora_r", 16], "float32"),
            weight_LB: R.Tensor([32, "weight_lora_r"], "float32"),
        ):
            R.func_attr({"tir_var_upper_bound": {"weight_lora_r": 16}})
            with R.dataflow():
                weight_lora_contrib = R.matmul(weight_LB, weight_LA)
                weight = R.add(weight_base, weight_lora_contrib)
                R.output(weight)

            x = R.matmul(weight, x)
            x = R.add(x, bias)
            return x


class TestInsertParamsAsRuntimeParams(Base):
    """Place LoRA weights after all base weights"""

    lora_param_order = "end_of_runtime_params"

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            weight: R.Tensor([32, 16], "float32"),
            bias: R.Tensor([32], "float32"),
        ):
            R.func_attr({"num_input": 1})
            x = R.matmul(weight, x)
            x = R.add(x, bias)
            return x

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            weight_LA: R.Tensor(["weight_lora_r", 16], "float32"),
            weight_LB: R.Tensor([32, "weight_lora_r"], "float32"),
            weight_base: R.Tensor([32, 16], "float32"),
            bias: R.Tensor([32], "float32"),
        ):
            R.func_attr(
                {
                    "num_input": 3,
                    "tir_var_upper_bound": {"weight_lora_r": 16},
                }
            )
            with R.dataflow():
                weight_lora_contrib = R.matmul(weight_LB, weight_LA)
                weight = R.add(weight_base, weight_lora_contrib)
                R.output(weight)

            x = R.matmul(weight, x)
            x = R.add(x, bias)
            return x


class TestInsertParamsAsRuntimeParamsWithoutNumInputs(Base):
    """Place LoRA weights in the runtime weights

    If the `attr::kNumInput` attribute is absent, all parameters are
    runtime parameters.
    """

    lora_param_order = "end_of_runtime_params"

    Before = TestAppendParamsToEnd.Before
    Expected = TestAppendParamsToEnd.Expected


if __name__ == "__main__":
    tvm.testing.main()
