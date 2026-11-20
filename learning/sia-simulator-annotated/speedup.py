# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导读 封装给定节点数和副本数的 Goodput 查询。支持批量输入、相对基线的加速比和可选缓存。
# 导读 来源 speedup.py。以下带「导读」的行是阅读注释。
from re import L
import numpy as np
from goodput import GoodputFunctionPMP


class SpeedupFunction(object):
    # 导读 SpeedupFunction.__init__（原源码第 21 行）
    # 导读 保存 Goodput 函数、批量范围和累积参数，并设置计算加速比所需的基准。
    # 导读 参数 goodput_fn 表示根据训练配置评估 Goodput 的函数对象，max_batch_size 表示全局 batch 的上界，atomic_bsz_range 表示单次局部 batch 的上下界。
    # 导读 参数 accumulation 表示是否允许梯度累积，tune_bsz 表示是否在给定范围中选择 batch。
    # 导读 其余显式参数为 mem_size，默认值见函数签名。
    # 导读 函数体中的相关调用包括 goodput_fn.optimize。
    # 导读 函数体直接赋值的实例字段包括 self._goodput_fn、self._max_batch_size、self._atomic_bsz_range、self._accumulation、self._mem_size、self._base_goodput、self.tune_bsz、self._mem_speedup。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, goodput_fn, max_batch_size=None, atomic_bsz_range=None,
                 accumulation=False, mem_size=32, tune_bsz=True):
        self._goodput_fn = goodput_fn
        self._max_batch_size = max_batch_size
        self._atomic_bsz_range = atomic_bsz_range
        self._accumulation = accumulation
        self._mem_size = mem_size
        self._base_goodput, _, _ = goodput_fn.optimize(
            num_nodes=1, num_replicas=1, max_batch_size=max_batch_size,
            atomic_bsz_range=atomic_bsz_range, accumulation=accumulation, tune_bsz=tune_bsz)
        self.tune_bsz = tune_bsz
        # Memoization for fast repeated queries.
        self._mem_speedup = -np.ones((mem_size, mem_size))
        self._mem_speedup[0, 0] = 0.0
        # self._mem_speedup[1, 1] = 1.0

    # 导读 SpeedupFunction.__call__（原源码第 37 行）
    # 导读 对象当函数调用时，通过 get_goodput 返回相对基线的加速比。
    # 导读 参数 num_nodes 表示当前配置占用或考虑的节点数，num_replicas 表示配置中的训练副本数，数据并行时一般对应 GPU 数。
    # 导读 函数体中的相关调用包括 self.get_goodput。
    # 导读 从返回表达式可以继续追踪 self.get_goodput(num_nodes, num_replicas, True)。
    def __call__(self, num_nodes, num_replicas):
        return self.get_goodput(num_nodes, num_replicas, True)

    # 导读 SpeedupFunction.get_goodput（原源码第 40 行）
    # 导读 把候选节点和副本数转成兼容形状，先查缓存，再批量计算缺失点，最后恢复标量或原形状。
    # 导读 参数 num_nodes 表示当前配置占用或考虑的节点数，num_replicas 表示配置中的训练副本数，数据并行时一般对应 GPU 数，return_speedup_only 表示控制是否返回相对基线的加速比。
    # 导读 函数体中的相关调用包括 self._goodput_fn.optimize。
    # 导读 从返回表达式可以继续追踪 goodput、speedup。
    def get_goodput(self, num_nodes, num_replicas, return_speedup_only=False):
        assert np.all(np.less_equal(0, num_nodes))
        assert np.all(np.less_equal(num_nodes, num_replicas))
        assert np.all((num_nodes > 0) == (num_replicas > 0))
        # Remember what the output shape/format should be and flatten inputs.
        # 导读 output_scalar 是记录原查询是否只对应一个标量配置。
        output_scalar = np.isscalar(num_nodes) and np.isscalar(num_replicas)
        # 导读 output_shape 是输入广播后的形状，用来把扁平计算结果还原。
        output_shape = np.broadcast(num_nodes, num_replicas).shape
        num_nodes = np.broadcast_to(num_nodes, output_shape).flatten()
        num_replicas = np.broadcast_to(num_replicas, output_shape).flatten()
        # Return values which will be filled out.
        # 导读 使用负值标记尚未填入缓存的加速比，后续仅计算缺失配置。
        speedup = -np.ones(output_shape).flatten()
        # Fill in any previously memoized results first.
        # 导读 indices 是在当前候选维度上选出的元素索引。
        indices = num_replicas < self._mem_size
        mem_idx = (num_nodes[indices], num_replicas[indices])
        speedup[indices] = self._mem_speedup[mem_idx]
        # Find the missing indices which still need to be computed.
        missing = speedup < 0
        if np.count_nonzero(missing) > 0:
            num_nodes, num_replicas = num_nodes[missing], num_replicas[missing]
            # Find unique inputs to reduce compuation.
            (num_nodes, num_replicas), inverse = np.unique(
                np.stack([num_nodes, num_replicas]),
                axis=1, return_inverse=True)
            # 导读 调用 Goodput 模型为缺失的节点数和副本数组合选择 batch。
            goodput, _, _ = self._goodput_fn.optimize(
                num_nodes, num_replicas,
                max_batch_size=self._max_batch_size,
                atomic_bsz_range=self._atomic_bsz_range,
                accumulation=self._accumulation,
                tune_bsz=self.tune_bsz)
            # Memoize results.
            # 导读 indices 是在当前候选维度上选出的元素索引。
            indices = num_replicas < self._mem_size
            mem_idx = (num_nodes[indices], num_replicas[indices])
            self._mem_speedup[mem_idx] = goodput[indices] / self._base_goodput
            # Fill in computed results.
            speedup[missing] = goodput[inverse] / self._base_goodput
        assert np.all(np.less_equal(0, speedup))
        # 导读 还原查询原本的形状，单个配置查询可以返回标量。
        speedup = speedup.reshape(output_shape)
        speedup = speedup.item() if output_scalar else speedup
        if not return_speedup_only:
            goodput = speedup * self._base_goodput
            return goodput
        else:
            return speedup

    # 导读 SpeedupFunction.set_base_goodput（原源码第 84 行）
    # 导读 用指定的基线节点与副本数重新计算归一化分母。
    # 导读 其余显式参数为 baseline_num_nodes、baseline_num_replicas，默认值见函数签名。
    # 导读 函数体中的相关调用包括 self._goodput_fn.optimize。
    # 导读 函数体直接赋值的实例字段包括 self._base_goodput。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def set_base_goodput(self, baseline_num_nodes, baseline_num_replicas):
        self._base_goodput, _, _ = self._goodput_fn.optimize(
            num_nodes=baseline_num_nodes,
            num_replicas=baseline_num_replicas,
            max_batch_size=self._max_batch_size,
            atomic_bsz_range=self._atomic_bsz_range,
            accumulation=self._accumulation)


class UncachedSpeedupFunction(object):
    # 导读 UncachedSpeedupFunction.__init__（原源码第 94 行）
    # 导读 保存 Goodput 函数、批量范围和累积参数，并设置计算加速比所需的基准。
    # 导读 参数 goodput_fn 表示根据训练配置评估 Goodput 的函数对象，max_batch_size 表示全局 batch 的上界，atomic_bsz_range 表示单次局部 batch 的上下界。
    # 导读 参数 accumulation 表示是否允许梯度累积，tune_bsz 表示是否在给定范围中选择 batch。
    # 导读 函数体中的相关调用包括 goodput_fn.optimize。
    # 导读 函数体直接赋值的实例字段包括 self._goodput_fn、self._max_batch_size、self._atomic_bsz_range、self._accumulation、self._base_goodput、self.tune_bsz。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, goodput_fn, max_batch_size=None, atomic_bsz_range=None,
                 accumulation=False, tune_bsz=True):
        self._goodput_fn = goodput_fn
        self._max_batch_size = max_batch_size
        self._atomic_bsz_range = atomic_bsz_range
        self._accumulation = accumulation
        self._base_goodput, _, _ = goodput_fn.optimize(
            num_nodes=1, num_replicas=1, max_batch_size=max_batch_size,
            atomic_bsz_range=atomic_bsz_range, accumulation=accumulation, tune_bsz=tune_bsz)
        self.tune_bsz = tune_bsz
        # print(f"{self.__class__.__name__} --> bsz tuning: {self.tune_bsz}")

    # 导读 UncachedSpeedupFunction.__call__（原源码第 106 行）
    # 导读 对象当函数调用时，通过 get_goodput 返回相对基线的加速比。
    # 导读 参数 num_nodes 表示当前配置占用或考虑的节点数，num_replicas 表示配置中的训练副本数，数据并行时一般对应 GPU 数。
    # 导读 函数体中的相关调用包括 self.get_goodput。
    # 导读 从返回表达式可以继续追踪 self.get_goodput(num_nodes, num_replicas, True)。
    def __call__(self, num_nodes, num_replicas):
        return self.get_goodput(num_nodes, num_replicas, True)

    # `num_nodes`, `num_replicas` MUST be numpy.ndarrays of type=np.float32/float64
    # YOU WILL GET WRONG RESULTS FOR dtype=np.uint32 for some reason`
    # 导读 UncachedSpeedupFunction.get_goodput（原源码第 111 行）
    # 导读 直接为候选节点和副本数调用 Goodput 优化，返回绝对 Goodput 或加速比。
    # 导读 参数 num_nodes 表示当前配置占用或考虑的节点数，num_replicas 表示配置中的训练副本数，数据并行时一般对应 GPU 数，return_speedup_only 表示控制是否返回相对基线的加速比。
    # 导读 函数体中的相关调用包括 self._goodput_fn.optimize。
    # 导读 从返回表达式可以继续追踪 goodputs / self._base_goodput、goodputs。
    def get_goodput(self, num_nodes, num_replicas, return_speedup_only=False):
        assert np.all(np.less_equal(0, num_nodes))
        assert np.all(np.less_equal(num_nodes, num_replicas))
        assert np.all((num_nodes > 0) == (num_replicas > 0))

        goodputs, _, _ = self._goodput_fn.optimize(
            num_nodes, num_replicas,
            max_batch_size=self._max_batch_size,
            atomic_bsz_range=self._atomic_bsz_range,
            accumulation=self._accumulation,
            tune_bsz=self.tune_bsz)
        # print(f"goodputs: {goodputs}")
        if return_speedup_only:
            return goodputs / self._base_goodput
        else:
            return goodputs

    # 导读 UncachedSpeedupFunction.set_base_goodput（原源码第 128 行）
    # 导读 用指定的基线节点与副本数重新计算归一化分母。
    # 导读 其余显式参数为 baseline_num_nodes、baseline_num_replicas，默认值见函数签名。
    # 导读 函数体中的相关调用包括 self._goodput_fn.optimize。
    # 导读 函数体直接赋值的实例字段包括 self._base_goodput。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def set_base_goodput(self, baseline_num_nodes, baseline_num_replicas):
        # fast return for [1,1] baseline config
        if baseline_num_nodes == 1 and baseline_num_replicas == 1:
            return

        self._base_goodput, _, _ = self._goodput_fn.optimize(
            num_nodes=baseline_num_nodes,
            num_replicas=baseline_num_replicas,
            max_batch_size=self._max_batch_size,
            atomic_bsz_range=self._atomic_bsz_range,
            accumulation=self._accumulation)
