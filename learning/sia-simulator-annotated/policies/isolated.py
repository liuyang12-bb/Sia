# 导读 隔离运行参考策略。计算按隔离份额运行时的资源分配和吞吐，作为其他公平策略的参考。
# 导读 来源 policies/isolated.py。以下带「导读」的行是阅读注释。
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np

from policy import Policy

class IsolatedPolicy(Policy):

    # 导读 IsolatedPolicy.__init__（原源码第 10 行）
    # 导读 构造 IsolatedPolicy 时执行。为实例建立 self._name 等状态。
    # 导读 函数体直接赋值的实例字段包括 self._name。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self):
        self._name = 'Isolated'

    # 导读 IsolatedPolicy.get_throughputs（原源码第 13 行）
    # 导读 用隔离份额和各类 worker 吞吐计算每个作业在参考条件下的有效吞吐。
    # 导读 参数 throughputs 表示作业在各 worker 类型或组合下的吞吐数据，index 表示将字典转换成矩阵时保存的行列索引，scale_factors 表示按作业索引的资源规模需求。
    # 导读 参数 cluster_spec 表示每类 worker 的可用数量。
    # 导读 函数体中的相关调用包括 self.scale_factors_array、self._get_allocation。
    # 导读 从返回表达式可以继续追踪 isolated_throughputs。
    def get_throughputs(self, throughputs, index, scale_factors,
                        cluster_spec):
        if throughputs is None: return None
        (job_ids, worker_types) = index
        (m, n) = throughputs.shape

        # 导读 scale_factors_array 是按矩阵顺序排列的作业资源需求。
        scale_factors_array = self.scale_factors_array(
            scale_factors, job_ids, m, n)
        for i in range(m):
            for j in range(n):
                scale_factors_array[i, j] = scale_factors[job_ids[i]]

        x_isolated = self._get_allocation(
            throughputs, index,
            scale_factors_array,
            cluster_spec)
        # 导读 isolated_throughputs 是隔离运行参考方案给出的作业吞吐。
        isolated_throughputs = np.sum(np.multiply(throughputs, x_isolated),
                                      axis=1).reshape((m, 1))
        return isolated_throughputs

    # 导读 IsolatedPolicy._get_allocation（原源码第 33 行）
    # 导读 实现本策略的内部求解步骤，返回用于后续恢复字典结构的分配结果。
    # 导读 参数 throughputs 表示作业在各 worker 类型或组合下的吞吐数据，index 表示将字典转换成矩阵时保存的行列索引，scale_factors_array 表示与当前矩阵作业索引顺序对应的资源需求。
    # 导读 参数 cluster_spec 表示每类 worker 的可用数量。
    # 导读 从返回表达式可以继续追踪 x。
    def _get_allocation(self, throughputs, index, scale_factors_array,
                        cluster_spec):
        (_, worker_types) = index
        (m, n) = throughputs.shape

        # Split cluster over users (m). By construction,
        # \sum_i (x[i, j] * scale_factor[i]) = num_workers[j].
        # Normalize to ensure \sum_j x[i, j] <= 1 for all i.
        x = np.array([[cluster_spec[worker_type] / m for worker_type in worker_types]
                      for i in range(m)])
        x = x / scale_factors_array
        per_row_sum = np.sum(x, axis=1)
        per_row_sum = np.maximum(per_row_sum, np.ones(per_row_sum.shape))
        x = x / per_row_sum[:, None]

        return x

    # 导读 IsolatedPolicy.get_allocation（原源码第 50 行）
    # 导读 本策略的分配入口。接收上层传入的状态，并使用 self.scale_factors_array 等调用生成分配。
    # 导读 参数 unflattened_throughputs 表示求解前以作业或组合、worker 类型为键组织的吞吐字典，scale_factors 表示按作业索引的资源规模需求，cluster_spec 表示每类 worker 的可用数量。
    # 导读 函数体中的相关调用包括 self.scale_factors_array、self._get_allocation。
    # 导读 从返回表达式可以继续追踪 super().unflatten(x, index)。
    def get_allocation(self, unflattened_throughputs, scale_factors,
                       cluster_spec):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (job_ids, worker_types) = index
        (m, n) = throughputs.shape

        # 导读 scale_factors_array 是按矩阵顺序排列的作业资源需求。
        scale_factors_array = self.scale_factors_array(
            scale_factors, job_ids, m, n)

        x = self._get_allocation(throughputs, index,
                                 scale_factors_array,
                                 cluster_spec)

        return super().unflatten(x, index)
