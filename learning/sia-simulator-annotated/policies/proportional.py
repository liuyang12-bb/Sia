# 导读 比例共享参考策略。基于资源规模生成份额并可把份额转成参考吞吐。
# 导读 来源 policies/proportional.py。以下带「导读」的行是阅读注释。
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np

from policy import Policy

class ProportionalPolicy(Policy):

    # 导读 ProportionalPolicy.__init__（原源码第 10 行）
    # 导读 构造 ProportionalPolicy 时执行。为实例建立 self._name 等状态。
    # 导读 函数体直接赋值的实例字段包括 self._name。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self):
        self._name = 'Proportional'

    # 导读 ProportionalPolicy.get_throughputs（原源码第 13 行）
    # 导读 先计算比例共享份额，再按 worker 维度汇总该份额对应的参考吞吐。
    # 导读 参数 throughputs 表示作业在各 worker 类型或组合下的吞吐数据，index 表示将字典转换成矩阵时保存的行列索引，cluster_spec 表示每类 worker 的可用数量。
    # 导读 函数体中的相关调用包括 self._get_allocation。
    # 导读 从返回表达式可以继续追踪 proportional_throughputs。
    def get_throughputs(self, throughputs, index,
                        cluster_spec):
        if throughputs is None: return None
        (job_ids, worker_types) = index
        (m, n) = throughputs.shape

        x_proportional = self._get_allocation(
            throughputs, index,
            cluster_spec)
        proportional_throughputs = np.sum(np.multiply(throughputs, x_proportional),
                                          axis=1).reshape((m, 1))
        return proportional_throughputs

    # 导读 ProportionalPolicy._get_allocation（原源码第 26 行）
    # 导读 实现本策略的内部求解步骤，返回用于后续恢复字典结构的分配结果。
    # 导读 参数 throughputs 表示作业在各 worker 类型或组合下的吞吐数据，index 表示将字典转换成矩阵时保存的行列索引，cluster_spec 表示每类 worker 的可用数量。
    # 导读 从返回表达式可以继续追踪 x。
    def _get_allocation(self, throughputs, index,
                        cluster_spec):
        (_, worker_types) = index
        (m, n) = throughputs.shape

        # Split cluster over users (m).
        # x[i, j] proportional to num_workers[j].
        # \sum_j x[i, j] <= 1 for all i.
        # \sum_i x[i, j] <= 1 for all j.
        x = np.array([[cluster_spec[worker_type] / m for worker_type in worker_types]
                      for i in range(m)])
        max_per_row_sum = np.sum(x, axis=1).max()
        x = x / max_per_row_sum

        return x

    # 导读 ProportionalPolicy.get_allocation（原源码第 42 行）
    # 导读 本策略的分配入口。接收上层传入的状态，并使用 self._get_allocation 等调用生成分配。
    # 导读 参数 unflattened_throughputs 表示求解前以作业或组合、worker 类型为键组织的吞吐字典，cluster_spec 表示每类 worker 的可用数量。
    # 导读 函数体中的相关调用包括 self._get_allocation。
    # 导读 从返回表达式可以继续追踪 super().unflatten(x, index)。
    def get_allocation(self, unflattened_throughputs,
                       cluster_spec):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (job_ids, worker_types) = index
        (m, n) = throughputs.shape

        x = self._get_allocation(throughputs, index,
                                 cluster_spec)

        return super().unflatten(x, index)
