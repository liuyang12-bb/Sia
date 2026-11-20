# 导读 策略证明相关的最大最小公平对照包装，处理该对照所需的输入与分配。
# 导读 来源 policies/max_min_fairness_strategy_proof.py。以下带「导读」的行是阅读注释。
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import copy
import cvxpy as cp
import numpy as np

from policy import Policy, PolicyWithPacking
from proportional import ProportionalPolicy

class MaxMinFairnessStrategyProofPolicy(Policy):

    # 导读 MaxMinFairnessStrategyProofPolicy.__init__（原源码第 13 行）
    # 导读 构造 MaxMinFairnessStrategyProofPolicy 时执行。为实例建立 self._name、self._max_min_fairness_perf_policy 等状态。
    # 导读 参数 solver 表示选用的数值求解器。
    # 导读 函数体直接赋值的实例字段包括 self._name、self._max_min_fairness_perf_policy。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, solver):
        self._name = 'MaxMinFairness'
        self._max_min_fairness_perf_policy = \
            MaxMinFairnessPolicyWithPerf(solver)

    # 导读 MaxMinFairnessStrategyProofPolicy.get_allocation（原源码第 18 行）
    # 导读 本策略的分配入口。接收上层传入的状态，并使用 self._max_min_fairness_perf_policy.get_allocation 等调用生成分配。
    # 导读 参数 unflattened_throughputs 表示求解前以作业或组合、worker 类型为键组织的吞吐字典，scale_factors 表示按作业索引的资源规模需求，priority_weights 表示求解目标使用的作业优先权重。
    # 导读 参数 cluster_spec 表示每类 worker 的可用数量。
    # 导读 函数体中的相关调用包括 self._max_min_fairness_perf_policy.get_allocation。
    # 导读 从返回表达式可以继续追踪 self._max_min_fairness_perf_policy.get_allocation(new_unflattened_throughputs, scale_factors, priority_weights, cluster_spec)。
    def get_allocation(self, unflattened_throughputs, scale_factors,
                       priority_weights, cluster_spec):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (job_ids, worker_types) = index

        new_unflattened_throughputs = {}
        for job_id in unflattened_throughputs:
            new_unflattened_throughputs[job_id] = {}
            for worker_type in unflattened_throughputs[job_id]:
                 new_unflattened_throughputs[job_id][worker_type] = 1.0

        return self._max_min_fairness_perf_policy.get_allocation(
            new_unflattened_throughputs, scale_factors, priority_weights,
            cluster_spec)


class MaxMinFairnessStrategyProofPolicyWithPerf(Policy):

    # 导读 MaxMinFairnessStrategyProofPolicyWithPerf.__init__（原源码第 38 行）
    # 导读 构造 MaxMinFairnessStrategyProofPolicyWithPerf 时执行。为实例建立 self._name、self._proportional_policy 等状态。
    # 导读 参数 solver 表示选用的数值求解器。
    # 导读 函数体直接赋值的实例字段包括 self._name、self._proportional_policy。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, solver):
        Policy.__init__(self, solver)
        self._name = 'MaxMinFairness_Perf'
        self._proportional_policy = ProportionalPolicy()

    # 导读 MaxMinFairnessStrategyProofPolicyWithPerf.get_allocation（原源码第 43 行）
    # 导读 本策略的分配入口。接收上层传入的状态，并使用 self.get_allocation 等调用生成分配。
    # 导读 参数 unflattened_throughputs 表示求解前以作业或组合、worker 类型为键组织的吞吐字典，scale_factors 表示按作业索引的资源规模需求，unflattened_priority_weights 表示按作业 ID 组织的优先权重。
    # 导读 参数 cluster_spec 表示每类 worker 的可用数量。
    # 导读 其余显式参数为 recurse_deeper，默认值见函数签名。
    # 导读 函数体中的相关调用包括 self.get_allocation、self.scale_factors_array、self._proportional_policy.get_throughputs、cp.Maximize、self.get_base_constraints、cp.Problem。
    # 导读 从返回表达式可以继续追踪 throughputs_dict、(super().unflatten(discounted_allocation.clip(min=0.0).clip(max=1.0), index), discount_factors)。
    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights, cluster_spec, recurse_deeper=True):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (m, n) = throughputs.shape
        (job_ids, worker_types) = index

        if recurse_deeper:
            all_throughputs_minus_job = []
            for job_id in job_ids:
                unflattened_throughputs_minus_job = copy.copy(unflattened_throughputs)
                del unflattened_throughputs_minus_job[job_id]
                throughputs_minus_job = self.get_allocation(
                    unflattened_throughputs_minus_job, scale_factors,
                    unflattened_priority_weights, cluster_spec,
                    recurse_deeper=False)
                all_throughputs_minus_job.append(throughputs_minus_job)

        # Row i of scale_factors_array is the scale_factor of job i
        # repeated len(worker_types) times.
        # 导读 scale_factors_array 是按矩阵顺序排列的作业资源需求。
        scale_factors_array = self.scale_factors_array(
             scale_factors, job_ids, m, n)

        priority_weights = np.array(
            [1. / unflattened_priority_weights[job_id]
             for job_id in job_ids])

        proportional_throughputs = self._proportional_policy.get_throughputs(
            throughputs, index, cluster_spec)
        priority_weights = np.multiply(priority_weights.reshape((m, 1)),
                                       1.0 / proportional_throughputs.reshape((m, 1)))

        # 导读 CVXPY 在这里创建求解变量。其形状和整数或布尔类型由调用参数给出，取值约束在后面构造。
        x = cp.Variable(throughputs.shape)
        # Multiply throughputs by scale_factors to ensure that scale_factor
        # is taken into account while allocating times to different jobs.
        # A job run on 1 GPU should receive `scale_factor` more time than
        # a job run on `scale_factor` GPUs if throughputs are equal.
        objective = cp.Maximize(
            cp.geo_mean(cp.sum(cp.multiply(
                np.multiply(throughputs * priority_weights.reshape((m, 1)),
                            scale_factors_array), x), axis=1)))
        # Make sure that the allocation can fit in the cluster.
        constraints = self.get_base_constraints(x, scale_factors_array)
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver=self._solver)

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        throughputs = np.sum(np.multiply(throughputs, x.value), axis=1)
        throughputs_dict = {job_ids[i]: throughputs[i] for i in range(len(job_ids))}
        if not recurse_deeper:
            return throughputs_dict

        discount_factors = np.zeros(len(job_ids))
        for i, job_id in enumerate(job_ids):
            discount_factor = 1.0
            for other_job_id in all_throughputs_minus_job[i]:
                discount_factor *= (
                    throughputs_dict[other_job_id] / all_throughputs_minus_job[i][other_job_id])
            discount_factors[i] = discount_factor

        discounted_allocation = np.multiply(x.value.T, discount_factors).T

        return super().unflatten(discounted_allocation.clip(min=0.0).clip(max=1.0), index), \
            discount_factors
