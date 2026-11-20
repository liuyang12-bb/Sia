# 导读 最大化总体吞吐的策略族，包含适配异构 worker 和组合放置的版本。
# 导读 来源 policies/max_sum_throughput.py。以下带「导读」的行是阅读注释。
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import cvxpy as cp
import numpy as np

from policy import Policy, PolicyWithPacking

class ThroughputSumWithPerf(Policy):

    # 导读 ThroughputSumWithPerf.__init__（原源码第 11 行）
    # 导读 构造 ThroughputSumWithPerf 时执行。为实例建立 self._name、self._policy 等状态。
    # 导读 参数 solver 表示选用的数值求解器。
    # 导读 函数体直接赋值的实例字段包括 self._name、self._policy。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, solver):
        self._name = 'ThroughputSumWithPerf'
        self._policy = ThroughputNormalizedByCostSumWithPerfSLOs(solver)

    # 导读 ThroughputSumWithPerf.get_allocation（原源码第 15 行）
    # 导读 调用吞吐优化策略，以各作业的吞吐和容量生成分配份额。
    # 导读 参数 unflattened_throughputs 表示求解前以作业或组合、worker 类型为键组织的吞吐字典，scale_factors 表示按作业索引的资源规模需求，cluster_spec 表示每类 worker 的可用数量。
    # 导读 函数体中的相关调用包括 self._policy.get_allocation。
    # 导读 从返回表达式可以继续追踪 self._policy.get_allocation(unflattened_throughputs, scale_factors, cluster_spec)。
    def get_allocation(self, unflattened_throughputs, scale_factors,
                       cluster_spec):
        return self._policy.get_allocation(unflattened_throughputs,
                                           scale_factors,
                                           cluster_spec)

class ThroughputNormalizedByCostSumWithPerf(Policy):
    # 导读 ThroughputNormalizedByCostSumWithPerf.__init__（原源码第 22 行）
    # 导读 构造 ThroughputNormalizedByCostSumWithPerf 时执行。为实例建立 self._name、self._policy 等状态。
    # 导读 参数 solver 表示选用的数值求解器。
    # 导读 函数体直接赋值的实例字段包括 self._name、self._policy。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, solver):
        self._name = 'ThroughputNormalizedByCostSum_Perf'
        self._policy = ThroughputNormalizedByCostSumWithPerfSLOs(solver)

    # 导读 ThroughputNormalizedByCostSumWithPerf.get_allocation（原源码第 26 行）
    # 导读 本策略的分配入口。接收上层传入的状态，并使用 self._policy.get_allocation 等调用生成分配。
    # 导读 参数 unflattened_throughputs 表示求解前以作业或组合、worker 类型为键组织的吞吐字典，scale_factors 表示按作业索引的资源规模需求，cluster_spec 表示每类 worker 的可用数量。
    # 导读 参数 instance_costs 表示各类 worker 的使用成本。
    # 导读 函数体中的相关调用包括 self._policy.get_allocation。
    # 导读 从返回表达式可以继续追踪 self._policy.get_allocation(unflattened_throughputs, scale_factors, cluster_spec, instance_costs=instance_costs)。
    def get_allocation(self, unflattened_throughputs, scale_factors,
                       cluster_spec, instance_costs):
        return self._policy.get_allocation(unflattened_throughputs,
                                           scale_factors,
                                           cluster_spec,
                                           instance_costs=instance_costs)

class ThroughputNormalizedByCostSumWithPerfSLOs(Policy):

    # 导读 ThroughputNormalizedByCostSumWithPerfSLOs.__init__（原源码第 35 行）
    # 导读 构造 ThroughputNormalizedByCostSumWithPerfSLOs 时执行。为实例建立 self._name 等状态。
    # 导读 参数 solver 表示选用的数值求解器。
    # 导读 函数体直接赋值的实例字段包括 self._name。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, solver):
        Policy.__init__(self, solver)
        self._name = 'ThroughputNormalizedByCostSum_PerfSLOs'

    # 导读 ThroughputNormalizedByCostSumWithPerfSLOs.get_allocation（原源码第 39 行）
    # 导读 用运行成本归一化吞吐，并结合传入的完成时间要求构造分配问题。
    # 导读 参数 unflattened_throughputs 表示求解前以作业或组合、worker 类型为键组织的吞吐字典，scale_factors 表示按作业索引的资源规模需求，cluster_spec 表示每类 worker 的可用数量。
    # 导读 参数 instance_costs 表示各类 worker 的使用成本，SLOs 表示作业的服务目标，例如完成时间要求，num_steps_remaining 表示每个作业剩余的训练迭代数。
    # 导读 函数体中的相关调用包括 self.scale_factors_array、cp.Maximize、self.get_base_constraints、cp.Problem、cvxprob.solve。
    # 导读 从返回表达式可以继续追踪 super().unflatten(x.value.round(decimals=5).clip(min=0.0).clip(max=1.0), index)。
    def get_allocation(self, unflattened_throughputs, scale_factors,
                       cluster_spec, instance_costs=None, SLOs={},
                       num_steps_remaining={}):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (m, n) = throughputs.shape
        (job_ids, worker_types) = index

        scale = 1.0 / throughputs.sum(axis=1)

        # Row i of scale_factors_array is the scale_factor of job
        # combination i repeated len(worker_types) times.
        # 导读 scale_factors_array 是按矩阵顺序排列的作业资源需求。
        scale_factors_array = self.scale_factors_array(
             scale_factors, job_ids, m, n)

        # 导读 CVXPY 在这里创建求解变量。其形状和整数或布尔类型由调用参数给出，取值约束在后面构造。
        x = cp.Variable(throughputs.shape)
        instance_costs_array = np.ones((1, n))
        if instance_costs is not None:
            for i in range(n):
                instance_costs_array[0, i] = instance_costs[worker_types[i]]
        objective = \
            cp.Maximize(cp.sum(cp.sum(cp.multiply(throughputs /
                                                  instance_costs_array, x),
                                      axis=1)))

        # Make sure that a given job is not over-allocated resources.
        constraints = self.get_base_constraints(x, scale_factors_array)
        SLO_constraints = []
        for job_id in SLOs:
            i = job_ids.index(job_id)
            assert(job_id in num_steps_remaining)
            SLO_constraints.append(
                cp.sum(cp.multiply(throughputs[i], x[i])) >=
                    (num_steps_remaining[job_id] / SLOs[job_id])
            )
        cvxprob = cp.Problem(objective, constraints + SLO_constraints)
        result = cvxprob.solve(solver=self._solver)

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        if x.value is None:
            print('WARNING: No allocation possible with provided SLOs!')
            cvxprob = cp.Problem(objective, constraints)
            result = cvxprob.solve(solver=self._solver)

        # print(x.shape)
        # print(x.value)
        # exit()

        return super().unflatten(x.value.round(decimals=5).clip(min=0.0).clip(max=1.0), index)

class ThroughputNormalizedByCostSumWithPackingSLOs(PolicyWithPacking):

    # 导读 ThroughputNormalizedByCostSumWithPackingSLOs.__init__（原源码第 94 行）
    # 导读 构造 ThroughputNormalizedByCostSumWithPackingSLOs 时执行。为实例建立 self._name 等状态。
    # 导读 参数 solver 表示选用的数值求解器。
    # 导读 函数体直接赋值的实例字段包括 self._name。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, solver):
        Policy.__init__(self, solver)
        self._name = 'ThroughputNormalizedByCostSum_PackingSLOs'

    # 导读 ThroughputNormalizedByCostSumWithPackingSLOs.get_allocation（原源码第 98 行）
    # 导读 本策略的分配入口。接收上层传入的状态，并使用 self.scale_factors_array 等调用生成分配。
    # 导读 参数 unflattened_throughputs 表示求解前以作业或组合、worker 类型为键组织的吞吐字典，scale_factors 表示按作业索引的资源规模需求，cluster_spec 表示每类 worker 的可用数量。
    # 导读 参数 instance_costs 表示各类 worker 的使用成本，SLOs 表示作业的服务目标，例如完成时间要求，num_steps_remaining 表示每个作业剩余的训练迭代数。
    # 导读 函数体中的相关调用包括 self.scale_factors_array、cp.Maximize、self.get_base_constraints、cp.Problem、cvxprob.solve。
    # 导读 从返回表达式可以继续追踪 super().unflatten(x.value.clip(min=0.0).clip(max=1.0), index)。
    def get_allocation(self, unflattened_throughputs, scale_factors, cluster_spec,
                       instance_costs=None, SLOs={}, num_steps_remaining={}):
        all_throughputs, index = super().flatten(unflattened_throughputs,
                                                 cluster_spec)
        if all_throughputs is None or len(all_throughputs) == 0: return None
        (m, n) = all_throughputs[0].shape
        (job_ids, single_job_ids, worker_types, relevant_combinations) = index

        # Row i of scale_factors_array is the scale_factor of job
        # combination i repeated len(worker_types) times.
        # 导读 scale_factors_array 是按矩阵顺序排列的作业资源需求。
        scale_factors_array = self.scale_factors_array(
            scale_factors, job_ids, m, n)

        # 导读 CVXPY 在这里创建求解变量。其形状和整数或布尔类型由调用参数给出，取值约束在后面构造。
        x = cp.Variable((m, n))
        instance_costs_array = np.ones((m, n))
        if instance_costs is not None:
            for i in range(m):
                for j in range(n):
                    instance_costs_array[i,j] = \
                        instance_costs[worker_types[j]]

        objective_terms = []
        for i in range(len(single_job_ids)):
            indexes = relevant_combinations[single_job_ids[i]]
            objective_terms.append(cp.sum(cp.multiply(
                all_throughputs[i][indexes] /\
                    instance_costs_array[indexes], x[indexes])))

        if len(objective_terms) == 1:
            objective = cp.Maximize(objective_terms[0])
        else:
            objective = cp.Maximize(cp.sum(cp.hstack(objective_terms)))

        # Make sure that a given job is not over-allocated resources.
        constraints = self.get_base_constraints(x, single_job_ids,
                                                scale_factors_array,
                                                relevant_combinations)

        SLO_constraints = []
        per_job_throughputs = []
        per_job_SLOs = []
        for job_id in SLOs:
            i = job_ids.index(job_id)
            assert(job_id in num_steps_remaining)
            indexes = relevant_combinations[single_job_ids[i]]
            throughput = cp.sum(cp.multiply(
                all_throughputs[i][indexes], x[indexes]))
            per_job_throughputs.append(throughput)
            per_job_SLOs.append(num_steps_remaining[job_id] / SLOs[job_id])
        if len(per_job_throughputs) > 0:
            SLO_constraints.append(cp.vstack(per_job_throughputs) >=
                                   cp.vstack(per_job_SLOs))
        cvxprob = cp.Problem(objective, constraints + SLO_constraints)
        result = cvxprob.solve(solver=self._solver)

        if x.value is None:
            print('WARNING: No allocation possible with provided SLOs!')
            cvxprob = cp.Problem(objective, constraints)
            result = cvxprob.solve(solver=self._solver)

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        return super().unflatten(x.value.clip(min=0.0).clip(max=1.0), index)
