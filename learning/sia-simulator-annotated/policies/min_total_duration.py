# 导读 优化全部作业总完成所需时长的策略族，尝试不同时间界下的可行分配。
# 导读 来源 policies/min_total_duration.py。以下带「导读」的行是阅读注释。
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import cvxpy as cp
import numpy as np

from policy import Policy, PolicyWithPacking

class MinTotalDurationPolicy(Policy):

    # 导读 MinTotalDurationPolicy.__init__（原源码第 11 行）
    # 导读 构造 MinTotalDurationPolicy 时执行。为实例建立 self._name、self._min_total_duration_perf_policy 等状态。
    # 导读 参数 solver 表示选用的数值求解器。
    # 导读 函数体直接赋值的实例字段包括 self._name、self._min_total_duration_perf_policy。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, solver):
        self._name = 'MinTotalDuration'
        self._min_total_duration_perf_policy = \
            MinTotalDurationPolicyWithPerf(solver)

    # 导读 MinTotalDurationPolicy.get_allocation（原源码第 16 行）
    # 导读 本策略的分配入口。接收上层传入的状态，并使用 self._min_total_duration_perf_policy.get_allocation 等调用生成分配。
    # 导读 参数 unflattened_throughputs 表示求解前以作业或组合、worker 类型为键组织的吞吐字典，scale_factors 表示按作业索引的资源规模需求，num_steps_remaining 表示每个作业剩余的训练迭代数。
    # 导读 参数 cluster_spec 表示每类 worker 的可用数量。
    # 导读 函数体中的相关调用包括 self._min_total_duration_perf_policy.get_allocation。
    def get_allocation(self, unflattened_throughputs, scale_factors,
                       num_steps_remaining, cluster_spec):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (job_ids, worker_types) = index

        new_unflattened_throughputs = {}
        for job_id in unflattened_throughputs:
            new_unflattened_throughputs[job_id] = {}
            for worker_type in unflattened_throughputs[job_id]:
                 new_unflattened_throughputs[job_id][worker_type] = \
                     unflattened_throughputs[job_id]['v100']

        return self._min_total_duration_perf_policy.get_allocation(
            new_unflattened_throughputs, scale_factors, num_steps_remaining,
            cluster_spec)


class MinTotalDurationPolicyWithPerf(Policy):

    # 导读 MinTotalDurationPolicyWithPerf.__init__（原源码第 37 行）
    # 导读 构造 MinTotalDurationPolicyWithPerf 时执行。为实例建立 self._name 等状态。
    # 导读 参数 solver 表示选用的数值求解器。
    # 导读 函数体直接赋值的实例字段包括 self._name。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, solver):
        Policy.__init__(self, solver)
        self._name = 'MinTotalDuration_Perf'

    # 导读 MinTotalDurationPolicyWithPerf.get_allocation_helper（原源码第 41 行）
    # 导读 在给定时间边界或分配参数下测试可行性，供外层时长搜索反复调用。
    # 导读 参数 throughputs 表示作业在各 worker 类型或组合下的吞吐数据，scale_factors_array 表示与当前矩阵作业索引顺序对应的资源需求。
    # 导读 其余显式参数为 T，默认值见函数签名。
    # 导读 函数体中的相关调用包括 cp.Maximize、self.get_base_constraints、cp.Problem、cvxprob.solve。
    # 导读 从返回表达式可以继续追踪 (cvxprob.status, x)。
    def get_allocation_helper(self, throughputs, scale_factors_array, T):
        # 导读 CVXPY 在这里创建求解变量。其形状和整数或布尔类型由调用参数给出，取值约束在后面构造。
        x = cp.Variable(throughputs.shape)
        objective = cp.Maximize(1)
        # Make sure the allocation can fit in the cluster, and that the
        # currently active jobs can finish in time T.
        constraints = self.get_base_constraints(x, scale_factors_array)
        constraints.append(
            cp.sum(cp.multiply(throughputs, x), axis=1) >= (
                self._num_steps_remaining / T)
        )
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver=self._solver)

        return cvxprob.status, x

    # 导读 MinTotalDurationPolicyWithPerf.get_allocation（原源码第 56 行）
    # 导读 本策略的分配入口。接收上层传入的状态，并使用 self.scale_factors_array 等调用生成分配。
    # 导读 参数 unflattened_throughputs 表示求解前以作业或组合、worker 类型为键组织的吞吐字典，scale_factors 表示按作业索引的资源规模需求，num_steps_remaining 表示每个作业剩余的训练迭代数。
    # 导读 参数 cluster_spec 表示每类 worker 的可用数量。
    # 导读 函数体中的相关调用包括 self.scale_factors_array、self.get_allocation_helper。
    # 导读 函数体直接赋值的实例字段包括 self._num_steps_remaining。
    # 导读 从返回表达式可以继续追踪 super().unflatten(last_feasible_x.value.clip(min=0.0).clip(max=1.0), index)。
    def get_allocation(self, unflattened_throughputs, scale_factors,
                       num_steps_remaining, cluster_spec):
        # TODO: Might not want to normalize throughputs here.
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if index is None: return None
        (m, n) = throughputs.shape
        (job_ids, _) = index
        self._num_steps_remaining = np.array([num_steps_remaining[job_id]
                                              for job_id in job_ids])
        if throughputs is None: return None

        # Row i of scale_factors_array is the scale_factor of job i
        # repeated len(worker_types) times.
        # 导读 scale_factors_array 是按矩阵顺序排列的作业资源需求。
        scale_factors_array = self.scale_factors_array(
             scale_factors, job_ids, m, n)

        # Units are in seconds.
        max_T = 1000000.
        min_T = 100.
        last_max_T = max_T
        status = None
        last_feasible_x = None
        while last_feasible_x is None:
            # Binary search for the smallest T that gives an optimal solution.
            while (1.05 * min_T) < max_T:  # TODO: Can tweak the 1.05 in this loop.
                T = (min_T + max_T) / 2.
                status, x = self.get_allocation_helper(throughputs,
                                                       scale_factors_array, T)
                if status == "optimal":
                    last_feasible_x = x
                    max_T = T
                else:
                    min_T = T
            max_T = last_max_T * 10.
            min_T = last_max_T
            last_max_T *= 10

        assert(last_feasible_x is not None)
        return super().unflatten(
            last_feasible_x.value.clip(min=0.0).clip(max=1.0), index)


class MinTotalDurationPolicyWithPacking(PolicyWithPacking):

    # 导读 MinTotalDurationPolicyWithPacking.__init__（原源码第 101 行）
    # 导读 构造 MinTotalDurationPolicyWithPacking 时执行。为实例建立 self._name 等状态。
    # 导读 参数 solver 表示选用的数值求解器。
    # 导读 函数体直接赋值的实例字段包括 self._name。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, solver):
        PolicyWithPacking.__init__(self, solver)
        self._name = 'MinTotalDuration_Packing'

    # 导读 MinTotalDurationPolicyWithPacking.get_allocation_helper（原源码第 105 行）
    # 导读 在给定时间边界或分配参数下测试可行性，供外层时长搜索反复调用。
    # 导读 参数 all_throughputs 表示当前组合或作业全集的吞吐矩阵，job_ids 表示用于保持结果顺序的作业标识，scale_factors_array 表示与当前矩阵作业索引顺序对应的资源需求。
    # 导读 其余显式参数为 single_job_ids、T、relevant_combinations，默认值见函数签名。
    # 导读 函数体中的相关调用包括 cp.Maximize、self.get_base_constraints、cp.Problem、cvxprob.solve。
    # 导读 从返回表达式可以继续追踪 (cvxprob.status, x)。
    def get_allocation_helper(self, all_throughputs, job_ids,
                              single_job_ids, scale_factors_array, T,
                              relevant_combinations):
        # 导读 CVXPY 在这里创建求解变量。其形状和整数或布尔类型由调用参数给出，取值约束在后面构造。
        x = cp.Variable(all_throughputs[0].shape)
        objective = cp.Maximize(1)
        # Make sure the allocation can fit in the cluster.
        constraints = self.get_base_constraints(x, single_job_ids,
                                                scale_factors_array,
                                                relevant_combinations)

        # See if passed in T is feasible.
        for i, (throughputs, num_steps_remaining) in \
            enumerate(zip(all_throughputs, self._num_steps_remaining)):
            indexes = relevant_combinations[single_job_ids[i]]
            # Ensure that every job satisfies its throughput constraint,
            # and can finish in time T.
            constraints.append(
                cp.sum(cp.multiply(throughputs[indexes], x[indexes])) >=
                    (num_steps_remaining / T))
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver=self._solver)

        return cvxprob.status, x

    # 导读 MinTotalDurationPolicyWithPacking.get_allocation（原源码第 129 行）
    # 导读 本策略的分配入口。接收上层传入的状态，并使用 self.scale_factors_array 等调用生成分配。
    # 导读 参数 unflattened_throughputs 表示求解前以作业或组合、worker 类型为键组织的吞吐字典，scale_factors 表示按作业索引的资源规模需求，num_steps_remaining 表示每个作业剩余的训练迭代数。
    # 导读 参数 cluster_spec 表示每类 worker 的可用数量。
    # 导读 函数体中的相关调用包括 self.scale_factors_array、self.get_allocation_helper。
    # 导读 函数体直接赋值的实例字段包括 self._num_steps_remaining。
    # 导读 从返回表达式可以继续追踪 super().unflatten(last_feasible_x.value.clip(min=0.0).clip(max=1.0), index)。
    def get_allocation(self, unflattened_throughputs, scale_factors,
                       num_steps_remaining, cluster_spec):
        all_throughputs, index = super().flatten(unflattened_throughputs,
                                                 cluster_spec)
        if all_throughputs is None or len(all_throughputs) == 0: return None
        if index is None: return None
        (job_ids, single_job_ids, worker_types, relevant_combinations) = index
        self._num_steps_remaining = [num_steps_remaining[single_job_id]
                                     for single_job_id in single_job_ids]

        # Row i of scale_factors_array is the scale_factor of job
        # combination i repeated len(worker_types) times.
        (m, n) = all_throughputs[0].shape
        # 导读 scale_factors_array 是按矩阵顺序排列的作业资源需求。
        scale_factors_array = self.scale_factors_array(
            scale_factors, job_ids, m, n)

        # Units are in seconds.
        max_T = 1000000.
        min_T = 100.
        last_max_T = max_T
        status = None
        last_feasible_x = None
        while last_feasible_x is None:
            # Binary search for the smallest T that gives an optimal solution.
            while (1.05 * min_T) < max_T:  # TODO: Can tweak the 1.05 in this loop.
                T = (min_T + max_T) / 2.
                status, x = self.get_allocation_helper(all_throughputs,
                                                       job_ids, single_job_ids,
                                                       scale_factors_array, T,
                                                       relevant_combinations)
                if status == "optimal":
                    last_feasible_x = x
                    max_T = T
                else:
                    min_T = T
            max_T = last_max_T * 10.
            min_T = last_max_T
            last_max_T *= 10

        assert(last_feasible_x is not None)
        return super().unflatten(
            last_feasible_x.value.clip(min=0.0).clip(max=1.0), index)
