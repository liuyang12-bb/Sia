# 导读 完成时间公平策略。比较共享集群中的完成时间与参考资源下的完成时间。
# 导读 来源 policies/finish_time_fairness.py。以下带「导读」的行是阅读注释。
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import copy
import cvxpy as cp
import numpy as np

from policy import Policy, PolicyWithPacking
from isolated import IsolatedPolicy

class FinishTimeFairnessPolicy(Policy):

    # 导读 FinishTimeFairnessPolicy.__init__（原源码第 13 行）
    # 导读 构造 FinishTimeFairnessPolicy 时执行。为实例建立 self._name、self._finish_time_fairness_perf_policy 等状态。
    # 导读 参数 solver 表示选用的数值求解器。
    # 导读 函数体直接赋值的实例字段包括 self._name、self._finish_time_fairness_perf_policy。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, solver):
        self._name = 'FinishTimeFairness'
        self._finish_time_fairness_perf_policy = \
            FinishTimeFairnessPolicyWithPerf(solver)

    # 导读 FinishTimeFairnessPolicy.get_allocation（原源码第 18 行）
    # 导读 本策略的分配入口。接收上层传入的状态，并使用 self._finish_time_fairness_perf_policy.get_allocation 等调用生成分配。
    # 导读 参数 unflattened_throughputs 表示求解前以作业或组合、worker 类型为键组织的吞吐字典，scale_factors 表示按作业索引的资源规模需求，unflattened_priority_weights 表示按作业 ID 组织的优先权重。
    # 导读 参数 times_since_start 表示按作业记录的已过去时间，num_steps_remaining 表示每个作业剩余的训练迭代数，total_steps 表示计划或声明的总训练迭代数。
    # 导读 参数 cluster_spec 表示每类 worker 的可用数量。
    # 导读 其余显式参数为 prestart_overhead，默认值见函数签名。
    # 导读 函数体中的相关调用包括 self._finish_time_fairness_perf_policy.get_allocation。
    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights,
                       times_since_start,
                       num_steps_remaining, total_steps, prestart_overhead, cluster_spec):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (job_ids, worker_types) = index

        new_unflattened_throughputs = {}
        for job_id in unflattened_throughputs:
            new_unflattened_throughputs[job_id] = {}
            for worker_type in unflattened_throughputs[job_id]:
                 # Hardcode worker_type to v100 since all other worker types
                 # have a throughput of 0 for some job.
                 new_unflattened_throughputs[job_id][worker_type] = \
                     unflattened_throughputs[job_id]['v100']

        return self._finish_time_fairness_perf_policy.get_allocation(
            new_unflattened_throughputs, scale_factors,
            unflattened_priority_weights,
            times_since_start,
            num_steps_remaining, cluster_spec)


class FinishTimeFairnessPolicyWithPerf(Policy):

    # 导读 FinishTimeFairnessPolicyWithPerf.__init__（原源码第 45 行）
    # 导读 构造 FinishTimeFairnessPolicyWithPerf 时执行。为实例建立 self._name、self._isolated_policy、self._cumulative_isolated_time、self._isolated_throughputs_prev_iteration、self._num_steps_remaining_prev_iteration 等状态。
    # 导读 参数 solver 表示选用的数值求解器。
    # 导读 函数体直接赋值的实例字段包括 self._name、self._isolated_policy、self._cumulative_isolated_time、self._isolated_throughputs_prev_iteration、self._num_steps_remaining_prev_iteration。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, solver):
        Policy.__init__(self, solver)
        self._name = 'FinishTimeFairness_Perf'
        self._isolated_policy = IsolatedPolicy()
        self._cumulative_isolated_time = {}
        self._isolated_throughputs_prev_iteration = {}
        self._num_steps_remaining_prev_iteration = {}

    # 导读 FinishTimeFairnessPolicyWithPerf.get_allocation（原源码第 53 行）
    # 导读 本策略的分配入口。接收上层传入的状态，并使用 self.scale_factors_array 等调用生成分配。
    # 导读 参数 unflattened_throughputs 表示求解前以作业或组合、worker 类型为键组织的吞吐字典，scale_factors 表示按作业索引的资源规模需求，unflattened_priority_weights 表示按作业 ID 组织的优先权重。
    # 导读 参数 times_since_start 表示按作业记录的已过去时间，num_steps_remaining 表示每个作业剩余的训练迭代数，total_steps 表示计划或声明的总训练迭代数。
    # 导读 参数 cluster_spec 表示每类 worker 的可用数量。
    # 导读 其余显式参数为 prestart_overhead，默认值见函数签名。
    # 导读 函数体中的相关调用包括 self.scale_factors_array、self._isolated_policy.get_throughputs、cp.Minimize、self.get_base_constraints、cp.Problem、cvxprob.solve。
    # 导读 函数体直接赋值的实例字段包括 self._isolated_throughputs_prev_iteration、self._num_steps_remaining_prev_iteration。
    # 导读 从返回表达式可以继续追踪 self._isolated_policy.get_allocation(unflattened_throughputs, scale_factors, cluster_spec)、super().unflatten(x.value.clip(min=0.0).clip(max=1.0), index)。
    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights,
                       times_since_start,
                       num_steps_remaining, total_steps, prestart_overhead, cluster_spec):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None:
            self._isolated_throughputs_prev_iteration = {}
            self._num_steps_remaining_prev_iteration = {}
            return None
        (m, n) = throughputs.shape
        (job_ids, worker_types) = index

        # Row i of scale_factors_array is the scale_factor of job i
        # repeated len(worker_types) times.
        # 导读 scale_factors_array 是按矩阵顺序排列的作业资源需求。
        scale_factors_array = self.scale_factors_array(
             scale_factors, job_ids, m, n)

        # TODO: Do something with these priority_weights.
        priority_weights = np.array(
            [1. / unflattened_priority_weights[job_id]
             for job_id in job_ids])

        # Create allocation variable, and isolated allocation.
        # 导读 CVXPY 在这里创建求解变量。其形状和整数或布尔类型由调用参数给出，取值约束在后面构造。
        x = cp.Variable(throughputs.shape)
        # 导读 isolated_throughputs 是隔离运行参考方案给出的作业吞吐。
        isolated_throughputs = self._isolated_policy.get_throughputs(
            throughputs, index, scale_factors, cluster_spec)
        expected_time_fractions = []
        for i in range(len(job_ids)):
            if job_ids[i] not in self._cumulative_isolated_time:
                self._cumulative_isolated_time[job_ids[i]] = 0
            if job_ids[i] in self._num_steps_remaining_prev_iteration:
                self._cumulative_isolated_time[job_ids[i]] += (
                    self._num_steps_remaining_prev_iteration[job_ids[i]] -
                    num_steps_remaining[job_ids[i]]) / \
                    self._isolated_throughputs_prev_iteration[job_ids[i]]

            allocation_throughput = cp.sum(cp.multiply(throughputs[i], x[i]))

            assert (isolated_throughputs[i] > 0)
            # expected_time_isolated = self._cumulative_isolated_time[job_ids[i]] + \
            #     (num_steps_remaining[job_ids[i]] / isolated_throughputs[i])
            expected_time_isolated = prestart_overhead[job_ids[i]] + total_steps[job_ids[i]] / isolated_throughputs[i]

            expected_time_allocation = times_since_start[job_ids[i]] + \
                (num_steps_remaining[job_ids[i]] * cp.inv_pos(allocation_throughput))
            expected_time_fraction = expected_time_allocation / expected_time_isolated
            expected_time_fractions.append(expected_time_fraction)
        if len(expected_time_fractions) == 1:
            objective = cp.Minimize(expected_time_fractions[0])
        else:
            objective = cp.Minimize(cp.maximum(*expected_time_fractions))

        # Make sure that the allocation can fit in the cluster.
        constraints = self.get_base_constraints(x, scale_factors_array)

        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver=self._solver)

        print(f'### ilp solution')
        print(f'objective: {objective.value}')
        for i in range(len(expected_time_fractions)):
            allocation_throughput = np.sum(np.multiply(throughputs[i], x.value[i]))
            expected_time_allocation = (times_since_start[job_ids[i]] + (num_steps_remaining[job_ids[i]] * cp.inv_pos(allocation_throughput))).value
            expected_time_isolated = prestart_overhead[job_ids[i]] + total_steps[job_ids[i]] / isolated_throughputs[i]

            print(f'{job_ids[i]}')
            print(f'vars: {x.value[i]}')
            print(f'\tallocated througput: {allocation_throughput}')
            print(f'\texpected time allocation: {times_since_start[job_ids[i]]} + {num_steps_remaining[job_ids[i]]} * {cp.inv_pos(allocation_throughput).value} = {expected_time_allocation}')
            print(f'\tisolated time: {expected_time_isolated}')
            print(f'\tftf: {expected_time_allocation} / {expected_time_isolated} = {expected_time_fractions[i].value} ({expected_time_allocation/expected_time_isolated})')

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        self._num_steps_remaining_prev_iteration = copy.copy(num_steps_remaining)
        self._isolated_throughputs_prev_iteration = {}
        for i in range(m):
            self._isolated_throughputs_prev_iteration[job_ids[i]] = \
                isolated_throughputs[i]

        if x.value is None:
            return self._isolated_policy.get_allocation(
                unflattened_throughputs, scale_factors, cluster_spec)
        return super().unflatten(x.value.clip(min=0.0).clip(max=1.0), index)


class FinishTimeFairnessPolicyWithPacking(PolicyWithPacking):

    # 导读 FinishTimeFairnessPolicyWithPacking.__init__（原源码第 143 行）
    # 导读 构造 FinishTimeFairnessPolicyWithPacking 时执行。为实例建立 self._name、self._isolated_policy、self._cumulative_isolated_time、self._isolated_throughputs_prev_iteration、self._num_steps_remaining_prev_iteration 等状态。
    # 导读 参数 solver 表示选用的数值求解器。
    # 导读 函数体直接赋值的实例字段包括 self._name、self._isolated_policy、self._cumulative_isolated_time、self._isolated_throughputs_prev_iteration、self._num_steps_remaining_prev_iteration。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, solver):
        PolicyWithPacking.__init__(self, solver)
        self._name = 'FinishTimeFairness_Packing'
        self._isolated_policy = IsolatedPolicy()
        self._cumulative_isolated_time = {}
        self._isolated_throughputs_prev_iteration = {}
        self._num_steps_remaining_prev_iteration = {}

    # 导读 FinishTimeFairnessPolicyWithPacking.get_allocation（原源码第 151 行）
    # 导读 本策略的分配入口。接收上层传入的状态，并使用 self.flatten 等调用生成分配。
    # 导读 参数 unflattened_throughputs 表示求解前以作业或组合、worker 类型为键组织的吞吐字典，scale_factors 表示按作业索引的资源规模需求，unflattened_priority_weights 表示按作业 ID 组织的优先权重。
    # 导读 参数 times_since_start 表示按作业记录的已过去时间，num_steps_remaining 表示每个作业剩余的训练迭代数，cluster_spec 表示每类 worker 的可用数量。
    # 导读 函数体中的相关调用包括 self.flatten、self.scale_factors_array、self._isolated_policy.get_throughputs、cp.Minimize、self.get_base_constraints、cp.Problem。
    # 导读 函数体直接赋值的实例字段包括 self._isolated_throughputs_prev_iteration、self._num_steps_remaining_prev_iteration。
    # 导读 从返回表达式可以继续追踪 self.unflatten(x.value.clip(min=0.0).clip(max=1.0), index)。
    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights,
                       times_since_start,
                       num_steps_remaining, cluster_spec):
        all_throughputs, index = \
            self.flatten(d=unflattened_throughputs,
                         cluster_spec=cluster_spec,
                         priority_weights=unflattened_priority_weights)
        if all_throughputs is None or len(all_throughputs) == 0:
            self._isolated_throughputs_prev_iteration = {}
            self._num_steps_remaining_prev_iteration = {}
            return None

        (m, n) = all_throughputs[0].shape
        (job_ids, single_job_ids, worker_types, relevant_combinations) = index
        # 导读 CVXPY 在这里创建求解变量。其形状和整数或布尔类型由调用参数给出，取值约束在后面构造。
        x = cp.Variable((m, n))

        # Row i of scale_factors_array is the scale_factor of job
        # combination i repeated len(worker_types) times.
        # 导读 scale_factors_array 是按矩阵顺序排列的作业资源需求。
        scale_factors_array = self.scale_factors_array(
            scale_factors, job_ids, m, n)

        throughputs_no_packed_jobs = np.zeros((len(single_job_ids), n))
        for i, single_job_id in enumerate(single_job_ids):
            for j, worker_type in enumerate(worker_types):
                throughputs_no_packed_jobs[i, j] = \
                    unflattened_throughputs[single_job_id][worker_type]
        # 导读 isolated_throughputs 是隔离运行参考方案给出的作业吞吐。
        isolated_throughputs = self._isolated_policy.get_throughputs(
            throughputs_no_packed_jobs,
            (single_job_ids, worker_types),
            scale_factors,
            cluster_spec)

        single_throughputs = np.zeros((len(single_job_ids), n))
        expected_time_fractions = []
        for i in range(len(all_throughputs)):
            if single_job_ids[i] not in self._cumulative_isolated_time:
                self._cumulative_isolated_time[single_job_ids[i]] = 0
            if single_job_ids[i] in self._num_steps_remaining_prev_iteration:
                self._cumulative_isolated_time[single_job_ids[i]] += (
                    self._num_steps_remaining_prev_iteration[single_job_ids[i]] -
                    num_steps_remaining[single_job_ids[i]]) / \
                    self._isolated_throughputs_prev_iteration[single_job_ids[i]]

            indexes = relevant_combinations[single_job_ids[i]]
            isolated_throughput = isolated_throughputs[i]
            allocation_throughput = cp.sum(cp.multiply(
                all_throughputs[i][indexes],
                x[indexes]))
            expected_time_isolated = self._cumulative_isolated_time[single_job_ids[i]] + \
                (num_steps_remaining[single_job_ids[i]] / isolated_throughput)
            expected_time_allocation = times_since_start[single_job_ids[i]] + \
                (num_steps_remaining[single_job_ids[i]] * cp.inv_pos(allocation_throughput))
            expected_time_fraction = expected_time_allocation / expected_time_isolated
            expected_time_fractions.append(expected_time_fraction)
        if len(expected_time_fractions) == 1:
            objective = cp.Minimize(expected_time_fractions[0])
        else:
            objective = cp.Minimize(cp.maximum(*expected_time_fractions))

        # Make sure the allocation can fit in the cluster.
        constraints = self.get_base_constraints(x, single_job_ids,
                                                scale_factors_array,
                                                relevant_combinations)

        # Explicitly constrain all allocation values with an effective scale
        # factor of 0 to be 0.
        # NOTE: This is not strictly necessary because these allocation values
        # do not affect the optimal allocation for nonzero scale factor
        # combinations.
        for i in range(m):
            for j in range(n):
                if scale_factors_array[i,j] == 0:
                    constraints.append(x[i,j] == 0)
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver=self._solver)

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        self._num_steps_remaining_prev_iteration = copy.copy(num_steps_remaining)
        self._isolated_throughputs_prev_iteration = {}
        for i in range(len(all_throughputs)):
            self._isolated_throughputs_prev_iteration[single_job_ids[i]] = \
                isolated_throughputs[i]

        return self.unflatten(x.value.clip(min=0.0).clip(max=1.0), index)
