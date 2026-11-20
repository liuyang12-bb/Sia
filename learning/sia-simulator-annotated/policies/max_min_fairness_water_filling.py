# 导读 渐进填充式最大最小公平。逐轮识别瓶颈作业、固定已达瓶颈的分配并继续优化其余作业。
# 导读 来源 policies/max_min_fairness_water_filling.py。以下带「导读」的行是阅读注释。
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import copy
import cvxpy as cp
import numpy as np

from policy import Policy, PolicyWithPacking
from proportional import ProportionalPolicy


class WaterFillingAlgorithm:

    # 导读 WaterFillingAlgorithm.__init__（原源码第 14 行）
    # 导读 构造 WaterFillingAlgorithm 时执行。为实例建立 self._previous_priority_weights、self._priority_reweighting_policies、self._lp、self._milp 等状态。
    # 导读 参数 priority_reweighting_policies 表示计算或调整优先权重所用的参考策略。
    # 导读 函数体直接赋值的实例字段包括 self._previous_priority_weights、self._priority_reweighting_policies、self._lp、self._milp。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, priority_reweighting_policies):
        self._previous_priority_weights = None
        self._priority_reweighting_policies = priority_reweighting_policies
        self._lp = None
        self._milp = None

    # 导读 WaterFillingAlgorithm._compute_priority_weights（原源码第 20 行）
    # 导读 为渐进填充过程计算作业优先权重。
    # 导读 参数 entity_weights 表示应用到当前求解实体的优先权重，priority_weights 表示求解目标使用的作业优先权重，entity_to_job_mapping 表示策略实体与组成它的作业之间的映射。
    # 导读 参数 job_ids 表示用于保持结果顺序的作业标识。
    # 导读 其余显式参数为 final_normalized_effective_throughputs，默认值见函数签名。
    # 导读 从返回表达式可以继续追踪 priority_weights、returned_priority_weights。
    def _compute_priority_weights(self, entity_weights, priority_weights, entity_to_job_mapping,
                                  final_normalized_effective_throughputs, job_ids):
        returned_priority_weights = {}
        if self._priority_reweighting_policies is None:
            # Do nothing if reweighting policy is None.
            return priority_weights
        if entity_to_job_mapping is None:
            raise ValueError("entity_to_job_mapping cannot be None when "
                             "priority_reweighting_policies is not None!")
        for entity_id in entity_to_job_mapping:
            priority_reweighting_policy = self._priority_reweighting_policies[
                entity_id]
            if priority_reweighting_policy == 'fairness':
                # The sum of final priority weights for all jobs in an entity
                # should be equal to the priority weight of that entity. The
                # final priority weight of a job should be proportional to the
                # passed-in weight. A job does not contribute its priority once
                # saturated.
                entity_weight = entity_weights[entity_id]
                total_job_priority_in_entity = 0.0
                for job_id in entity_to_job_mapping[entity_id]:
                    if job_id in final_normalized_effective_throughputs:
                        continue
                    total_job_priority_in_entity += float(priority_weights[job_id])
                for job_id in entity_to_job_mapping[entity_id]:
                    if job_id in final_normalized_effective_throughputs:
                        returned_priority_weights[job_id] = 0.0
                    else:
                        returned_priority_weights[job_id] = \
                            entity_weight * (float(priority_weights[job_id]) /
                                             total_job_priority_in_entity)
            elif priority_reweighting_policy == 'fifo':
                # Active jobs are given the corresponding entity's weight.
                # Jobs become active in FIFO order within an entity.
                entity_weight = entity_weights[entity_id]
                total_job_priority_in_entity = 0.0
                entity_to_job_mapping[entity_id].sort()
                done = False
                for job_id in entity_to_job_mapping[entity_id]:
                    if job_id in final_normalized_effective_throughputs:
                        returned_priority_weights[job_id] = 0.0
                    elif not done:
                        returned_priority_weights[job_id] = entity_weight
                        done = True
                    else:
                        returned_priority_weights[job_id] = 0.0
            else:
                raise ValueError("Unknown priority reweighting policy!")
        return returned_priority_weights


    # 导读 WaterFillingAlgorithm._get_allocation（原源码第 71 行）
    # 导读 实现本策略的内部求解步骤，返回用于后续恢复字典结构的分配结果。
    # 导读 参数 job_ids 表示用于保持结果顺序的作业标识，priority_weights 表示求解目标使用的作业优先权重，proportional_throughputs 表示按比例共享资源时的参考吞吐。
    # 导读 参数 scale_factors_array 表示与当前矩阵作业索引顺序对应的资源需求。
    # 导读 其余显式参数为 m、n、final_normalized_effective_throughputs、normalized_effective_throughputs_so_far，默认值见函数签名。
    # 导读 函数体中的相关调用包括 self._get_effective_throughputs、cp.Maximize、self._get_constraints、cp.Problem、self._lp.solve。
    # 导读 函数体直接赋值的实例字段包括 self._lp_variables_and_parameters、self._lp_objective、self._lp。
    # 导读 从返回表达式可以继续追踪 (x.value, self._lp_objective.value, mask)。
    def _get_allocation(self, job_ids, priority_weights,
                        proportional_throughputs,
                        scale_factors_array, m, n,
                        final_normalized_effective_throughputs,
                        normalized_effective_throughputs_so_far):
        M = self._M

        if self._lp is None:
            self._lp_variables_and_parameters = {}
            # 导读 CVXPY 在这里创建求解变量。其形状和整数或布尔类型由调用参数给出，取值约束在后面构造。
            self._lp_variables_and_parameters['x'] = cp.Variable((m, n))
            self._lp_variables_and_parameters[
                'normalized_effective_throughputs_so_far_parameter'] = cp.Parameter(len(job_ids))
            self._lp_variables_and_parameters[
                'normalized_effective_throughputs_lower_bounds_parameter'] = cp.Parameter(len(job_ids))
            self._lp_variables_and_parameters[
                'multiplicative_terms_parameter'] = cp.Parameter(len(job_ids))
            self._lp_variables_and_parameters[
                'additive_terms_parameter'] = cp.Parameter(len(job_ids))

        x = self._lp_variables_and_parameters['x']
        normalized_effective_throughputs_so_far_parameter = \
            self._lp_variables_and_parameters[
                'normalized_effective_throughputs_so_far_parameter']
        normalized_effective_throughputs_lower_bounds_parameter = \
            self._lp_variables_and_parameters[
                'normalized_effective_throughputs_lower_bounds_parameter']
        multiplicative_terms_parameter = \
            self._lp_variables_and_parameters['multiplicative_terms_parameter']
        additive_terms_parameter = \
            self._lp_variables_and_parameters['additive_terms_parameter']

        # 导读 effective_throughputs 是将候选吞吐按当前分配份额汇总得到的有效吞吐。
        effective_throughputs = self._get_effective_throughputs(x)
        normalized_effective_throughputs = cp.multiply(
            effective_throughputs,
            1.0 / proportional_throughputs.reshape(len(job_ids)))

        # Solve max-min optimization problem over all jobs with priority
        # weight > 0.
        normalized_effective_throughputs_so_far_parameter.value = \
            normalized_effective_throughputs_so_far
        objective_terms = normalized_effective_throughputs - \
            normalized_effective_throughputs_so_far_parameter
        multiplicative_terms = np.zeros(len(job_ids))
        mask = np.zeros(len(job_ids))
        additive_terms = np.zeros(len(job_ids))
        for i, job_id in enumerate(job_ids):
            if job_id not in final_normalized_effective_throughputs:
                if priority_weights[i] > 0.0:
                    multiplicative_terms[i] = priority_weights[i] * scale_factors_array[i, 0]
                    mask[i] = 1.0 / multiplicative_terms[i]
                else:
                    additive_terms[i] = M
            else:
                additive_terms[i] = M
        multiplicative_terms_parameter.value = multiplicative_terms
        additive_terms_parameter.value = additive_terms

        normalized_effective_throughputs_lower_bounds = np.zeros(len(job_ids))
        for i, job_id in enumerate(job_ids):
            if job_id in final_normalized_effective_throughputs:
                normalized_effective_throughputs_lower_bounds[i] = \
                    final_normalized_effective_throughputs[job_id]
            else:
                normalized_effective_throughputs_lower_bounds[i] = \
                    normalized_effective_throughputs_so_far[i]
        normalized_effective_throughputs_lower_bounds_parameter.value = \
            normalized_effective_throughputs_lower_bounds

        if self._lp is None:
            self._lp_objective = cp.Maximize(cp.min(
                cp.multiply(objective_terms, multiplicative_terms_parameter) +
                additive_terms_parameter))
            # Specify constraints.
            constraints = self._get_constraints(x, scale_factors_array)
            constraints.append(normalized_effective_throughputs >=
                normalized_effective_throughputs_lower_bounds_parameter)

            self._lp = cp.Problem(self._lp_objective, constraints)

        result = self._lp.solve(solver='ECOS', warm_start=True)

        return x.value, self._lp_objective.value, mask

    # 导读 WaterFillingAlgorithm._get_bottleneck_jobs（原源码第 154 行）
    # 导读 在当前填充分配下识别已经受限的作业。
    # 导读 参数 job_ids 表示用于保持结果顺序的作业标识，priority_weights 表示求解目标使用的作业优先权重，proportional_throughputs 表示按比例共享资源时的参考吞吐。
    # 导读 参数 scale_factors_array 表示与当前矩阵作业索引顺序对应的资源需求。
    # 导读 其余显式参数为 m、n、final_normalized_effective_throughputs、normalized_effective_throughputs_so_far、slack，默认值见函数签名。
    # 导读 函数体中的相关调用包括 self._get_effective_throughputs、cp.Maximize、self._get_constraints、cp.Problem、self._milp.solve。
    # 导读 函数体直接赋值的实例字段包括 self._milp_variables_and_parameters、self._milp。
    # 导读 从返回表达式可以继续追踪 z.value。
    def _get_bottleneck_jobs(self, job_ids, priority_weights,
                             proportional_throughputs,
                             scale_factors_array, m, n,
                             final_normalized_effective_throughputs,
                             normalized_effective_throughputs_so_far, slack=1.0001):
        M = self._M
        epsilon = 1e-5

        if self._milp is None:
            self._milp_variables_and_parameters = {}
            # 导读 CVXPY 在这里创建求解变量。其形状和整数或布尔类型由调用参数给出，取值约束在后面构造。
            self._milp_variables_and_parameters['x'] = cp.Variable((m, n))
            # 导读 CVXPY 在这里创建求解变量。其形状和整数或布尔类型由调用参数给出，取值约束在后面构造。
            self._milp_variables_and_parameters['z'] = cp.Variable(len(job_ids), boolean=True)
            self._milp_variables_and_parameters[
                'normalized_effective_throughputs_so_far_parameter'] = cp.Parameter(len(job_ids))
            self._milp_variables_and_parameters[
                'normalized_effective_throughputs_lower_bounds_parameter'] = cp.Parameter(len(job_ids))
            self._milp_variables_and_parameters['mask_parameter'] = cp.Parameter(len(job_ids))

        x = self._milp_variables_and_parameters['x']
        z = self._milp_variables_and_parameters['z']
        normalized_effective_throughputs_so_far_parameter = \
            self._milp_variables_and_parameters[
                'normalized_effective_throughputs_so_far_parameter']
        normalized_effective_throughputs_lower_bounds_parameter = \
            self._milp_variables_and_parameters[
                'normalized_effective_throughputs_lower_bounds_parameter']
        mask_parameter = self._milp_variables_and_parameters['mask_parameter']

        # 导读 effective_throughputs 是将候选吞吐按当前分配份额汇总得到的有效吞吐。
        effective_throughputs = self._get_effective_throughputs(x)
        normalized_effective_throughputs = cp.multiply(
            effective_throughputs,
            1.0 / proportional_throughputs.reshape(len(job_ids)))

        normalized_effective_throughputs_so_far_parameter.value = \
            normalized_effective_throughputs_so_far

        normalized_effective_throughputs_lower_bounds = np.zeros(len(job_ids))
        for i, job_id in enumerate(job_ids):
            if job_id in final_normalized_effective_throughputs:
                normalized_effective_throughputs_lower_bounds[i] = \
                    final_normalized_effective_throughputs[job_id]
            else:
                normalized_effective_throughputs_lower_bounds[i] = \
                    normalized_effective_throughputs_so_far[i]
        normalized_effective_throughputs_lower_bounds_parameter.value = \
            normalized_effective_throughputs_lower_bounds

        mask = np.zeros(len(job_ids))
        for i, job_id in enumerate(job_ids):
            if job_id in final_normalized_effective_throughputs:
                mask[i] = 1.0
            else:
                if priority_weights[i] == 0.0:
                    mask[i] = 1.0
        mask_parameter.value = mask

        if self._milp is None:
            objective = cp.Maximize(cp.sum(z))
            # Specify constraints.
            constraints = self._get_constraints(x, scale_factors_array)
            constraints.append(
                (M * z) >=
                    (normalized_effective_throughputs -
                     (normalized_effective_throughputs_so_far_parameter * slack) +
                     epsilon))
            constraints.append(
                 (M * (1 - z)) >=
                    ((normalized_effective_throughputs_so_far_parameter * slack) -
                     normalized_effective_throughputs))
            constraints.append(normalized_effective_throughputs >=
                normalized_effective_throughputs_lower_bounds_parameter)
            constraints.append(cp.multiply(mask_parameter, z) == 0)

            self._milp = cp.Problem(objective, constraints)

        result = self._milp.solve(solver='GLPK_MI', warm_start=True)
        if self._milp.status != "optimal":
            raise Exception("WARNING: Non-optimal allocation in _get_bottleneck_jobs()!")

        return z.value

    # 导读 WaterFillingAlgorithm._run_get_allocation_iterations（原源码第 235 行）
    # 导读 反复求解，锁定瓶颈作业并为未锁定作业继续分配资源。
    # 导读 参数 job_ids 表示用于保持结果顺序的作业标识，proportional_throughputs 表示按比例共享资源时的参考吞吐，scale_factors_array 表示与当前矩阵作业索引顺序对应的资源需求。
    # 导读 参数 entity_weights 表示应用到当前求解实体的优先权重，unflattened_priority_weights 表示按作业 ID 组织的优先权重，cluster_spec 表示每类 worker 的可用数量。
    # 导读 参数 entity_to_job_mapping 表示策略实体与组成它的作业之间的映射。
    # 导读 其余显式参数为 m、n、verbose，默认值见函数签名。
    # 导读 函数体中的相关调用包括 self._compute_priority_weights、self._get_allocation、self._get_bottleneck_jobs。
    # 导读 函数体直接赋值的实例字段包括 self._previous_priority_weights。
    # 导读 从返回表达式可以继续追踪 x。
    def _run_get_allocation_iterations(self, job_ids, m, n,
                                       proportional_throughputs,
                                       scale_factors_array,
                                       entity_weights,
                                       unflattened_priority_weights, cluster_spec,
                                       entity_to_job_mapping, verbose):
        done = False

        final_normalized_effective_throughputs = {}
        normalized_effective_throughputs_so_far = np.zeros(len(job_ids))
        num_iterations = 0
        c, x, mask = 0, None, None
        while not done:
            priority_weights = self._compute_priority_weights(
                entity_weights,
                unflattened_priority_weights,
                entity_to_job_mapping,
                final_normalized_effective_throughputs, job_ids)
            previous_priority_weights = copy.copy(priority_weights)
            if verbose:
                print("Using the following as priority weights:", np.array(
                    [priority_weights[job_id] for job_id in job_ids]))
            priority_weights = np.array(
                [1. / priority_weights[job_id] if priority_weights[job_id] > 0 else 0.0
                 for job_id in job_ids])

            old_x, old_c, old_mask = np.copy(x), c, np.copy(mask)
            try:
                x, c, mask = self._get_allocation(
                    job_ids, priority_weights, proportional_throughputs,
                    scale_factors_array,
                    m, n, final_normalized_effective_throughputs,
                    normalized_effective_throughputs_so_far)
                if x is None:
                    x, c, mask = old_x, old_c, old_mask
                    done = True
                else:
                    normalized_effective_throughputs_so_far += np.multiply(mask, c)
            except:
                x, c, mask = old_x, old_c, old_mask
                done = True

            self._previous_priority_weights = previous_priority_weights
            if verbose:
                if num_iterations == 0:
                    print("Objective value: %.3f" % c)

            # Find bottleneck job_ids.
            try:
                z = self._get_bottleneck_jobs(
                    job_ids, priority_weights,
                    proportional_throughputs,
                    scale_factors_array, m, n,
                    final_normalized_effective_throughputs,
                    normalized_effective_throughputs_so_far)
            except Exception as e:
                z = np.zeros(len(job_ids))
            old_len_effective_throughputs = len(final_normalized_effective_throughputs)
            for i, job_id in enumerate(job_ids):
                if job_id not in final_normalized_effective_throughputs and (z is None or not z[i]) \
                    and priority_weights[i] > 0.0:
                    if verbose:
                        print("Iteration %d:" % num_iterations, job_id)
                    final_normalized_effective_throughputs[job_id] = \
                        normalized_effective_throughputs_so_far[i]
            if old_len_effective_throughputs == len(final_normalized_effective_throughputs):
                done = True
            num_iterations += 1
            if len(final_normalized_effective_throughputs) == m:
                done = True
        if verbose:
            print("Number of iterations: %d" % num_iterations)

        return x


class MaxMinFairnessWaterFillingPolicy(Policy, WaterFillingAlgorithm):

    # 导读 MaxMinFairnessWaterFillingPolicy.__init__（原源码第 313 行）
    # 导读 构造 MaxMinFairnessWaterFillingPolicy 时执行。为实例建立 self._name、self._max_min_fairness_perf_policy 等状态。
    # 导读 参数 priority_reweighting_policies 表示计算或调整优先权重所用的参考策略。
    # 导读 函数体直接赋值的实例字段包括 self._name、self._max_min_fairness_perf_policy。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, priority_reweighting_policies=None):
        self._name = 'MaxMinFairnessWaterFilling'
        self._max_min_fairness_perf_policy = \
            MaxMinFairnessWaterFillingPolicyWithPerf(priority_reweighting_policies)

    # 导读 MaxMinFairnessWaterFillingPolicy.get_allocation（原源码第 318 行）
    # 导读 整理当前作业矩阵和优先级，调用渐进填充的迭代求解，随后将分配还原为作业和 worker 字典。
    # 导读 参数 unflattened_throughputs 表示求解前以作业或组合、worker 类型为键组织的吞吐字典，scale_factors 表示按作业索引的资源规模需求，unflattened_priority_weights 表示按作业 ID 组织的优先权重。
    # 导读 参数 cluster_spec 表示每类 worker 的可用数量，entity_weights 表示应用到当前求解实体的优先权重，entity_to_job_mapping 表示策略实体与组成它的作业之间的映射。
    # 导读 其余显式参数为 verbose、return_effective_throughputs，默认值见函数签名。
    # 导读 函数体中的相关调用包括 self._max_min_fairness_perf_policy.get_allocation、self._max_min_fairness_perf_policy._proportional_policy.get_throughputs。
    # 导读 从返回表达式可以继续追踪 (normalized_effective_throughputs, job_ids)、unflattened_x。
    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights, cluster_spec,
                       entity_weights=None,
                       entity_to_job_mapping=None, verbose=False,
                       return_effective_throughputs=False):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (job_ids, worker_types) = index
        m, n = len(job_ids), len(worker_types)

        new_unflattened_throughputs = {}
        for job_id in unflattened_throughputs:
            new_unflattened_throughputs[job_id] = {}
            for worker_type in unflattened_throughputs[job_id]:
                 new_unflattened_throughputs[job_id][worker_type] = 1.0

        unflattened_x = \
            self._max_min_fairness_perf_policy.get_allocation(
                new_unflattened_throughputs, scale_factors,
                unflattened_priority_weights,
                cluster_spec,
                entity_weights=entity_weights,
                entity_to_job_mapping=entity_to_job_mapping,
                verbose=verbose,
                return_effective_throughputs=False)
        x = np.zeros((len(job_ids), len(worker_types)))
        for i, job_id in enumerate(job_ids):
            for j, worker_type in enumerate(worker_types):
                x[i, j] = unflattened_x[job_id][worker_type]
        if return_effective_throughputs:
            # 导读 effective_throughputs 是将候选吞吐按当前分配份额汇总得到的有效吞吐。
            effective_throughputs = np.sum(np.multiply(
                throughputs, x), axis=1)
            proportional_throughputs = \
                self._max_min_fairness_perf_policy._proportional_policy.get_throughputs(
                    throughputs, index, cluster_spec)
            normalized_effective_throughputs = np.multiply(
                effective_throughputs,
                1.0 / proportional_throughputs.reshape(m))
            return normalized_effective_throughputs, job_ids
        return unflattened_x

class MaxMinFairnessWaterFillingPolicyWithPerf(Policy, WaterFillingAlgorithm):

    # 导读 MaxMinFairnessWaterFillingPolicyWithPerf.__init__（原源码第 362 行）
    # 导读 构造 MaxMinFairnessWaterFillingPolicyWithPerf 时执行。为实例建立 self._name、self._proportional_policy 等状态。
    # 导读 参数 priority_reweighting_policies 表示计算或调整优先权重所用的参考策略。
    # 导读 函数体直接赋值的实例字段包括 self._name、self._proportional_policy。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, priority_reweighting_policies=None):
        WaterFillingAlgorithm.__init__(self, priority_reweighting_policies)
        Policy.__init__(self, solver=None)
        self._name = 'MaxMinFairnessWaterFilling_Perf'
        self._proportional_policy = ProportionalPolicy()

    # 导读 MaxMinFairnessWaterFillingPolicyWithPerf._get_constraints（原源码第 368 行）
    # 导读 构造当前策略变体在填充求解中需要的约束。
    # 导读 参数 scale_factors_array 表示与当前矩阵作业索引顺序对应的资源需求。
    # 导读 其余显式参数为 x，默认值见函数签名。
    # 导读 函数体中的相关调用包括 self.get_base_constraints。
    # 导读 从返回表达式可以继续追踪 self.get_base_constraints(x, scale_factors_array)。
    def _get_constraints(self, x, scale_factors_array):
        return self.get_base_constraints(x, scale_factors_array)

    # 导读 MaxMinFairnessWaterFillingPolicyWithPerf.get_allocation（原源码第 371 行）
    # 导读 本策略的分配入口。接收上层传入的状态，并使用 self.scale_factors_array 等调用生成分配。
    # 导读 参数 unflattened_throughputs 表示求解前以作业或组合、worker 类型为键组织的吞吐字典，scale_factors 表示按作业索引的资源规模需求，unflattened_priority_weights 表示按作业 ID 组织的优先权重。
    # 导读 参数 cluster_spec 表示每类 worker 的可用数量，entity_weights 表示应用到当前求解实体的优先权重，entity_to_job_mapping 表示策略实体与组成它的作业之间的映射。
    # 导读 其余显式参数为 verbose、return_effective_throughputs，默认值见函数签名。
    # 导读 函数体中的相关调用包括 self.scale_factors_array、self._proportional_policy.get_throughputs、self._run_get_allocation_iterations。
    # 导读 函数体直接赋值的实例字段包括 self._M、self._get_effective_throughputs、self._lp、self._milp。
    # 导读 从返回表达式可以继续追踪 (normalized_effective_throughputs, job_ids)、super().unflatten(x.clip(min=0.0).clip(max=1.0), index)。
    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights, cluster_spec,
                       entity_weights=None,
                       entity_to_job_mapping=None, verbose=False,
                       return_effective_throughputs=False):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None

        (job_ids, worker_types) = index
        (m, n) = throughputs.shape

        # Row i of scale_factors_array is the scale_factor of job i
        # repeated len(worker_types) times.
        # 导读 scale_factors_array 是按矩阵顺序排列的作业资源需求。
        scale_factors_array = self.scale_factors_array(
             scale_factors, job_ids, m, n)
        proportional_throughputs = self._proportional_policy.get_throughputs(
            throughputs, index, cluster_spec)

        self._M = np.max(
            np.multiply(throughputs * (1.0 / proportional_throughputs).reshape((m, 1)),
                        scale_factors_array))
        self._get_effective_throughputs = lambda x: \
            cp.sum(cp.multiply(throughputs, x), axis=1)

        x = self._run_get_allocation_iterations(
            job_ids, m, n, proportional_throughputs,
            scale_factors_array,
            entity_weights,
            unflattened_priority_weights, cluster_spec,
            entity_to_job_mapping=entity_to_job_mapping, verbose=verbose)

        priority_weights = np.array(
            [1. / unflattened_priority_weights[job_id]
             for job_id in job_ids])
        priority_weights = np.multiply(
            priority_weights.reshape((m, 1)),
            1.0 / proportional_throughputs.reshape((m, 1)))
        # 导读 effective_throughputs 是将候选吞吐按当前分配份额汇总得到的有效吞吐。
        effective_throughputs = np.sum(np.multiply(
            throughputs, x), axis=1)
        normalized_effective_throughputs = np.multiply(
            effective_throughputs,
            1.0 / proportional_throughputs.reshape(m))

        if verbose:
            print("Normalized effective throughputs:",
                normalized_effective_throughputs)
            print("Constraints:",
                np.multiply(x, scale_factors_array).sum(axis=0),
                x.sum(axis=1))

        if return_effective_throughputs:
            return normalized_effective_throughputs, job_ids

        self._lp = None
        self._milp = None

        return super().unflatten(x.clip(min=0.0).clip(max=1.0), index)


class MaxMinFairnessWaterFillingPolicyWithPacking(PolicyWithPacking, WaterFillingAlgorithm):

    # 导读 MaxMinFairnessWaterFillingPolicyWithPacking.__init__（原源码第 433 行）
    # 导读 构造 MaxMinFairnessWaterFillingPolicyWithPacking 时执行。为实例建立 self._name、self._proportional_policy 等状态。
    # 导读 参数 priority_reweighting_policies 表示计算或调整优先权重所用的参考策略。
    # 导读 函数体直接赋值的实例字段包括 self._name、self._proportional_policy。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, priority_reweighting_policies=None):
        WaterFillingAlgorithm.__init__(self, priority_reweighting_policies)
        PolicyWithPacking.__init__(self, solver=None)
        self._name = 'MaxMinFairnessWaterFilling_Packing'
        self._proportional_policy = ProportionalPolicy()

    # 导读 MaxMinFairnessWaterFillingPolicyWithPacking._get_constraints（原源码第 439 行）
    # 导读 构造当前策略变体在填充求解中需要的约束。
    # 导读 参数 scale_factors_array 表示与当前矩阵作业索引顺序对应的资源需求。
    # 导读 其余显式参数为 x，默认值见函数签名。
    # 导读 函数体中的相关调用包括 self.get_base_constraints。
    # 导读 从返回表达式可以继续追踪 self.get_base_constraints(x, self._single_job_ids, scale_factors_array, self._relevant_combinations)。
    def _get_constraints(self, x, scale_factors_array):
        return self.get_base_constraints(x, self._single_job_ids,
                                         scale_factors_array,
                                         self._relevant_combinations)

    # 导读 MaxMinFairnessWaterFillingPolicyWithPacking._get_M（原源码第 444 行）
    # 导读 构造组合放置相关的矩阵，将组合配置关联到对应的作业。
    # 导读 参数 all_throughputs 表示当前组合或作业全集的吞吐矩阵，index 表示将字典转换成矩阵时保存的行列索引，proportional_throughputs 表示按比例共享资源时的参考吞吐。
    # 导读 参数 scale_factors_array 表示与当前矩阵作业索引顺序对应的资源需求。
    # 导读 从返回表达式可以继续追踪 np.max(np.array(max_throughputs))。
    def _get_M(self, all_throughputs, index, proportional_throughputs,
               scale_factors_array):
        (_, single_job_ids, _, relevant_combinations) = index
        max_throughputs = []
        for i, single_job_id in enumerate(single_job_ids):
            indexes = relevant_combinations[single_job_id]
            max_throughputs.append(np.max(
                np.multiply(all_throughputs[i][indexes], scale_factors_array[indexes])))
            max_throughputs[-1] /= proportional_throughputs[i]
        return np.max(np.array(max_throughputs))

    # 导读 MaxMinFairnessWaterFillingPolicyWithPacking._get_effective_throughputs_helper（原源码第 455 行）
    # 导读 计算当前分配下的有效吞吐，供公平性约束使用。
    # 导读 参数 all_throughputs 表示当前组合或作业全集的吞吐矩阵，index 表示将字典转换成矩阵时保存的行列索引。
    # 导读 其余显式参数为 x，默认值见函数签名。
    # 导读 从返回表达式可以继续追踪 cp.hstack(effective_throughputs)。
    def _get_effective_throughputs_helper(self, x, all_throughputs, index):
        (_, single_job_ids, _, relevant_combinations) = index
        # 导读 effective_throughputs 是将候选吞吐按当前分配份额汇总得到的有效吞吐。
        effective_throughputs = []
        for i, single_job_id in enumerate(single_job_ids):
            indexes = relevant_combinations[single_job_id]
            effective_throughputs.append(cp.sum(cp.multiply(
                all_throughputs[i][indexes],
                x[indexes])))
        return cp.hstack(effective_throughputs)

    # 导读 MaxMinFairnessWaterFillingPolicyWithPacking.get_allocation（原源码第 465 行）
    # 导读 本策略的分配入口。接收上层传入的状态，并使用 self.flatten 等调用生成分配。
    # 导读 参数 unflattened_throughputs 表示求解前以作业或组合、worker 类型为键组织的吞吐字典，scale_factors 表示按作业索引的资源规模需求，unflattened_priority_weights 表示按作业 ID 组织的优先权重。
    # 导读 参数 cluster_spec 表示每类 worker 的可用数量，entity_weights 表示应用到当前求解实体的优先权重，entity_to_job_mapping 表示策略实体与组成它的作业之间的映射。
    # 导读 其余显式参数为 verbose、return_effective_throughputs，默认值见函数签名。
    # 导读 函数体中的相关调用包括 self.flatten、self.scale_factors_array、self._proportional_policy.get_throughputs、self._get_M、self._run_get_allocation_iterations。
    # 导读 函数体直接赋值的实例字段包括 self._single_job_ids、self._relevant_combinations、self._M、self._get_effective_throughputs、self._lp、self._milp。
    # 导读 从返回表达式可以继续追踪 (normalized_effective_throughputs, single_job_ids)、super().unflatten(x.clip(min=0.0).clip(max=1.0), index)。
    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights, cluster_spec,
                       entity_weights=None,
                       entity_to_job_mapping=None, verbose=False,
                       return_effective_throughputs=False):
        all_throughputs, index = \
            self.flatten(d=unflattened_throughputs,
                         cluster_spec=cluster_spec)
        if all_throughputs is None or len(all_throughputs) == 0: return None

        (job_ids, single_job_ids, worker_types, relevant_combinations) = index
        self._single_job_ids = single_job_ids
        self._relevant_combinations = relevant_combinations
        (m, n) = all_throughputs[0].shape

        # Row i of scale_factors_array is the scale_factor of job i
        # repeated len(worker_types) times.
        # 导读 scale_factors_array 是按矩阵顺序排列的作业资源需求。
        scale_factors_array = self.scale_factors_array(
             scale_factors, job_ids, m, n)
        throughputs_no_packed_jobs = np.zeros((len(single_job_ids), n))
        for i, single_job_id in enumerate(single_job_ids):
            for j, worker_type in enumerate(worker_types):
                throughputs_no_packed_jobs[i, j] = \
                    unflattened_throughputs[single_job_id][worker_type]
        proportional_throughputs = self._proportional_policy.get_throughputs(
            throughputs_no_packed_jobs,
            (single_job_ids, worker_types),
            cluster_spec)

        self._M = self._get_M(all_throughputs, index, proportional_throughputs,
                              scale_factors_array)
        self._get_effective_throughputs = lambda x: \
            self._get_effective_throughputs_helper(x, all_throughputs, index)

        x = self._run_get_allocation_iterations(
            single_job_ids, m, n, proportional_throughputs,
            scale_factors_array,
            entity_weights,
            unflattened_priority_weights, cluster_spec,
            entity_to_job_mapping=entity_to_job_mapping, verbose=verbose)

        priority_weights = np.array(
            [1. / unflattened_priority_weights[single_job_id]
             for single_job_id in single_job_ids])
        priority_weights = np.multiply(
            priority_weights.reshape((len(single_job_ids), 1)),
            1.0 / proportional_throughputs.reshape((len(single_job_ids), 1)))
        # 导读 effective_throughputs 是将候选吞吐按当前分配份额汇总得到的有效吞吐。
        effective_throughputs = np.zeros(len(single_job_ids))
        for i, single_job_id in enumerate(single_job_ids):
            indexes = relevant_combinations[single_job_id]
            effective_throughputs[i] = np.sum(np.multiply(
                all_throughputs[i][indexes],
                x[indexes]))
        normalized_effective_throughputs = np.multiply(
            effective_throughputs,
            1.0 / proportional_throughputs.reshape(len(single_job_ids)))

        if verbose:
            print("Normalized effective throughputs:",
                normalized_effective_throughputs)
            print("Constraints:",
                np.multiply(x, scale_factors_array).sum(axis=0),
                x.sum(axis=1))

        if return_effective_throughputs:
            return normalized_effective_throughputs, single_job_ids

        self._lp = None
        self._milp = None

        return super().unflatten(x.clip(min=0.0).clip(max=1.0), index)
