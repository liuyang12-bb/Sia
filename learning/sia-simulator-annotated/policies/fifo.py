# 导读 FIFO 对照策略及其变体。用作业顺序和每类 worker 的剩余容量组织分配。
# 导读 来源 policies/fifo.py。以下带「导读」的行是阅读注释。
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import copy
import random

import job_id_pair
from policy import Policy, PolicyWithPacking

class FIFOPolicy(Policy):
    # 导读 FIFOPolicy.__init__（原源码第 11 行）
    # 导读 构造 FIFOPolicy 时执行。为实例建立 self._name、self._mode、self._allocation、self._scale_factors、self._rng、self._packing_threshold 等状态。
    # 导读 参数 seed 表示随机数生成所用的种子。
    # 导读 其余显式参数为 mode、packing_threshold，默认值见函数签名。
    # 导读 函数体中的相关调用包括 self._rng.seed。
    # 导读 函数体直接赋值的实例字段包括 self._name、self._mode、self._allocation、self._scale_factors、self._rng、self._packing_threshold。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, mode='base', seed=None, packing_threshold=1.5):
        self._name = 'FIFO'
        self._mode = mode
        self._allocation = {}
        self._scale_factors = {}
        if mode == 'base':
            self._rng = random.Random()
            if seed is not None:
                self._rng.seed(seed)
        elif mode == 'packing':
            self._packing_threshold = packing_threshold

    # 导读 FIFOPolicy._pack（原源码第 23 行）
    # 导读 根据剩余 worker 容量为给定作业装入可用份额，供 FIFO 分配逻辑使用。
    # 导读 参数 throughputs 表示作业在各 worker 类型或组合下的吞吐数据，scale_factors 表示按作业索引的资源规模需求。
    # 导读 其余显式参数为 queue，默认值见函数签名。
    # 导读 函数体直接赋值的实例字段包括 self._output。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def _pack(self, queue, throughputs, scale_factors):
        while len(queue) > 0:
            # Only make a packing decision if combined normalized
            # throughput would provide a signficant gain.
            max_packed_throughput = self._packing_threshold
            job_id_to_pack_with = None
            job_id_to_schedule = queue.pop(0)

            # Find the already scheduled job with which the next job on
            # the queue will pack best with.
            for scheduled_job_id in self._allocation:
                assert scheduled_job_id != job_id_to_schedule
                assert scheduled_job_id in throughputs
                if scheduled_job_id.is_pair():
                    continue
                if (scale_factors[scheduled_job_id] !=\
                        scale_factors[job_id_to_schedule]):
                    continue
                worker_type = self._allocation[scheduled_job_id]
                merged_job_id = \
                        job_id_pair.JobIdPair(scheduled_job_id[0],
                                              job_id_to_schedule[0])
                packed_throughput = throughputs[merged_job_id][worker_type]
                normalized_packed_throughput = 0.0
                for i, single_job_id in enumerate(merged_job_id.singletons()):
                    if packed_throughput[i] <= 0.0:
                        continue
                    isolated_throughput = \
                            throughputs[single_job_id][worker_type]
                    normalized_packed_throughput += \
                            packed_throughput[i] / isolated_throughput
                if normalized_packed_throughput > max_packed_throughput:
                    max_packed_throughput = normalized_packed_throughput
                    job_id_to_pack_with = scheduled_job_id
            if job_id_to_pack_with is None:
                # Terminate when we cannot find a job to pack with.
                # This respects the FIFO property of no jobs being able
                # to jump ahead in the queue.
                break
            else:
                # Transfer the allocation for the single job to the
                # packed job.
                self._output = None
                merged_job_id = \
                        job_id_pair.JobIdPair(job_id_to_pack_with[0],
                                              job_id_to_schedule[0])
                worker_type = self._allocation[job_id_to_pack_with]
                del self._allocation[job_id_to_pack_with]
                self._allocation[merged_job_id] = worker_type


    # 导读 FIFOPolicy.get_allocation（原源码第 74 行）
    # 导读 保持作业排队顺序，用剩余 worker 容量给队列分配资源。节点类型选择与实际安排在本方法的队列处理逻辑中。
    # 导读 参数 throughputs 表示作业在各 worker 类型或组合下的吞吐数据，scale_factors 表示按作业索引的资源规模需求，cluster_spec 表示每类 worker 的可用数量。
    # 导读 函数体中的相关调用包括 self._allocation.keys、self._rng.randrange、self._pack、self._allocation.items。
    # 导读 函数体直接赋值的实例字段包括 self._allocation。
    # 导读 从返回表达式可以继续追踪 final_allocation。
    def get_allocation(self, throughputs, scale_factors, cluster_spec):
        available_workers = copy.deepcopy(cluster_spec)
        queue = []

        # Update the internal representation of scale_factors.
        for job_id in scale_factors:
            self._scale_factors[job_id] = scale_factors[job_id]

        # Reset the allocation when running in performance-aware mode.
        if self._mode != 'base':
            self._allocation = {}

        # Add all jobs that have not been allocated already to the queue.
        # Jobs should be added in order of arrival (i.e. according to Job ID).
        for job_id in sorted(list(throughputs.keys())):
            if job_id not in self._allocation and not job_id.is_pair():
                queue.append(job_id)

        # Find all completed jobs and schedule jobs off the queue to replace
        # them. Also determine how many workers are available.
        # NOTE: In performance-aware mode, this loop should be a no-op
        # because the allocation is reset.
        for scheduled_job_id in sorted(list(self._allocation.keys())):
            worker_type = self._allocation[scheduled_job_id]
            # Check if job has completed.
            if scheduled_job_id not in throughputs:
                # If only one job in a pair of co-located jobs completed, then
                # add the other job back to the queue.
                for single_job_id in scheduled_job_id.singletons():
                    if single_job_id in throughputs:
                        queue.append(single_job_id)
                        queue.sort()
                if len(queue) > 0:
                    job_id_to_schedule = queue[0]
                    if (scale_factors[job_id_to_schedule] <=
                            available_workers[worker_type]):
                        worker_type = self._allocation[scheduled_job_id]
                        if throughputs[job_id_to_schedule][worker_type] > 0.0:
                            queue.pop(0)
                            self._allocation[job_id_to_schedule] = worker_type
                            available_workers[worker_type] -= \
                                scale_factors[job_id_to_schedule]
                del self._allocation[scheduled_job_id]
                del self._scale_factors[scheduled_job_id]
            else:
                # Job has not completed, subtract its allocated workers
                # from available_workers.
                available_workers[worker_type] -= \
                    scale_factors[scheduled_job_id]

        # Find all available workers.
        available_worker_types = []
        for worker_type in available_workers:
            if available_workers[worker_type] > 0:
                available_worker_types.append(worker_type)
        available_worker_types.sort()

        # Allocate resources to as many jobs as possible.
        while len(queue) > 0 and len(available_worker_types) > 0:
            job_id_to_schedule = queue.pop(0)
            scale_factor = scale_factors[job_id_to_schedule]
            available_worker_types_with_scale_factor = []
            original_available_worker_types_mapping = []
            for i, worker_type in enumerate(available_worker_types):
                if available_workers[worker_type] >= scale_factor:
                    available_worker_types_with_scale_factor.append(worker_type)
                    original_available_worker_types_mapping.append(i)
            if len(available_worker_types_with_scale_factor) == 0:
                break
            if self._mode == 'base':
                worker_type_idx = self._rng.randrange(
                        len(available_worker_types_with_scale_factor))
            else:
                # Find the worker_type with best performance for this job.
                worker_type = None
                worker_type_idx = None
                max_throughput = -1
                for i, x in enumerate(available_worker_types_with_scale_factor):
                    throughput = throughputs[job_id_to_schedule][x]
                    if throughput > max_throughput:
                        max_throughput = throughput
                        worker_type = x
                        worker_type_idx = i
            if throughputs[job_id_to_schedule][worker_type] > 0.0:
                self._allocation[job_id_to_schedule] = worker_type
                available_workers[worker_type] -= scale_factors[job_id_to_schedule]
                if available_workers[worker_type] == 0:
                    worker_type_idx =\
                        original_available_worker_types_mapping[worker_type_idx]
                    available_worker_types.pop(worker_type_idx)

        if self._mode == 'packing':
            self._pack(queue, throughputs, scale_factors)

        # Construct output allocation.
        final_allocation = {}
        for job_id in throughputs:
            final_allocation[job_id] = \
                    {worker_type: 0.0 for worker_type in cluster_spec}
        for job_id, worker_type in self._allocation.items():
            final_allocation[job_id][worker_type] = 1.0

        return final_allocation

class FIFOPolicyWithPerf(Policy):
    # 导读 FIFOPolicyWithPerf.__init__（原源码第 179 行）
    # 导读 构造 FIFOPolicyWithPerf 时执行。为实例建立 self._name、self._packing、self._policy 等状态。
    # 导读 其余显式参数为 packing，默认值见函数签名。
    # 导读 函数体直接赋值的实例字段包括 self._name、self._packing、self._policy。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, packing=False):
        self._name = 'FIFO_Perf'
        self._packing = packing
        self._policy = FIFOPolicy(mode='perf')

    # 导读 FIFOPolicyWithPerf.get_allocation（原源码第 184 行）
    # 导读 本策略的分配入口。接收上层传入的状态，并使用 self._policy.get_allocation 等调用生成分配。
    # 导读 参数 throughputs 表示作业在各 worker 类型或组合下的吞吐数据，scale_factors 表示按作业索引的资源规模需求，cluster_spec 表示每类 worker 的可用数量。
    # 导读 函数体中的相关调用包括 self._policy.get_allocation。
    # 导读 从返回表达式可以继续追踪 self._policy.get_allocation(throughputs, scale_factors, cluster_spec)。
    def get_allocation(self, throughputs, scale_factors, cluster_spec):
        return self._policy.get_allocation(throughputs, scale_factors,
                                           cluster_spec)

class FIFOPolicyWithPacking(PolicyWithPacking):
    # 导读 FIFOPolicyWithPacking.__init__（原源码第 189 行）
    # 导读 构造 FIFOPolicyWithPacking 时执行。为实例建立 self._name、self._policy 等状态。
    # 导读 其余显式参数为 packing_threshold，默认值见函数签名。
    # 导读 函数体直接赋值的实例字段包括 self._name、self._policy。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, packing_threshold=1.5):
        self._name = 'FIFO_Packing'
        self._policy = FIFOPolicy(mode='packing',
                                  packing_threshold=packing_threshold)

    # 导读 FIFOPolicyWithPacking.get_allocation（原源码第 194 行）
    # 导读 本策略的分配入口。接收上层传入的状态，并使用 self._policy.get_allocation 等调用生成分配。
    # 导读 参数 throughputs 表示作业在各 worker 类型或组合下的吞吐数据，scale_factors 表示按作业索引的资源规模需求，cluster_spec 表示每类 worker 的可用数量。
    # 导读 函数体中的相关调用包括 self._policy.get_allocation。
    # 导读 从返回表达式可以继续追踪 self._policy.get_allocation(throughputs, scale_factors, cluster_spec)。
    def get_allocation(self, throughputs, scale_factors, cluster_spec):
        return self._policy.get_allocation(throughputs, scale_factors,
                                           cluster_spec)
