# 导读 AlloX 对照策略。根据吞吐与工作量构造分配问题，返回作业在各 worker 类型上的分配。
# 导读 来源 policies/allox.py。以下带「导读」的行是阅读注释。
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import copy
import numpy as np
import random
from scipy.optimize import linear_sum_assignment

from policy import Policy, PolicyWithPacking

class AlloXPolicy(Policy):
    # 导读 AlloXPolicy.__init__（原源码第 12 行）
    # 导读 构造 AlloXPolicy 时执行。为实例建立 self._name、self._alpha、self._prev_allocation 等状态。
    # 导读 其余显式参数为 alpha，默认值见函数签名。
    # 导读 函数体直接赋值的实例字段包括 self._name、self._alpha、self._prev_allocation。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, alpha=1.0):
        self._name = 'AlloX_Perf'
        self._alpha = alpha
        self._prev_allocation = {}

    # 导读 AlloXPolicy.get_allocation（原源码第 17 行）
    # 导读 本策略的分配入口。根据输入吞吐、资源与策略约束计算分配结果。
    # 导读 参数 unflattened_throughputs 表示求解前以作业或组合、worker 类型为键组织的吞吐字典，scale_factors 表示按作业索引的资源规模需求，times_since_start 表示按作业记录的已过去时间。
    # 导读 参数 num_steps_remaining 表示每个作业剩余的训练迭代数，cluster_spec 表示每类 worker 的可用数量。
    # 导读 函数体直接赋值的实例字段包括 self._prev_allocation。
    # 导读 从返回表达式可以继续追踪 allocation。
    def get_allocation(self, unflattened_throughputs,
                       scale_factors, times_since_start,
                       num_steps_remaining,
                       cluster_spec):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (m, n) = throughputs.shape
        (job_ids, worker_types) = index

        # Make sure all scale factors are 1, since AlloX only supports jobs
        # with scale factors of 1.
        for job_id in scale_factors:
            assert(scale_factors[job_id] == 1)

        # m is the number of jobs, n is the total number of available workers
        # (not the total number of worker types).
        unallocated_job_ids = []
        already_allocated_job_ids = []
        for job_id in unflattened_throughputs:
            if job_id not in self._prev_allocation:
                unallocated_job_ids.append(job_id)
            else:
                total_allocation = 0.0
                for worker_type in worker_types:
                    total_allocation += self._prev_allocation[job_id][worker_type]
                if total_allocation == 1.0:
                    already_allocated_job_ids.append(job_id)
                else:
                    unallocated_job_ids.append(job_id)

        m = len(unallocated_job_ids)
        n = 0
        worker_id_to_worker_type_mapping = {}
        for worker_type in worker_types:
            # 导读 num_workers 是当前统计或建模所用的 worker 数量。
            num_workers = cluster_spec[worker_type]
            for already_allocated_job_id in already_allocated_job_ids:
                if self._prev_allocation[already_allocated_job_id][worker_type] == 1.0:
                    num_workers -= 1
            for worker_id in range(n, n+num_workers):
                worker_id_to_worker_type_mapping[worker_id] = worker_type
                n += 1

        # Sort job IDs according to arrival time.
        unallocated_job_ids.sort(key=lambda x: -times_since_start[x])
        unallocated_job_ids = unallocated_job_ids[:max(int(self._alpha * m), n)]
        m = len(unallocated_job_ids)

        # Construct matrix of processing times for each job on each worker,
        # taking into account the type of each worker.
        q_base = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                worker_type = worker_id_to_worker_type_mapping[j]
                throughput = unflattened_throughputs[unallocated_job_ids[i]][worker_type]
                if throughput == 0.0:
                    throughput = 1e-10
                q_base[i, j] = num_steps_remaining[unallocated_job_ids[i]] / \
                    throughput
        # q is computed as [q_base q_base*2 q_base*3 ... q_base*n].
        q = np.copy(q_base)
        for i in range(2, m+1):
            scaled_q_base = i * q_base
            q = np.concatenate((q, scaled_q_base), axis=1)

        # Construct matrix of delay times for each job on each worker.
        d_base = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                d_base[i, j] = times_since_start[unallocated_job_ids[i]]
        # d is computed as [d_base d_base d_base ... d_base].
        d = np.copy(d_base)
        for i in range(2, m+1):
            d = np.concatenate((d, d_base), axis=1)

        # Add d to q.
        q = q + d

        # Solve assignment problem using Hungarian method (implemented in scipy).
        row_indices, col_indices = linear_sum_assignment(q)

        # Extract assignment of jobs to worker types.
        per_worker_id_job_assignment = {i: [] for i in range(n)}
        for (row_index, col_index) in zip(row_indices, col_indices):
            job_id = unallocated_job_ids[row_index]
            worker_id = col_index % n
            worker_type = worker_id_to_worker_type_mapping[worker_id]
            worker_type_order = col_index // n
            per_worker_id_job_assignment[worker_id].append((job_id, worker_type_order))
        for worker_id in range(n):
            per_worker_id_job_assignment[worker_id] = [
                (x[0], len(per_worker_id_job_assignment[worker_id]) -1 - x[1])
                for x in per_worker_id_job_assignment[worker_id]]
            per_worker_id_job_assignment[worker_id].sort(key=lambda x: x[1])

        # Construct allocation. Don't remember allocations beyond the first
        # for each worker, since these can be recomputed the next time the
        # policy is run. Copy over allocations from already running jobs whose
        # allocations have already been computed.
        allocation = {}
        for job_id in job_ids:
            allocation[job_id] = \
                {worker_type: 0.0 for worker_type in cluster_spec}
        for job_id in job_ids:
            if job_id in self._prev_allocation:
                allocation[job_id] = copy.copy(self._prev_allocation[job_id])
        for worker_id in range(n):
            if len(per_worker_id_job_assignment[worker_id]) > 0:
                job_id = per_worker_id_job_assignment[worker_id][0][0]
                worker_type = worker_id_to_worker_type_mapping[worker_id]
                allocation[job_id][worker_type] = 1.0
        total_workers_allocated = 0
        for job_id in job_ids:
            for worker_type in worker_types:
                total_workers_allocated += allocation[job_id][worker_type]
        self._prev_allocation = copy.copy(allocation)

        return allocation
