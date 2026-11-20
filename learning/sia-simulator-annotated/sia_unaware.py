# 导读 异构性不感知的 Sia 对照策略。在策略可见的资源视图中选择 GPU 配置与放置。
# 导读 来源 sia_unaware.py。以下带「导读」的行是阅读注释。
import copy
import logging
import re
import cvxpy as cp
import numpy as np
import pymoo.model.crossover
import pymoo.model.mutation
import pymoo.model.problem
import pymoo.model.repair
import pymoo.optimize
import time as time

from collections import OrderedDict
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.operators.crossover.util import crossover_mask
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

NUM_GPUS_PER_NODE = 4
from speedup import *

RESTART_PENALTY = 30

class SiaUnaware(object):
    # 导读 SiaUnaware.__init__（原源码第 24 行）
    # 导读 构造 SiaUnaware 时执行。为实例建立 self.lambda_n、self.lambda_a、self.num_gpus、self.alloc_configs、self.p_fairness、self.prev_alloc_binary 等状态。
    # 导读 参数 p_fairness 表示Sia 候选效用所用的幂指数，num_gpus 表示GPU 张数，lambda_n 表示未分配作业相关的配置系数。
    # 导读 参数 lambda_a 表示分配变动相关的配置系数，是否进入目标取决于具体策略实现。
    # 导读 函数体直接赋值的实例字段包括 self.lambda_n、self.lambda_a、self.num_gpus、self.alloc_configs、self.p_fairness、self.prev_alloc_binary、self.prev_allocs、self.prev_placements。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, p_fairness = 0.5, num_gpus=4, lambda_n=None, lambda_a=None):
        # regularization params
        self.lambda_n = lambda_n
        self.lambda_a = lambda_a

        # num gpus per node
        self.num_gpus = num_gpus

        # possible configs to alloc from
        self.alloc_configs = None

        self.p_fairness = p_fairness

        # prev-allocs
        self.prev_alloc_binary = dict()
        self.prev_allocs = dict()
        self.prev_placements = dict()

    # 导读 SiaUnaware.populate_valid_configs（原源码第 42 行）
    # 导读 根据机器类型、容量和物理节点限制建立候选索引，供本策略构造目标矩阵和容量约束。
    # 导读 参数 max_num_nodes 表示画像筛选或资源配置允许的最大节点数。
    # 导读 函数体直接赋值的实例字段包括 self.alloc_configs。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def populate_valid_configs(self, max_num_nodes):
        num_nodes = np.asarray([1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 24, 28, 32], dtype=np.uint32)
        num_replicas = np.asarray([1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 80, 96, 112, 128], dtype=np.uint32)
        # 导读 valid_configs 是适合当前作业的合法配置索引或掩码。
        valid_configs = num_nodes < max_num_nodes
        self.alloc_configs = (num_nodes[valid_configs], num_replicas[valid_configs])

    # boundary-aware placement
    # 导读 SiaUnaware.alloc_to_placement（原源码第 49 行）
    # 导读 把策略选定的节点数与 GPU 数转换为具体的节点编号列表。
    # 导读 参数 job_allocs 表示按作业索引的配置或分配，node_remaining_gpus 表示以节点为索引的剩余 GPU 张数。
    # 导读 从返回表达式可以继续追踪 placements。
    def alloc_to_placement(self, job_allocs, node_remaining_gpus):
        max_num_nodes = len(node_remaining_gpus)
        cur_node_id = 0
        placements = {}
        # sort by num gpus needed
        job_order = sorted(job_allocs.keys(), key=lambda x : job_allocs[x][1], reverse=True)
        # try to alloc distributed jobs on different nodes first
        for jobname in job_order:
            num_nodes, num_gpus = job_allocs[jobname]
            node_id = cur_node_id
            num_full_nodes = num_gpus // self.num_gpus

            # corner case
            job_placement = []
            if num_gpus == 0:
                placements[jobname] = job_placement
                continue

            # check if num_full_nodes number of nodes are available
            if num_full_nodes > 0:
                num_checked = 0
                while num_checked < max_num_nodes and num_full_nodes > 0:
                    node_gpus = node_remaining_gpus[node_id]
                    # can take full node
                    if node_gpus == self.num_gpus:
                        node_remaining_gpus[node_id] -= self.num_gpus
                        num_gpus -= self.num_gpus
                        job_placement.extend([node_id] * self.num_gpus)
                        num_full_nodes -= 1
                    node_id = (node_id + 1) % max_num_nodes
                    num_checked += 1

            # alloc any needed gpus anywhere
            while num_gpus > 0:
                node_gpus = node_remaining_gpus[node_id]
                if node_gpus != 0:
                    can_take_gpus = min(num_gpus, node_gpus)
                    num_gpus -= can_take_gpus
                    node_remaining_gpus[node_id] -= can_take_gpus
                    job_placement.extend([node_id] * can_take_gpus)
                # advance node pointer
                node_id = (node_id + 1) % max_num_nodes
            # record placement
            placements[jobname] = job_placement
            # advance cur_node_id ptr
            cur_node_id = node_id
                
        return placements

    # 导读 SiaUnaware.optimize_mip（原源码第 98 行）
    # 导读 完成一轮配置选择和节点放置，读取作业、节点和旧分配，返回以作业名索引的分配。
    # 导读 参数 jobs 表示本轮策略可见的作业信息，nodes 表示节点信息及资源容量，base_allocations 表示本轮求解的已有分配。
    # 导读 其余显式参数为 is_baseline_fair，默认值见函数签名。
    # 导读 函数体中的相关调用包括 self.__solve_mip_rescaled、self.alloc_to_placement。
    # 导读 函数体直接赋值的实例字段包括 self.prev_placements。
    # 导读 从返回表达式可以继续追踪 (job_placements, len(nodes))。
    def optimize_mip(self, jobs, nodes, base_allocations, is_baseline_fair=False):
        joblist, jobnames = [], []
        for i, (jobname, job) in enumerate(jobs.items()):
            joblist.append(job)
            jobnames.append(jobname)

        # filter jobs to only contain active jobs first
        num_jobs = len(jobs)
        num_gpus = len(nodes) * 4 # total_num_gpus
        if is_baseline_fair:
            fair_num_gpus = num_gpus // num_jobs
            fair_num_gpus = fair_num_gpus if fair_num_gpus > 0 else 1
            fair_num_nodes = (fair_num_gpus // NUM_GPUS_PER_NODE)
            if fair_num_gpus % NUM_GPUS_PER_NODE != 0:
                fair_num_nodes += 1
        else:
            fair_num_gpus = 1
            fair_num_nodes = 1

        # get all configs
        alloc_configs = self.alloc_configs
        alloc_num_nodes, alloc_num_replicas = alloc_configs
        num_configs = len(alloc_num_nodes)

        speedup_matrix = np.zeros((num_jobs, num_configs), dtype=np.float32)
        realloc_factors = []
        for i, job in enumerate(joblist):
            # compute _fair_ goodput
            speedup_fn = job.speedup_fn
            max_replicas = job.max_replicas
            min_replicas = job.min_replicas
            if min_replicas > 1:
                print(f"Min replicas: {min_replicas}")
            valid_config_idxs = (alloc_num_replicas <= max_replicas) & (alloc_num_replicas >= min_replicas)
            assert np.any(valid_config_idxs), "no valid configs found for min_replicas: {min_replicas}, max_replicas: {max_replicas}, alloc_replicas: {alloc_num_replicas}"

            # change notion of fairness to include max_replicas
            job_fair_ngpus = max(min_replicas, min(fair_num_gpus, max_replicas))
            job_fair_nnodes = job_fair_ngpus // 4
            if job_fair_ngpus % 4 > 0:
                job_fair_nnodes += 1

            job_fair_ngpus, job_fair_nnodes = 1, 1

            if isinstance(speedup_fn, SpeedupFunction) or isinstance(speedup_fn, UncachedSpeedupFunction):
                # speedup_fn.set_base_goodput(fair_num_nodes, fair_num_gpus)
                speedup_list = np.asarray(speedup_fn(alloc_num_nodes[valid_config_idxs], alloc_num_replicas[valid_config_idxs]), dtype=np.float32)
                fair_speedup = speedup_fn(job_fair_nnodes, job_fair_ngpus)
                if job_fair_ngpus > 1:
                    fair_speedup /= job_fair_ngpus
                single_speedup = speedup_fn(np.asarray([1]), np.asarray([1]))
                # print(f"Job: {jobnames[i]}, \nvalid idxs: {alloc_num_replicas[valid_config_idxs]}")
                # print(f"speedups: {speedup_list}")
                # enforce max-replicas
                speedup_matrix[i, valid_config_idxs] = speedup_list / fair_speedup
            else:
                print(f"JOb: {jobnames[i]}, missing speedup")
                speedup_list = alloc_num_replicas[valid_config_idxs].astype(np.float32)
                fair_speedup = float(job_fair_ngpus)
            speedup_matrix[i, 0] = 1
            # 导读 SiaUnaware.optimize_mip.check_speedups（原源码第 158 行）
            # 导读 检查候选配置的性能分数或有效性。
            # 导读 其余显式参数为 speedup_list，默认值见函数签名。
            # 导读 从返回表达式可以继续追踪 all_valid_speedups。
            def check_speedups(speedup_list):
                all_valid_speedups = np.all(speedup_list <= num_gpus)
                at_least_one_exists = np.any(speedup_list > 0)
                if not (all_valid_speedups and at_least_one_exists):
                    print(f"fair replicas: {fair_num_gpus}")
                    print(f"input alloc list: {alloc_configs[valid_config_idxs]}")
                    print(f"bad speedup list: {speedup_list}")
                    print(f"fair-speedup: {fair_speedup}, fair_nnodes={fair_num_nodes}, fair_ngpus={fair_num_gpus}")
                return all_valid_speedups
            assert check_speedups(speedup_matrix[i, valid_config_idxs]), f"borked speedup"
            # corner case for some reason?? issue in throughput pred for (1,1) config
            speedup_matrix[i, 0] = 1.0
            # re-alloc factor
            # 导读 realloc_factor 是用于给迁移或重配置成本折算效用的系数。
            realloc_factor = max((job.age - job.num_restarts*RESTART_PENALTY), 0) / (job.age + RESTART_PENALTY)
            # realloc_factor = 1.0
            realloc_factors.append(realloc_factor)
            # speedup_matrix[i, :] *= realloc_factor

        # job weights 
        # 导读 job_weights 是与作业顺序对应的目标权重。
        job_weights = np.ones((1, num_jobs), dtype=np.float32)

        optim_st_time = time.time()
        allocs = self.__solve_mip_rescaled(speedup_matrix, alloc_configs, 
                                           num_gpus, job_weights, jobnames, realloc_factors)
        optim_ed_time = time.time()
        # print(f"OPTIM TIME: {(optim_ed_time - optim_st_time) * 1000}ms")

        # turn one-hot allocs into job-allocs, compute realized goodputs
        job_allocs = dict()
        for i, job in enumerate(joblist):
            if np.all(allocs[i, :] == 0):
                job_allocs[jobnames[i]] = (0, 0)
            else:
                alloc_config_idx = np.argwhere(allocs[i, :] > 0)[0][0]
                job_nodes, job_replicas = alloc_num_nodes[alloc_config_idx], alloc_num_replicas[alloc_config_idx]
                job_allocs[jobnames[i]] = (job_nodes, job_replicas)

        # convert allocs to placements
        job_placements = {}
        node_remaining_gpus = np.asarray([4] * len(nodes), dtype=np.uint32)
        job_placements = self.alloc_to_placement(job_allocs, node_remaining_gpus)

        # update prev placement to new placement
        self.prev_placements = job_placements
        return (job_placements, len(nodes))

    # speedups scaled for reallocation 
    # 导读 SiaUnaware.__solve_mip_rescaled（原源码第 205 行）
    # 导读 把候选效用、容量、锁定与选择约束写成整数优化模型，求解后解码为作业配置。
    # 导读 参数 speedup_matrix 表示行对应作业、列对应候选配置或类型的性能评分矩阵，job_weights 表示按作业顺序排列的目标权重，jobnames 表示候选矩阵或计算结果采用的作业名顺序。
    # 导读 参数 realloc_factors 表示各作业改变配置时采用的性能分数折损系数。
    # 导读 其余显式参数为 configs、max_num_gpus，默认值见函数签名。
    # 导读 函数体中的相关调用包括 self.prev_alloc_binary.get、cp.Maximize、cp.Problem、problem.solve。
    # 导读 函数体直接赋值的实例字段包括 self.prev_alloc_binary。
    # 导读 从返回表达式可以继续追踪 np.asarray(np.round(x.value), dtype=np.uint32)。
    def __solve_mip_rescaled(self, speedup_matrix, configs, max_num_gpus=64, job_weights = None, jobnames=None, realloc_factors=None):
        # print(f"realloc_factors: {realloc_factors}")

        # print(f"MIP: speedup_matrix.shape: {speedup_matrix.shape}")
        config_ngpus = configs[1]
        num_gpus_max = max_num_gpus
        
        # ones vec
        num_jobs, num_configs = speedup_matrix.shape
        ones_jobvec = np.ones((1, num_jobs), dtype=np.float32)
        ones_configvec = np.ones((num_configs, 1), dtype=np.float32)

        # job weights
        if job_weights is None:
            # 导读 job_weights 是与作业顺序对应的目标权重。
            job_weights = ones_jobvec

        # rescale speedups for differing allocations
        for i, jobname in enumerate(jobnames):
            prev_alloc = self.prev_alloc_binary.get(jobname, None)
            # rescale speedups to incorporate this slowdown
            if prev_alloc is not None and realloc_factors[i] > 0:
                speedup_matrix[i, :] = np.where(prev_alloc, speedup_matrix[i, :], realloc_factors[i] * speedup_matrix[i, :])

        # power-up speedup matrix 
        A = np.power(speedup_matrix, self.p_fairness)

        A = np.round(A, decimals=2)
        # print(f"RESTARTS: Input matrix: {A}")
        # print(f"RESTARTS: Jobnames: {jobnames}")


        # regularization parameter
        opt_lambda_a = self.lambda_a if self.lambda_a else -0.05
        opt_lambda_n = self.lambda_n if self.lambda_n else -1
        constraints = []

        # construct variable to optimize over
        # 导读 CVXPY 在这里创建求解变量。其形状和整数或布尔类型由调用参数给出，取值约束在后面构造。
        x = cp.Variable(shape=A.shape, integer=True)
        # construct objective : weighted sum of mul(x, A) with weights
        obj_expr = cp.sum(job_weights @ cp.multiply(x, A))
        if jobnames is not None:
            for i, jobname in enumerate(jobnames):
                # no previous allocation
                if jobname not in self.prev_alloc_binary:
                    continue
                # pre-alloc exists
                prev_alloc = self.prev_alloc_binary[jobname]
                # zero-alloc last interval
                if np.sum(prev_alloc) < 1:
                    job_weights[0, i] += 1
                # print(f"prev_alloc[{jobname}] = {prev_alloc}")
                # penalty for no-alloc
                obj_expr += opt_lambda_n * (1 - cp.sum(x[i, :]))
                # print(f"prev_alloc[{jobname}] = {prev_alloc}")
                # 导读 CVXPY 在这里创建求解变量。其形状和整数或布尔类型由调用参数给出，取值约束在后面构造。
                t_job = cp.Variable(shape=A[i, :].shape)
                # momentum for allocation
                obj_expr += opt_lambda_a * cp.sum(t_job)
                constraints.append(t_job >= (prev_alloc - x[i, :]))
                constraints.append(t_job >= (x[i, :] - prev_alloc))
                constraints.append(t_job <= 1)

        obj = cp.Maximize(obj_expr)
        
        # add constraints
        # constrain range of x
        constraints.append(x >= 0)
        constraints.append(x <= 1)
        # constrain max-number of gpus alloc'ed
        constraints.append(cp.sum(x @ config_ngpus) <= num_gpus_max)
        # constraint only one config per job
        constraints.append((x @ ones_configvec) <= ones_jobvec)

        problem = cp.Problem(obj, constraints=constraints)
        st_time = time.time()
        problem.solve(solver=cp.GLPK_MI)
        ed_time = time.time()
        # print(f"RESTARTS: Optim Problem: {problem}")

        if problem.status != 'optimal':
            print(f"Solver time: {ed_time - st_time}s")
            print("Status: ", problem.status)
            print(f"Problem: {problem}")
            print("The optimal value is", problem.value)
            print("A solution x is")
            print(np.round(x.value))
        
        # record new allocs as prev-alloc for next iter
        self.prev_alloc_binary = {jobnames[i] : np.round(x.value[i, :], decimals=0) for i in range(len(jobnames))}
        # print(f"RESTARTS: Optim Solution: {x.value}")
        return np.asarray(np.round(x.value), dtype=np.uint32)

    # 导读 SiaUnaware.optimize（原源码第 296 行）
    # 导读 接收策略可见的资源和作业信息，求解异构性不感知的 GPU 选择。
    # 导读 参数 jobs 表示本轮策略可见的作业信息，nodes 表示节点信息及资源容量，base_allocations 表示本轮求解的已有分配。
    # 导读 其余显式参数为 node_template，默认值见函数签名。
    # 导读 函数体中的相关调用包括 self.optimize_mip。
    # 导读 从返回表达式可以继续追踪 self.optimize_mip(jobs, nodes, base_allocations)。
    def optimize(self, jobs, nodes, base_allocations, node_template):
        return self.optimize_mip(jobs, nodes, base_allocations)