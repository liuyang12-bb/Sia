# python multi_simulator.py ../workloads/workload-1.csv --policy=weighted_mip_fix --policy_p_val=0.5 --mip_lambda_a=-0.02 --mip_lambda_n=-0.5 --project_throughput --share_max_replicas --timeshare_sched_window=300 --output=../output_mip_heterogeneous/workloads_no_ncf
# 导读 固定副本数的 Sia 对照策略。比较不同 GPU 类型上的性能，并包含时间共享相关的求解分支。
# 导读 来源 sia_fix.py。以下带「导读」的行是阅读注释。
import cvxpy as cp
import numpy as np
import time as time

from speedup import *

# CONFIGS_4GPU = (np.asarray([1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
#                 np.asarray([1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]))

CONFIGS_4GPU = (np.ceil(np.divide(np.asarray(list(range(1, 65))), 4)),
                np.asarray(list(range(1, 65))))

# CONFIGS_8GPU = (np.asarray([1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8]),
#                 np.asarray([1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]))

CONFIGS_8GPU = (np.ceil(np.divide(np.asarray(list(range(1, 65))), 8)),
                np.asarray(list(range(1, 65))))


class SiaFixPolicy(object):
    # 导读 SiaFixPolicy.__init__（原源码第 22 行）
    # 导读 构造 SiaFixPolicy 时执行。为实例建立 self.p_fairness、self.prev_allocs、self.prev_cluster、self.restart_penalty、self.migrate_penalty、self.alloc_configs 等状态。
    # 导读 参数 p_fairness 表示Sia 候选效用所用的幂指数，lambda_a 表示分配变动相关的配置系数，是否进入目标取决于具体策略实现，lambda_n 表示未分配作业相关的配置系数。
    # 导读 其余显式参数为 restart_penalty、migrate_penalty、project_throughputs、share_max_replicas、timeshare_penalty_window、sched_interval，默认值见函数签名。
    # 导读 函数体直接赋值的实例字段包括 self.p_fairness、self.prev_allocs、self.prev_cluster、self.restart_penalty、self.migrate_penalty、self.alloc_configs、self.lambda_a、self.lambda_n、self.cluster_ordering。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, p_fairness=0.5, restart_penalty=30,
                 migrate_penalty=45, lambda_a=-0.02, lambda_n=-1,
                 project_throughputs=False,
                 share_max_replicas=False,
                 timeshare_penalty_window=None, sched_interval=60):

        self.p_fairness = p_fairness

        # prev-allocs (config)
        self.prev_allocs = dict()
        # prev cluster (0 - slow, 1 - fast)
        self.prev_cluster = dict()

        # penalties for restart / migration
        self.restart_penalty = restart_penalty
        self.migrate_penalty = migrate_penalty

        # possible configs to alloc from
        self.alloc_configs = None

        # optimization params
        self.lambda_a = lambda_a
        self.lambda_n = lambda_n

        # cluster ordering
        self.cluster_ordering = None

        # use max replicas across all clusters
        self.share_max_replicas = share_max_replicas

        # generalize throughputs to other clusters
        self.project_throughputs = project_throughputs
        self.gput_ratios = dict()
        if self.project_throughputs:
            print(f"Project speedups active")
            self.share_max_replicas = True
        if self.share_max_replicas:
            self.share_max_replicas = True

        # track allocations over a window
        self.apply_timeshare_penalty = timeshare_penalty_window is not None
        self.window_prev_allocs = dict()
        self.window_len = timeshare_penalty_window
        self.sched_interval = sched_interval

    # 导读 SiaFixPolicy.populate_valid_configs（原源码第 67 行）
    # 导读 根据机器类型、容量和物理节点限制建立候选索引，供本策略构造目标矩阵和容量约束。
    # 导读 参数 cluster_num_nodes 表示用于构建策略候选的各类节点数量，cluster_num_gpus 表示用于构建策略候选的各类每节点 GPU 数，cluster_max_physical_nnodes 表示配置筛选使用的物理节点数上限。
    # 导读 函数体中的相关调用包括 self.cluster_ordering.append。
    # 导读 函数体直接赋值的实例字段包括 self.configs、self.cluster_ordering、self.cluster_num_nodes、self.cluster_ngpus_per_node。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def populate_valid_configs(self, cluster_num_nodes, cluster_num_gpus, cluster_max_physical_nnodes=None):
        self.configs = dict()
        self.cluster_ordering = []
        print(f"Unique configs:")
        for cluster_name in cluster_num_gpus.keys():
            self.cluster_ordering.append(cluster_name)
            nnodes, ngpus = cluster_num_nodes[cluster_name], cluster_num_gpus[cluster_name]
            nnodes = min(nnodes, cluster_max_physical_nnodes[cluster_name])
            alloc_configs = CONFIGS_4GPU if ngpus == 4 else CONFIGS_8GPU
            valid_config_idxs = alloc_configs[0] <= nnodes
            num_valid_nodes = alloc_configs[0][valid_config_idxs]
            num_valid_gpus = alloc_configs[1][valid_config_idxs]
            alloc_configs = (num_valid_nodes, num_valid_gpus)
            self.configs[cluster_name] = alloc_configs
            print(
                f"Cluster: {cluster_name}, Configs: {self.configs[cluster_name]}")
        # store cluster config for future use
        self.cluster_num_nodes, self.cluster_ngpus_per_node = cluster_num_nodes, cluster_num_gpus

    # 导读 SiaFixPolicy.update_timeshare_penalties（原源码第 86 行）
    # 导读 根据时间共享状态更新调度惩罚。
    # 导读 其余显式参数为 new_allocs，默认值见函数签名。
    # 导读 函数体中的相关调用包括 self.window_prev_allocs.setdefault(jobname, list()).append、self.window_prev_allocs.setdefault、self.window_prev_allocs.get。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def update_timeshare_penalties(self, new_allocs):
        max_window_num_obs = (self.window_len // self.sched_interval)
        for jobname, new_alloc in new_allocs.items():
            # new alloc is no-alloc?
            no_alloc = np.sum(new_alloc) < 1
            # could be change of alloc
            if jobname in self.window_prev_allocs:
                alloc_change = np.sum(
                    np.abs(self.prev_allocs[jobname] - new_alloc)) > 0
            else:
                # no alloc -> alloc
                alloc_change = True
            # get no service
            if no_alloc:
                self.window_prev_allocs.setdefault(jobname, list()).append(0)
            # get `sched_interval` service minus restart_penalty
            elif alloc_change:
                self.window_prev_allocs.setdefault(jobname, list()).append(
                    self.sched_interval - self.restart_penalty)
            # continue same allocation, get `sched_interval` service
            else:
                self.window_prev_allocs.setdefault(
                    jobname, list()).append(self.sched_interval)

            if len(self.window_prev_allocs.get(jobname, list())) > max_window_num_obs:
                self.window_prev_allocs[jobname] = self.window_prev_allocs[jobname][-max_window_num_obs:]

    # 导读 SiaFixPolicy.get_timeshare_penalties（原源码第 113 行）
    # 导读 取得求解阶段使用的时间共享惩罚。
    # 导读 参数 jobs 表示本轮策略可见的作业信息。
    # 导读 其余显式参数为 job_ordering，默认值见函数签名。
    # 导读 函数体中的相关调用包括 self.window_prev_allocs.get。
    # 导读 从返回表达式可以继续追踪 (penalty_no_alloc, penalty_change_alloc, penalty_no_change)。
    def get_timeshare_penalties(self, jobs, job_ordering):
        max_window_num_obs = (self.window_len // self.sched_interval)
        penalty_no_alloc, penalty_change_alloc, penalty_no_change = list(), list(), list()
        for jobname in job_ordering:
            job = jobs[jobname]
            window_elapsed_time = min(
                job.age, max_window_num_obs * self.sched_interval)
            window_active_time = sum(self.window_prev_allocs.get(jobname, []))
            # no_alloc_penalty = window_active_time / (window_elapsed_time + self.sched_interval)
            job_alloced = np.asarray(
                self.window_prev_allocs.get(jobname, [])) > 0
            no_alloc_penalty = 1
            if len(job_alloced) > 0:
                if job_alloced[-1]:
                    no_alloc_penalty += (max_window_num_obs - sum(job_alloced))
                else:
                    if sum(job_alloced) == 0:
                        no_alloc_penalty += max_window_num_obs
            # print(f"{jobname} : {no_alloc_penalty}")
            alloc_change_penalty = (window_active_time + self.sched_interval - self.restart_penalty) \
                / (window_elapsed_time + self.sched_interval)
            no_change_penalty = (window_active_time + self.sched_interval) / \
                (window_elapsed_time + self.sched_interval)
            penalty_no_alloc.append(no_alloc_penalty)
            penalty_change_alloc.append(alloc_change_penalty)
            penalty_no_change.append(no_change_penalty)

        # convert to numpy arrays
        penalty_no_alloc = np.asarray(penalty_no_alloc, dtype=np.float32)
        penalty_change_alloc = np.asarray(
            penalty_change_alloc, dtype=np.float32)
        penalty_no_change = np.asarray(penalty_no_change, dtype=np.float32)
        return penalty_no_alloc, penalty_change_alloc, penalty_no_change

    # job_allocs: {jobname : (num_nodes, num_gpus)}
    # 导读 SiaFixPolicy.alloc_to_placement（原源码第 148 行）
    # 导读 把策略选定的节点数与 GPU 数转换为具体的节点编号列表。
    # 导读 参数 cluster_name 表示GPU 集群或机器类型名，job_allocs 表示按作业索引的配置或分配，node_remaining_gpus 表示以节点为索引的剩余 GPU 张数。
    # 导读 从返回表达式可以继续追踪 placements。
    def alloc_to_placement(self, cluster_name, job_allocs, node_remaining_gpus):
        max_num_nodes = len(node_remaining_gpus)
        cur_node_id = 0
        placements = {}
        # sort by num gpus needed
        job_order = sorted(list(job_allocs.items()),
                           key=lambda x: x[1][1], reverse=True)
        ngpus_per_node = np.max(node_remaining_gpus)
        # try to alloc distributed jobs on different nodes first
        for jobname, alloc in job_order:
            num_nodes, num_gpus = alloc
            node_id = cur_node_id
            num_full_nodes = num_gpus // ngpus_per_node

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
                    if node_gpus == ngpus_per_node:
                        node_remaining_gpus[node_id] -= ngpus_per_node
                        num_gpus -= ngpus_per_node
                        job_placement.extend([node_id] * ngpus_per_node)
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

    # 导读 SiaFixPolicy._compute_goodputs（原源码第 199 行）
    # 导读 用作业的性能函数或跨 GPU 类型的预测逻辑计算给定配置的 Goodput。
    # 导读 参数 job_info 表示调度器使用的作业摘要，cluster_name 表示GPU 集群或机器类型名，num_nodes 表示当前配置占用或考虑的节点数。
    # 导读 参数 num_replicas 表示配置中的训练副本数，数据并行时一般对应 GPU 数。
    # 导读 从返回表达式可以继续追踪 goodput_arr、output_arr。
    def _compute_goodputs(self, job_info, cluster_name, num_nodes, num_replicas):
        speedup_fn = job_info.speedup_fn.get(cluster_name, None)
        if speedup_fn is None and not self.project_throughputs:
            return None
        if speedup_fn is not None:
            # speedup_fn exists for job in `cluster_name` cluster
            goodput_arr = np.asarray(speedup_fn.get_goodput(num_nodes.astype(
                np.float32), num_replicas.astype(np.float32)), dtype=np.float32)
            return goodput_arr

        # self.project_throughputs and speedup_fn is None:
        # check if some speedup fn is not None
        any_speedup_fn = any(
            [v is not None for v in job_info.speedup_fn.values()])
        if not any_speedup_fn:
            return None
        # take any speedup_fn
        dest_cluster = [k for k in job_info.speedup_fn.keys(
        ) if job_info.speedup_fn[k] is not None][0]
        is_dest_cluster_4gpu = self.cluster_ngpus_per_node[dest_cluster] == 4
        is_src_cluster_4gpu = self.cluster_ngpus_per_node[cluster_name] == 4
        # both 8-GPU:
        if is_src_cluster_4gpu and is_dest_cluster_4gpu:
            translated_num_nodes, translated_num_replicas = num_nodes, num_replicas
        elif is_src_cluster_4gpu and not is_dest_cluster_4gpu:
            # 4 -> 8 GPU conversion
            translated_num_nodes = np.ceil(num_replicas / 8).astype(np.uint32)
            translated_num_replicas = num_replicas.astype(np.uint32)
        elif not is_src_cluster_4gpu and is_dest_cluster_4gpu:
            # 4 -> 8 GPU conversion
            translated_num_nodes = np.ceil(num_replicas / 4).astype(np.uint32)
            translated_num_replicas = num_replicas.astype(np.uint32)
        elif not is_src_cluster_4gpu and not is_dest_cluster_4gpu:
            translated_num_nodes, translated_num_replicas = num_nodes, num_replicas
        # remove configs that exceed cluster size
        max_dest_cluster_size = self.cluster_ngpus_per_node[dest_cluster] * \
            self.cluster_num_nodes[dest_cluster]
        valid_dest_configs_idxs = (translated_num_replicas < max_dest_cluster_size) & (
            translated_num_nodes < self.cluster_num_nodes[dest_cluster])
        # multiplier for throughput projection
        multiplier = job_info.cluster_throughput_ratios[cluster_name][dest_cluster]

        # actual speedup_fn being evaluated
        dest_speedup_fn = job_info.speedup_fn[dest_cluster]

        # match output shape to input
        output_arr = np.zeros_like(num_nodes)
        translated_num_nodes = translated_num_nodes[valid_dest_configs_idxs]
        translated_num_replicas = translated_num_replicas[valid_dest_configs_idxs]

        goodput_arr = np.asarray(dest_speedup_fn.get_goodput(translated_num_nodes.astype(
            np.float32), translated_num_replicas.astype(np.float32)), dtype=np.float32)
        output_arr[valid_dest_configs_idxs] = goodput_arr * multiplier
        print(
            f"Imputated goodput for {cluster_name} from {dest_cluster}: {num_nodes}, {num_replicas} = {goodput_arr} -> {output_arr}")
        return output_arr

    # 导读 SiaFixPolicy.save_current_gput_ratios（原源码第 256 行）
    # 导读 保存从当前 Goodput 矩阵得到的跨机器类型比率，供后续查询。
    # 导读 其余显式参数为 cluster_matrices、clusters，默认值见函数签名。
    # 导读 函数体直接赋值的实例字段包括 self.gput_ratios。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def save_current_gput_ratios(self, cluster_matrices, clusters):
        gput_ratios = dict()
        for i, dst_cluster in enumerate(clusters):
            for j, src_cluster in enumerate(clusters):
                if dst_cluster == src_cluster:
                    continue
                else:
                    src_val = np.mean(cluster_matrices[src_cluster][:, 0])
                    dst_val = np.mean(cluster_matrices[dst_cluster][:, 0])
                    ratio = src_val / dst_val
                    gput_ratios.setdefault(dst_cluster, dict())[
                        src_cluster] = ratio
        self.gput_ratios = gput_ratios

    # 导读 SiaFixPolicy.get_current_gput_ratios（原源码第 270 行）
    # 导读 返回已经保存的跨机器类型 Goodput 比率。
    # 导读 从返回表达式可以继续追踪 self.gput_ratios if self.gput_ratios else dict()。
    def get_current_gput_ratios(self):
        return self.gput_ratios if self.gput_ratios else dict()

    # 导读 SiaFixPolicy.optimize_mip（原源码第 273 行）
    # 导读 完成一轮配置选择和节点放置，读取作业、节点和旧分配，返回以作业名索引的分配。
    # 导读 参数 jobs 表示本轮策略可见的作业信息，nodes 表示节点信息及资源容量，prev_allocations 表示上一轮已经生效的作业分配。
    # 导读 函数体中的相关调用包括 self.configs.items、self.get_timeshare_penalties、self._compute_goodputs、self.save_current_gput_ratios、self.__solve_mip_timeshare、self.__solve_mip_rescaled。
    # 导读 从返回表达式可以继续追踪 job_placements。
    def optimize_mip(self, jobs, nodes, prev_allocations):
        print("### input jobs:")
        print(list(jobs.keys()))
        np.set_printoptions(suppress=True)
        joblist, jobnames = [], []
        for i, (jobname, job) in enumerate(jobs.items()):
            joblist.append(job)
            jobnames.append(jobname)

        # filter jobs to only contain active jobs first
        num_jobs = len(jobs)

        num_gpus = {}
        for k, k_nodes in nodes.items():
            num_gpus[k] = 0
            for node_idx, node in k_nodes.items():
                num_gpus[k] += node.resources["nvidia.com/gpu"]
        num_configs = {k: len(v[1]) for k, v in self.configs.items()}
        total_num_configs = sum(num_configs.values())

        # single-cluster speedup-matrix
        # 导读 cluster_goodput_matrices 是不同机器类型的候选评分表，按类型分别建立。
        cluster_goodput_matrices = {k: np.zeros((num_jobs, num_configs[k]), dtype=np.float32)
                                    for k in num_configs.keys()
                                    }
        realloc_factors, migrate_factors = [], []

        # job weights
        # 导读 job_weights 是与作业顺序对应的目标权重。
        job_weights = np.ones((1, num_jobs), dtype=np.float32)

        if self.apply_timeshare_penalty:
            timeshare_penalties = self.get_timeshare_penalties(jobs, jobnames)
        else:
            timeshare_penalties = None

        # compute raw speedup matrix (irrespective of slow/fast cluster)
        for i, job in enumerate(joblist):
            # compute _fair_ goodput
            for cluster in self.cluster_ordering:
                speedup_fn = job.speedup_fn[cluster]
                if self.share_max_replicas:
                    max_replicas = max(job.max_replicas.values())
                    min_replicas = min(job.min_replicas.values())
                else:
                    max_replicas = job.max_replicas[cluster]
                    min_replicas = job.min_replicas[cluster]
                assert max_replicas == min_replicas

                # if min_replicas > 1:
                #     print(f"Min replicas: {min_replicas}")

                if speedup_fn is None:
                    if not self.project_throughputs:
                        continue

                    # check if any throughput model exists to extrapolate from
                    any_speedup_fn = any(
                        [fn is not None for fn in job.speedup_fn.values()])
                    if not any_speedup_fn:
                        continue

                # cluster-specific configs
                alloc_num_nodes, alloc_num_replicas = self.configs[cluster]
                # 导读 valid_configs 是适合当前作业的合法配置索引或掩码。
                valid_configs = (alloc_num_replicas <= max_replicas) & (
                    alloc_num_replicas >= min_replicas)

                valid_nnodes, valid_ngpus = alloc_num_nodes[valid_configs], alloc_num_replicas[valid_configs]

                goodput_matrix = cluster_goodput_matrices[cluster]
                # 导读 valid_configs_goodput 是与 valid_configs 顺序对齐的 Goodput 值。
                valid_configs_goodput = self._compute_goodputs(
                    job, cluster, valid_nnodes, valid_ngpus)
                # print(f"{jobnames[i]}, {cluster}: {valid_configs_goodput}")
                goodput_matrix[i, valid_configs] = valid_configs_goodput
                # print(jobnames[i], goodput_matrix[i])

            # fill in (1, 1) config for each cluster
            min_goodputs = []
            for cluster in self.cluster_ordering:
                if np.sum(cluster_goodput_matrices[cluster][i, :]) == 0:
                    min_goodputs.append(0)
                else:
                    min_goodputs.append(np.min(cluster_goodput_matrices[cluster][i, :][np.nonzero(
                        cluster_goodput_matrices[cluster][i, :])]))
            min_goodputs = np.asarray(min_goodputs)
            if max(min_goodputs) == 0:
                min_goodput = 1
            else:
                # take lowest non-zero goodput value
                min_goodput = np.min(min_goodputs[np.nonzero(min_goodputs)])
            for cluster in self.cluster_ordering:
                # if goodput vector is empty, set first config to be [1]
                cluster_goodput_matrices[cluster][i, :] /= min_goodput
                if max(cluster_goodput_matrices[cluster][i, :]) == 0:
                    cluster_goodput_matrices[cluster][i,
                                                      max_replicas-1] = max_replicas
                # assert max(cluster_goodput_matrices[cluster][i, :]) < 500, "bad speedup values"

            # re-alloc/migrate factor
            job_lost_gpu_seconds = (
                job.num_restarts * self.restart_penalty) + (job.num_migrations * self.migrate_penalty)
            # 导读 realloc_factor 是用于给迁移或重配置成本折算效用的系数。
            realloc_factor = max(
                (job.age - job_lost_gpu_seconds), 0) / (job.age + self.restart_penalty)
            migrate_factor = max(
                (job.age - job_lost_gpu_seconds), 0) / (job.age + self.migrate_penalty)
            realloc_factors.append(realloc_factor)
            migrate_factors.append(migrate_factor)

        self.save_current_gput_ratios(
            cluster_goodput_matrices, self.cluster_ordering)

        # for cluster in self.cluster_ordering:
        #     print(f"\t{cluster}")
        #     print(f"\t{cluster_goodput_matrices[cluster]}")

        # append slow and fast speedup matrices
        final_speedup_matrix = np.hstack(
            tuple([cluster_goodput_matrices[cluster] for cluster in self.cluster_ordering]))

        # print("### speedup matrix")
        # for i, job in enumerate(joblist):
        #     print(f"{jobnames[i]}")
        #     for cluster in self.cluster_ordering:
        #         print(f"\t{cluster} {cluster_goodput_matrices[cluster][i]}")

        # print(f"speedup matrix: {final_speedup_matrix}")
        optim_st_time = time.time()
        if self.apply_timeshare_penalty:
            job_allocs, cluster_allocs = self.__solve_mip_timeshare(final_speedup_matrix, num_gpus, job_weights,
                                                                    jobnames, realloc_factors, migrate_factors,
                                                                    timeshare_penalties)
        else:
            job_allocs, cluster_allocs = self.__solve_mip_rescaled(final_speedup_matrix, num_gpus, job_weights,
                                                                   jobnames, realloc_factors, migrate_factors)
        optim_ed_time = time.time()
        #  TIME: {(optim_ed_time - optim_st_time) * 1000}ms")

        # cluster-specific job placements
        cluster_job_placements = dict()
        for cluster in self.cluster_ordering:
            node_remaining_gpus = np.asarray(
                [node.resources["nvidia.com/gpu"] for idx, node in nodes[cluster].items()], dtype=np.uint32)
            if cluster in cluster_allocs:
                cluster_job_placements[cluster] = self.alloc_to_placement(
                    cluster, cluster_allocs[cluster], node_remaining_gpus)
            else:
                cluster_job_placements[cluster] = dict()

        # merge allocs
        job_placements = {}
        for k, v in job_allocs.items():
            if v is None:
                job_placements[k] = (None, ())
            else:
                cluster_name, alloc = v
                job_placements[k] = (
                    cluster_name, cluster_job_placements[cluster_name][k])

        return job_placements

    # alternate formulation with speedups scaled for reallocation
    # cluster_num_gpus = (num_slow_gpus, num_fast_gpus)
    # 导读 SiaFixPolicy.__solve_mip_rescaled（原源码第 433 行）
    # 导读 把候选效用、容量、锁定与选择约束写成整数优化模型，求解后解码为作业配置。
    # 导读 参数 speedup_matrix 表示行对应作业、列对应候选配置或类型的性能评分矩阵，num_gpus 表示GPU 张数，job_weights 表示按作业顺序排列的目标权重。
    # 导读 参数 jobnames 表示候选矩阵或计算结果采用的作业名顺序，realloc_factors 表示各作业改变配置时采用的性能分数折损系数。
    # 导读 其余显式参数为 migrate_factors、timeshare_penalties，默认值见函数签名。
    # 导读 函数体中的相关调用包括 self.prev_allocs.get、self.prev_cluster.get、cp.Maximize、cp.Problem、problem.solve。
    # 导读 从返回表达式可以继续追踪 (job_allocs, cluster_allocs)。
    def __solve_mip_rescaled(self, speedup_matrix, num_gpus, job_weights=None, jobnames=None,
                             realloc_factors=None, migrate_factors=None, timeshare_penalties=None):
        # ones vec
        num_jobs, num_configs = speedup_matrix.shape
        cluster_config_offset, cluster_num_configs = {}, {}
        idx = 0
        for cluster in self.cluster_ordering:
            cluster_config_offset[cluster] = idx
            cluster_num_configs[cluster] = len(self.configs[cluster][0])
            idx += cluster_num_configs[cluster]
        ones_jobvec = np.ones((1, num_jobs), dtype=np.float32)
        ones_configvec = np.ones((num_configs, 1), dtype=np.float32)

        # slow/fast config selectors (one-hot vectors)
        # cluster selectors
        cluster_selectors = dict()
        cluster_selectors["none"] = np.ones((num_configs), dtype=np.bool8)
        for cluster in self.cluster_ordering:
            cluster_config_selector = np.zeros((num_configs), dtype=np.bool8)
            low, high = cluster_config_offset[cluster], cluster_config_offset[cluster] + \
                cluster_num_configs[cluster]
            cluster_config_selector[low: high] = True
            cluster_selectors[cluster] = cluster_config_selector

        # job weights
        if job_weights is None:
            # 导读 job_weights 是与作业顺序对应的目标权重。
            job_weights = ones_jobvec

        # rescale speedups for differing allocations
        for i, jobname in enumerate(jobnames):
            prev_alloc = self.prev_allocs.get(jobname, None)
            prev_cluster = self.prev_cluster.get(jobname, None)
            same_cluster_selector = cluster_selectors[prev_cluster] if prev_cluster else cluster_selectors["none"]
            migrate_cluster_selector = ~same_cluster_selector

            # rescale speedups projecting reallocations / migrations
            if prev_cluster is not None:
                same_cluster_realloc = same_cluster_selector & ~prev_alloc
                migrate_cluster_realloc = migrate_cluster_selector & ~prev_alloc

                # no penalty for keeping same alloc; penalty for realloc within same cluster
                speedup_matrix[i, :] = np.where(
                    same_cluster_realloc, realloc_factors[i] * speedup_matrix[i, :], speedup_matrix[i, :])

                # rescale speedups projecting migrations
                speedup_matrix[i, :] = np.where(
                    migrate_cluster_realloc, migrate_factors[i] * speedup_matrix[i, :], speedup_matrix[i, :])

        # power-up speedup matrix
        A = np.power(speedup_matrix, self.p_fairness)
        A = np.round(A, decimals=2)
        # replcate 0 with -10 (<no_alloc_penalty) to prevent allocate a job with less than its specified num_gpus due to the no_alloc_penalty
        # TODO: change this p < 0
        A[A==0] = -10 if (self.lambda_n if self.lambda_n else -1) < 0 else 10
        # print(f"OPTIM: Jobnames: {jobnames}")
        print(f"OPTIM: Input matrix: \n{A}")

        # regularization parameter
        opt_lambda_alloc_change = self.lambda_a if self.lambda_a else -0.02
        opt_lambda_no_alloc = self.lambda_n if self.lambda_n else -1
        constraints = []

        # construct variable to optimize over
        # 导读 CVXPY 在这里创建求解变量。其形状和整数或布尔类型由调用参数给出，取值约束在后面构造。
        x = cp.Variable(shape=A.shape, integer=True)

        # construct objective : weighted sum of mul(x, A) with weights
        obj_expr = cp.sum(job_weights @ cp.multiply(x, A))
        if jobnames is not None:
            for i, jobname in enumerate(jobnames):
                # penalty for no-alloc in both sub-clusters
                obj_expr += opt_lambda_no_alloc * (1 - cp.sum(x[i, :]))

                # no previous allocation
                if jobname not in self.prev_allocs:
                    continue
                else:
                    prev_alloc = self.prev_allocs[jobname]

                # penalty for change of allocation
                # 导读 CVXPY 在这里创建求解变量。其形状和整数或布尔类型由调用参数给出，取值约束在后面构造。
                t_job = cp.Variable(shape=A[i, :].shape)
                obj_expr += opt_lambda_alloc_change * cp.sum(t_job)
                constraints.append(t_job >= (prev_alloc - x[i, :]))
                constraints.append(t_job >= (x[i, :] - prev_alloc))
                constraints.append(t_job <= 1)

        obj = cp.Maximize(obj_expr)

        # add constraints
        # constrain range of x
        constraints.append(x >= 0)
        constraints.append(x <= 1)

        # constrain max-number of gpus alloc'ed per sub-cluster
        # slow sub-cluster
        for cluster in self.cluster_ordering:
            start_offset, end_offset = cluster_config_offset[
                cluster], cluster_config_offset[cluster] + cluster_num_configs[cluster]
            config_nnodes, config_ngpus = self.configs[cluster]
            constraints.append(
                cp.sum(x[:, start_offset: end_offset] @ config_ngpus) <= num_gpus[cluster])

        # constraint only one config per job
        constraints.append((x @ ones_configvec) <= ones_jobvec)

        problem = cp.Problem(obj, constraints=constraints)
        st_time = time.time()
        problem.solve(solver=cp.GLPK_MI, glpk={
                      'msg_lev': 'GLP_MSG_OFF'}, verbose=False)
        ed_time = time.time()
        # print(f"OPTIM: Problem: {problem}")

        if problem.status != 'optimal':
            print(f"Solver time: {ed_time - st_time}s")
            print("Status: ", problem.status)
            print(f"Problem: {problem}")
            print("The optimal value is", problem.value)
            print("A solution x is")
            print(np.round(x.value))

        # record new allocs as prev-alloc for next iter
        output_soln = np.round(x.value, decimals=0).astype(np.uint32)

        # convert binary solution to allocations
        job_allocs, cluster_allocs = dict(), dict()
        cluster_config_offset_list = np.asarray(
            [cluster_config_offset[cluster] for cluster in self.cluster_ordering])
        for i, jobname in enumerate(jobnames):
            soln = output_soln[i, :]
            if np.sum(soln) == 0:
                job_allocs[jobname] = None
                cluster_name = None
            else:
                # map solution to allocation
                alloc_config_idx = np.nonzero(soln)[0][0]
                cluster_id = np.nonzero(
                    alloc_config_idx >= cluster_config_offset_list)[0][-1]
                cluster_name = self.cluster_ordering[cluster_id]
                config_idx = alloc_config_idx - \
                    cluster_config_offset_list[cluster_id]
                nnodes, ngpus = self.configs[cluster_name][0][config_idx], self.configs[cluster_name][1][config_idx]
                job_allocs[jobname] = (cluster_name, (nnodes, ngpus))
                if cluster_name not in cluster_allocs:
                    cluster_allocs[cluster_name] = dict()
                cluster_allocs[cluster_name][jobname] = (nnodes, ngpus)

            self.prev_allocs[jobnames[i]] = soln
            self.prev_cluster[jobnames[i]] = cluster_name
        return job_allocs, cluster_allocs

    # improved version that attempts to rotate resources between jobs when num jobs > num resources
    # 导读 SiaFixPolicy.__solve_mip_timeshare（原源码第 583 行）
    # 导读 为时间共享模式构造并求解分配问题。
    # 导读 参数 speedup_matrix 表示行对应作业、列对应候选配置或类型的性能评分矩阵，num_gpus 表示GPU 张数，job_weights 表示按作业顺序排列的目标权重。
    # 导读 参数 jobnames 表示候选矩阵或计算结果采用的作业名顺序，realloc_factors 表示各作业改变配置时采用的性能分数折损系数。
    # 导读 其余显式参数为 migrate_factors、timeshare_penalties，默认值见函数签名。
    # 导读 函数体中的相关调用包括 self.prev_allocs.get、self.prev_cluster.get、cp.Maximize、cp.Problem、problem.solve、self.update_timeshare_penalties。
    # 导读 从返回表达式可以继续追踪 (job_allocs, cluster_allocs)。
    def __solve_mip_timeshare(self, speedup_matrix, num_gpus, job_weights=None, jobnames=None,
                              realloc_factors=None, migrate_factors=None, timeshare_penalties=None):
        # ones vec
        num_jobs, num_configs = speedup_matrix.shape
        cluster_config_offset, cluster_num_configs = {}, {}
        idx = 0
        for cluster in self.cluster_ordering:
            cluster_config_offset[cluster] = idx
            cluster_num_configs[cluster] = len(self.configs[cluster][0])
            idx += cluster_num_configs[cluster]
        ones_jobvec = np.ones((1, num_jobs), dtype=np.float32)
        ones_configvec = np.ones((num_configs, 1), dtype=np.float32)

        # slow/fast config selectors (one-hot vectors)
        # cluster selectors
        cluster_selectors = dict()
        cluster_selectors["none"] = np.ones((num_configs), dtype=np.bool8)
        for cluster in self.cluster_ordering:
            cluster_config_selector = np.zeros((num_configs), dtype=np.bool8)
            low, high = cluster_config_offset[cluster], cluster_config_offset[cluster] + \
                cluster_num_configs[cluster]
            cluster_config_selector[low: high] = True
            cluster_selectors[cluster] = cluster_config_selector

        # job weights
        if job_weights is None:
            # 导读 job_weights 是与作业顺序对应的目标权重。
            job_weights = ones_jobvec

        # rescale speedups for differing allocations
        for i, jobname in enumerate(jobnames):
            prev_alloc = self.prev_allocs.get(jobname, None)
            prev_cluster = self.prev_cluster.get(jobname, None)
            same_cluster_selector = cluster_selectors[prev_cluster] if prev_cluster else cluster_selectors["none"]
            migrate_cluster_selector = ~same_cluster_selector

            # rescale speedups projecting reallocations / migrations
            if prev_cluster is not None:
                same_cluster_realloc = same_cluster_selector & ~prev_alloc
                migrate_cluster_realloc = migrate_cluster_selector & ~prev_alloc

                # no penalty for keeping same alloc; penalty for realloc within same cluster
                speedup_matrix[i, :] = np.where(
                    same_cluster_realloc, realloc_factors[i] * speedup_matrix[i, :], speedup_matrix[i, :])

                # rescale speedups projecting migrations
                speedup_matrix[i, :] = np.where(
                    migrate_cluster_realloc, migrate_factors[i] * speedup_matrix[i, :], speedup_matrix[i, :])

        # power-up speedup matrix
        A = np.power(speedup_matrix, self.p_fairness)

        A = np.round(A, decimals=2)
        # print(f"OPTIM: Jobnames: {jobnames}")
        # print(f"OPTIM: Input matrix: {A}")

        # regularization parameter
        opt_lambda_alloc_change = self.lambda_a if self.lambda_a else -0.02
        opt_lambda_no_alloc = self.lambda_n if self.lambda_n else -1
        constraints = []

        # construct variable to optimize over
        # 导读 CVXPY 在这里创建求解变量。其形状和整数或布尔类型由调用参数给出，取值约束在后面构造。
        x = cp.Variable(shape=A.shape, integer=True)
        # x[A == 0] = 0
        # print(x[A==0])
        # for row in range(A.shape[0]):
        #     for col in range(A.shape[1]):
        #         if A[row, col] == 0:
        #             x[row, col] = 0
        # print(x[A==0])
        constraints.append(x[A == 0] == 0)

        if timeshare_penalties is not None:
            penalty_no_alloc, penalty_change_alloc, penalty_no_change = timeshare_penalties

        # construct objective : weighted sum of mul(x, A) with weights
        obj_expr = cp.sum(job_weights @ cp.multiply(x, A))
        if jobnames is not None:
            for i, jobname in enumerate(jobnames):
                # penalty for no-alloc in both sub-clusters
                no_alloc_gain = -1 * \
                    penalty_no_alloc[i] * (1 - cp.sum(x[i, :]))
                obj_expr += no_alloc_gain

                # no previous allocation
                if jobname not in self.prev_allocs:
                    continue
                else:
                    prev_alloc = self.prev_allocs[jobname]

                # gain of service
                # job_window_gain_vec = np.where(prev_alloc, penalty_no_change[i], penalty_change_alloc[i])
                # job_change_gain = (1 - cp.sum(cp.multiply(x[i, :], job_window_gain_vec)))
                # obj_expr += job_change_gain

        obj = cp.Maximize(obj_expr)

        # add constraints
        # constrain range of x
        constraints.append(x >= 0)
        constraints.append(x <= 1)

        # constrain max-number of gpus alloc'ed per sub-cluster
        # slow sub-cluster
        for cluster in self.cluster_ordering:
            start_offset, end_offset = cluster_config_offset[
                cluster], cluster_config_offset[cluster] + cluster_num_configs[cluster]
            config_nnodes, config_ngpus = self.configs[cluster]
            constraints.append(
                cp.sum(x[:, start_offset: end_offset] @ config_ngpus) <= num_gpus[cluster])

        # constraint only one config per job
        constraints.append((x @ ones_configvec) <= ones_jobvec)

        problem = cp.Problem(obj, constraints=constraints)
        st_time = time.time()
        problem.solve(solver=cp.GLPK_MI, verbose=False)
        ed_time = time.time()
        # print(f"OPTIM: Problem: {problem}")

        if problem.status != 'optimal':
            print(f"Solver time: {ed_time - st_time}s")
            print("Status: ", problem.status)
            print(f"Problem: {problem}")
            print("The optimal value is", problem.value)
            print("A solution x is")
            print(np.round(x.value))

        # print(problem)
        # input('\nPress Enter to continue...')

        # record new allocs as prev-alloc for next iter
        output_soln = np.round(x.value, decimals=0).astype(np.uint32)

        # update penalties for next iteration
        self.update_timeshare_penalties(
            {jobnames[i]: output_soln[i, :] for i in range(len(jobnames))})

        # convert binary solution to allocations
        job_allocs, cluster_allocs = dict(), dict()
        cluster_config_offset_list = np.asarray(
            [cluster_config_offset[cluster] for cluster in self.cluster_ordering])
        effective_gpus = dict()
        for i, jobname in enumerate(jobnames):
            soln = output_soln[i, :]
            if np.sum(soln) == 0:
                job_allocs[jobname] = None
                cluster_name = None
                effective_gpus[jobname] = 0
            else:
                # map solution to allocation
                alloc_config_idx = np.nonzero(soln)[0][0]
                cluster_id = np.nonzero(
                    alloc_config_idx >= cluster_config_offset_list)[0][-1]
                cluster_name = self.cluster_ordering[cluster_id]
                config_idx = alloc_config_idx - \
                    cluster_config_offset_list[cluster_id]
                nnodes, ngpus = self.configs[cluster_name][0][config_idx], self.configs[cluster_name][1][config_idx]
                job_allocs[jobname] = (cluster_name, (nnodes, ngpus))
                if cluster_name not in cluster_allocs:
                    cluster_allocs[cluster_name] = dict()
                cluster_allocs[cluster_name][jobname] = (nnodes, ngpus)
                effective_gpus[jobname] = speedup_matrix[i, alloc_config_idx]

            self.prev_allocs[jobnames[i]] = soln
            self.prev_cluster[jobnames[i]] = cluster_name
        # print(f"Effective GPUs: {sum(effective_gpus.values())}")
        return job_allocs, cluster_allocs

    # 导读 SiaFixPolicy.optimize（原源码第 751 行）
    # 导读 固定配置策略入口，按当前参数选择相应优化分支并返回作业分配。
    # 导读 参数 jobs 表示本轮策略可见的作业信息，nodes 表示节点信息及资源容量，base_allocations 表示本轮求解的已有分配。
    # 导读 函数体中的相关调用包括 self.optimize_mip。
    # 导读 从返回表达式可以继续追踪 self.optimize_mip(jobs, nodes, base_allocations)。
    def optimize(self, jobs, nodes, base_allocations):
        return self.optimize_mip(jobs, nodes, base_allocations)
