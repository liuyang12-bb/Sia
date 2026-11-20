# 导读 异构集群的协调层。保存时钟、Job 列表、GPU 分配和日志。每轮先执行旧分配，再调用策略生成新分配。
# 导读 来源 cluster.py。以下带「导读」的行是阅读注释。
import json
from multiprocessing import Pool
import numpy as np
import time

from applications import APPLICATIONS
from gavel import GavelPolicy
from job import Job
from sia import SiaPolicy
from sia_fix import SiaFixPolicy
from utils import JobInfo, NodeInfo
import simulator_config as sim_config

class Cluster(object):
    # num_nodes, ngpus_per_node are dicts with key=cluster_name
    # 导读 Cluster.__init__（原源码第 16 行）
    # 导读 接收作业表、策略和节点拓扑。把 CSV 每一行变成 Job，建立应用索引，初始化时钟、分配与日志。
    # 导读 参数 workload 表示作业输入表或工作负载路径，取决于调用入口，policy 表示调度策略对象或用于选择策略的名称，num_nodes 表示当前配置占用或考虑的节点数。
    # 导读 参数 ngpus_per_node 表示每节点 GPU 数，异构接口可按类型索引，max_physical_nodes 表示各类型的物理节点数量上限。
    # 导读 函数体中的相关调用包括 self.num_nodes.keys、self.jobs.append、self.compute_throughput_ratios、self.compute_throughputs。
    # 导读 函数体直接赋值的实例字段包括 self.workload、self.policy、self.num_nodes、self.ngpus_per_node、self.current_time、self.clusters、self.max_physical_nodes、self.rescale_events、self.migrate_events。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, workload, policy, num_nodes, ngpus_per_node, max_physical_nodes):
        print(f"Simulated cluster config: \nnnodes: {num_nodes}, ngpus_per_node:{ngpus_per_node}")
        # workload is a pandas dataframe
        # 导读 workload 是原始作业表，jobs 是据此创建的对象列表。policy 是入口传入的调度策略。
        self.workload = workload
        # policy is a Policy object
        self.policy = policy
        # 导读 两个字典都用机器类型作键。num_nodes 是台数，ngpus_per_node 是每台的 GPU 张数。
        self.num_nodes, self.ngpus_per_node = num_nodes, ngpus_per_node
        # 导读 全局模拟时钟从 0 开始，按调度轮推进。
        self.current_time = 0
        self.clusters = list(self.num_nodes.keys())
        self.max_physical_nodes = max_physical_nodes
        self.rescale_events, self.migrate_events = [], []  #创建用于记录资源调整和迁移事件的空列表

        # create inverse map like {app-name: {cluster_name: cluster_app}}
        # 导读 cluster_applications 按应用名再按机器类型索引。APPLICATIONS 原本按机器类型再按应用名索引。

        #原来的形式
#         APPLICATIONS = {
#     "aws": {
#         "cifar10": aws_cifar10,
#         "bert": aws_bert
#     },
#     "rtx": {
#         "cifar10": rtx_cifar10,
#         "bert": rtx_bert
#     }
# }
        #转换为如下的形式
#         {
#     "cifar10": {
#         "aws": aws_cifar10,
#         "rtx": rtx_cifar10
#     },
#     "bert": {
#         "aws": aws_bert,
#         "rtx": rtx_bert
#     }
# }
        self.cluster_applications = dict()
        for cluster_name, cluster_apps in APPLICATIONS.items():
            for app_name, app in cluster_apps.items():
                if app_name not in self.cluster_applications:
                    self.cluster_applications[app_name] = dict()
                self.cluster_applications[app_name][cluster_name] = app

        if isinstance(policy, SiaPolicy):
            # cache_speedups = isinstance(policy, PolluxPolicy)
            cache_speedups = False
            # old code
            '''
            self.jobs = [Job(row.name, self.cluster_applications[row.application],
                             row.time, cache_speedups=cache_speedups,
                             target_num_replicas=row.num_replicas,  # store in case we need it
                             target_batch_size=row.batch_size) for row in workload.itertuples()]
            '''
            # 导读 初始化所有作业对象。作业是否已到达由后面的调度筛选决定。
            self.jobs = []
            adaptive_jobnames = list()
            rigid_jobnames = list()
            scale_gpu_jobnames = list()
            for row in workload.itertuples():
                # 导读 category 列可选。rigid 固定副本和 batch，scale-gpus 保留 GPU 调整路径并关闭 batch 调整。
                category = row.category if 'category' in workload.columns else None
                # 导读 CSV 的 time 传给 Job 的 submission_time，num_replicas 和 batch_size 是作业的目标配置。
                cur_job = Job(row.name, self.cluster_applications[row.application],
                              row.time, cache_speedups=cache_speedups, target_num_replicas=row.num_replicas,
                              target_batch_size=row.batch_size, category=category)
                if category is not None:
                    if category == 'scale-gpus':
                        scale_gpu_jobnames.append(cur_job.name)
                        cur_job.disable_bsz_tuning() #表示以后更新训练配置时，不再为这个 Job 自适应搜索最优 Batch Size，而是使用固定 Batch Size 路径
                    elif category == 'adaptive':
                        adaptive_jobnames.append(cur_job.name)
                    elif category == 'rigid':
                        rigid_jobnames.append(cur_job.name)
                        cur_job.disable_bsz_tuning()
                self.jobs.append(cur_job)
            cname = next(iter(self.num_nodes.keys()))
            for job in self.jobs:
                if job.applications[cname].name == "ncf":
                    job.target_batch_size = 32768
            print(f"Initialized job state for {len(self.jobs)} jobs")
            print(f"Scaling only GPU count (scale-gpu): {scale_gpu_jobnames}")
            print(f"Scaling bsz, GPU count (adaptive): {adaptive_jobnames}")
            print(f"Fixed bsz, GPU count (rigid): {rigid_jobnames}")
            # enable bsz adaptation for Sia by default
            self.enable_bsz_tuning = True
            #即使 Cluster 的开关为 True，刚才的 Rigid 和 Scale-GPUs Job 也可以因为自身的开关为 False，而不进行 Batch Size 自适应。
        elif isinstance(policy, GavelPolicy):
            # change the target num_replicas to fit into the smallest cluster
            smallest_cluster_size = min(
                [self.num_nodes[k] * self.ngpus_per_node[k] for k in self.num_nodes.keys()])
            self.jobs = [Job(row.name, self.cluster_applications[row.application], row.time,
                             target_num_replicas=row.num_replicas, target_batch_size=row.batch_size)
                         for row in workload.itertuples()]
            # disable bsz adaptation for Gavel
            self.enable_bsz_tuning = False
            for job in self.jobs:
                job.fix_minibatch_size()
        elif isinstance(policy, SiaFixPolicy):
            smallest_cluster_size = min(
                [self.num_nodes[k] * self.ngpus_per_node[k] for k in self.num_nodes.keys()])
            self.jobs = [Job(row.name, self.cluster_applications[row.application],
                             row.time, cache_speedups=False,
                             target_num_replicas=min(
                                 row.num_replicas, smallest_cluster_size),
                             target_batch_size=row.batch_size)
                         for row in workload.itertuples()]
            cname = next(iter(self.num_nodes.keys()))
            for job in self.jobs:
                if job.applications[cname].name == "ncf":
                    job.target_batch_size = 32768
        else:
            assert False, f"unsupported policy {policy.__class__.__name__}"
        # 导读 allocations 按作业名索引。值包含机器类型和节点编号序列，例如 ('aws', [0, 0]) 表示节点 0 分到两张卡。
        self.allocations = {}
        # 导读 logs 按时间保留每轮结束的记录。
        self.logs = []
        self.utility = []
        self.compute_throughput_ratios(init=False)
        self.compute_throughputs()

        # multi-processing for job.step() in parallel
        self.process_pool = Pool(sim_config.num_parallel_jobs) if sim_config.num_parallel_jobs > 1 else None

    # disable bsz tuning for MIP
    # 导读 Cluster.disable_bsz_tuning（原源码第 111 行）
    # 导读 关闭 batch 自动调整。集群版本向每个 Job 传递这一设置，Job 版本更新本作业的开关。
    # 导读 函数体直接赋值的实例字段包括 self.enable_bsz_tuning。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def disable_bsz_tuning(self):
        self.enable_bsz_tuning = False
        for job in self.jobs:
            job.disable_bsz_tuning()

    # simulate one scheduling round
    # 导读 Cluster.step（原源码第 117 行）
    # 导读 入口每轮调用。seconds 是模拟秒。顺序为推进旧分配、计算并应用下一轮分配、记录状态。返回值为 None，结果保存在对象中。
    # 导读 参数 seconds 表示需要推进的模拟秒数。
    # 导读 函数体中的相关调用包括 self.step_jobs_parallel、self.step_jobs_sequential、self.optimize_policy、self.rescale_events.append、self.migrate_events.append、self.step_log。
    def step(self, seconds=60):
        # advance job state for `seconds` seconds
        # 导读 time.time() 用于测量 Python 执行所花的实际时间。模拟时间由 current_time 和 seconds 推进。
        start_t = time.time()
        if self.process_pool:
            self.step_jobs_parallel(seconds)
        else:
            # 导读 默认单进程从这里推进作业，作业使用轮开始时已有的放置。
            self.step_jobs_sequential(seconds)
        end_t = time.time()
        print(f"Job Step time: {(end_t - start_t)*1000}ms")

        # invoke scheduler policy to optimize allocations for next round
        # returns a list of changes to allocations as rescale and migrate events for logging
        # 导读 时钟已经来到轮结束边界。此时新到达的作业可参加下一轮选择。
        step_rescale_events, step_migrate_events = self.optimize_policy()

        # step_rescale_events：本轮发生的资源重分配事件。
        # step_migrate_events：本轮发生的 GPU 类型迁移事件。
        # log cluster state and metrics
        self.rescale_events.append(step_rescale_events)
        self.migrate_events.append(step_migrate_events)
        # 导读 日志同时记录刚推进完的进度和下一轮使用的新分配。
        self.step_log()

    # 导读 Cluster.step_jobs_sequential（原源码第 136 行）
    # 导读 依次让各 Job 推进相同的 seconds，并在所有作业推进后将集群时钟加一次 seconds。
    # 导读 参数 seconds 表示需要推进的模拟秒数。
    # 导读 函数体中的相关调用包括 self.allocations.items、self.allocations.get、job.step。
    # 导读 函数体直接赋值的实例字段包括 self.current_time。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def step_jobs_sequential(self, seconds=60):
        # interference check disabled
        # 导读 这里的干扰检查在当前版本默认关闭。
        check_interference = False
        # TODO: fix interference check code
        if check_interference:
            interfere_nodes = set(idx for idx in range(self.num_nodes)
                                  if sum(len(set(val)) > 1 and idx in val
                                         for key, val in self.allocations.items()) > 1)
        # 导读 每个 Job 都接收同一个模拟区间。例如 3 个作业各推进 60 秒，集群只过了 60 秒。
        for job in self.jobs:
            if check_interference:
                alloc_set = set(self.allocations.get(job.name, [])[1])
                interference = 0.0 if len(alloc_set) > 1 and any(
                    idx in interfere_nodes for idx in alloc_set) else 0.0
            else:
                interference = 0.0
            # 导读 Job.step 自己处理完成、未分到卡、恢复和训练等分支。
            job.step(seconds, interference=interference)

        # 导读 所有作业推进完成后，统一更新全局时钟。
        self.current_time += seconds

    # WARNING:: disables batch size & interference check (see other version above)
    # 导读 Cluster.step_jobs_parallel（原源码第 156 行）
    # 导读 把同一轮的 Job 推进分发给进程池，用返回的 Job 对象更新列表，再推进全局时钟。
    # 导读 参数 seconds 表示需要推进的模拟秒数。
    # 导读 函数体中的相关调用包括 self.process_pool.starmap。
    # 导读 函数体直接赋值的实例字段包括 self.jobs、self.current_time。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def step_jobs_parallel(self, seconds=60):
        self.jobs = self.process_pool.starmap(_helper_step_job_single, [(job, seconds) for job in self.jobs])
        self.current_time += seconds

    # 导读 Cluster.optimize_policy（原源码第 160 行）
    # 导读 收集本轮可调度作业与节点，调用策略，然后把选择结果应用到 Job，保存当前分配。
    # 导读 函数体中的相关调用包括 self.get_job_infos、self.get_node_infos、self.allocations.items、self.policy.optimize、self.allocations.get、job.migrate。
    # 导读 函数体直接赋值的实例字段包括 self.allocations。
    # 导读 从返回表达式可以继续追踪 ([], [])、(rescale_events, migrate_events)。

#     变量	类型	主要作用
# self.jobs	Job 对象列表	保存全部任务
# job_infos	Job 名称 → JobInfo 的字典	保存本轮待调度任务信息
# node_infos	GPU 类型 → 节点 → NodeInfo 的字典	保存集群 GPU 容量
# self.allocations	字典	保存上一轮 GPU 分配
# allocations	字典	保存本轮新计算出的 GPU 分配
# policy_optim_time	浮点数	记录真实调度计算耗时
    def optimize_policy(self):
        # 导读 job_infos 是已到达且未完成作业的摘要。若为空，本轮无需调度。
        job_infos = self.get_job_infos()
        policy_optim_time, alloc_event_time = 0, 0

        # policy_optim_time：调用调度策略计算分配方案花费的真实时间。
        # alloc_event_time：处理新分配、重配置和迁移花费的真实时间。

        if not job_infos:
            return [], []
        
        # Optimize allocations.
        # 导读 node_infos 按机器类型和节点 ID 提供 GPU 容量。
        node_infos = self.get_node_infos()
        self.allocations = {
            k: v for k, v in self.allocations.items() if k in job_infos}

        # run scheduling policy
        p_start_t = time.time()
        # 导读 策略接收作业、节点与旧分配，返回下一轮每个作业的节点分配。
        allocations = self.policy.optimize(
            job_infos, node_infos, self.allocations)
        policy_optim_time = time.time() - p_start_t
        # collect any change in allocations
        # rescale: change in ngpus, migrate: change in ngpus and/or cluster
        p_start_t = time.time()
        rescale_events, migrate_events = [], []
        for job in self.jobs:
            if job.submission_time <= self.current_time and job.completion_time is None:
                job.contention.append(len(job_infos))
                # print(f"job: {job.name}, contention: {job.contention}")
                if isinstance(self.policy, SiaPolicy) or isinstance(self.policy, GavelPolicy) or isinstance(self.policy, SiaFixPolicy):
                    # 导读 从新分配字典取出机器类型与每张卡所在的节点编号。
                    cluster, alloc = allocations.get(job.name, (None, ()))
                    if isinstance(self.policy, SiaFixPolicy) or isinstance(self.policy, GavelPolicy):
                        assert len(alloc) == 0 or len(
                            alloc) == job.target_num_replicas, f'{job.name} target_num_replicas: {job.target_num_replicas} alloc: {alloc}'
                    # change in resources
                    if allocations.get(job.name) != self.allocations.get(job.name):
                        # 导读 连续相同的节点编号折成卡数，例如 [0, 0, 1] 转成 [2, 1]。
                        placement = []
                        for i in range(len(alloc)):
                            if i == 0 or alloc[i] != alloc[i - 1]:
                                placement.append(1)
                            else:
                                placement[-1] += 1
                        # fix placement if bad
                        nnodes = len(placement)
                        nreplicas = sum(placement)
                        if cluster is not None and nnodes > self.max_physical_nodes[cluster]:
                            n_reps_per_node = int(np.ceil(
                                nreplicas / self.max_physical_nodes[cluster]))
                            new_placement = []
                            while nreplicas > 0:
                                n_reps = min(n_reps_per_node, nreplicas)
                                new_placement.append(n_reps)
                                nreplicas -= n_reps
                            print(f"WARNING:: FIXING PLACEMENT FOR LARGE CLUSTER: {cluster}, old placement:{placement}, new/fixed placement: {new_placement}")
                            placement = new_placement
                        # 导读 机器类型变化走 migrate，同类型放置变化走 reallocate。两者把分配写回 Job。
                        if job.current_cluster != cluster:
                            # migrated to another cluster
                            migrate_events.append(job.migrate(cluster, placement))
                        elif self.allocations.get(job.name) != alloc:
                            # reallocated within same cluster
                            rescale_events.append(job.reallocate(placement))
                else:
                    assert False, "other policies not implemented"
        # make a copy of allocations
        # 导读 保存策略返回的字典，下一轮调用会把它作为已有分配传回策略。
        self.allocations = allocations
        alloc_event_time = time.time() - p_start_t
        print(f"Policy optimization took {policy_optim_time*1000:.2f}ms, reallocating+migrating jobs took {alloc_event_time*1000:.2f}ms")
        if sim_config.log_sched_decisions:
            print(f"Rescale events: {rescale_events}")
            print(f"Migrate events: {migrate_events}")
        return rescale_events, migrate_events

    # 导读 Cluster.compute_throughput_ratios（原源码第 227 行）
    # 导读 按应用和两种机器类型生成吞吐比例，供不同类型之间的性能预测迁移使用。
    # 导读 其余显式参数为 init，默认值见函数签名。
    # 导读 函数体中的相关调用包括 self.cluster_applications.keys。
    # 导读 函数体直接赋值的实例字段包括 self.cluster_throughput_ratios。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def compute_throughput_ratios(self, init=False):
        # throughput conversion ratios
        self.cluster_throughput_ratios = dict()
        for app_name in self.cluster_applications.keys():
            if app_name == "gpt_pmp":
                continue
            # print(f"app: {app_name}")
            app_xput_ratios = dict()
            for cluster in self.clusters:
                ratios = dict()
                for dest_cluster in self.clusters:
                    # ignore any clusters that dont have speedup_fn ready
                    if dest_cluster == cluster:
                        continue
                    elif not init:
                        dest_df = self.cluster_applications[app_name][dest_cluster].placements
                        src_df = self.cluster_applications[app_name][cluster].placements
                        src_num_stages = self.cluster_applications[app_name][cluster].num_stages
                        dest_num_stages = self.cluster_applications[app_name][dest_cluster].num_stages
                        # get slice of placements with min number of stages (=1 for standard DP jobs)
                        if src_num_stages == dest_num_stages == 1:
                            src_df = src_df[src_df.num_replicas == 1]
                            dest_df = dest_df[dest_df.num_replicas == 1]
                            mean_src_atomic_xput = (
                                src_df.local_bsz.max() / src_df.step_time.max())
                            mean_dest_atomic_xput = (
                                dest_df.local_bsz.max() / dest_df.step_time.max())
                        else:
                            # PMP jobs
                            src_df = src_df[src_df.num_replicas ==
                                            src_num_stages]
                            dest_df = dest_df[dest_df.num_replicas ==
                                              dest_num_stages]
                            # mean_src_atomic_xput = xput of PMP job with `src_num_stages` GPUs in a single pipeline on GPU type `src_cluster`
                            # mean_dest_atomic_xput = xput of PMP job with `dest_num_stages` GPUs in a single pipeline on GPU type `dest_cluster`
                            mean_src_atomic_xput = (
                                src_df.local_bsz.max() / src_df.step_time.max())
                            mean_dest_atomic_xput = (
                                dest_df.local_bsz.max() / dest_df.step_time.max())
                            print(
                                f"src_cluster: {cluster}, dest_cluster: {dest_cluster}")
                            print(
                                f"mean_src_atomic_xput: {mean_src_atomic_xput}, mean_dest_atomic_xput: {mean_dest_atomic_xput}")
                        # record ratio as (mean xput in src_cluster) / (mean xput in dest_cluster)
                        # 导读 比值按源类型吞吐除以目标类型吞吐保存，使用时注意字典的源与目标顺序。
                        ratio = mean_src_atomic_xput / mean_dest_atomic_xput
                        ratios[dest_cluster] = ratio
                    else:
                        ratios[dest_cluster] = 1.0
                app_xput_ratios[cluster] = ratios
            self.cluster_throughput_ratios[app_name] = app_xput_ratios
        if sim_config.log_cluster_verbose:
            print(f"Throughput ratios: {self.cluster_throughput_ratios}")

    # 导读 Cluster.compute_throughputs（原源码第 280 行）
    # 导读 用应用画像生成各类型最小配置的参考吞吐，供冷启动或固定配置等路径使用。
    # 导读 函数体中的相关调用包括 self.cluster_applications.keys。
    # 导读 函数体直接赋值的实例字段包括 self.cluster_throughputs。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def compute_throughputs(self):
        # throughput conversion ratios
        self.cluster_throughputs = dict()
        for app_name in self.cluster_applications.keys():
            if app_name == "gpt_pmp":
                continue
            # print(f"app: {app_name}")
            app_xputs = dict()
            for cluster in self.clusters:
                cluster_profiles = self.cluster_applications[app_name][cluster].placements
                cluster_num_stages = self.cluster_applications[app_name][cluster].num_stages
                # standard DP jobs
                if cluster_num_stages == 1:
                    min_profile = cluster_profiles[cluster_profiles.num_replicas == 1]
                    mean_atomic_xput = (min_profile.local_bsz.max() / min_profile.step_time.max())
                # PMP jobs
                else:
                    min_profile = cluster_profiles[cluster_profiles.num_replicas == cluster_num_stages]
                    mean_atomic_xput = (min_profile.local_bsz.max() / min_profile.step_time.max())
                # atomic xput = xput with minimum number of replicas
                app_xputs[cluster] = mean_atomic_xput
            self.cluster_throughputs[app_name] = app_xputs
        if sim_config.log_cluster_verbose:
            print(f"min-GPU throughputs: {self.cluster_throughputs}")


    # 导读 Cluster.step_log（原源码第 306 行）
    # 导读 把当前模拟时刻、作业状态和分配整理为本轮记录并追加到 logs。
    # 导读 函数体中的相关调用包括 self.allocations.get、self.logs.append。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def step_log(self):
        # 导读 Cluster.step_log.get_bsz（原源码第 307 行）
        # 导读 从节点放置、单次计算 batch 和累积次数计算日志中的全局 batch。
        # 导读 参数 job 表示当前要处理的作业对象。
        # 导读 从返回表达式可以继续追踪 job.atomic_bsz * (job.accum_steps + 1) * num_replicas。
        def get_bsz(job):
            ngpus = int(sum(job.placement))
            num_replicas = ngpus
            # if job is a pipeline, divide by number of stages
            if job.num_stages > 1:
                num_replicas //= job.num_stages
            return job.atomic_bsz * (job.accum_steps + 1) * num_replicas
        # 导读 step_log 是本轮状态字典。timestamp 是模拟时刻，submitted_jobs 按作业保存状态。
        step_log = {
            "timestamp": self.current_time,
            "num_nodes": self.num_nodes,
            "submitted_jobs": [
                {
                    "name": job.name,
                    "epoch": job.epoch,
                    "progress": job.progress,
                    "num_restarts": job.num_restarts,
                    "cluster": job.current_cluster,
                    "allocation": self.allocations.get(job.name, tuple()),
                    "placement": job.placement,
                    "batch_size": get_bsz(job),
                    "accum_steps": job.accum_steps,
                    "stages": job.num_stages,
                    "submission_time": job.submission_time,
                    "start_time": job.start_time,
                    "completion_time": job.completion_time,
                    "n_avg": np.mean(np.asarray(job.contention)),
                    "category": job.category,
                }
                for job in self.jobs if job.submission_time <= self.current_time
            ],
        }
        self.logs.append(step_log)

    # 导读 Cluster.get_job_infos（原源码第 340 行）
    # 导读 筛选已提交且未完成的 Job，为它们构造策略需要的摘要，以作业名作为字典键。
    # 导读 函数体中的相关调用包括 self.get_sia_multi_job_info、self.get_gavel_job_info、self.get_sia_fix_multi_job_info、self.get_pollux_job_info。
    # 导读 从返回表达式可以继续追踪 job_infos。
    def get_job_infos(self):
        # 导读 job_infos 是策略输入的作业摘要集合，按作业名或 ID 索引。
        job_infos = {}
        for job in self.jobs:
            # 导读 对象可以提前创建，只有提交时刻已到且还未完成的作业进入调度摘要。
            if self.current_time >= job.submission_time and job.completion_time is None:
                if isinstance(self.policy, SiaPolicy):
                    job_infos[job.name] = self.get_sia_multi_job_info(
                        job)
                elif isinstance(self.policy, GavelPolicy):
                    job_infos[job.name] = self.get_gavel_job_info(job)
                elif isinstance(self.policy, SiaFixPolicy):
                    job_infos[job.name] = self.get_sia_fix_multi_job_info(
                        job)
                else:
                    job_infos[job.name] = self.get_pollux_job_info(job)
        return job_infos

    # 导读 Cluster.get_pollux_job_info（原源码第 356 行）
    # 导读 把单个 Job 的性能函数、年龄或服务量及副本范围打包给 Pollux。
    # 导读 参数 job 表示当前要处理的作业对象。
    # 导读 从返回表达式可以继续追踪 job_info。
    def get_pollux_job_info(self, job):
        job_info = JobInfo(
            resources={"nvidia.com/gpu": 1},
            speedup_fn=job.get_speedup_fn(),
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            min_replicas=0,
            max_replicas=min(max(2 * job.max_profiled_replicas, 1), 64,  # simulator can't handle more.
                             job.application.max_batch_size // job.application.min_local_bsz),
            preemptible=True,
        )
        if job.application.name == "ncf":
            job_info.max_replicas = 1
        job_info.num_restarts = job.num_restarts or 0
        job_info.age = self.current_time - job.submission_time
        return job_info

    # 导读 Cluster.get_sia_multi_job_info（原源码第 373 行）
    # 导读 按 GPU 类型收集性能函数、可用副本范围和运行状态，构造 Sia 用的 JobInfo。
    # 导读 参数 job 表示当前要处理的作业对象。
    # 导读 从返回表达式可以继续追踪 job_info。
    def get_sia_multi_job_info(self, job):
        # 导读 不同 GPU 类型可以有各自的性能预测函数。模型未建立时值可能为 None。
        speedup_fns = {cname: job.get_speedup_fn(cname) for cname in self.clusters}
        scale_factor = 2
        # 导读 候选副本数同时受画像规模、实现上限和局部 batch 下限限制。
        max_replicas = {cname: min(max(scale_factor * job.max_profiled_replicas(cname), 1), 64,  # simulator can't handle more.
                        job.applications[cname].max_batch_size // job.applications[cname].min_local_bsz)
                        for cname in self.clusters}
        # 导读 JobInfo 是策略用的摘要。resources 的 GPU=1 是每个 GPU 单位的需求表示。
        job_info = JobInfo(
            resources={"nvidia.com/gpu": 1},
            speedup_fn=speedup_fns,
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            min_replicas={cname: 0 for cname in self.clusters},
            max_replicas=max_replicas,
            preemptible=True,
        )

        # attach used+wasted gpu seconds to job_info
        job_info.used_gpu_seconds = job.used_gpu_seconds
        job_info.wasted_gpu_seconds = job.wasted_gpu_seconds

        # record job category
        job_info.category = job.category
        if job_info.category == "rigid" or job_info.category == "scale-gpus":
            job_info.max_replicas = {
                cname: job.target_num_replicas for cname in self.clusters}

        # treat NCF jobs as rigid with max_replicas = 1
        if 'ncf' in job.applications:
            job_info.max_replicas = {cname: 1 for cname in self.clusters}
            job_info.category = "rigid"

        job_info.num_restarts = job.num_restarts or 0
        job_info.num_migrations = job.num_migrations or 0
        job_info.age = self.current_time - job.submission_time

        app_name = job.name.split('-')[0]

        # throughput conversion ratios
        job_info.cluster_throughput_ratios = self.cluster_throughput_ratios[app_name]
        job_info.cluster_throughputs = self.cluster_throughputs[app_name]

        # Rigid jobs ==> throughput ratios == throughputs of the job
        # since num_replicas = target_num_replicas and stays fixed
        if job_info.category == "rigid":
            job_info.cluster_throughput_ratios = job.get_throughputs(
                self.ngpus_per_node)
            print(
                f"Rigid job: {job.name}, throughput ratios: {job_info.cluster_throughput_ratios}")

        # if job is currently in rescale stage
        if sim_config.preserve_ckpt_allocs:
            job_info.scaling_underway = job.rescale_time > 0
        else:
            job_info.scaling_underway = False

        # scale units for this job
        job_info.scale_unit = job.get_scale_units()

        return job_info

    # 导读 Cluster.get_sia_fix_multi_job_info（原源码第 434 行）
    # 导读 为固定 GPU 数量的 Sia 变体构造 JobInfo，将各类型的最小与最大副本设为目标值。
    # 导读 参数 job 表示当前要处理的作业对象。
    # 导读 从返回表达式可以继续追踪 job_info。
    def get_sia_fix_multi_job_info(self, job):
        speedup_fns = {cname: job.get_speedup_fn(
            cname) for cname in self.clusters}
        target_replicas = {
            cname: job.target_num_replicas for cname in self.clusters}
        job_info = JobInfo(
            resources={"nvidia.com/gpu": 1},
            speedup_fn=speedup_fns,
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            min_replicas=target_replicas,
            max_replicas=target_replicas,
            preemptible=True,
        )

        for cname, capp in job.applications.items():
            if capp.name == "ncf":
                job_info.max_replicas[cname] = 1
        job_info.num_restarts = job.num_restarts or 0
        job_info.num_migrations = job.num_migrations or 0
        job_info.age = self.current_time - job.submission_time
        # job_info.target_batch_size = job.target_batch_size

        app_name = job.name.split('-')[0]

        # throughput conversion ratios
        job_info.cluster_throughput_ratios = self.cluster_throughput_ratios[app_name]

        return job_info

    # 导读 Cluster.get_gavel_job_info（原源码第 464 行）
    # 导读 为 Gavel 准备副本数、性能和运行状态等字段。
    # 导读 参数 job 表示当前要处理的作业对象。
    # 导读 从返回表达式可以继续追踪 job_info。
    def get_gavel_job_info(self, job):
        speedup_fns = {cname: job.get_speedup_fn(
            cname) for cname in self.clusters}
        max_replicas = {cname: min(max(2 * job.max_profiled_replicas(cname), 1), 64,  # simulator can't handle more.
                        job.applications[cname].max_batch_size // job.applications[cname].min_local_bsz)
                        for cname in self.clusters}
        job_info = JobInfo(
            resources={"nvidia.com/gpu": 1},
            speedup_fn=speedup_fns,
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            min_replicas={cname: 0 for cname in self.clusters},
            # max_replicas=min(max(2 * job.max_profiled_replicas, 1), 64,  # simulator can't handle more.
            #                 job.target_batch_size // job.application.min_local_bsz),
            max_replicas=max_replicas,
            preemptible=True,
        )
        for cname, capp in job.applications.items():
            if capp.name == "ncf":
                job_info.max_replicas[cname] = 1
        job_info.applications = job.applications
        job_info.epoch = job.epoch
        job_info.target_batch_size = job.target_batch_size
        job_info.scale_factor = job.target_num_replicas
        job_info.age = self.current_time - job.submission_time
        job_app = list(job.applications.values())[0]
        job_info.cur_step = job_app.get_cur_iteration(
            job.target_batch_size, job.epoch, job.progress)
        job_info.total_steps = job_app.get_iteration(
            job.target_batch_size, job.completion_epoch)
        job_info.remain_steps = max(
            1, job_info.total_steps - job_info.cur_step)
        job_info.rescale_time = job_app.rescale_time
        job_info.submission_time = job.submission_time
        job_info.slowdown_factor = job.calibration_factor
        assert job_app.rescale_time > 0
        return job_info

    # 导读 Cluster.get_node_infos（原源码第 502 行）
    # 导读 将模拟节点的容量转换成 NodeInfo，供策略求解和放置使用。
    # 导读 参数 num_nodes 表示当前配置占用或考虑的节点数。
    # 导读 从返回表达式可以继续追踪 cluster_node_info。
    # 导读 NodeInfo 的 GPU 值是节点总容量，策略会结合已有分配计算使用情况。
    def get_node_infos(self, num_nodes=None):
        cluster_node_info = dict()
        for cluster_name in self.clusters:
            cluster_info = {idx: NodeInfo({"nvidia.com/gpu": self.ngpus_per_node[cluster_name]}, preemptible=False)
                            for idx in range(self.num_nodes[cluster_name])}
            cluster_node_info[cluster_name] = cluster_info
        return cluster_node_info

    # 导读 Cluster.all_complete（原源码第 510 行）
    # 导读 检查 jobs 中每个作业是否已有 completion_time，返回布尔值。
    # 导读 从返回表达式可以继续追踪 all((job.completion_time is not None for job in self.jobs))。
    def all_complete(self):
        return all(job.completion_time is not None for job in self.jobs)

    # 导读 Cluster.output_logs（原源码第 513 行）
    # 导读 把累积日志写入 path，按一轮一条 JSON 记录的方式输出。
    # 导读 参数 path 表示输入或输出文件路径。
    # 导读 函数体中的相关调用包括 json.dump。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def output_logs(self, path):
        with open(path, "w") as f:
            for record in self.logs:
                json.dump(record, f)
                f.write("\n")

    # 导读 Cluster.get_jcts（原源码第 519 行）
    # 导读 从已记录的完成时刻减去提交时刻，得到以作业名索引的 JCT 字典。
    # 导读 从返回表达式可以继续追踪 {}。
    def get_jcts(self):
        if len(self.logs) > 0:
            return {
                # 导读 JCT 从提交到完成，包含排队和恢复开销。按最后一轮的已完成作业生成字典。
                val["name"]: val["completion_time"] - val["submission_time"]
                for val in self.logs[-1]["submitted_jobs"]
                if val["completion_time"] is not None
            }
        else:
            return {}

# 导读 _helper_step_job_single（原源码第 529 行）
# 导读 进程池的顶层包装函数，推进一个 Job 并返回更新后的对象。
# 导读 参数 job 表示当前要处理的作业对象，seconds 表示需要推进的模拟秒数。
# 导读 函数体中的相关调用包括 job.step。
# 导读 从返回表达式可以继续追踪 job。
def _helper_step_job_single(job, seconds):
    job.step(seconds=seconds, interference=0)
    # important to return job object as all changes are process-local
    return job
