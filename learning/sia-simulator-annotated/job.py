# 导读 单个训练作业的状态与推进逻辑。维护 GPU 放置、批量、性能观测、训练进度、迁移与恢复计时。
# 导读 来源 job.py。以下带「导读」的行是阅读注释。
import math
import time
import numpy as np
import random

from applications import APPLICATIONS
import simulator_config as sim_config
from simulator_config import SLOWDOWNS
from goodput import GoodputFunction, GoodputFunctionPMP, fit_perf_params, PerfParams, GradParams
from speedup import SpeedupFunction, UncachedSpeedupFunction
from utils import JobInfo, NodeInfo

MAX_SLOWDOWN = 15

class Job(object):
    # 导读 Job.__init__（原源码第 16 行）
    # 导读 初始化单个作业。输入包含提交时刻、各机器的 Application、目标副本数、目标 batch 和类别。后续 step、reallocate、migrate 会更新其状态。
    # 导读 参数 name 表示此对象或待查询条目的名称，applications 表示机器类型到 Application 对象的映射，submission_time 表示作业提交的模拟时刻。
    # 导读 参数 target_num_replicas 表示工作负载指定的目标副本数，target_batch_size 表示工作负载指定的目标 batch，category 表示作业类别，例如 rigid 或 adaptive。
    # 导读 其余显式参数为 cache_speedups、h_unaware，默认值见函数签名。
    # 导读 函数体中的相关调用包括 self.applications.keys、self.reference_app.get_completion_epoch。
    # 导读 函数体直接赋值的实例字段包括 self.name、self.app_name、self.applications、self.category、self.cluster_throughputs、self.submission_time、self.target_num_replicas、self.target_batch_size、self.enable_bsz_tuning。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, name, applications, submission_time,
                 target_num_replicas=None, target_batch_size=None,
                 cache_speedups=False, h_unaware=False, category=None):
        ## Job attributes
        self.name = name
        self.app_name = list(applications.values())[0].name
        # applications is a dict w/ key=cluster_name, val=cluster_specific_App
        self.applications = applications
        self.category = category
        
        if self.category == "rigid":
            self.cluster_throughputs = None
        
        # submission_time: when it was submitted
        # 导读 提交时间来自 workload 的 time 列，单位为模拟秒。
        self.submission_time = submission_time
        # target_num_replicas: requested num replicas
        self.target_num_replicas = target_num_replicas
        # target_batch_size: requested batch size
        self.target_batch_size = target_batch_size
        # switch to control bsz tuning
        self.enable_bsz_tuning = True

        # switch to control if job is heterogeneity-aware
        # 导读 h_unaware 控制画像与性能参数是跨机器类型共用，还是按类型分别保存。
        self.h_unaware = h_unaware
        # profile is also a dict w/ key=cluster_name, val=cluster_specific_profile
        # if h_unaware is True, profiles is just all profiles merged into one dict
        # 导读 profiles 以配置三元组 (节点数, 副本数, 单次局部 batch) 保存计时二元组 (step_time, sync_time)。
        self.profiles = dict()
        # perf_params is also a dict w/ key=cluster_name, val=cluster_specific_perf_params

        # Application 中虽然有参考性能数据，但新建的 Job 不会默认把全部参考数据都当作自己已经观察到的信息。
        if self.h_unaware:
            self.perf_params = None
        else:
            self.perf_params = dict()
            for cluster in self.applications.keys():
                self.profiles[cluster] = dict()
                self.perf_params[cluster] = None
        # Optimization state
        # 导读 atomic_bsz 是单个副本一次计算的样本数。accum_steps 是额外累积次数，计算一次更新共做 accum_steps + 1 次。
        self.atomic_bsz = 0
        self.accum_steps = 0
        self.grad_params = None
        self.best_metric = None
        # 导读 progress 是模型的累计训练进度，epoch 是所在的训练阶段。两个值的对应关系由 Application 提供。
        self.progress = 0.0
        self.epoch = 0
        self.reference_app = self.applications[sim_config.calibrate_cluster]
        
        ## Job state
        # start_time: when it started running
        self.start_time = None
        # completion_time: when it finished running
        # 导读 None 表示作业尚未完成。完成时会记录具体模拟秒数。
        self.completion_time = None
        # current_time: current wall-clock time
        # 导读 Job 自己的时钟用于记录区间内部的完成时刻，例如 120 到 180 秒之间的第 145 秒。
        self.current_time = 0
        # (overhead) time to checkpoint + restore job
        # >0 if an operation is pending/executing, 0 otherwise
        # 导读 rescale_time 是剩余恢复秒数。Job.step 优先消耗它。
        self.rescale_time = 0
        self.max_rescale_time = self.reference_app.rescale_time
        # N-tuple: (x_1, x_2, ..., x_N) where x_i is the number of replicas in node i
        # 导读 placement 是每个已占用节点的卡数元组，(2, 1) 表示两个节点上分别用两张和一张卡，空元组表示无分配。
        self.placement = ()
        # GPU-seconds allocated
        # 导读 attained_service 以 GPU·秒累计已获得服务。恢复时间也有 GPU 占用，并另记到 wasted_gpu_seconds。
        self.attained_service = 0
        # GPU-seconds used by job
        self.used_gpu_seconds = 0
        # GPU-seconds wasted by job
        self.wasted_gpu_seconds = 0

        # number of jobs contending for resources (one value per round)
        self.contention = []
        # number of times the job has been restarted
        self.num_restarts = None
        # number of times the job has been migrated between different clusters
        # note: num_migrations <= num_restarts
        self.num_migrations = None
        # events corresponding to job state changes
        self.bsz_update_events = []
        self.rescale_events = []
        self.migrate_events = []
        # num_stages > 1 for Pipeline-Model-Parallelism(PMP) jobs
        self.num_stages = self.reference_app.num_stages

        # cluster that this job is allocated to
        # 导读 current_cluster 保存机器类型名，具体节点上的 GPU 张数保存在 placement。
        self.current_cluster = None
        self.speedup_fn_class = SpeedupFunction if cache_speedups else UncachedSpeedupFunction

        # shockwave addons
        self.execution_time = 0
        self.epoch_duration = []
        self.completion_epoch = self.reference_app.get_completion_epoch(
            self.target_batch_size) if self.target_batch_size is not None else None

        # calibration factor for this job (calibration step to account for real-world runs being slower than simulator)
        # 导读 通过应用特定的 SLOWDOWNS 调整模拟速率，例如固定作业吞吐和训练进度都会使用这个校准因子。
        self.calibration_factor = 1 / SLOWDOWNS[self.app_name]
        if sim_config.log_cluster_verbose: 
            print(f"{self.name}: SpeedupFnClass={self.speedup_fn_class.__name__}, calibration-factor={self.calibration_factor:.2f}")
    
    # optimizes perf params given an input perf profile
    # perf_profile is a dict w/ key=(num_nodes, num_replicas, local_bsz), val=(step_time, sync_time)
    # returns a PerfParams object
    # 导读 Job.optimize_perf_params（原源码第 112 行）
    # 导读 把画像中每个配置的节点数、副本数、局部 batch 和耗时提取成数组，拟合 PerfParams。
    # 导读 其余显式参数为 perf_profile，默认值见函数签名。
    # 导读 函数体中的相关调用包括 fit_perf_params。
    # 导读 从返回表达式可以继续追踪 perf_params。
    def optimize_perf_params(self, perf_profile):
        if perf_profile is None:
            return None
        # 导读 把画像字典拆成等长数组，同一索引指向同一次配置观测。
        # 例如：
        # perf_profile = {
        #                     (1, 1, 16): (0.12, 0.01),
        #                     (1, 2, 16): (0.15, 0.03),
        #                     (2, 4, 32): (0.28, 0.08),
        #                     (2, 8, 32): (0.42, 0.15),
        #                 }                   
        num_nodes = np.array([key[0] for key in perf_profile])
        num_replicas = np.array([key[1] for key in perf_profile])
        local_bsz = np.array([key[2] for key in perf_profile])
        step_time = np.array([val[0] for val in perf_profile.values()])
        sync_time = np.array([val[1] for val in perf_profile.values()])
        # 导读 compute_time 用整步耗时减去同步耗时，作为纯计算观测。
        compute_time = step_time - sync_time
        # NumPy 自动对相同位置的元素做减法

        perf_params = fit_perf_params(
                num_nodes, num_replicas, local_bsz, compute_time, step_time)
        return perf_params


    # 性能测试数据是怎样从 CSV 文件变成 Job 内部的性能字典的？
    # 遍历 Job 对应的不同 GPU 集群。
    # 从 Application 中读取性能测试数据。
    # 按最大节点数和最大 GPU 副本数筛选。
    # 把 Pandas DataFrame 转换成 NumPy 数组。
    # 将配置及其性能数据保存到 self.profiles[cluster]。

    # 最大节点数和最大副本数在这个函数中限制的是初始性能数据的覆盖范围，不是直接规定任务最多只能使用多少 GPU。
    def seed_profiles(self, max_num_nodes, max_num_replicas):
        print(f"Seeding profiles for job: {self.name}")
        for cluster, cluster_app in self.applications.items():
            self.profiles[cluster] = dict()
            profile = self.profiles[cluster]

            # add placements data
            if max_num_nodes > 0:
                placements_selector = (cluster_app.placements.num_nodes <= max_num_nodes) & (
                    cluster_app.placements.num_replicas <= max_num_replicas)
                df = cluster_app.placements[placements_selector]
            else:
                df = cluster_app.placements

            num_nodes, num_replicas, local_bsz, step_time, sync_time = df.num_nodes.to_numpy(
            ), df.num_replicas.to_numpy(), df.local_bsz.to_numpy(), df.step_time.to_numpy(), df.sync_time.to_numpy()
            for i in range(len(num_nodes)):
                self.profiles[cluster][num_nodes[i], num_replicas[i],
                                       local_bsz[i]] = step_time[i], sync_time[i]
            # add scalability data
            if max_num_nodes > 0:
                scalability_selector = (cluster_app.scalability.num_nodes <= max_num_nodes) & (
                    cluster_app.scalability.num_replicas <= max_num_replicas)
                df = cluster_app.scalability[scalability_selector]
            else:
                df = cluster_app.scalability

            num_nodes, num_replicas, local_bsz, step_time, sync_time = df.num_nodes.to_numpy(
            ), df.num_replicas.to_numpy(), df.local_bsz.to_numpy(), df.step_time.to_numpy(), df.sync_time.to_numpy()
            for i in range(len(num_nodes)):
                profile_key = (num_nodes[i], num_replicas[i], local_bsz[i])
                if profile_key not in profile:
                    profile[profile_key] = step_time[i], sync_time[i]

            # update perf params for cluster
            self.perf_params[cluster] = self.optimize_perf_params(profile)
    # CSV → DataFrame → 条件筛选 → NumPy 数组 → Python 字典 → 性能模型参数

    # 导读 Job.seed_profiles_rigid（原源码第 163 行）
    # 导读 根据固定作业的副本需求和每节点 GPU 容量选已有计时，初始化各机器类型的性能画像。
    # 导读 参数 cluster_ngpus_per_node 表示各机器类型每节点 GPU 数。
    # 导读 函数体中的相关调用包括 self.optimize_perf_params。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    # 导读 对固定 GPU 数的作业，按需要的节点和副本上界载入已有画像，建立初始性能参数。
    def seed_profiles_rigid(self, cluster_ngpus_per_node):
        print(
            f"Seeding profiles for job: {self.name}, target bsz: {self.target_batch_size}, target num replicas: {self.target_num_replicas}")
        for cluster in cluster_ngpus_per_node.keys():
            cluster_app = self.applications[cluster]
            self.profiles[cluster] = dict()
            profile = self.profiles[cluster]
            job_num_nodes = np.ceil(self.target_num_replicas /
                                    cluster_ngpus_per_node[cluster])
            job_num_replicas = self.target_num_replicas

            # add placements data if exists
            placements_selector = (cluster_app.placements.num_nodes == job_num_nodes) & (
                cluster_app.placements.num_replicas <= job_num_replicas)
            df = cluster_app.placements[placements_selector]
            if len(df) == 0:
                print(f"No placements data for job: {self.name}, \
                        cluster: {cluster}, num_nodes: {job_num_nodes}, num_replicas: {job_num_replicas}")
            else:
                print(
                    f"Num placement profiles: {len(df)} for job: {self.name}")
                num_nodes, num_replicas, local_bsz, step_time, sync_time = df.num_nodes.to_numpy(
                ), df.num_replicas.to_numpy(), df.local_bsz.to_numpy(), df.step_time.to_numpy(), df.sync_time.to_numpy()
                for i in range(len(num_nodes)):
                    self.profiles[cluster][num_nodes[i], num_replicas[i],
                                           local_bsz[i]] = step_time[i], sync_time[i]
            # add scalability data
            scalability_selector = (cluster_app.scalability.num_nodes == job_num_nodes) & (
                cluster_app.scalability.num_replicas <= job_num_replicas)
            df = cluster_app.scalability[scalability_selector]
            if len(df) == 0:
                print(f"No scalability data for job: {self.name}, \
                        cluster: {cluster}, num_nodes: {job_num_nodes}, num_replicas: {job_num_replicas}")
            else:
                print(
                    f"Num scalability profiles: {len(df)} for job: {self.name}")
                num_nodes, num_replicas, local_bsz, step_time, sync_time = df.num_nodes.to_numpy(
                ), df.num_replicas.to_numpy(), df.local_bsz.to_numpy(), df.step_time.to_numpy(), df.sync_time.to_numpy()
                for i in range(len(num_nodes)):
                    profile_key = (num_nodes[i], num_replicas[i], local_bsz[i])
                    if profile_key not in profile:
                        profile[profile_key] = step_time[i], sync_time[i]
            if not profile:
                print(
                    f"WARNING: No data for job: {self.name}, cluster: {cluster}")
                continue

            # compute perf params for cluster
            self.perf_params[cluster] = self.optimize_perf_params(profile)
            print(
                f"Initialized goodput fn for job: {self.name}, cluster: {cluster}")

    # returns the maximum number of replicas profiled for a given cluster
    # 导读 Job.max_profiled_replicas（原源码第 216 行）
    # 导读 查询已有画像覆盖到的最大副本数，供策略设定候选范围。
    # 导读 参数 cluster_name 表示GPU 集群或机器类型名。
    # 导读 从返回表达式可以继续追踪 max_val。

#     max_profiled_replicas() 返回的是性能模型目前掌握的数据范围.为性能外推提供一个参考范围，   
#     self.profiles["aws"] = {
#     (1, 2, 16): (0.15, 0.03),
#     (2, 4, 32): (0.28, 0.08),
#     (2, 8, 32): (0.42, 0.15)
# }
#     (1, 2, 16) → 2
#     (2, 4, 32) → 4
#     (2, 8, 32) → 8
#     max_val = 8
    def max_profiled_replicas(self, cluster_name=None):
        max_val = 0
        if not cluster_name and self.h_unaware:
            max_val = max((k[1] for k in self.profiles), default=0)
        if cluster_name in self.profiles:
            max_val = max((k[1]
                          for k in self.profiles[cluster_name]), default=0)
        return max_val

    # 导读 Job.get_goodput_fn（原源码第 225 行）
    # 导读 拿指定 GPU 类型的性能参数与当前梯度统计构造 GoodputFunction。模型尚未建立时返回 None。
    # 导读 参数 cluster_name 表示GPU 集群或机器类型名。
    # 导读 从返回表达式可以继续追踪 GoodputFunctionPMP(perf_params, grad_params, app.init_batch_size, app.num_stages, app.num_microbatches)、GoodputFunction(perf_params, grad_params, app.init_batch_size)。
    def get_goodput_fn(self, cluster_name=None):
        app = self.applications[cluster_name if cluster_name else "aws"]
        if self.h_unaware:
            perf_params, grad_params = self.perf_params, self.grad_params
        else:
            perf_params, grad_params = self.perf_params[cluster_name], self.grad_params

        # 导读 估计 Goodput 需要性能参数和梯度统计。缺任意一项时返回 None。
        # no throughput model yet
        if grad_params is None or perf_params is None:
            return None
        elif app.num_stages > 1:
            return GoodputFunctionPMP(perf_params, grad_params, app.init_batch_size, app.num_stages, app.num_microbatches)
        else:
            return GoodputFunction(perf_params, grad_params, app.init_batch_size)

    # 导读 Job.get_speedup_fn（原源码第 240 行）
    # 导读 包装 Goodput 模型，并带上 batch 范围、梯度累积和是否调 batch 的配置，供策略查询候选。
    # 导读 参数 cluster_name 表示GPU 集群或机器类型名。
    # 导读 函数体中的相关调用包括 self.speedup_fn_class、self.get_goodput_fn。
    # 导读 从返回表达式可以继续追踪 lambda n, r: r。
    def get_speedup_fn(self, cluster_name=None):
        if self.h_unaware:
            if self.grad_params is None:
                return lambda n, r: r
        else:
            perf_params = self.perf_params[cluster_name]
            if self.grad_params is None or perf_params is None:
                return None
        app = self.applications[cluster_name if cluster_name else "aws"]
        max_batch_size = app.max_batch_size if self.target_batch_size is None else self.target_batch_size
        bsz_range = (app.min_local_bsz, app.max_local_bsz)
        # 导读 这个包装对象让调度策略可以按节点数和副本数批量查询性能。
        return self.speedup_fn_class(self.get_goodput_fn(cluster_name), max_batch_size,
                                     bsz_range, accumulation=True, tune_bsz=self.enable_bsz_tuning)

    # returns throughput of job for different GPU types in examples/sec
    # 导读 Job.get_throughputs（原源码第 255 行）
    # 导读 固定作业在各类型上按目标副本数与 batch 查询吞吐，再乘作业校准系数。重复查询可读缓存。
    # 导读 参数 cluster_ngpus_per_node 表示各机器类型每节点 GPU 数。
    # 导读 函数体中的相关调用包括 self.applications[cname].get_throughput_with_accum。
    # 导读 函数体直接赋值的实例字段包括 self.cluster_throughputs。
    # 导读 从返回表达式可以继续追踪 {cname: self.cluster_throughputs[cname] for cname in cluster_ngpus_per_node}。

    #这个函数的目标是计算：同一个固定配置的训练任务，在不同 GPU 集群上的吞吐量是多少
    def get_throughputs(self, cluster_ngpus_per_node):
        if self.category != "rigid":
            print(
                f"ERROR:: invalid job category: {self.category} for get_throughput call")
        # cluster throughputs are not computed
        if not self.cluster_throughputs:
            self.cluster_throughputs = dict()
            for cname, cngpus_per_node in cluster_ngpus_per_node.items():
                if self.target_num_replicas <= cngpus_per_node:
                    placement = [self.target_num_replicas]
                else:
                    num_whole_nodes = self.target_num_replicas // cngpus_per_node
                    placement = [cngpus_per_node] * num_whole_nodes
                    need_partial = self.target_num_replicas % cngpus_per_node != 0
                    if need_partial:
                        placement.append(
                            self.target_num_replicas % cngpus_per_node)
                # 导读 固定作业查询目标 batch 下的样本每秒，并乘校准因子。
                self.cluster_throughputs[cname] = self.applications[cname].get_throughput_with_accum(
                    placement, self.target_batch_size) * self.calibration_factor
            print(
                f"Initialized cluster throughputs for job: {self.name} = {self.cluster_throughputs}")
        else:
            print(
                f"Using cached cluster throughputs for job: {self.name} = {self.cluster_throughputs}")
        return {cname: self.cluster_throughputs[cname] for cname in cluster_ngpus_per_node}

    # 导读 Job.get_scale_units（原源码第 281 行）
    # 导读 按应用的流水线阶段数建立 GPU 数与训练副本数的换算单位。
    # 导读 函数体中的相关调用包括 self.applications.items。
    # 导读 函数体直接赋值的实例字段包括 self.scale_units。
    # 导读 从返回表达式可以继续追踪 self.scale_units。
    def get_scale_units(self):
        self.scale_units = {
            cluster: app.num_stages for cluster, app in self.applications.items()}
        if sim_config.sia_log_goodputs:
            print(f"Scale units for job: {self.name} = {self.scale_units}")
        return self.scale_units

    # fixes batch size to obey memory/profiling constraints
    # needed for gavel
    # 导读 Job.fix_minibatch_size（原源码第 290 行）
    # 导读 根据固定副本数、目标 batch 和设备可承受的局部 batch 调整目标批量。
    # 导读 函数体中的相关调用包括 self.applications.items。
    # 导读 函数体直接赋值的实例字段包括 self.target_batch_size。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def fix_minibatch_size(self):
        print(f"WARNING:: bypassing fix for minibatch size")
        return
        if self.target_num_replicas is not None and self.target_batch_size is not None:
            max_atomic_bsz = math.ceil(
                self.target_batch_size / self.target_num_replicas - 1e-8)
            for cluster, cluster_app in self.applications.items():
                if self.target_num_replicas in cluster_app.placements.num_replicas.values:
                    df = cluster_app.placements[cluster_app.placements.num_replicas ==
                                                self.target_num_replicas]
                    new_bsz = int(min(max_atomic_bsz, df.local_bsz.max()))
                    if new_bsz < max_atomic_bsz:
                        print(
                            f"{self.name}: correcting atomic_bsz: {max_atomic_bsz} -> {new_bsz}")
                        max_atomic_bsz = new_bsz
            target_batch_size = self.target_num_replicas * max_atomic_bsz
            self.target_batch_size = min(
                self.target_batch_size, target_batch_size)

    # call this function to disable bsz tuning
    # 导读 Job.disable_bsz_tuning（原源码第 310 行）
    # 导读 关闭 batch 自动调整。集群版本向每个 Job 传递这一设置，Job 版本更新本作业的开关。
    # 导读 函数体直接赋值的实例字段包括 self.enable_bsz_tuning。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def disable_bsz_tuning(self):
        self.enable_bsz_tuning = False
        print(f"Disabled bsz tuning for job: {self.name}")

    # given a placement, computes per-GPU batch size, # accumulations
    # uses user-supplied batch size if bsz tuning is disabled
    # NOTE: if bsz tuning disabled ==> the job scales using strong scaling
    # 导读 Job.update_local_bsz（原源码第 317 行）
    # 导读 根据当前机器类型及 placement，写入 atomic_bsz 和 accum_steps。覆盖已有 Goodput 模型、冷启动与固定 batch 分支。
    # 导读 参数 placement 表示每个已使用节点上的 GPU 或副本数。
    # 导读 函数体中的相关调用包括 self.get_goodput_fn、goodput_fn.optimize。
    # 导读 函数体直接赋值的实例字段包括 self.atomic_bsz、self.accum_steps。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。

    # 但是，光知道有 6 张 GPU，还不能开始模拟训练。
    # 我们还需要确定：
    # - 每张 GPU 一次处理多少个样本？
    # - 是否需要梯度累积？
    # - 每完成一次参数更新，整个任务处理多少个样本？
    # update_local_bsz() 就是负责解决这些问题的。

    def update_local_bsz(self, placement):
        if self.current_cluster is None:
            assert False, "updating local bsz before assigning cluster"
        app = self.applications[self.current_cluster]
        if app.num_stages > 1:
            self.atomic_bsz = 1
            self.accum_steps = app.num_microbatches
        placement = tuple(filter(None, placement)) 
        # 删除placement里面的空节点
        # 导读 去掉零之后，元组长度是占用节点数，和是总 GPU 数。流水线分支稍后再折算副本。
        num_nodes, num_replicas = len(placement), sum(placement)
        # use target batch size by default
        # 导读 有固定 batch 时先保留目标值。允许调 batch 时用 None 表示后续需要重新选择。
        batch_size = self.target_batch_size

        # if bsz tuning is enabled, reset batch size and optimize again
        if self.enable_bsz_tuning:
            # print(f"update_local_bsz: bsz tuning ****enabled**** for job: {self.name}")
            batch_size = None
        else:
            # print(f"update_local_bsz: bsz tuning disabled for job: {self.name}")
            pass

        perf_params = self.perf_params if self.h_unaware else self.perf_params[
            self.current_cluster]
        grad_params = self.grad_params
        # 导读 max_local_bsz 来自 Application 的放置相关 batch 上限。
        max_local_bsz = app.get_max_local_bsz(placement)

        # handle PMP jobs gracefully
        if 'pmp' in app.name:
            # obtain pipeline params from app directly
            # 导读 流水线应用由阶段数和 microbatch 配置确定每副本资源和批量。
            self.atomic_bsz, self.accum_steps = app.microbatch_size, (
                app.num_microbatches - 1)
            num_stages = app.num_stages
            # each replica runs on num_stages GPUs
            num_replicas = num_replicas // num_stages
            # obtain global bsz using num_replicas
            bsz_per_replica = self.atomic_bsz * (self.accum_steps + 1)
            batch_size = bsz_per_replica * num_replicas
            max_local_bsz = self.atomic_bsz

        # initial batch size if job has not yet started running
        # 导读 模型还不完整时，按应用的初始 batch 和每副本最低 batch 初始化。
        if batch_size is None and (grad_params is None or perf_params is None):
            batch_size = max(app.init_batch_size,
                             app.min_local_bsz * num_replicas)
        # if we have a goodput function, use it to optimize batch size
        if batch_size is None:
            # 导读 使用性能模型搜索合法 batch，取对应的单次 batch 和额外累积次数。
            goodput_fn = self.get_goodput_fn(self.current_cluster)
            # use standard goodput function if exists
            _, self.atomic_bsz, self.accum_steps = goodput_fn.optimize(
                num_nodes, num_replicas, app.max_batch_size,
                (app.min_local_bsz, max_local_bsz), accumulation=True)
        else:
            # otherwise, use the batch size specified by the user
            # 导读 固定 batch 路径先分给各副本，再拆分成满足单次容量的多个计算步。
            local_bsz = math.ceil(batch_size / num_replicas - 1e-8)
            self.accum_steps = math.ceil(local_bsz / max_local_bsz - 1e-8) - 1
            if num_replicas == 1 and batch_size > app.init_batch_size:
                self.accum_steps = max(1, self.accum_steps)
            self.atomic_bsz = math.ceil(
                local_bsz / (self.accum_steps + 1) - 1e-8)

        # correct self.atomic_bsz to take into account memory constraints
        # 导读 如果画像给出了这个副本数的支持范围，再把单次 batch 限制在该范围。
        if num_replicas in app.placements.num_replicas.values and 'gpt' not in app.name:
            df = app.placements[app.placements.num_replicas == num_replicas]
            new_bsz = int(min(self.atomic_bsz, df.local_bsz.max()))
            if new_bsz < self.atomic_bsz:
                print(
                    f"WARNING #------>{self.name}: correcting atomic_bsz: {self.atomic_bsz} -> {new_bsz}")
                self.atomic_bsz = new_bsz

        # 导读 count 是每次更新包含的副本计算步总数，乘 atomic_bsz 得到全局 batch。
        count = num_replicas * (self.accum_steps + 1)
        self.atomic_bsz = min(self.atomic_bsz, int(app.max_batch_size / count))
        # print(f"update_local_bsz({self.name}): atomic_bsz={self.atomic_bsz}, accum_steps={self.accum_steps},num_replicas={num_replicas}, batch_size={batch_size}")

    # 导读 Job.update_params（原源码第 388 行）
    # 导读 接收本次模拟执行得到的计时与梯度信息。写入梯度统计，对未见过的配置记录画像并重新拟合性能参数。
    # 导读 参数 num_nodes 表示当前配置占用或考虑的节点数，num_replicas 表示配置中的训练副本数，数据并行时一般对应 GPU 数，local_bsz 表示传入当前方法的每副本局部 batch。
    # 导读 参数 step_time 表示一次完整训练步骤的计时，单位为秒，sync_time 表示其中的同步计时，单位为秒，grad_sqr 表示梯度信号平方项。
    # 导读 参数 grad_var 表示梯度方差项。
    # 导读 函数体中的相关调用包括 self.optimize_perf_params。
    # 导读 函数体直接赋值的实例字段包括 self.grad_params、self.perf_params。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。

    # seed_profiles() → 初始化已知的性能数据。

    # optimize_perf_params() → 根据性能数据拟合模型。

    # 现在的 update_params() 则负责：任务运行期间，将当前配置下获得的性能信息更新到 Job 内部，并在遇到新配置时重新拟合性能模型。
    def update_params(self, num_nodes, num_replicas, local_bsz,
                      step_time, sync_time, grad_sqr, grad_var):
        assert self.current_cluster is not None, "current_cluster is None??"
        # 导读 梯度统计随训练阶段更新。对于已有的相同配置，保留已记录的性能样本。
        self.grad_params = (grad_sqr, grad_var)
        if self.h_unaware:
            profile = self.profiles
        else:
            profile = self.profiles[self.current_cluster]

        if (num_nodes, num_replicas, local_bsz) in profile:
            return

        if self.h_unaware:
            self.profiles[num_nodes, num_replicas,
                          local_bsz] = step_time, sync_time
        else:
            self.profiles[self.current_cluster][num_nodes,
                                                num_replicas, local_bsz] = step_time, sync_time

        # get PerfParams for these profiles
        # 导读 只有加入新配置观测后才重新拟合该类型的性能模型。
        perf_params = self.optimize_perf_params(profile)
        if self.h_unaware:
            self.perf_params = perf_params
        else:
            self.perf_params[self.current_cluster] = perf_params

    # 导读 Job.step（原源码第 414 行）
    # 导读 推进一份 Job 的 seconds 模拟秒。无资源时只走时钟，有资源时先扣恢复倒计时，再按有效速率推进进度与 epoch。状态写回 Job。
    # 导读 参数 seconds 表示需要推进的模拟秒数，interference 表示执行速率上的干扰折损比例。
    # 导读 函数体中的相关调用包括 application.get_throughput、self.update_params、self.epoch_duration.append、self.update_local_bsz。
    # 导读 函数体直接赋值的实例字段包括 self.current_time、self.attained_service、self.wasted_gpu_seconds、self.rescale_time、self.used_gpu_seconds、self.progress、self.execution_time、self.epoch、self.completion_epoch。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def step(self, seconds, interference=0.0):
        # 导读 已经完成的 Job 保持完成时刻。
        if self.completion_time is not None:
            return
        # 导读 没有放置时，只增加该作业的模拟时钟。
        if not self.placement:
            # No resources are allocated to this job.
            self.current_time += seconds
            return
        # job does not use GPUs for `delay` seconds (checkpoint+restore)
        # 导读 一轮能消耗的恢复秒数取剩余恢复时间与本轮时间中的较小值。
        delay = min(self.rescale_time, seconds)
        # step delay time
        self.current_time += delay
        self.attained_service += delay * sum(self.placement)
        self.wasted_gpu_seconds += delay * sum(self.placement)
        
        # update rescale time
        self.rescale_time -= delay

        # simulate training for any remaining seconds
        # 导读 从可用时间中扣除恢复耗时，剩余的秒数才进入训练推进循环。
        seconds -= delay
        if seconds > 0:
            self.used_gpu_seconds += (seconds * sum(self.placement))
        # 导读 一次调度轮可能跨越几个 epoch，所以用循环分段推进。
        while seconds > 0 and self.completion_time is None:
            assert self.current_cluster is not None, "stepping on job without current_cluster set"
            application = self.applications[self.current_cluster]
            assert self.epoch < application.max_epochs
            # print(f"job: {self.name}, placement: {self.placement}")
            # Calculate current job configurations.
            placement = tuple(filter(None, self.placement))
            num_nodes, num_gpus = len(placement), sum(placement)
            # 导读 local_bsz 是一个副本在一次更新中合计计算的样本数。
            local_bsz = self.atomic_bsz * (self.accum_steps + 1)
            # one PMP replica can span many stages
            if self.num_stages > 1:
                num_replicas = num_gpus // self.num_stages
            else:
                num_replicas = num_gpus
            # compute minibatch size
            # 导读 batch_size 是所有数据并行副本合起来的一次更新的样本数。
            batch_size = num_replicas * local_bsz

            scale = batch_size / application.init_batch_size
            # scale 是为下一步计算统计效率准备的中间变量

            # Calculate true (simulated) efficiency.
            # 导读 根据 batch 和 epoch 查梯度统计，gain 表示本次更新相对参考 batch 的有效进展。
            grad_sqr, grad_var = application.get_grad_stats(
                batch_size, self.epoch)
            gain = (grad_var + grad_sqr) / (grad_var / scale + grad_sqr)

            # Calculate true (simulated) throughput.
            # query xput with atomic_bsz (bsz per pipeline replica)
            query_bsz = self.atomic_bsz
            # check if job is PMP
            if self.num_stages > 1:
                # query xput with local_bsz (bsz per pipeline replica)
                query_bsz = local_bsz
            # get throughput for current placement with query_bsz per replica
            # 导读 用执行中的实际放置和单次局部 batch 查询耗时，把该结果作为模拟观测。
            step_time, sync_time = application.get_throughput(
                placement, query_bsz)
            # compute time per accumulation step
            accum_time = step_time - sync_time
            # Update the estimated throughput/efficiency parameters.
            # 导读 把已发生的配置和耗时交回性能模型，供后续调度预测使用。
            self.update_params(num_nodes, num_replicas, query_bsz,
                               step_time, sync_time, grad_sqr, grad_var)
            # Calculate true (simulated) goodput.
            # do not count accum steps for PMP, only for DP jobs
            accum_steps = self.accum_steps if self.num_stages == 1 else 0

            # compute time per iter
            # 导读 每次更新包含一次完整步骤和若干额外的纯计算累积步。
            total_time = step_time + accum_time * accum_steps
            # 导读 这里的 goodput 用作训练进度每秒，基于本段模拟观测。策略候选评分来自独立的预测路径。
            goodput = gain / total_time * (1.0 - interference)

            # goodput multiplier
            # goodput = self.multiplier * goodput
            # slowdown for job
            goodput = goodput * self.calibration_factor

            # Update current epoch and progress.
            # 导读 next_progress 是下一个 epoch 边界的目标进度。与本段可增加的进度比较决定是否跨边界。
            next_progress = application.get_progress(self.epoch + 1)
            if self.progress + goodput * seconds < next_progress:
                # Used up the entire time interval without finishing an epoch.
                self.progress += goodput * seconds
                self.current_time += seconds
                self.attained_service += seconds * sum(self.placement)
                self.execution_time += seconds
                seconds = 0
            else:
                # Crossed an epoch boundary before finishing the time interval.
                # update epoch duration
                assert len(self.epoch_duration) == self.epoch
                #计算下一个epoch完成总共要花费的时间
                duration = round(float((application.get_progress(
                    self.epoch + 1) - application.get_progress(self.epoch)) / goodput))
                self.epoch_duration.append(duration)
                if self.epoch == application.max_epochs:
                    print(
                        f"Epoch durations: {self.name} -> {self.epoch_durations}")
                self.epoch += 1
                # 导读 delta 是走到当前 epoch 边界所需的模拟秒数。四舍五入后用于记录区间内完成时刻。
                #计算目前的进度走到下一个epoch边界的时间
                delta = round(float((next_progress - self.progress) / goodput))
                assert delta <= seconds
                # 导读 completion_epoch 是用于判断作业完成的 epoch 阈值。
                completion_epoch = application.get_completion_epoch(batch_size)
                self.completion_epoch = completion_epoch
                if self.epoch > completion_epoch:
                    # 导读 completion_time 写成 Job 时钟加 delta，保留区间内的实际模拟完成时刻。
                    self.completion_time = self.current_time + delta
                self.progress = next_progress
                self.best_metric = application.get_best_metric(
                    batch_size, self.epoch)
                self.current_time += delta
                self.attained_service += delta * sum(self.placement)
                self.execution_time += delta
                seconds -= delta
                # Re-scale batch size between epochs.
            # 导读 到达边界后可重新计算 batch，剩余 seconds 用新状态继续。
            self.update_local_bsz(self.placement)
        self.current_time += seconds  # Add any remaining time.

    # 导读 Job.reallocate（原源码第 524 行）
    # 导读 在当前机器类型内更新节点放置。配置变化时更新 batch、恢复计时和事件，返回重配记录。
    # 导读 参数 placement 表示每个已使用节点上的 GPU 或副本数。
    # 导读 函数体中的相关调用包括 self.update_local_bsz、self.rescale_events.append。
    # 导读 函数体直接赋值的实例字段包括 self.num_stages、self.placement、self.rescale_time、self.num_restarts、self.atomic_bsz。
    # 导读 从返回表达式可以继续追踪 new_rescale_event。
    def reallocate(self, placement):
        old_placement, new_placement = self.placement, tuple(placement)
        if placement:
            if old_placement != new_placement:
                # print(f"RESCALE: job: {self.name}, cluster: {self.current_cluster}, placement: {old_placement} -> {new_placement}")

                # Update placement, num_stages, batch size per replica
                self.num_stages = self.applications[self.current_cluster].num_stages
                self.placement = new_placement
                self.update_local_bsz(self.placement)

                # Start startup/re-scale countdown
                self.rescale_time = self.applications[self.current_cluster].rescale_time or 30
                if self.num_restarts is None:
                    # starting for the first time
                    self.num_restarts = 0
                else:
                    # restarting
                    self.num_restarts += 1
        else:
            # De-allocate all resources.
            # print(f"SUSPEND: job: {self.name}")
            self.placement = ()
            self.atomic_bsz = 0
        
        # Record a rescale event
        new_rescale_event = (self.name, self.current_time, self.current_cluster, old_placement, new_placement, self.rescale_time)
        self.rescale_events.append(new_rescale_event)
        return new_rescale_event

    # 导读 Job.migrate（原源码第 554 行）
    # 导读 变更作业所在机器类型及放置，更新恢复时间、迁移统计和事件记录，返回迁移记录。
    # 导读 参数 new_cluster 表示作业迁移后的机器类型，new_placement 表示变更后的逐节点卡数。
    # 导读 函数体中的相关调用包括 self.update_local_bsz、self.migrate_events.append。
    # 导读 函数体直接赋值的实例字段包括 self.current_cluster、self.placement、self.rescale_time、self.num_restarts、self.num_stages、self.atomic_bsz。
    # 导读 从返回表达式可以继续追踪 new_migrate_event。
    def migrate(self, new_cluster, new_placement):
        # set current cluster
        prev_cluster = self.current_cluster
        # print(f"MIGRATE:: {self.name}, cluster: {prev_cluster} -> {new_cluster}")
        # update current cluster
        self.current_cluster = new_cluster
        if new_placement:
            self.placement = tuple(new_placement)
            self.update_local_bsz(self.placement)
            # Start startup/re-scale countdown.
            self.rescale_time = self.applications[self.current_cluster].rescale_time or 30
            # print(f"RESCALE_TIME: {self.name} --> {new_cluster}, {self.rescale_time}")
            if self.num_restarts is None:
                self.num_restarts = 0
            else:
                self.num_restarts += 1
            # get num stages for this GPU type
            self.num_stages = self.applications[self.current_cluster].num_stages
        else:
            print(f"SUSPEND: job: {self.name}")
            # De-allocate all resources.
            self.placement = ()
            self.atomic_bsz = 0
        # Record a migrate event
        new_migrate_event = (self.name, self.current_time, prev_cluster, new_cluster, new_placement, self.rescale_time)
        self.migrate_events.append(new_migrate_event)
        return new_migrate_event