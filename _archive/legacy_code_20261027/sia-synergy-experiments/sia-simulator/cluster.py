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
    def __init__(self, workload, policy, num_nodes, ngpus_per_node,
                 max_physical_nodes, auxiliary_allocator=None):
        print(f"Simulated cluster config: \nnnodes: {num_nodes}, ngpus_per_node:{ngpus_per_node}")
        # workload is a pandas dataframe
        self.workload = workload
        # policy is a Policy object
        self.policy = policy
        self.num_nodes, self.ngpus_per_node = num_nodes, ngpus_per_node
        self.current_time = 0
        self.clusters = list(self.num_nodes.keys())
        self.max_physical_nodes = max_physical_nodes
        self.auxiliary_allocator = auxiliary_allocator
        self.rescale_events, self.migrate_events = [], []
        self.sia_gpu_decision_changes = 0
        self.auxiliary_resource_seconds = {
            cluster: {
                str(node): {"cpu_core_seconds": 0.0, "ram_gb_seconds": 0.0}
                for node in range(num_nodes[cluster])
            }
            for cluster in num_nodes
        }

        # create inverse map like {app-name: {cluster_name: cluster_app}}
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
            self.jobs = []
            adaptive_jobnames = list()
            rigid_jobnames = list()
            scale_gpu_jobnames = list()
            for row in workload.itertuples():
                category = row.category if 'category' in workload.columns else None
                cur_job = Job(row.name, self.cluster_applications[row.application],
                              row.time, cache_speedups=cache_speedups,                             target_num_replicas=row.num_replicas,
                              target_batch_size=row.batch_size, category=category)
                if category is not None:
                    if category == 'scale-gpus':
                        scale_gpu_jobnames.append(cur_job.name)
                        cur_job.disable_bsz_tuning()
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
        self.allocations = {}
        self.logs = []
        self.utility = []
        self.compute_throughput_ratios(init=False)
        self.compute_throughputs()

        # multi-processing for job.step() in parallel
        self.process_pool = Pool(sim_config.num_parallel_jobs) if sim_config.num_parallel_jobs > 1 else None

    # disable bsz tuning for MIP
    def disable_bsz_tuning(self):
        self.enable_bsz_tuning = False
        for job in self.jobs:
            job.disable_bsz_tuning()

    # simulate one scheduling round
    def step(self, seconds=60):
        # advance job state for `seconds` seconds
        start_t = time.time()
        self._account_auxiliary_resources(seconds)
        if self.process_pool:
            self.step_jobs_parallel(seconds)
        else:
            self.step_jobs_sequential(seconds)
        end_t = time.time()
        print(f"Job Step time: {(end_t - start_t)*1000}ms")

        # invoke scheduler policy to optimize allocations for next round
        # returns a list of changes to allocations as rescale and migrate events for logging
        step_rescale_events, step_migrate_events = self.optimize_policy()
        
        # log cluster state and metrics
        self.rescale_events.append(step_rescale_events)
        self.migrate_events.append(step_migrate_events)
        self.step_log()

    def step_jobs_sequential(self, seconds=60):
        # interference check disabled
        check_interference = False
        # TODO: fix interference check code
        if check_interference:
            interfere_nodes = set(idx for idx in range(self.num_nodes)
                                  if sum(len(set(val)) > 1 and idx in val
                                         for key, val in self.allocations.items()) > 1)
        for job in self.jobs:
            if check_interference:
                alloc_set = set(self.allocations.get(job.name, [])[1])
                interference = 0.0 if len(alloc_set) > 1 and any(
                    idx in interfere_nodes for idx in alloc_set) else 0.0
            else:
                interference = 0.0
            auxiliary_multiplier = (self.auxiliary_allocator.multiplier_for(job.name)
                                    if self.auxiliary_allocator else 1.0)
            job.step(seconds, interference=interference,
                     auxiliary_multiplier=auxiliary_multiplier)

        self.current_time += seconds

    # WARNING:: disables batch size & interference check (see other version above)
    def step_jobs_parallel(self, seconds=60):
        self.jobs = self.process_pool.starmap(
            _helper_step_job_single,
            [(job, seconds,
              self.auxiliary_allocator.multiplier_for(job.name)
              if self.auxiliary_allocator else 1.0)
             for job in self.jobs])
        self.current_time += seconds

    def _account_auxiliary_resources(self, seconds):
        if not self.auxiliary_allocator:
            return
        for cluster, nodes in self.auxiliary_allocator.last_result["node_usage"].items():
            for node, usage in nodes.items():
                counters = self.auxiliary_resource_seconds[cluster][node]
                counters["cpu_core_seconds"] += usage["cpu"] * seconds
                counters["ram_gb_seconds"] += usage["ram_gb"] * seconds

    def optimize_policy(self):
        job_infos = self.get_job_infos()
        policy_optim_time, alloc_event_time = 0, 0
        if not job_infos:
            return [], []
        
        # Optimize allocations.
        node_infos = self.get_node_infos()
        self.allocations = {
            k: v for k, v in self.allocations.items() if k in job_infos}

        # run scheduling policy
        p_start_t = time.time()
        allocations = self.policy.optimize(
            job_infos, node_infos, self.allocations)
        policy_optim_time = time.time() - p_start_t
        # collect any change in allocations
        # rescale: change in ngpus, migrate: change in ngpus and/or cluster
        p_start_t = time.time()
        rescale_events, migrate_events = [], []
        decision_changes = 0
        for job in self.jobs:
            if job.submission_time <= self.current_time and job.completion_time is None:
                job.contention.append(len(job_infos))
                # print(f"job: {job.name}, contention: {job.contention}")
                if isinstance(self.policy, SiaPolicy) or isinstance(self.policy, GavelPolicy) or isinstance(self.policy, SiaFixPolicy):
                    cluster, alloc = allocations.get(job.name, (None, ()))
                    if isinstance(self.policy, SiaFixPolicy) or isinstance(self.policy, GavelPolicy):
                        assert len(alloc) == 0 or len(
                            alloc) == job.target_num_replicas, f'{job.name} target_num_replicas: {job.target_num_replicas} alloc: {alloc}'
                    # change in resources
                    if allocations.get(job.name) != self.allocations.get(job.name):
                        decision_changes += 1
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
                        if job.current_cluster != cluster:
                            # migrated to another cluster
                            migrate_events.append(job.migrate(cluster, placement))
                        elif self.allocations.get(job.name) != alloc:
                            # reallocated within same cluster
                            rescale_events.append(job.reallocate(placement))
                else:
                    assert False, "other policies not implemented"
        # make a copy of allocations
        self.allocations = allocations
        self.sia_gpu_decision_changes += decision_changes
        if self.auxiliary_allocator:
            jobs = {job.name: job for job in self.jobs}
            self.auxiliary_allocator.allocate(
                self.allocations, jobs, self.current_time)
        alloc_event_time = time.time() - p_start_t
        print(f"Policy optimization took {policy_optim_time*1000:.2f}ms, reallocating+migrating jobs took {alloc_event_time*1000:.2f}ms")
        if sim_config.log_sched_decisions:
            print(f"Rescale events: {rescale_events}")
            print(f"Migrate events: {migrate_events}")
        return rescale_events, migrate_events

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
                        ratio = mean_src_atomic_xput / mean_dest_atomic_xput
                        ratios[dest_cluster] = ratio
                    else:
                        ratios[dest_cluster] = 1.0
                app_xput_ratios[cluster] = ratios
            self.cluster_throughput_ratios[app_name] = app_xput_ratios
        if sim_config.log_cluster_verbose:
            print(f"Throughput ratios: {self.cluster_throughput_ratios}")

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


    def step_log(self):
        def get_bsz(job):
            ngpus = int(sum(job.placement))
            num_replicas = ngpus
            # if job is a pipeline, divide by number of stages
            if job.num_stages > 1:
                num_replicas //= job.num_stages
            return job.atomic_bsz * (job.accum_steps + 1) * num_replicas
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
        if self.auxiliary_allocator:
            step_log["synthetic_resource_model"] = self.auxiliary_allocator.last_result
            step_log["sia_gpu_decision_changes"] = self.sia_gpu_decision_changes
            for job_record in step_log["submitted_jobs"]:
                resource_record = self.auxiliary_allocator.last_result["jobs"].get(
                    job_record["name"])
                job_record["auxiliary_multiplier"] = (
                    resource_record["multiplier"] if resource_record else 1.0)
                job_record["auxiliary_profile"] = (
                    resource_record["profile"] if resource_record else None)
        self.logs.append(step_log)

    def get_job_infos(self):
        job_infos = {}
        for job in self.jobs:
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

    def get_sia_multi_job_info(self, job):
        speedup_fns = {cname: job.get_speedup_fn(
            cname) for cname in self.clusters}
        scale_factor = 2
        max_replicas = {cname: min(max(scale_factor * job.max_profiled_replicas(cname), 1), 64,  # simulator can't handle more.
                        job.applications[cname].max_batch_size // job.applications[cname].min_local_bsz)
                        for cname in self.clusters}
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

    def get_node_infos(self, num_nodes=None):
        cluster_node_info = dict()
        for cluster_name in self.clusters:
            cluster_info = {idx: NodeInfo({"nvidia.com/gpu": self.ngpus_per_node[cluster_name]}, preemptible=False)
                            for idx in range(self.num_nodes[cluster_name])}
            cluster_node_info[cluster_name] = cluster_info
        return cluster_node_info

    def all_complete(self):
        return all(job.completion_time is not None for job in self.jobs)

    def output_logs(self, path):
        with open(path, "w") as f:
            for record in self.logs:
                json.dump(record, f)
                f.write("\n")

    def get_synthetic_metrics(self):
        if not self.auxiliary_allocator:
            raise ValueError("synthetic resource model is not enabled")
        completed = [job for job in self.jobs if job.completion_time is not None]
        jcts = np.asarray([
            job.completion_time - job.submission_time for job in completed
        ], dtype=float)
        first_submission = min((job.submission_time for job in self.jobs), default=0.0)
        last_completion = max((job.completion_time for job in completed), default=first_submission)
        cpu_core_seconds = sum(
            values["cpu_core_seconds"]
            for nodes in self.auxiliary_resource_seconds.values()
            for values in nodes.values())
        ram_gb_seconds = sum(
            values["ram_gb_seconds"]
            for nodes in self.auxiliary_resource_seconds.values()
            for values in nodes.values())
        per_node_utilization = {}
        for cluster, nodes in self.auxiliary_resource_seconds.items():
            capacity = self.auxiliary_allocator.capacities[cluster]
            per_node_utilization[cluster] = {}
            for node, values in nodes.items():
                elapsed = max(float(self.current_time), 1.0)
                per_node_utilization[cluster][node] = {
                    "cpu": values["cpu_core_seconds"] /
                           (float(capacity["cpu_per_node"]) * elapsed),
                    "ram": values["ram_gb_seconds"] /
                           (float(capacity["ram_gb_per_node"]) * elapsed),
                }
        jobs = []
        for job in self.jobs:
            jobs.append({
                "name": job.name,
                "submission_time": job.submission_time,
                "completion_time": job.completion_time,
                "jct_seconds": (job.completion_time - job.submission_time
                                if job.completion_time is not None else None),
                "num_restarts": job.num_restarts or 0,
                "num_migrations": job.num_migrations or 0,
                "used_gpu_seconds": job.used_gpu_seconds,
                "wasted_gpu_seconds": job.wasted_gpu_seconds,
                "synthetic_profile": self.auxiliary_allocator.profile_for(job.name),
            })
        result = self.auxiliary_allocator.metadata()
        result.update({
            "completed_jobs": len(completed),
            "total_jobs": len(self.jobs),
            "mean_jct_seconds": float(jcts.mean()) if jcts.size else None,
            "p95_jct_seconds": float(np.percentile(jcts, 95)) if jcts.size else None,
            "p99_jct_seconds": float(np.percentile(jcts, 99)) if jcts.size else None,
            "makespan_seconds": float(last_completion - first_submission),
            "simulation_end_time_seconds": float(self.current_time),
            "gpu_hours": sum(job.used_gpu_seconds + job.wasted_gpu_seconds
                             for job in self.jobs) / 3600.0,
            "productive_gpu_hours": sum(job.used_gpu_seconds
                                        for job in self.jobs) / 3600.0,
            "restart_gpu_hours": sum(job.wasted_gpu_seconds
                                     for job in self.jobs) / 3600.0,
            "cpu_core_hours": cpu_core_seconds / 3600.0,
            "ram_gb_hours": ram_gb_seconds / 3600.0,
            "per_node_utilization": per_node_utilization,
            "restart_count": sum(job.num_restarts or 0 for job in self.jobs),
            "migration_count": sum(job.num_migrations or 0 for job in self.jobs),
            "sia_gpu_decision_changes": self.sia_gpu_decision_changes,
            "capacity_violation_count": len(
                self.auxiliary_allocator.last_result["capacity_violations"]),
            "jobs": jobs,
        })
        return result

    def get_jcts(self):
        if len(self.logs) > 0:
            return {
                val["name"]: val["completion_time"] - val["submission_time"]
                for val in self.logs[-1]["submitted_jobs"]
                if val["completion_time"] is not None
            }
        else:
            return {}

def _helper_step_job_single(job, seconds, auxiliary_multiplier=1.0):
    job.step(seconds=seconds, interference=0,
             auxiliary_multiplier=auxiliary_multiplier)
    # important to return job object as all changes are process-local
    return job
