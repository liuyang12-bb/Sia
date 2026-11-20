"""Isolated Stage2C: fixed offline R0, E-legacy, the same L2 decision semantics.

Offline throughput is NEVER passed to Job.seed_profiles or a Goodput model.
Only CPU/RAM demand scales are replaced. F is a GPU-only single proposal,
not the Sia MIP. Historical modes and Stage2B guards remain available.
"""
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import simulator_config as sim_config
from .joint_profile import JointProfileOracle
from .joint_integration import JointCluster
from .neutral_liveness import L2Admission, NeutralL2Policy

APPLICATION_TEMPLATES={'bert':'bert_1.py','cifar10':'alexnet_1.py',
    'deepspeech2':'deepspeech_1.py','imagenet':'res50_1.py','yolov3':'mobilenet_1.py'}
REGISTRY_SHA256='78F676E7695CBE09AA9A3DB71F539FE2A2CEF1DBC243E57CC064E4E294DCF1C9'


class R0Oracle(JointProfileOracle):
    def __init__(self,registry_path,models,job_applications,response):
        if response not in ('neutral','medium'): raise ValueError('unregistered response')
        raw=Path(registry_path).read_bytes()
        self.registry_sha256=hashlib.sha256(raw).hexdigest().upper()
        if self.registry_sha256!=REGISTRY_SHA256: raise ValueError('R0_REGISTRY_HASH_MISMATCH')
        rows=json.loads(raw); self.registry={r['key']:r for r in rows}
        domain={f'{a}/{t}:1:{g}' for a in APPLICATION_TEMPLATES
                for t,gs in [('aws',(1,2,4)),('dgx-ext',(1,2,4,8)),('rtx',(1,2,4,8))] for g in gs}
        if len(rows)!=55 or set(self.registry)!=domain: raise ValueError('R0_DOMAIN_MISMATCH')
        self.job_applications=dict(job_applications)
        super().__init__(models,'fixed_reference',(3.,62.5),response=='neutral',0. if response=='neutral' else .5)
        for template in APPLICATION_TEMPLATES.values(): self.template(template)

    def reference(self,job_id,config_id):
        key=f'{self.job_applications.get(job_id)}/{config_id}'
        if key not in self.registry: raise ValueError(f'UNCOVERED_R0: {key}')
        return self.registry[key]

    def query(self,job,config,cpu_per_gpu,ram_gb_per_gpu,proportional_point):
        r=self.reference(job.job_id,config.config_id)
        if (config.cpu_demand_scale,config.memory_demand_scale)!=(r['phi_c'],r['phi_m']):
            raise ValueError('REGISTRY_SCALE_MISMATCH')
        if job.template_file!=r['template']: raise ValueError('R0_TEMPLATE_MISMATCH')
        return super().query(job,config,cpu_per_gpu,ram_gb_per_gpu,proportional_point)

    def provenance_digest(self):
        return hashlib.sha256((super().provenance_digest()+self.registry_sha256).encode()).hexdigest().upper()


class R0Admission(L2Admission):
    MAX_ACTIVE_JOBS=20
    REQUIRE_NEUTRAL=False


class R0Policy(NeutralL2Policy):
    ADMISSION_CLASS=R0Admission
    REQUIRE_NEUTRAL=False

    def _job_specs(self,jobs,base_allocations):
        specs,provenance=super()._job_specs(jobs,base_allocations)
        fixed=[]
        for job in specs:
            configs=[]
            for c in job.configs:
                r=self.oracle.reference(job.job_id,c.config_id)
                configs.append(replace(c,cpu_demand_scale=r['phi_c'],memory_demand_scale=r['phi_m']))
                provenance[job.job_id][c.config_id].update(cpu_demand_scale=r['phi_c'],memory_demand_scale=r['phi_m'],
                    response_scale_source='FIXED_R0_FULL_55_DOMAIN',registry_key=r['key'],
                    registry_sha256=self.oracle.registry_sha256,reference_state=r['reference_state'])
            fixed.append(replace(job,configs=tuple(configs)))
        return tuple(fixed),provenance

    def optimize(self,jobs,nodes,base_allocations):
        allocations=super().optimize(jobs,nodes,base_allocations)
        self.last_decision['shared_contract'].update(rho='R0_E_LEGACY',response_exponent=self.oracle.response_exponent,
            reference=[3.,62.5],registry_sha256=self.oracle.registry_sha256,max_active_jobs=20)
        # Per-round delta, not quadratic copies of all past event history.
        # Queue/services/seen/sequence are unchanged; logs retain every event.
        self.liveness.events=[]
        return allocations


class R0Cluster(JointCluster):
    def optimize_policy(self):
        self.policy.now=self.current_time
        self.policy.productive={j.name:j.execution_time for j in self.jobs if j.submission_time<=self.current_time}
        for j in self.jobs:
            for s in j.joint_step_segments:
                if s['seconds']>0 and (not math.isfinite(s['raw_rate_before_rho']) or s['raw_rate_before_rho']<=0):
                    raise RuntimeError(f'ZERO_EXECUTION_RATE: {j.name}')
                if not math.isclose(s['effective_rate'],s['raw_rate_before_rho']*s['rho'],rel_tol=1e-12):
                    raise RuntimeError(f'EXECUTION_RESPONSE_ONCE: {j.name}')
        return super().optimize_policy()


def make_cluster(config,workload,registry,models,cell,response,seed):
    sim_config.num_parallel_jobs=1; sim_config.num_parallel_traces=1
    sim_config.sia_solver_verbose=False; sim_config.preserve_ckpt_allocs=True
    cc=config['cluster']; rc=config['response']; sc=config['simulation']
    apps=dict(zip(workload.name,workload.application))
    oracle=R0Oracle(registry,models,apps,response)
    policy=R0Policy(mode=cell,oracle=oracle,node_capacities=cc['capacity_scenarios']['cpu_constrained'],
        template_map={j:APPLICATION_TEMPLATES[a] for j,a in apps.items()},
        aux_cpu_points=rc['aux_cpu_points'],aux_ram_points=rc['aux_ram_points'],
        minimum_cpu_per_gpu=rc['minimum_legal_profile_point']['cpu_per_gpu'],
        minimum_ram_gb_per_gpu=rc['minimum_legal_profile_point']['ram_gb_per_gpu'],
        p_fairness=sc['p_fairness'],restart_penalty=30,lambda_a=sc['lambda_a'],lambda_n=sc['lambda_n'],
        project_throughputs=sc['project_throughputs'],share_max_replicas=sc['share_max_replicas'],
        sched_interval=sc['interval_seconds'],enable_bsz_tuning=not sc['disable_batch_size_tuning'])
    policy.populate_valid_configs(cc['nnodes'],cc['ngpus_per_node'],cc['max_physical_nodes'])
    cluster=R0Cluster(workload.copy(),policy,cc['nnodes'],cc['ngpus_per_node'],cc['max_physical_nodes'])
    if seed:
        # Preserve the PRIOR frozen Sia experiment's official seeding convention;
        # never use R0 T0/phi as performance samples. Tests explicitly seed=False.
        for job in cluster.jobs: job.seed_profiles(sc['seed_profiles_num_nodes'],sc['seed_profiles_num_replicas'])
    return cluster
