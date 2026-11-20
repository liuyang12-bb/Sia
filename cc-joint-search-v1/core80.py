"""Opt-in80-job scale and diagnostics adapter; historical files unchanged."""
import collections,copy,hashlib,json,math,sys,time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'sia-simulator')]
from joint_prototype.clean_ablation import CleanAdmission,CleanPolicy,install as install_clean
from joint_prototype.small_mechanism import EvidenceMixin,ProbeSia
from joint_prototype.completion_time import install_aux
from joint_prototype.completion_continuity import install as install_completion
from joint_prototype.tail_recovery import install as install_tail
from joint_prototype.physical_domain import restrict_single_node_domain
from joint_prototype.full160 import make_cluster
from breakpoint_study import install as install_breakpoints
from exact_optional_proof import ExactOptionalProofMixin
from joint_prototype.bounded_correction import BoundedCorrectionMixin

METHODS=('SIA_P','FULL','SIA_AUX','PROP','CPU','MEM','FO','A00','A01','A10','A11',
         'LAZY_FO','LOCAL')
GPUS={'aws':4,'dgx-ext':8,'rtx':8}

def compact_column(c):
    """Logging-only serialization, no optimizer reads these dictionaries."""
    return dict(column_id=c.column_id,job_id=c.job_id,config=asdict(c.config),
        placement=c.placement.node_gpus,cpu_per_gpu=c.query.cpu_per_gpu,
        ram_gb_per_gpu=c.query.ram_gb_per_gpu,rho=c.query.rho,
        scheduling_utility=c.scheduling_utility,query_round=c.query_round,
        candidate_kind=c.candidate_kind,node_usage=c.node_usage)

def ledger_view(live):
    if live is None:return None
    return dict(ready=list(live.ready),services={j:{k:(compact_column(v) if k=='column' else v)
        for k,v in h.items()} for j,h in live.services.items()})

def efficient_evidence(self,jobs,nodes,base):
    # Exact parent decision, skip three UNUSED diagnostic enumerations from
    # EvidenceMixin.optimize. The actual exchange/aux candidate list is still
    # recorded by its original wrapper after this call returns.
    previous=copy.deepcopy(self.current_resources);live=getattr(self,'liveness',None)
    before=ledger_view(live)
    result=super(EvidenceMixin,self).optimize(jobs,nodes,base)
    self.last_decision['probe']=dict(specs=[asdict(j) for j in self.probe_specs],
        provenance=copy.deepcopy(self.probe_provenance),
        job_state={j:dict(age=v.age,num_restarts=v.num_restarts,used_gpu_seconds=v.used_gpu_seconds,wasted_gpu_seconds=v.wasted_gpu_seconds) for j,v in jobs.items()},
        previous_resources=previous,l2_before=before,l2_after=ledger_view(getattr(self,'liveness',None)),
        joint_columns=[],proportional_columns=[],floor_rejected=[],
        restart_factors={j.job_id:{c.config_id:self._restart_factor(j.job_id,jobs[j.job_id],c.config_id,j.current_config_id) for c in j.configs} for j in self.probe_specs},
        contract=dict(p=self.p_fairness,lambda_n=self.lambda_n,allocation_bonus=0,progress_floor=False,response=self.oracle.response_exponent),
        diagnostic_enumeration='UNUSED_HISTORICAL_REBUILD_SKIPPED; actual candidate set untouched')
    return result

class EfficientTraceMixin:
    def optimize_policy(self):
        from joint_prototype import exchange_netprogress,completion_time
        from scale_storage import storage_context
        from joint_prototype.joint_scheduler import JointScheduler,_proportional_point
        original_columns=JointScheduler.build_joint_columns
        policy=self.policy
        started = time.perf_counter()
        previous_decision = getattr(policy, 'last_decision', None)
        if getattr(policy, '_search_mode', None) is not None:
            policy._search_started = started
        def existing_cold_filter_early(s,*args,**kwargs):
            # Completion scoring ALREADY discards non-proportional unknown-ETA
            # columns. Move only this existing dead-path filter before object
            # construction. Restore the complete trusted grid before auditing.
            from joint_prototype.completion_time import predict_job
            if getattr(s, 'lazy_search', None) is not None:
                return original_columns(s,*args,**kwargs)
            live=getattr(policy,'completion_jobs',{})
            if not live:return original_columns(s,*args,**kwargs)
            cold={(j.job_id,c.config_id) for j in s.jobs for c in j.configs
                if predict_job(live[j.job_id],c)['raw_rate'] is None}
            if not cold:return original_columns(s,*args,**kwargs)
            provider=getattr(s,'resource_point_provider',None)
            def points(current,j,c,p):
                if (j.job_id,c.config_id) in cold and not current.lock_active(j):
                    cpu,ram=current.allocation_floor(_proportional_point(p,current.node_map))
                    return (cpu,),(ram,)
                return provider(current,j,c,p) if provider else (current.aux_cpu_points,current.aux_ram_points)
            s.resource_point_provider=points
            try:return original_columns(s,*args,**kwargs)
            finally:
                if provider is None:del s.resource_point_provider
                else:s.resource_point_provider=provider
        with ExitStack() as stack:
            if getattr(self.policy,'_engineering_repair',False):
                from engineering_repair import repair_context
                stack.enter_context(repair_context(self.policy.oracle))
            else:stack.enter_context(storage_context(self.policy.oracle))
            stack.enter_context(patch.object(JointScheduler,'build_joint_columns',existing_cold_filter_early))
            stack.enter_context(patch.object(EvidenceMixin,'optimize',efficient_evidence))
            stack.enter_context(patch.object(exchange_netprogress,'asdict',compact_column))
            stack.enter_context(patch.object(completion_time,'asdict',compact_column))
            result = super().optimize_policy()
        if policy.last_decision is not previous_decision:
            policy.last_decision['engineering_decision_seconds'] = time.perf_counter() - started
        return result

class Core80Admission(ExactOptionalProofMixin,CleanAdmission):
    MAX_ACTIVE_JOBS=80
    def _decide(self,s,columns,now,productive):
        mode=getattr(self,'resource_factor','FULL')
        # Post-score restriction: allfour resource factors share the complete
        # universe normalization. Genuine checkpoint geometry/quotas stay exact.
        if mode in ('CPU','MEM'):
            columns=[c for c in columns if s.lock_active(s.job_map[c.job_id]) or
                (abs(c.query.ram_gb_per_gpu-c.proportional_ram_per_gpu)<1e-8 if mode=='CPU' else
                 abs(c.query.cpu_per_gpu-c.proportional_cpu_per_gpu)<1e-8)]
        result=super()._decide(s,columns,now,productive)
        if result.audit.get('search', {}).get('local', {}).get('local_status') == 'accepted':
            result.status = 'FEASIBLE_LOCAL_IMPROVEMENT'
        result.audit.update(scale80_active_jobs=len(s.jobs),scale80_candidate_columns=len(columns),resource_factor=mode)
        return result

class Core80Policy(CleanPolicy):
    def optimize(self,*args):
        result=super().optimize(*args)
        self.last_decision['shared_contract']['max_active_jobs']=80
        self.last_decision['clean_contract']['resource_factor']=self.liveness.resource_factor
        if hasattr(self,'_repair_factor_certificate'):self.last_decision['engineering_repair']=self._repair_factor_certificate
        search = self.last_decision['result']['audit'].get('search')
        if search is not None:
            elapsed = time.perf_counter() - self._search_started
            search.update(decision_seconds=elapsed,
                          budget_exhausted=elapsed >= self._search_budget_seconds)
        return result

class CorrectedCore80Admission(BoundedCorrectionMixin,Core80Admission):
    pass

def install80(c,method):
    if method.startswith('A'):
        install_clean(c,'A',int(method[1]),int(method[2]),'joint','prop')
    else:install_clean(c,'B',1,1,'fixed' if method=='FO' else 'joint','prop' if method=='PROP' else 'opt')
    c.policy.__class__=Core80Policy;c.policy.liveness.__class__=Core80Admission
    c.policy.liveness.resource_factor=method
    if method in ('FO','FULL'):
        c.policy.liveness.__class__=CorrectedCore80Admission
    return c

def scaled_installer80(c):
    """Same original installation, explicit tested80 limit instead of20.

    No jobs are hidden from the installer; its normal single-node, empty-ledger
    checks still execute. Core80Admission is attached before the first decision.
    """
    from joint_prototype.adaptive_probe import install as adaptive_install
    from joint_prototype.scaled_adaptive import ScaledJoint,ScaledAdmission
    if not 1<=len(c.jobs)<=80:raise ValueError('CORE80_INSTALLER_SCOPE')
    adaptive_install(c)
    c.policy.__class__=ScaledJoint;c.policy.liveness.__class__=ScaledAdmission
    return c

def saturation_per_gpu():
    from joint_prototype.joint_profile import load_synergy_template,CPU_COORDS,RAM_COORDS
    import numpy as np
    rows=json.loads((ROOT/'inputs/registry.json').read_text());out={t:dict(cpu=3.,ram=62.5) for t in GPUS};evidence=[]
    for r in rows:
        if r['application'] not in ('cifar10','deepspeech2','bert'):continue
        typ,nn,g=r['config_id'].split(':')
        if int(nn)!=1 or int(g)>GPUS[typ]:continue
        a=load_synergy_template(ROOT/'inputs/models'/r['template']).matrix;peak=float(a.max())
        cpu=float(CPU_COORDS[np.flatnonzero(a[:,-1]>=peak-1e-12)[0]])*r['phi_c']
        ram=float(RAM_COORDS[np.flatnonzero(a[-1,:]>=peak-1e-12)[0]])*r['phi_m']
        out[typ]['cpu']=max(out[typ]['cpu'],cpu);out[typ]['ram']=max(out[typ]['ram'],ram)
        evidence.append(dict(key=r['key'],template=r['template'],peak=peak,cpu_saturation=cpu,ram_saturation=ram,phi_c=r['phi_c'],phi_m=r['phi_m']))
    return out,evidence

def build(method,response='medium',scenario='dual',workload_path=None,config_path=None,seed=True,efficient=True,topology=None,engineering=True,search_budget_seconds=5.0,recovery_precheck=True):
    import pandas as pd
    if method not in METHODS or scenario not in ('dual','cpu_only','ram_only'):raise ValueError('UNREGISTERED_CORE80_CELL')
    search_mode = method if method in ('LAZY_FO', 'LOCAL') else None
    if search_mode is not None:
        if not efficient or not engineering:
            raise ValueError('DEFERRED_SEARCH_REQUIRES_ENGINEERING_CONTEXT')
        if not math.isfinite(search_budget_seconds) or search_budget_seconds <= 0:
            raise ValueError('INVALID_SEARCH_BUDGET')
        method = 'FO'
    cfg=json.loads(Path(config_path or ROOT/'inputs/core80_config.json').read_text(encoding='utf-8'));cfg=copy.deepcopy(cfg)
    if topology is not None:
        cfg['cluster'].update(nnodes={t:1 for t in topology},ngpus_per_node=dict(topology),max_physical_nodes={t:1 for t in topology})
    sizes=cfg['cluster']['ngpus_per_node'];caps={t:dict(cpu_cores=3*g,ram_gb=62.5*g) for t,g in sizes.items()}
    if scenario!='dual':
        sat,_=saturation_per_gpu()
        for t,g in sizes.items():
            if scenario=='cpu_only':caps[t]['ram_gb']=sat[t]['ram']*g
            else:caps[t]['cpu_cores']=sat[t]['cpu']*g
    cfg['cluster']['capacity_scenarios']={'main':caps}
    c=make_cluster(cfg,pd.read_csv(workload_path or ROOT/'inputs/core80_workload.csv'),ROOT/'inputs/registry.json',ROOT/'inputs/models',
        'sia_p' if method.startswith('SIA') else 'joint_o',response,'main',seed)
    c.policy.node_capacities=caps;restrict_single_node_domain(c.policy)
    c.policy.aux_cpu_points=tuple(sorted(set(c.policy.aux_cpu_points)|{8.}))
    if method=='SIA_P':c.policy.__class__=ProbeSia
    elif method=='SIA_AUX':
        install_aux(c);install_breakpoints(c,'CC_FULL')
    else:
        from joint_prototype import bundle_adaptation
        with patch.object(bundle_adaptation,'install_scaled',scaled_installer80):
            install_completion(c,proportional=method.startswith('A'))
        install_tail(c)
        if not method.startswith('A'):install_breakpoints(c,'CC_FULL')
        install80(c,method)
    if efficient:c.__class__=type('Core80'+c.__class__.__name__,(EfficientTraceMixin,c.__class__),{})
    c.policy._engineering_repair=bool(engineering)
    if search_mode is not None:
        c.policy._search_mode = search_mode
        c.policy._search_budget_seconds = float(search_budget_seconds)
        c.policy._recovery_precheck = search_mode == 'LAZY_FO' and bool(recovery_precheck)
    return c
