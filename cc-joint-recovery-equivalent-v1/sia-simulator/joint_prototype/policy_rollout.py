"""Current-model forecast using the ACTUAL scheduler and cost/lock state machine.

Never calls Job.step/Cluster.step, true throughput, or future gradient data.
The proposed initial assignment is a hypothetical, independently legal boundary
intervention. All subsequent decisions use the unmodified parent FO/FULL path.
Only already-arrived, unfinished jobs are copied. Learned models stay frozen.
"""
import contextvars,copy,math
from types import SimpleNamespace
from dataclasses import asdict
from .completion_time import predict_job
from .joint_profile import GpuConfig
from .neutral_liveness import geometry
from .release_evaluation import ReleaseEvaluationMixin,evaluate_plan

CURRENT=contextvars.ContextVar('current_policy_rollout_cluster',default=None)
RATE_FACTORS=contextvars.ContextVar('prediction_rate_factors',default=None)

def validate_hypothetical(prediction_only,audit,columns_match):
    if not prediction_only:raise RuntimeError('HYPOTHETICAL_NOT_PREDICTION')
    if not audit.get('valid') or not columns_match:raise RuntimeError('HYPOTHETICAL_INVALID_INITIAL')
    return True

def advance_model(cluster,interval):
    start=cluster.current_time;events=[]
    for j in cluster.jobs:
        j.joint_step_segments=[]
        if j.completion_time is not None:continue
        if j.submission_time>start:raise ValueError('FUTURE_JOB_IN_PREDICTION')
        j.current_time=start+interval
        if not j.placement:continue
        delay=min(interval,j.rescale_time);j.rescale_time-=delay
        j.attained_service+=delay*sum(j.placement);j.wasted_gpu_seconds+=delay*sum(j.placement)
        time_left=interval-delay
        if not time_left:
            events.append(dict(job=j.name,time=start,checkpoint=delay,training=0.,raw_rate=None,rho=None));continue
        resource=cluster.policy.current_resources[j.name]
        typ,nn,g=resource['config_id'].split(':')
        config=SimpleNamespace(config_id=resource['config_id'],gpu_type=typ,num_nodes=int(nn),gpus=int(g))
        pred=predict_job(j,config);raw=pred['raw_rate'];rho=resource['rho']
        if raw is None:raise ValueError('UNKNOWN_MODEL_IN_PREDICTION')
        from .uncertainty import calibration_key
        key=calibration_key(j,config,pred)
        factors=RATE_FACTORS.get()
        if factors is not None:
            if key not in factors:raise ValueError('UNCOVERED_SCENARIO_KEY '+key)
            raw*=factors[key]
        if not math.isfinite(raw*rho) or raw*rho<=0:raise ValueError('INVALID_PREDICTION_RATE')
        work=max(0.,cluster.policy.declared_work[j.name]['total_progress']-j.progress)
        train=min(time_left,work/(raw*rho));j.progress+=train*raw*rho;j.execution_time+=train
        j.attained_service+=train*sum(j.placement);j.used_gpu_seconds+=train*sum(j.placement)
        j.joint_step_segments=[dict(seconds=train,raw_rate_before_rho=raw,rho=rho,effective_rate=raw*rho)]
        if work<=time_left*raw*rho+1e-8:j.completion_time=start+delay+train
        events.append(dict(job=j.name,time=start,checkpoint=delay,training=train,raw_rate=raw,rho=rho,calibration_key=key,
                           work_remaining=max(0.,work-train*raw*rho),completion=j.completion_time))
    cluster.current_time+=interval
    return events

def clone_current(cluster):
    # Applications contain immutable offline tables. Sharing them saves memory;
    # no forecasting path is allowed to query their future training throughput.
    memo={id(app):app for j in cluster.jobs for app in j.applications.values()}
    v=copy.deepcopy(cluster,memo)
    original_policy=getattr(v.policy,'_uncertainty_base_policy',None)
    if original_policy is not None:v.policy.__class__=original_policy
    v.jobs=[j for j in v.jobs if j.submission_time<=v.current_time and j.completion_time is None]
    for name in ('logs','joint_round_logs','rescale_events','migrate_events'):setattr(v,name,[])
    base=cluster.policy.liveness._rollout_base_class
    v.policy.liveness.__class__=base
    v.policy.liveness._prediction_only=True
    return v,base

def predict_with_policy(cluster,initial,interval):
    v,base=clone_current(cluster);origin=v.current_time
    initial_keys={c.job_id:geometry(c) for c in initial}
    initial_fingerprint={j.name:(j.progress,j.execution_time,j.rescale_time) for j in cluster.jobs}
    def force(self,s,columns,legacy,now):
        if not getattr(self,'_initial_pending',False):return legacy,None
        selected=[next(c for c in columns if geometry(c)==key) for key in self._initial_keys.values()]
        validate_hypothetical(self._prediction_only,s.audit(selected),
            {c.job_id for c in selected}=={c.job_id for c in legacy})
        self._initial_pending=False
        return selected,dict(status='HYPOTHETICAL_INITIAL',columns={c.job_id:c.column_id for c in selected})
    v.policy.liveness.__class__=type('HypotheticalInitial'+base.__name__,(base,),{'refine_completion_bundle':force})
    v.policy.liveness._initial_pending=True;v.policy.liveness._initial_keys=initial_keys
    traces=[]
    def decide():
        # Outer real optimize already owns the identical sparse/engineering
        # adapters. Re-entering their global patch stack would inspect a
        # patched function as source. Reuse that active stack, not a new one.
        from core80 import EfficientTraceMixin
        token=CURRENT.set(v)
        try:super(EfficientTraceMixin,v).optimize_policy()
        finally:CURRENT.reset(token)
        d=v.policy.last_decision;a=d['result']['audit']
        if not a['valid']:raise RuntimeError('PREDICTED_COMMIT_INVALID')
        for j,ledger in a.get('service_delay_ledgers',{}).items():
            limit=v.policy.liveness.delay_bound+a.get('tail_grants',{}).get(j,{}).get('extra_budget',0.)
            if ledger['delay_charged']>limit+1e-7:raise RuntimeError('PREDICTED_BUDGET_INVALID')
        traces.append(dict(time=v.current_time-origin,selected=copy.deepcopy(d['result']['selected']),
            recovery=copy.deepcopy(a.get('tail_recoveries',[])),locks=a.get('checkpoint_locks',[]),
            eligibility=copy.deepcopy(a.get('shared_eligibility',{})),
            ledgers=copy.deepcopy(a.get('service_delay_ledgers',{})),tail_grants=copy.deepcopy(a.get('tail_grants',{}))))
    decide();executions=[]
    # The same absolute 6000-round observation window; no extrapolated JCT.
    remaining=max(0,6000-int(origin/interval))
    for _ in range(remaining):
        executions.extend(advance_model(v,interval))
        if all(j.completion_time is not None for j in v.jobs):break
        decide()
        v.logs.clear();v.joint_round_logs.clear();v.rescale_events.clear();v.migrate_events.clear()
    if not all(j.completion_time is not None for j in v.jobs):return dict(status='PREDICTION_CENSORED')
    assert initial_fingerprint=={j.name:(j.progress,j.execution_time,j.rescale_time) for j in cluster.jobs}
    done={j.name:j.completion_time-origin for j in v.jobs}
    return dict(status='OK',sum_completion=sum(done.values()),completion_times=done,events=traces,
        model_execution=executions,continuation='ACTUAL_PARENT_FO_FULL_OPTIMIZE_POLICY',
        shared_candidate_lock_budget_rules=True,future_arrivals_read=False,
        future_profile_observations_read=False,learned_rate_model='FROZEN_CURRENT_PARAMETERS',global_optimum=False)

class PolicyRolloutMixin(ReleaseEvaluationMixin):
    def evaluate_completion_plan(self,s,columns,scores,bundle,queue,interval,context):
        checked=evaluate_plan(s,columns,scores,bundle,queue,interval)
        if checked['status']!='OK':return checked
        cluster=CURRENT.get()
        if cluster is None:raise RuntimeError('MISSING_LIVE_CLUSTER_CONTEXT')
        key=tuple(sorted(geometry(c) for c in bundle))
        if key not in self._rollout_cache:self._rollout_cache[key]=predict_with_policy(cluster,bundle,interval)
        return copy.deepcopy(self._rollout_cache[key])
    def refine_completion_bundle(self,*args):
        self._rollout_cache={}
        try:
            selected,evidence=super().refine_completion_bundle(*args)
            evidence['version']='SHARED_POLICY_COMPLETION_ROLLOUT_V3'
            return selected,evidence
        finally:self._rollout_cache={}

def install(cluster):
    base=cluster.policy.liveness.__class__
    cluster.policy.liveness.__class__=type('PolicyRollout'+base.__name__,(PolicyRolloutMixin,base),{'_rollout_base_class':base})
    old=cluster.__class__
    class ContextCluster(old):
        def optimize_policy(self):
            token=CURRENT.set(self)
            try:return super().optimize_policy()
            finally:CURRENT.reset(token)
    cluster.__class__=ContextCluster
    return cluster
