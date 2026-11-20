"""Pre-decision residual envelope and paired endpoint stress screen.

An empirical screen, NOT a confidence interval or global robust guarantee.
Missing support never becomes a factor of one. Actual execution is unchanged.
"""
import copy,itertools,json,math
from types import SimpleNamespace
from .completion_time import predict_job
from .policy_rollout import PolicyRolloutMixin,CURRENT,RATE_FACTORS,predict_with_policy

def calibration_key(job,config,pred):
    return json.dumps([job.app_name,config.config_id,int(pred['atomic_bsz']),int(pred['accum_steps']),int(job.epoch)],separators=(',',':'))

class ResidualBank:
    def __init__(self):self.rows=[];self.identities=set()
    def add(self,key,origin,end,ratio,now,job):
        if not origin<end<=now or not math.isfinite(ratio) or ratio<=0:raise ValueError('INVALID_OR_FUTURE_LABEL')
        identity=(job,origin)
        if identity in self.identities:raise ValueError('DUPLICATE_CALIBRATION_INTERVAL')
        self.identities.add(identity);self.rows.append(dict(key=key,origin=origin,observed_end=end,observed_at=now,ratio=ratio,job=job))
    def bounds(self,key,now):
        rows=[r for r in self.rows if r['key']==key and r['observed_end']<=now and r['observed_at']<=now]
        origins={r['origin'] for r in rows}
        if len(origins)<2:return None
        return dict(low=min([1.]+[r['ratio'] for r in rows]),high=max([1.]+[r['ratio'] for r in rows]),
                    count=len(rows),distinct_intervals=len(origins),latest_observation=max(r['observed_end'] for r in rows))

def endpoint_screen(bounds,evaluate):
    keys=sorted(bounds);checked=[];worst=math.inf
    for values in itertools.product(*(sorted({bounds[k]['low'],bounds[k]['high']}) for k in keys)):
        world=dict(zip(keys,values))
        try:gain=evaluate(world)
        except (ValueError,RuntimeError) as e:
            return dict(accepted=False,reason=str(e),worlds=len(checked),checked=checked)
        checked.append(dict(factors=world,gain=gain));worst=min(worst,gain)
        if not math.isfinite(gain) or gain<=1e-7:
            return dict(accepted=False,reason='NONPOSITIVE_ENDPOINT_GAIN',worlds=len(checked),checked=checked)
    return dict(accepted=True,reason='ALL_EMPIRICAL_ENDPOINTS_POSITIVE',worlds=len(checked),worst_gain=worst,checked=checked)

def prepare(cluster):
    pending={}
    for j in cluster.jobs:
        resource=cluster.policy.current_resources.get(j.name)
        if j.completion_time is not None or not j.placement or resource is None:continue
        typ,nn,g=resource['config_id'].split(':');cfg=SimpleNamespace(config_id=resource['config_id'],gpu_type=typ,num_nodes=int(nn),gpus=int(g))
        pred=predict_job(j,cfg)
        if pred['raw_rate'] is None:continue
        if pred['atomic_bsz']!=j.atomic_bsz or pred['accum_steps']!=j.accum_steps:continue
        pending[j.name]=dict(key=calibration_key(j,cfg,pred),origin=cluster.current_time,epoch=j.epoch,raw=pred['raw_rate'])
    return pending

def observe(cluster):
    pending=getattr(cluster,'_calibration_pending',None)
    if pending is None:return
    for j in cluster.jobs:
        p=pending.get(j.name);segments=j.joint_step_segments
        if not p or not segments:continue
        seg=segments[0]
        if seg['epoch_before']!=p['epoch'] or seg['seconds']<=0:continue
        cluster._residual_bank.add(p['key'],p['origin'],seg['start_time']+seg['seconds'],
            seg['raw_rate_before_rho']/p['raw'],cluster.current_time,j.name)
        cluster._residual_bank.rows[-1].update(predicted_raw=p['raw'],observed_raw=seg['raw_rate_before_rho'],
            observed_at=cluster.current_time,epoch_before=seg['epoch_before'],segment_start=seg['start_time'],
            source='PRE_INTERVAL_MODEL_AND_FIRST_REAL_TRAINING_SEGMENT')
    cluster._calibration_pending=None

class ReliableMixin(PolicyRolloutMixin):
    def refine_completion_bundle(self,s,columns,legacy,now):
        try:chosen,ev=super().refine_completion_bundle(s,columns,legacy,now)
        except (ValueError,RuntimeError) as error:
            if not s.audit(legacy)['valid']:raise
            return legacy,dict(version='PREDECISION_EMPIRICAL_UNCERTAINTY_V4',status='PREDICTION_FAILURE_KEEP_LEGACY',
                selected_plan='LEGACY',uncertainty=dict(accepted=False,reason='POINT_FORECAST_FAILURE',error=repr(error)))
        ev['version']='PREDECISION_EMPIRICAL_UNCERTAINTY_V4'
        if ev['status']!='OK':
            ev['uncertainty']=dict(reason='UNKNOWN_POINT_MODEL',accepted=False);return chosen,ev
        if ev['selected_plan']=='LEGACY':
            ev['uncertainty']=dict(reason='POINT_PREFERS_LEGACY',accepted=False);return chosen,ev
        owner=CURRENT.get();bank=owner._residual_bank
        alternative=ev['selected'];baseline=ev['baseline']
        keys=sorted({e['calibration_key'] for plan in (baseline,alternative) for e in plan['model_execution'] if e['raw_rate'] is not None})
        bounds={k:bank.bounds(k,now) for k in keys};missing=[k for k,v in bounds.items() if v is None]
        if missing:screen=dict(accepted=False,reason='MISSING_PREDECISION_SUPPORT',missing=missing,worlds=0)
        else:
            def evaluate(world):
                token=RATE_FACTORS.set(world)
                try:a=predict_with_policy(owner,legacy,self.interval);b=predict_with_policy(owner,chosen,self.interval)
                finally:RATE_FACTORS.reset(token)
                if a['status']!='OK' or b['status']!='OK':raise ValueError('UNCERTAINTY_PREDICTION_CENSORED')
                return a['sum_completion']-b['sum_completion']
            screen=endpoint_screen(bounds,evaluate)
        ev['uncertainty']=dict(**screen,bounds=bounds,decision_time=now,available_observations=len(bank.rows),
            point_selected_plan=ev['selected_plan'],point_gain=ev['predicted_sum_completion_saved'],confidence_guarantee=False)
        if screen['accepted']:return chosen,ev
        ev['selected_plan']='LEGACY';ev['selected']=baseline;ev['predicted_sum_completion_saved']=0.
        ev['per_job_predicted_saved']={j:0. for j in s.job_map};ev['receiver_gain']=ev['donor_loss']=0.
        return legacy,ev

def install(cluster):
    from .policy_rollout import install as install_policy
    install_policy(cluster)
    original=cluster.policy.liveness._rollout_base_class
    cluster.policy.liveness.__class__=type('Reliable'+original.__name__,(ReliableMixin,original),{'_rollout_base_class':original})
    cluster._residual_bank=ResidualBank();cluster._calibration_pending=None
    base=cluster.__class__
    class ObservedCluster(base):
        def step(self,seconds=60):
            self._calibration_pending=prepare(self)
            return super().step(seconds)
        def optimize_policy(self):
            observe(self)
            return super().optimize_policy()
    cluster.__class__=ObservedCluster
    old_policy=cluster.policy.__class__
    class ReliablePolicy(old_policy):
        _uncertainty_base_policy=old_policy
        def optimize(self,*args):
            answer=super().optimize(*args);ev=self.last_decision['result']['audit'].get('release_completion')
            if ev is None:return answer
            self.last_decision['shared_contract'].update(policy_version='PREDECISION_UNCERTAINTY_V4',objective='EMPIRICAL_ENDPOINT_SCREENED_COMPLETION_GAIN',admission_and_service_unchanged=True)
            self.last_decision['legacy_utility_field_is_diagnostic_only']=True
            self.last_decision['result']['status']='UNCERTAINTY_'+ev.get('uncertainty',{}).get('reason',ev['status'])
            return answer
    cluster.policy.__class__=ReliablePolicy
    return cluster
