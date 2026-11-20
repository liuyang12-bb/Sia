"""One observed, cost-paying GPU recovery after the arrived queue drains.

No future arrivals, new candidates, response or scoring changes. A used permit
freezes subsequent GPU geometry while auxiliary quotas remain adaptive.
"""
import math
from .completion_continuity import ContinuityAdmission,CompletionContinuityPolicy
from .adaptive_service import gpu_geometry

class TailRecoveryAdmission(ContinuityAdmission):
    def delay_limit(self,name,service):
        pending=getattr(self,'_pending_tail',{}).get(name,{})
        extra=max(service.get('tail_extra_budget',0.),pending.get('extra_budget',0.))
        if not math.isfinite(extra) or not 0<=extra<=self.delay_bound:
            raise RuntimeError('INVALID_TAIL_DELAY_GRANT')
        return self.delay_bound+extra

    def eligible_held_columns(self,scheduler,job,service,columns,productive,now):
        legal,evidence=super().eligible_held_columns(scheduler,job,service,columns,productive,now)
        current=gpu_geometry(service['column']);name=job.job_id
        if service.get('tail_recovery_used',False):
            legal=[c for c in legal if gpu_geometry(c)==current]
            evidence.update(tail_rule='USED_ONCE_GPU_FROZEN',eligible_count=len(legal))
            return legal,evidence
        if (len(scheduler.job_map)!=1 or self.ready or scheduler.lock_active(job)
                or productive.get(name,0.)<=service['last_geometry_productive']+1e-9):
            evidence['tail_rule']='NOT_ELIGIBLE'
            return legal,evidence
        scores=getattr(scheduler,'current_completion_scores',{})
        def eta(c):
            x=scores.get(c.column_id,{})
            v=x.get('estimated_completion')
            return v if v is not None and math.isfinite(v) and v>0 and x.get('raw_rate') is not None else None
        incumbents=[c for c in legal if gpu_geometry(c)==current and eta(c) is not None]
        if not incumbents:
            evidence['tail_rule']='UNKNOWN_INCUMBENT_ETA';return legal,evidence
        best_current=min(eta(c) for c in incumbents)
        options=[]
        for c in columns:
            if c.job_id!=name or gpu_geometry(c)==current or eta(c) is None:continue
            if self.profile_sources.get(name,{}).get(c.config.config_id) not in {'direct_profile','projected_profile'}:continue
            if not scheduler.audit([c],require_locks=False)['valid']:continue
            cost=self._delay(name,c)
            # Scores already include this actual target startup exactly once.
            if scores[c.column_id]['delay']!=cost or eta(c)>=best_current-1e-9:continue
            options.append(c)
        if not options:
            evidence['tail_rule']='NO_COST_PAYING_IMPROVEMENT';return legal,evidence
        best=min(options,key=lambda c:(eta(c),c.column_id))
        if any(c.column_id==best.column_id for c in legal):
            evidence['tail_rule']='ORDINARY_POLICY_ALREADY_ALLOWS_BEST';return legal,evidence
        cost=self._delay(name,best)
        extra=max(0.,service['delay_charged']+cost-self.delay_bound)
        self._pending_tail[name]=dict(column_id=best.column_id,extra_budget=extra,
            actual_startup=cost,eta_before=best_current,eta_after=eta(best),time=now)
        legal=legal+[best]
        evidence.update(tail_rule='ONE_BOUNDED_RECOVERY_OFFERED',tail_offer=self._pending_tail[name],eligible_count=len(legal))
        return legal,evidence

    def _decide(self,scheduler,columns,now,productive):
        self._pending_tail={}
        result=super()._decide(scheduler,columns,now,productive)
        committed=[]
        for name,p in self._pending_tail.items():
            if result.selected[name].column_id!=p['column_id']:continue
            service=self.services[name]
            if service.get('tail_recovery_used',False):raise RuntimeError('TAIL_PERMIT_REUSED')
            service.update(tail_recovery_used=True,tail_extra_budget=p['extra_budget'])
            record=dict(job=name,**p,cumulative_delay=service['delay_charged'])
            if service['delay_charged']>self.delay_bound+p['extra_budget']+1e-9:raise RuntimeError('TAIL_BUDGET_EXCEEDED')
            self.events.append(dict(event='TAIL_RECOVERY',**record));committed.append(record)
        self._pending_tail={}
        result.audit.update(tail_recoveries=committed,
            tail_grants={n:dict(used=v.get('tail_recovery_used',False),extra_budget=v.get('tail_extra_budget',0.)) for n,v in self.services.items()},
            tail_rule='ONE_ARRIVED_JOB_EMPTY_READY_OBSERVED_STRICT_ETA_WITH_ACTUAL_STARTUP')
        return result

class TailRecoveryPolicy(CompletionContinuityPolicy):
    def factor_columns(self,*args):
        columns,proposals=super().factor_columns(*args)
        scheduler=args[2]
        scheduler.current_completion_scores={x['column_id']:x for x in self.continuity_scores}
        return columns,proposals
    def optimize(self,*args):
        allocations=super().optimize(*args)
        self.last_decision['shared_contract'].update(policy_version='CC_TAIL_ONE_RECOVERY_V1',
            tail_recovery=True,max_extra_gpu_recoveries_per_job=1,
            extra_delay_limit='ONE_ACTUAL_STARTUP_AT_MOST_D_BAR',candidate_grid_changed=False)
        return allocations

def install(cluster):
    if cluster.policy.liveness.seen:raise RuntimeError('INSTALL_BEFORE_FIRST_DECISION')
    cluster.policy.__class__=TailRecoveryPolicy
    cluster.policy.liveness.__class__=TailRecoveryAdmission
    return cluster
