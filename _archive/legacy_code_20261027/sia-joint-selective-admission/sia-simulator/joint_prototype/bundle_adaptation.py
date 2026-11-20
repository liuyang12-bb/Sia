"""Opt-in JOINT-G: global exchanges under unchanged service/delay protection.

No new utility weights, response model, grid, admission bonus or service clock.
Local no-slowdown and within-lease payback are not global optimality conditions.
The actual fairness objective (including its original restart factor) chooses
whole bundles. Actual startup cost is a secondary criterion ONLY on exact
primary ties; the cumulative actual-delay cap still protects productive service.
"""
import numpy as np
import hashlib
from scipy.optimize import LinearConstraint,Bounds
from . import adaptive_service as solver_backend
from .scaled_adaptive import ScaledAdmission, ScaledJoint, install as install_scaled
from .neutral_liveness import geometry
from .adaptive_service import gpu_geometry


class BundleAdmission(ScaledAdmission):
    def _milp(self,cost,columns,constraints):
        def payload():
            return dict(c=np.asarray(cost,dtype=float).tolist(),columns=[dict(job=c.job_id,column_id=c.column_id) for c in columns],constraints=[dict(
                A=np.asarray(v.A,dtype=float).tolist(),lb=np.asarray(v.lb,dtype=float).tolist(),ub=np.asarray(v.ub,dtype=float).tolist()) for v in constraints])
        def fingerprint():
            h=hashlib.sha256(np.asarray(cost,dtype=float).tobytes())
            for v in constraints:
                for a in [v.A,v.lb,v.ub]:h.update(np.asarray(a,dtype=float).tobytes())
            return h.hexdigest()
        def verify(result):
            x=getattr(result,'x',None)
            if x is None:return dict(valid=False,reason='NO_X',maximum_violation=None)
            x=np.asarray(x)
            if x.shape!=(len(columns),) or not np.all(np.isfinite(x)):
                return dict(valid=False,reason='NONFINITE_OR_SHAPE',maximum_violation=None)
            maximum=max(float(np.max(np.abs(x-np.round(x)))),float(np.max(-x)),float(np.max(x-1.)))
            violations=[]
            for ci,v in enumerate(constraints):
                for label,vec in [('raw',x),('rounded',np.round(x))]:
                    values=np.asarray(v.A@vec).reshape(-1);lower=np.broadcast_to(v.lb,values.shape);upper=np.broadcast_to(v.ub,values.shape)
                    delta=np.maximum(lower-values,values-upper);maximum=max(maximum,float(np.max(delta)))
                    for row in np.flatnonzero(delta>1e-7):
                        item=dict(constraint=ci,row=int(row),vector=label,value=float(values[row]),lb=float(lower[row]),ub=float(upper[row]),violation=float(delta[row]))
                        if ci==0 and row<len(self._diagnostic_job_order):item['job_id']=self._diagnostic_job_order[row]
                        violations.append(item)
            return dict(valid=maximum<=1e-7,maximum_violation=maximum,violations=violations,
                selected_jobs=[columns[i].job_id for i,v in enumerate(x) if v>.5])
        model_hash=fingerprint();result=super()._milp(cost,columns,constraints)
        if fingerprint()!=model_hash:raise RuntimeError('SOLVER_MUTATED_INPUT_MODEL')
        # A timeout or genuine infeasibility is NOT silently converted to success.
        if result.status!=0 or not result.success:return result
        original=verify(result)
        if original['valid']:return result
        record=dict(model_fingerprint=model_hash,original_status=int(result.status),original_message=str(result.message),
            original_validation=original,original_x=np.asarray(result.x).tolist(),model=payload(),
            retry_options=dict(time_limit=30.,mip_rel_gap=0.,presolve=False))
        retried=solver_backend.milp(c=cost,integrality=np.ones(len(columns)),bounds=Bounds(np.zeros(len(columns)),np.ones(len(columns))),
            constraints=constraints,options=record['retry_options'])
        record.update(retry_status=int(retried.status),retry_message=str(retried.message),retry_validation=verify(retried),
                      retry_x=np.asarray(retried.x).tolist() if retried.x is not None else None)
        self.numerical_solver_retries.append(record)
        if fingerprint()!=model_hash or retried.status!=0 or not retried.success or not record['retry_validation']['valid']:
            error=RuntimeError('INVALID_SOLVER_RESULT_AFTER_NUMERICAL_RETRY')
            error.solver_diagnostics=record
            raise error
        return retried

    def eligible_held_columns(self, scheduler, job, service, columns, productive, now):
        legal, legacy = super().eligible_held_columns(scheduler, job, service, columns, productive, now)
        by_id={c.column_id:c for c in columns if c.job_id==job.job_id}
        reclaimed=[];remaining=[]
        for item in legacy['rejected']:
            if item['reason'] in {'NO_REMAINING_SERVICE_PAYBACK','AUX_REDUCES_CURRENT_RATE'}:
                reclaimed.append(by_id[item['column_id']])
            else:remaining.append(item)
        # The legacy order already checked checkpoint, observed training,
        # provenance and cumulative actual startup budget for GPU changes.
        # CPU/RAM-only changes have no startup delay in the fixed execution model.
        result=legal+reclaimed
        evidence=dict(legacy)
        evidence.update(eligibility_rule='GLOBAL_BUNDLE_WITH_REAL_LOCK_PROFILE_AND_DELAY_BUDGET',
            legacy_rejection_counts=legacy['rejection_counts'],
            reintroduced=[dict(column_id=c.column_id,config=c.config.config_id,
                               effective_goodput=self._rate(c)) for c in reclaimed],
            rejected=remaining,eligible_count=len(result))
        from collections import Counter
        evidence['rejection_counts']=dict(Counter(v['reason'] for v in remaining))
        return result,evidence

    def _secondary_tie(self, scheduler, columns, required, constraints, call, chosen):
        costs=[]
        for c in columns:
            j=scheduler.job_map[c.job_id]
            same=(j.current_config_id==c.config.config_id and j.current_placement==c.placement.node_gpus)
            costs.append(0. if same else self._delay(c.job_id,c))
        costs=np.asarray(costs)
        candidate=call(costs)
        if candidate is None:raise RuntimeError('SECONDARY_COST_INCONSISTENT')
        best=sum(costs[i] for i,c in enumerate(columns) if any(v is c for v in candidate))
        constraints.append(LinearConstraint(costs,best-1e-9,best+1e-9))
        self.secondary_startup_cost=dict(minimum_actual_delay=best,
            rule='MIN_ACTUAL_STARTUP_ON_EXACT_PRIMARY_OPTIMUM')
        return candidate

    def _decide(self,s,columns,now,productive):
        self.secondary_startup_cost=dict(minimum_actual_delay=0.,rule='MIN_ACTUAL_STARTUP_ON_EXACT_PRIMARY_OPTIMUM')
        self.numerical_solver_retries=[]
        self._diagnostic_job_order=[j.job_id for j in s.jobs]
        held={j:v for j,v in self.services.items() if j in s.job_map and
              productive.get(j,0.)-v['productive_base']<self.quantum-1e-9}
        old_geometry={j:geometry(v['column']) for j,v in held.items()}
        result=super()._decide(s,columns,now,productive)
        baseline=[]
        for j,c in result.selected.items():
            baseline.append(next(v for v in columns if geometry(v)==old_geometry[j]) if j in held else c)
        audit=s.audit(baseline)
        comparison=dict(feasible=audit['valid'],same_required_jobs=True,
            baseline_objective=sum(c.scheduling_utility for c in baseline),
            chosen_objective=result.scheduling_objective)
        if audit['valid'] and result.scheduling_objective+1e-7<comparison['baseline_objective']:
            raise RuntimeError('GLOBAL_BUNDLE_OBJECTIVE_REGRESSION')
        if not audit['valid']:comparison['baseline_violations']=audit['violations']
        result.audit.update(global_bundle_baseline=comparison,secondary_tie=self.secondary_startup_cost,
            numerical_solver_retries=self.numerical_solver_retries)
        result.policy='JOINT_G';result.status='DRAIN_FOR_HEAD' if result.audit['drain'] else 'OPTIMAL_GLOBAL_BUNDLE'
        return result


class BundleJoint(ScaledJoint):
    def optimize(self,*args):
        result=super().optimize(*args)
        self.last_decision['shared_contract'].update(
            admission='FIFO_PRODUCTIVE_GLOBAL_BUNDLE_DELAY_BUDGET',
            eligibility='NO_LOCAL_PAYBACK_OR_PER_JOB_NONDEGRADATION_VETO',
            secondary='MIN_ACTUAL_STARTUP_ON_PRIMARY_TIES')
        return result


def install(cluster):
    install_scaled(cluster)
    cluster.policy.__class__=BundleJoint
    cluster.policy.liveness.__class__=BundleAdmission
    return cluster
