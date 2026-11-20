"""Stage2B-only Neutral L2 modes. Historical policies are not modified.

This module has no offline-phi registry dependency. All execution rho == 1.
Scope guard: at most three simultaneously active jobs; not a cohort runner.
"""
import copy
import math
import time
from dataclasses import replace

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

import simulator_config as sim_config
from sia import ZERO_ALLOC_GAIN
from .joint_integration import JointSiaPolicy, JointCluster
from .joint_scheduler import JointScheduler, PolicyResult


def geometry(c):
    return (c.job_id,c.config.config_id,str(c.placement.node_gpus),
            c.query.cpu_per_gpu,c.query.ram_gb_per_gpu)


def validate_productive_segments(job_id, segments):
    for s in segments:
        if s['rho'] != 1 or not math.isclose(s['raw_rate_before_rho'],s['effective_rate'],rel_tol=1e-12):
            raise RuntimeError(f'NON_NEUTRAL_EXECUTION: {job_id}')
        if s['seconds'] > 0 and (not math.isfinite(s['raw_rate_before_rho']) or s['raw_rate_before_rho'] <= 0):
            raise RuntimeError(f'ZERO_EXECUTION_RATE: {job_id}')


class FactorScheduler(JointScheduler):
    # Preserve literal P subset O. Pareto pruning the proportional point in
    # Neutral would invalidate the requested same-state inclusion contract.
    @staticmethod
    def _pareto_prune(columns):
        return sorted(columns,key=lambda c:c.column_id)


class L2Admission:
    MAX_ACTIVE_JOBS = 3
    REQUIRE_NEUTRAL = True
    def __init__(self, quantum, delay_bound, interval):
        if quantum<=0 or delay_bound<0 or interval<=0:
            raise ValueError('invalid service bounds')
        self.quantum=float(quantum); self.delay_bound=float(delay_bound); self.interval=float(interval)
        self.ready=[]; self.seen=set(); self.services={}; self.sequence=0; self.events=[]

    def decide(self,scheduler,columns,now,productive):
        # Nothing from a failed solve or validation is committed to the queue.
        trial=copy.deepcopy(self)
        result=trial._decide(scheduler,list(columns),now,productive)
        self.__dict__.update(trial.__dict__)
        return result

    def _decide(self,s,columns,now,productive):
        start=time.perf_counter(); jobs=s.job_map
        if len(jobs)>self.MAX_ACTIVE_JOBS: raise RuntimeError('STAGE2B_SCOPE_MAX_THREE_ACTIVE_JOBS' if self.MAX_ACTIVE_JOBS==3 else 'SCOPE_MAX_TWENTY_ACTIVE_JOBS')
        if any(j.arrival_time>now for j in s.jobs): raise RuntimeError('FUTURE_JOB_INPUT')
        if self.REQUIRE_NEUTRAL and any(c.query.rho!=1 for c in columns): raise RuntimeError('NON_NEUTRAL_COLUMN')
        if any(not math.isfinite(c.query.rho) or c.query.rho<=0 for c in columns): raise RuntimeError('INVALID_RESPONSE_COLUMN')
        self.ready=[j for j in self.ready if j in jobs]
        for name in list(self.services):
            if name not in jobs:
                self.events.append(dict(time=now,event='COMPLETED_SERVICE',job=name))
                del self.services[name]
        new=sorted((j for j in s.jobs if j.job_id not in self.seen),key=lambda j:(j.arrival_time,j.source_row_index,j.job_id))
        for j in new:
            self.ready.append(j.job_id); self.seen.add(j.job_id)
            self.events.append(dict(time=now,event='ARRIVE_READY',job=j.job_id))
        expired=[]
        for name,service in self.services.items():
            done=productive.get(name,0.)-service['productive_base']
            if done < -1e-8: raise RuntimeError('NON_MONOTONIC_SERVICE_COUNTER')
            if done>=self.quantum-1e-9:
                if s.lock_active(jobs[name]): raise RuntimeError('CHECKPOINT_AFTER_SERVICE_WITHOUT_RECONFIGURATION')
                expired.append(name)
            elif now-service['issued_time'] > self.delay_bound+self.quantum+self.interval+1e-8:
                raise RuntimeError(f'SERVICE_BOUND_EXCEEDED: {name}')
        for name in sorted(expired,key=lambda n:self.services[n]['sequence']):
            self.ready.append(name); del self.services[name]
            self.events.append(dict(time=now,event='REQUEUE_PRODUCTIVE_QUANTUM',job=name))
        locked={j.job_id for j in s.jobs if s.lock_active(j)}
        if locked-set(self.services):
            raise RuntimeError('LOCK_WITHOUT_SERVICE_EVIDENCE')
        held=set(self.services)
        # Held service is ordinary current eligibility, not a fake checkpoint.
        for name in held:
            old=self.services[name]['column']
            fresh=s.refresh_column(old)
            if jobs[name].current_config_id!=fresh.config.config_id or jobs[name].current_placement!=fresh.placement.node_gpus:
                raise RuntimeError(f'INFEASIBLE_COMMITMENT: {name}')
            matches=[c for c in columns if geometry(c)==geometry(fresh)]
            if not matches: raise RuntimeError(f'INFEASIBLE_COMMITMENT_CANDIDATE: {name}')
            columns=[c for c in columns if c.job_id!=name]+[matches[0]]
        columns=sorted(columns,key=geometry)
        traces=[]
        def solve(required, objective=False, tie=False):
            required=set(required)
            if not required:
                if locked: raise RuntimeError('INVALID_EMPTY_LOCK')
                return []
            if not columns or any(not any(c.job_id==j for c in columns) for j in required):
                return None
            matrix,upper=s._constraint_matrix(columns,('gpu','cpu','ram'))
            lower=np.full(len(upper),-np.inf)
            for i,j in enumerate(s.jobs): lower[i]=upper[i]=float(j.job_id in required)
            weights=np.asarray([c.scheduling_utility for c in columns])
            constraints=[LinearConstraint(matrix,lower,upper)]
            def call(cost):
                result=milp(c=cost,integrality=np.ones(len(columns)),
                    bounds=Bounds(np.zeros(len(columns)),np.ones(len(columns))),
                    constraints=constraints,options={'time_limit':30.,'mip_rel_gap':0.})
                traces.append(dict(required=sorted(required),status=int(result.status),message=str(result.message),phase='objective' if objective else 'feasibility'))
                if result.status==2: return None
                if result.x is None or not result.success or result.status!=0:
                    raise RuntimeError(f'SOLVER_UNDECIDED: {traces[-1]}')
                x=np.asarray(result.x)
                if len(x)!=len(columns) or not np.all(np.isfinite(x)) or np.any(abs(x-np.round(x))>1e-6):
                    raise RuntimeError('INVALID_SOLVER_ALLOCATION integrality')
                chosen=[c for c,v in zip(columns,x) if v>.5]
                audit=s.audit(chosen,require_locks=locked<=required)
                if not audit['valid'] or {c.job_id for c in chosen}!=required or len(chosen)!=len(required):
                    raise RuntimeError(f'INVALID_SOLVER_ALLOCATION: {audit}')
                return chosen
            chosen=call(-weights if objective else np.zeros(len(columns)))
            if chosen is None: return None
            if tie:
                optimum=sum(c.scheduling_utility for c in chosen)
                constraints.append(LinearConstraint(weights,optimum-1e-9,optimum+1e-9))
                # Same deterministic lexicographic tie resolution in all cells;
                # no epsilon or allocation bonus in the primary objective.
                for name in sorted(required):
                    cost=np.zeros(len(columns)); candidates=[i for i,c in enumerate(columns) if c.job_id==name]
                    for rank,i in enumerate(candidates): cost[i]=rank
                    chosen=call(cost)
                    if chosen is None: raise RuntimeError('LEXICOGRAPHIC_OBJECTIVE_INCONSISTENT')
                    c=next(c for c in chosen if c.job_id==name)
                    mask=np.asarray([float(v is c) for v in columns])
                    constraints.append(LinearConstraint(mask,1.,1.))
            return chosen
        admitted=set(held); drain=False
        # The current commitments alone must be jointly legal before checking head.
        if held and solve(held) is None: raise RuntimeError('INFEASIBLE_COMMITMENT')
        if self.ready:
            head=self.ready[0]
            # For solo feasibility ignore other checkpoint selection constraints;
            # retained columns still encode each protected job's exact geometry.
            if solve({head}) is None:
                raise RuntimeError(f'NO_SOLO_FEASIBLE_CANDIDATE: {head}')
            if solve(admitted|{head}) is None:
                drain=True
                self.events.append(dict(time=now,event='DRAIN_FOR_HEAD',job=head,held=sorted(held)))
            else:
                for name in self.ready:
                    if solve(admitted|{name}) is not None: admitted.add(name)
        chosen=solve(admitted,objective=True,tie=True)
        if chosen is None: raise RuntimeError('INFEASIBLE_FINAL_ADMISSION')
        independent=s.audit(chosen)
        if not independent['valid'] or any(geometry(c) not in {geometry(v) for v in columns} for c in chosen):
            raise RuntimeError(f'INVALID_FINAL_COMMIT: {independent}')
        selected={c.job_id:c for c in chosen}
        for name in sorted(admitted-held,key=lambda n:self.ready.index(n)):
            self.sequence+=1
            self.services[name]=dict(column=selected[name],issued_time=now,
                productive_base=productive.get(name,0.),sequence=self.sequence)
            self.events.append(dict(time=now,event='ISSUE_SERVICE',job=name,sequence=self.sequence))
        self.ready=[j for j in self.ready if j not in admitted]
        for name in held: self.services[name]['column']=selected[name]
        independent.update(ready=list(self.ready),checkpoint_locks=sorted(locked),
            service_commitments=sorted(self.services),drain=drain,solver_trace=traces,
            productive_service={j:productive.get(j,0.)-v['productive_base'] for j,v in self.services.items()})
        return PolicyResult('L2_NEUTRAL','DRAIN_FOR_HEAD' if drain else 'OPTIMAL_L2',selected,
            sorted(set(jobs)-admitted),sum(c.scheduling_utility for c in chosen),
            sum(c.query.effective_goodput for c in chosen),time.perf_counter()-start,audit=independent)


class NeutralL2Policy(JointSiaPolicy):
    MODES={'u_fp','u_fo','u_jp','u_jo'}
    ADMISSION_CLASS = L2Admission
    REQUIRE_NEUTRAL = True
    def __init__(self,*,mode,quantum=300.,delay_bound=250.,**kwargs):
        if self.REQUIRE_NEUTRAL and not kwargs['oracle'].neutral_response: raise ValueError('Stage2B requires Neutral oracle')
        kwargs.setdefault('minimum_cpu_per_gpu',1.)
        kwargs.setdefault('minimum_ram_gb_per_gpu',20.83)
        super().__init__(mode=mode,**kwargs)
        self.liveness=self.ADMISSION_CLASS(quantum,delay_bound,60.)
        self.productive={}; self.now=0.

    def _scheduler(self,nodes,specs):
        return FactorScheduler(nodes,specs,self.oracle,self.aux_cpu_points,self.aux_ram_points,
            allocation_bonus=0.,verify_exact=False,minimum_cpu_per_gpu=self.minimum_cpu_per_gpu,
            minimum_ram_gb_per_gpu=self.minimum_ram_gb_per_gpu,
            candidate_policy='b3_prop' if self.mode.endswith('p') else 'b3')

    def factor_columns(self,jobs,specs,s):
        prop=s.build_proportional_columns()
        columns=prop if self.mode.endswith('p') else s.build_joint_columns()
        proposals={}
        for spec in specs:
            service=self.liveness.services.get(spec.job_id)
            unexpired=service is not None and self.productive.get(spec.job_id,0.)-service['productive_base'] < self.liveness.quantum-1e-9
            if unexpired or s.lock_active(spec):
                proposals[spec.job_id]=spec.current_config_id
            else:
                feasible={c.config.config_id for c in prop if c.job_id==spec.job_id}
                configs=[c for c in spec.configs if c.config_id in feasible]
                if configs:
                    proposals[spec.job_id]=min(configs,key=lambda c:(
                        -c.sia_goodput*self._restart_factor(spec.job_id,jobs[spec.job_id],c.config_id,spec.current_config_id),c.config_id)).config_id
        if self.mode.startswith('u_f'):
            columns=[c for c in columns if c.config.config_id==proposals.get(c.job_id)]
        # Exactly the historical B3 final utility; not JointScheduler's prototype
        # bonus. Parent Goodputs were normalized BEFORE any factor filtering.
        by_job={j.job_id:j for j in specs}; transformed=[]
        for c in columns:
            factor=self._restart_factor(c.job_id,jobs[c.job_id],c.config.config_id,by_job[c.job_id].current_config_id)
            adjusted=max(c.query.effective_goodput*factor,ZERO_ALLOC_GAIN)
            powered=min(round(adjusted**self.p_fairness,2),sim_config.sia_goodput_clip_val)
            utility=(-1. if self.p_fairness<0 else 1.)*(powered-(self.lambda_n if self.lambda_n else -1.))
            transformed.append(replace(c,scheduling_utility=utility))
        return transformed,proposals

    def optimize(self,jobs,nodes,base_allocations):
        if len(jobs)>self.liveness.MAX_ACTIVE_JOBS: raise RuntimeError('STAGE2B_SCOPE_MAX_THREE_ACTIVE_JOBS' if self.liveness.MAX_ACTIVE_JOBS==3 else 'SCOPE_MAX_TWENTY_ACTIVE_JOBS')
        specs,provenance=self._job_specs(jobs,base_allocations)
        s=self._scheduler(self._nodes(nodes),specs); s.locked_columns()
        columns,proposals=self.factor_columns(jobs,specs,s)
        pending=copy.deepcopy(self.liveness)
        result=pending.decide(s,columns,self.now,self.productive)
        audit=s.audit(list(result.selected.values()))
        if not audit['valid']: raise RuntimeError(f'INVALID_PRECOMMIT: {audit}')
        result.audit['precommit_independent_audit']=audit
        allocations=self._allocations_from_result(result,jobs)
        self._update_b3_previous(jobs,result)
        self._commit_metadata(result,provenance)
        self.liveness=pending
        self.last_decision.update(factor_candidate_keys=[geometry(c) for c in columns],
            gpu_only_proposals=proposals,liveness_events=copy.deepcopy(pending.events),
            shared_contract=dict(p=self.p_fairness,lambda_n=self.lambda_n,
                quantum=pending.quantum,delay_bound=pending.delay_bound,interval=pending.interval,
                rho=1.,admission='L2_FIFO_PRODUCTIVE_DRAIN',placement='JOINT_ALL_CELLS',
                objective='STAGE1P1_B3_UTILITY',parent_normalization='BEFORE_FACTOR_FILTER',
                fallback='EXPLICIT_STOP_NO_PARTIAL_COMMIT',solver_time_limit=30.,mip_rel_gap=0.))
        return allocations


class NeutralL2Cluster(JointCluster):
    def optimize_policy(self):
        self.policy.now=self.current_time
        self.policy.productive={j.name:j.execution_time for j in self.jobs if j.submission_time<=self.current_time}
        for j in self.jobs:
            validate_productive_segments(j.name,j.joint_step_segments)
        return super().optimize_policy()
