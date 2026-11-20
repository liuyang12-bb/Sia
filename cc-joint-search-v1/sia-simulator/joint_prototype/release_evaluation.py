"""Opt-in completion-event evaluation of a bounded set of whole bundles.

No future arrivals, throughput lookup, Job.step, new quota points or service
preemption. V2 predicts legal survivor upgrades after completion releases,
with paid startup and lifetime budgets. The V1 evaluator remains available
for its original tests. This is a
receding-horizon heuristic, NOT a global optimum or a JCT guarantee.
"""
import math
import time
import numpy as np
from scipy.optimize import LinearConstraint
from .neutral_liveness import geometry

def evaluate_reconfig_plan(scheduler,columns,scores,initial,queue,interval,context):
    import copy
    validation=evaluate_plan(scheduler,columns,scores,initial,queue,interval)
    if validation['status']!='OK':return validation
    context=copy.deepcopy(context)
    if set(context)!=set(scheduler.job_map):raise ValueError('MISSING_ROLLOUT_BUDGET_CONTEXT')
    for ctx in context.values():
        if (not all(math.isfinite(ctx[k]) for k in ('spent','limit')) or
                ctx['spent']<0 or ctx['limit']<ctx['spent']-1e-9 or
                any(not math.isfinite(v) or v<=0 for v in ctx['costs'].values())):
            raise ValueError('INVALID_ROLLOUT_BUDGET_CONTEXT')
    by_job={j:[c for c in columns if c.job_id==j and scores.get(c.column_id,{}).get('raw_rate') is not None] for j in scheduler.job_map}
    def rate(c):return scores[c.column_id]['raw_rate']*c.query.rho
    def fits(cs):
        return all(all(sum(c.usage_for(n.node_id)[k] for c in cs)<=cap+1e-7
            for k,cap in enumerate((n.gpus,n.cpu_cores,n.ram_gb))) for n in scheduler.nodes)
    def record(c,delay):
        return dict(column=c,work=scores[c.column_id]['remaining_work'],delay=delay,productive=0.)
    running={c.job_id:record(c,scores[c.column_id]['delay']) for c in initial}
    queue=[j for j in queue if j not in running];events=[];done={};t=0.;previous=0.
    for _ in range(4*len(scheduler.jobs)+4):
        released=False
        for j,z in list(running.items()):
            dt=t-previous;startup=min(dt,z['delay']);z['delay']-=startup
            productive=max(0.,dt-startup);needed=z['work']/rate(z['column'])
            if needed<=productive+1e-8 and z['delay']<=1e-8:
                finish=previous+startup+needed;done[j]=finish;released=True
                events.append(dict(event='COMPLETE_RELEASE',job=j,time=finish,boundary=t));del running[j]
            else:
                used=min(productive,needed);z['work']=max(0.,z['work']-used*rate(z['column']));z['productive']+=used
        while queue and t>0:
            j=queue[0];ctx=context[j];resident=[z['column'] for z in running.values()]
            options=[c for c in by_job[j] if ctx['spent']+ctx['costs'][c.config.gpu_type]<=ctx['limit']+1e-9 and fits(resident+[c])]
            if not options:break
            c=min(options,key=lambda c:(ctx['costs'][c.config.gpu_type]+scores[c.column_id]['remaining_work']/rate(c),str(geometry(c))))
            delay=ctx['costs'][c.config.gpu_type];ctx['spent']+=delay
            running[j]=record(c,delay);queue.pop(0)
            events.append(dict(event='FIFO_START',job=j,time=t,column=c.column_id,startup=delay,spent_after=ctx['spent']))
        # Only a real predicted completion permits survivor geometry changes.
        # Peers remain allocated; no invented permit or additional resource.
        if released:
            moved=set()
            for attempt in range(len(running)):
                options=[]
                for j,z in running.items():
                    if j in moved or z['delay']>1e-9 or z['productive']<=1e-9:continue
                    old=z['column'];ctx=context[j];baseline=z['work']/rate(old)
                    for c in by_job[j]:
                        different=(old.config.config_id,old.placement.node_gpus)!=(c.config.config_id,c.placement.node_gpus)
                        if different and (ctx['frozen'] or c.config.config_id not in ctx['known']):continue
                        cost=ctx['costs'][c.config.gpu_type] if different else 0.
                        if ctx['spent']+cost>ctx['limit']+1e-9:continue
                        if not fits([v['column'] for k,v in running.items() if k!=j]+[c]):continue
                        saving=baseline-cost-z['work']/rate(c)
                        if saving>1e-7:options.append((saving,str(geometry(c)),j,c,cost))
                if not options:break
                saving,_,j,c,cost=min(options,key=lambda v:(-v[0],v[1]));z=running[j];old=z['column']
                context[j]['spent']+=cost;z['column']=c;z['delay']=cost
                if cost:z['productive']=0.
                moved.add(j)
                events.append(dict(event='SURVIVOR_RECONFIGURE',job=j,time=t,old_column=old.column_id,
                    column=c.column_id,startup=cost,spent_after=context[j]['spent'],budget=context[j]['limit'],predicted_own_seconds_saved=saving,
                    peer_allocations_unchanged=True))
        if not running:
            if queue:return dict(status='NO_CONTINUATION',blocked=queue[0],events=events)
            break
        next_time=interval*math.ceil((min(z['delay']+z['work']/rate(z['column']) for z in running.values())+t-1e-9)/interval)
        if t==0 and queue:next_time=min(next_time,interval)
        previous,t=t,max(t,next_time)
    else:raise RuntimeError('RECONFIG_ROLLOUT_EVENT_BOUND')
    return dict(status='OK',sum_completion=sum(done.values()),completion_times=done,events=events,
        continuation='COMPLETION_EVENT_LEGAL_SURVIVOR_UPGRADE_AND_FIFO',global_optimum=False,
        model_scope='CURRENT_KNOWN_ELIGIBLE_CANDIDATES_ONLY',final_spent={j:v['spent'] for j,v in context.items()})


def evaluate_plan(scheduler, columns, scores, initial, queue, interval=60.):
    jobs = set(scheduler.job_map)
    if interval <= 0 or not math.isfinite(interval):
        raise ValueError('INVALID_ROLLOUT_INTERVAL')
    initial_audit = scheduler.audit(initial)
    if not initial_audit['valid'] or len({c.job_id for c in initial}) != len(initial):
        raise ValueError('INVALID_INITIAL_PLAN')
    if any(c.job_id not in jobs for c in columns):
        raise ValueError('NONCURRENT_JOB_COLUMN')
    chosen = {c.job_id: c for c in initial}
    queue = [j for j in queue if j not in chosen]
    if len(queue) != len(set(queue)) or set(queue) != jobs-set(chosen):
        raise ValueError('QUEUE_DOES_NOT_COVER_CURRENT_JOBS')
    by_job = {j: [] for j in jobs}
    for c in columns:
        v = scores.get(c.column_id, {})
        raw, work, delay = v.get('raw_rate'), v.get('remaining_work'), v.get('delay')
        if raw is None or work is None or delay is None:
            continue
        if not all(math.isfinite(x) for x in (raw, work, delay, c.query.rho)):
            raise ValueError('NONFINITE_ROLLOUT_MODEL')
        if raw <= 0 or work < 0 or delay < 0 or c.query.rho <= 0:
            raise ValueError('INVALID_ROLLOUT_MODEL')
        if not math.isclose(v['rho'], c.query.rho, rel_tol=1e-12):
            raise ValueError('ROLLOUT_RHO_MISMATCH')
        by_job[c.job_id].append(c)
    unknown = sorted(j for j in jobs if not by_job[j])
    unknown += [j for j,c in chosen.items() if c not in by_job[j] and j not in unknown]
    if unknown:
        return dict(status='UNKNOWN_MODEL', unknown_jobs=sorted(unknown),
                    rule='NO_FAKE_ETA_NO_JOB_EXCLUSION')
    # Different columns of one job must use exactly the same remaining work.
    for j, cs in by_job.items():
        if len({scores[c.column_id]['remaining_work'] for c in cs}) != 1:
            raise ValueError('INCONSISTENT_REMAINING_WORK')
    def duration(c):
        v = scores[c.column_id]
        return float(v['delay'] + v['remaining_work']/(v['raw_rate']*c.query.rho))
    def fits(bundle):
        for node in scheduler.nodes:
            use = [sum(c.usage_for(node.node_id)[k] for c in bundle) for k in range(3)]
            if any(x > y+1e-7 for x,y in zip(use,(node.gpus,node.cpu_cores,node.ram_gb))):
                return False
        return True
    if not fits(initial):
        raise ValueError('INITIAL_RESOURCE_OVERFLOW')
    running = {j:dict(column=c, finish=duration(c)) for j,c in chosen.items()}
    completions = {}; events = []; t = 0.
    # Each iteration admits a job, completes a job, or advances to its release.
    for _ in range(4*len(jobs)+4):
        for j in sorted(list(running)):
            if running[j]['finish'] <= t+1e-8:
                completions[j] = running[j]['finish']
                events.append(dict(event='COMPLETE_RELEASE',job=j,time=running[j]['finish'],
                                   next_admission_boundary=t))
                del running[j]
        # Only `initial` is actually committed at this boundary. Do not give
        # an excluded waiter fictitious service at t=0, even if space is free.
        while queue and t > 0:
            j=queue[0]; resident=[v['column'] for v in running.values()]
            feasible=[c for c in by_job[j] if fits(resident+[c])]
            if not feasible:
                break  # FIFO head draining, no bypass or future reservation.
            c=min(feasible,key=lambda x:(duration(x),str(geometry(x))))
            running[j]=dict(column=c,finish=t+duration(c)); queue.pop(0)
            events.append(dict(event='FIFO_START',job=j,time=t,column=c.column_id,
                               duration=duration(c)))
        if not running:
            if queue:
                return dict(status='NO_CONTINUATION',blocked=queue[0],events=events)
            break
        earliest=min(v['finish'] for v in running.values())
        next_time=interval*math.ceil(max(0.,earliest-1e-9)/interval)
        if t == 0 and queue:
            next_time=min(next_time,interval)
        t=max(t,next_time)
    else:
        raise RuntimeError('ROLLOUT_EVENT_BOUND')
    if set(completions) != jobs:
        raise RuntimeError('INCOMPLETE_ROLLOUT')
    return dict(status='OK',sum_completion=sum(completions.values()),
                completion_times=completions,events=events,
                continuation='CURRENT_MODEL_FROZEN_SURVIVORS_FIFO_AT_INTERVAL_BOUNDARY',
                model_scope='CURRENT_ARRIVED_JOBS_ONLY',global_optimum=False)


class ReleaseEvaluationMixin:
    """Change only final bundle evaluation, before any ledger/atomic commit."""
    def refine_completion_bundle(self,s,columns,legacy,now):
        start=time.perf_counter()
        scores=s.current_completion_scores
        served={c.job_id for c in legacy}
        queue=[j for j in self.ready if j not in served]
        def evaluate(bundle):
            context={}
            for j,spec in s.job_map.items():
                h=self.services.get(j,getattr(self,'_lifetime_history',{}).get(j,{}))
                c=next((c for c in bundle if c.job_id==j),None)
                spent=float(h.get('delay_charged',0.));limit=self.delay_bound+h.get('tail_extra_budget',0.)
                frozen=bool(h.get('tail_recovery_used',False))
                if c is not None and (spec.current_config_id!=c.config.config_id or spec.current_placement!=c.placement.node_gpus):
                    spent+=self._delay(j,c)
                    offer=getattr(self,'_pending_tail',{}).get(j,{})
                    if offer.get('column_id')==c.column_id:
                        limit=self.delay_bound+max(h.get('tail_extra_budget',0.),offer.get('extra_budget',0.));frozen=True
                if spent>limit+1e-9:raise RuntimeError('ROLLOUT_INITIAL_BUDGET')
                context[j]=dict(spent=spent,limit=limit,frozen=frozen,
                    costs=self.delay_costs[j],known=[k for k,v in self.profile_sources.get(j,{}).items() if v in ('direct_profile','projected_profile')])
            evaluator=getattr(self,'evaluate_completion_plan',None)
            if evaluator is not None:
                return evaluator(s,columns,scores,bundle,queue,self.interval,context)
            return evaluate_reconfig_plan(s,columns,scores,bundle,queue,self.interval,context)
        evaluated=evaluate(legacy)
        evidence=dict(version='RELEASE_RECONFIG_EVALUATION_V2',status=evaluated['status'],
            fixed_service_set=sorted(served),future_arrivals_read=False,
            historical_reciprocal_is_proposal_only=True,baseline=evaluated,
            plans=[],selected_plan='LEGACY',guard_seconds_tolerance=1e-7)
        if evaluated['status'] != 'OK' or not legacy:
            evidence['reason']='UNKNOWN_OR_UNAVAILABLE_CONTINUATION_KEEP_VALIDATED_BOOTSTRAP'
            evidence['seconds']=time.perf_counter()-start
            return legacy,evidence
        fixed={c.job_id:c.config.config_id for c in legacy} if self.clean_gpu=='fixed' else None
        domain=[c for c in columns if c.job_id in served
                and (fixed is None or fixed[c.job_id]==c.config.config_id)
                and scores.get(c.column_id,{}).get('raw_rate') is not None]
        matrix,upper=s._constraint_matrix(domain,('gpu','cpu','ram'))
        lower=np.full(len(upper),-np.inf)
        for i,j in enumerate(s.jobs):lower[i]=upper[i]=float(j.job_id in served)
        cost=np.array([scores[c.column_id]['estimated_completion'] for c in domain])
        solve=self._milp(cost,domain,[LinearConstraint(matrix,lower,upper)])
        if solve.status!=0 or not solve.success or solve.x is None:
            raise RuntimeError('RELEASE_ETA_PROPOSAL_UNPROVEN')
        eta=[c for c,x in zip(domain,solve.x) if x>.5]
        if ({c.job_id for c in eta}!=served or not s.audit(eta)['valid']
                or np.any(abs(solve.x-np.rint(solve.x))>1e-6)):
            raise RuntimeError('RELEASE_ETA_PROPOSAL_INVALID')
        plans=[('LEGACY',legacy),('MIN_STATIC_SUM_ETA',eta)]
        prior=[]
        for c in legacy:
            old=self.services.get(c.job_id,{}).get('column')
            match=next((v for v in domain if old is not None and geometry(v)==geometry(old)),None)
            prior.append(match or c)
        if s.audit(prior)['valid']:
            plans.append(('CURRENT_ELIGIBLE_BUNDLE',prior))
        best=legacy;best_score=evaluated['sum_completion'];best_eval=evaluated
        for name,bundle in plans:
            ev=evaluated if name=='LEGACY' else evaluate(bundle)
            evidence['plans'].append(dict(name=name,evaluation=ev,
                columns={c.job_id:c.column_id for c in bundle},
                static_sum_eta=sum(scores[c.column_id]['estimated_completion'] for c in bundle)))
            if ev['status']=='OK' and ev['sum_completion']<best_score-1e-7:
                best,best_score,best_eval=bundle,ev['sum_completion'],ev
                evidence['selected_plan']=name
        if fixed is not None and {c.job_id:c.config.config_id for c in best}!=fixed:
            raise RuntimeError('RELEASE_SERIAL_GPU_CHANGED')
        # Independent repeat, not a result.x/fun-only acceptance.
        checked=evaluate(best)
        if (checked['status']!='OK' or checked['sum_completion']>evaluated['sum_completion']+1e-7
                or not s.audit(best)['valid']):
            raise RuntimeError('RELEASE_PLAN_ACCEPTANCE_FAILED')
        savings={j:evaluated['completion_times'][j]-best_eval['completion_times'][j]
                 for j in s.job_map}
        evidence.update(status='OK',selected=best_eval,
            predicted_sum_completion_saved=evaluated['sum_completion']-best_score,
            per_job_predicted_saved=savings,
            receiver_gain=sum(max(x,0.) for x in savings.values()),
            donor_loss=sum(max(-x,0.) for x in savings.values()),
            solver_status=int(solve.status),seconds=time.perf_counter()-start,
            fo_fixed_gpu=fixed,selected_validated=True)
        return best,evidence


def install(cluster):
    from core80 import Core80Policy
    old=cluster.policy.liveness.__class__
    cluster.policy.liveness.__class__=type('Release'+old.__name__,(ReleaseEvaluationMixin,old),{})
    class ReleasePolicy(Core80Policy):
        def optimize(self,*args):
            answer=super().optimize(*args)
            self.last_decision['shared_contract'].update(
                policy_version='COMPLETION_EVENT_RELEASE_RECONFIG_V2',
                objective='MIN_PREDICTED_SUM_COMPLETION_OVER_THREE_BUNDLES',
                proposal_objectives=['HISTORICAL_RECIPROCAL','STATIC_SUM_ETA','CURRENT_VALID_BUNDLE'],
                admission_and_service_unchanged=True,global_optimum=False)
            for key in ('clean_contract',):
                self.last_decision[key]['objective']='MIN_PREDICTED_SUM_COMPLETION_OVER_THREE_BUNDLES'
            self.last_decision['probe']['contract']['objective']='MIN_PREDICTED_SUM_COMPLETION_OVER_THREE_BUNDLES'
            ev=self.last_decision['result']['audit']['release_completion']
            self.last_decision['completion_event_objective_seconds']=ev.get('selected',{}).get('sum_completion')
            self.last_decision['legacy_utility_field_is_diagnostic_only']=True
            self.last_decision['result']['status']=('RELEASE_EVALUATED_FEASIBLE'
                if ev['status']=='OK' else 'RELEASE_ABSTAIN_'+ev['status'])
            return answer
    cluster.policy.__class__=ReleasePolicy
    return cluster
