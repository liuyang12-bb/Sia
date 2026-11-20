import copy,math,time
from collections import Counter
from dataclasses import replace
import numpy as np
from scipy.optimize import LinearConstraint
from .adaptive_service import AdaptiveAdmission,gpu_geometry
from .neutral_liveness import geometry
from .joint_scheduler import PolicyResult
from .tail_recovery import TailRecoveryAdmission,TailRecoveryPolicy
from .small_mechanism import utility

class CleanCore(AdaptiveAdmission):
    def _decide(self, s, columns, now, productive):
        start = time.perf_counter()
        jobs = s.job_map
        if len(jobs) > self.MAX_ACTIVE_JOBS:
            raise RuntimeError('ADAPTIVE_SCOPE_MAX_THREE_ACTIVE_JOBS')
        if any(j.arrival_time > now for j in s.jobs):
            raise RuntimeError('FUTURE_JOB_INPUT')
        if any(not math.isfinite(c.query.rho) or c.query.rho <= 0 for c in columns):
            raise RuntimeError('INVALID_RESPONSE_COLUMN')
        self.ready = [j for j in self.ready if j in jobs]
        for name in list(self.services):
            if name not in jobs:
                self.events.append(dict(time=now, event='COMPLETED_SERVICE', job=name))
                del self.services[name]
        new = sorted((j for j in s.jobs if j.job_id not in self.seen),
                     key=lambda j: (j.arrival_time, j.source_row_index, j.job_id))
        for j in new:
            self.ready.append(j.job_id)
            self.seen.add(j.job_id)
            self.events.append(dict(time=now, event='ARRIVE_READY', job=j.job_id))
        expired = []
        for name, service in self.services.items():
            done = productive.get(name, 0.) - service['productive_base']
            if done < -1e-8:
                raise RuntimeError('NON_MONOTONIC_SERVICE_COUNTER')
            if self.service_expired(name, service, productive):
                if s.lock_active(jobs[name]):
                    raise RuntimeError('CHECKPOINT_AFTER_SERVICE_WITHOUT_RECONFIGURATION')
                expired.append(name)
            else:
                self.check_service_clock(name, service, now, productive)
        for name in sorted(expired, key=lambda n: self.services[n]['sequence']):
            self.ready.append(name)
            previous = self.services.pop(name)
            self.events.append(dict(time=now, event='REQUEUE_PRODUCTIVE_QUANTUM', job=name,
                                    delay_charged=previous['delay_charged'],
                                    geometry_changes=previous['geometry_changes']))
        locked = {j.job_id for j in s.jobs if s.lock_active(j)}
        if locked - set(self.services):
            raise RuntimeError('LOCK_WITHOUT_SERVICE_EVIDENCE')
        held = set(self.services)
        # A COMMON history filter, independent of whether service expired.
        filters = {}
        for name, history in self._lifetime_history.items():
            if name not in jobs:
                continue
            eligible, filters[name] = self.shared_eligible(
                s, jobs[name], history, columns, productive, now)
            columns = [c for c in columns if c.job_id != name] + eligible
        if self.clean_aux == 'prop':
            columns = [c for c in columns if self.is_prop(c) or s.lock_active(jobs[c.job_id])]
        columns = sorted(columns, key=geometry)
        # A physically valid old bundle is a lower bound only if it belongs
        # to THIS round's complete eligibility/quota domain.
        old_keys={name:geometry(h['column']) for name,h in self.services.items()}
        incumbent_membership={name:False for name in old_keys}
        for c in columns:
            if old_keys.get(c.job_id)==geometry(c):incumbent_membership[c.job_id]=True
        traces = []
        search = getattr(s, 'lazy_search', None)
        deferred = search is not None and not search.recovery_eager
        if search is not None:
            search.bind_admission(self, columns, productive, now)

        # Original L2 equality admission, capacity matrix and exact-utility
        # lexicographic tie solving. No allocation bonus or utility change.
        def solve(required, objective=False, tie=False, optional=False, domain=None, serial_pass=False):
            # GPU-first proposal and final allocation share the same objective,
            # capacity/lock/admission rules. Stage2 freezes stage1's service set
            # and GPU type/count/node-count, while placement/aux remain free.
            active_columns = columns if domain is None else domain
            if self.clean_gpu == 'fixed' and not serial_pass:
                prop_domain = [c for c in active_columns if self.is_prop(c) or s.lock_active(jobs[c.job_id])]
                proposal = solve(required, objective, tie, optional, prop_domain, True)
                if proposal is None or not objective:
                    return proposal
                fixed = {c.job_id:c.config.config_id for c in proposal}
                if deferred:
                    final_domain = search.legal_local(search.expand(
                        {name: (config,) for name, config in fixed.items()}
                    ))
                else:
                    final_domain = [c for c in active_columns if fixed.get(c.job_id)==c.config.config_id]
                chosen = solve(set(fixed), True, tie, False, final_domain, True)
                if chosen is None:
                    raise RuntimeError('SERIAL_SECOND_STAGE_INFEASIBLE_DESPITE_PROP_WITNESS')
                if {c.job_id:c.config.config_id for c in chosen} != fixed:
                    raise RuntimeError('SERIAL_GPU_INTENT_CHANGED')
                self.serial_proposals.append(dict(fixed=fixed,
                    proposal=[dict(job=c.job_id,config=c.config.config_id,placement=c.placement.node_gpus,
                        cpu=c.query.cpu_per_gpu,ram=c.query.ram_gb_per_gpu,utility=c.scheduling_utility) for c in proposal],
                    selected=[dict(job=c.job_id,config=c.config.config_id,placement=c.placement.node_gpus,
                        cpu=c.query.cpu_per_gpu,ram=c.query.ram_gb_per_gpu,utility=c.scheduling_utility) for c in chosen]))
                return chosen
            columns_local = active_columns
            required = set(required)
            if not required and not optional:
                if locked:
                    raise RuntimeError('INVALID_EMPTY_LOCK')
                return []
            if not columns_local or any(not any(c.job_id == j for c in columns_local) for j in required):
                return None
            matrix, upper = s._constraint_matrix(columns_local, ('gpu', 'cpu', 'ram'))
            lower = np.full(len(upper), -np.inf)
            for i, j in enumerate(s.jobs):
                lower[i] = float(j.job_id in required)
                upper[i] = 1. if optional else lower[i]
            weights = np.asarray([c.scheduling_utility for c in columns_local])
            constraints = [LinearConstraint(matrix, lower, upper)]
            if optional and not required:
                # Shared nonempty-start safety. No allocation bonus, no FIFO
                # head requirement, no maximum-cardinality surrogate.
                constraints.append(LinearConstraint(np.ones(len(columns_local)), 1., np.inf))

            def call(cost):
                result = self._milp(cost, columns_local, constraints)
                traces.append(dict(required=sorted(required), status=int(result.status),
                                   message=str(result.message),
                                   phase='objective' if objective else 'feasibility'))
                if result.status == 2:
                    return None
                if result.x is None or not result.success or result.status != 0:
                    raise RuntimeError(f'SOLVER_UNDECIDED: {traces[-1]}')
                x = np.asarray(result.x)
                if (len(x) != len(columns_local) or not np.all(np.isfinite(x))
                        or np.any(abs(x - np.round(x)) > 1e-6)):
                    raise RuntimeError('INVALID_SOLVER_ALLOCATION integrality')
                chosen = [c for c, v in zip(columns_local, x) if v > .5]
                audit = s.audit(chosen, require_locks=locked <= required)
                selected_jobs = {c.job_id for c in chosen}
                set_valid = required <= selected_jobs if optional else selected_jobs == required
                if (not audit['valid'] or not set_valid
                        or len(chosen) != len(selected_jobs)):
                    raise RuntimeError(f'INVALID_SOLVER_ALLOCATION: {audit}')
                return chosen

            chosen = call(-weights if objective else np.zeros(len(columns_local)))
            if chosen is None:
                return None
            if tie:
                optimum = sum(c.scheduling_utility for c in chosen)
                constraints.append(LinearConstraint(weights, optimum - 1e-9, optimum + 1e-9))
                chosen = self._secondary_tie(s, columns_local, required, constraints, call, chosen)
                order = ([j.job_id for j in sorted(s.jobs,key=lambda j:(j.arrival_time,j.source_row_index,j.job_id))]
                         if optional else sorted(required))
                for name in order:
                    if optional and name not in required:
                        presence = np.asarray([float(c.job_id == name) for c in columns_local])
                        candidate = call(-presence)
                        if candidate is None:
                            raise RuntimeError('OPTIONAL_TIE_INCONSISTENT')
                        present = float(any(c.job_id == name for c in candidate))
                        constraints.append(LinearConstraint(presence,present,present))
                        if not present:
                            chosen = candidate
                            continue
                    cost = np.zeros(len(columns_local))
                    candidates = [i for i, c in enumerate(columns_local) if c.job_id == name]
                    for rank, i in enumerate(candidates):
                        cost[i] = rank
                    chosen = call(cost)
                    if chosen is None:
                        raise RuntimeError('LEXICOGRAPHIC_OBJECTIVE_INCONSISTENT')
                    c = next(c for c in chosen if c.job_id == name)
                    mask = np.asarray([float(v is c) for v in columns_local])
                    constraints.append(LinearConstraint(mask, 1., 1.))
            return chosen

        admitted = set(held)
        drain = False
        if held and solve(held) is None:
            raise RuntimeError('INFEASIBLE_COMMITMENT')
        if self.ready and self.factors[1]:
            head = self.ready[0]
            if solve({head}) is None:
                raise RuntimeError(f'NO_SOLO_FEASIBLE_CANDIDATE: {head}')
            if solve(admitted | {head}) is None:
                drain = True
                self.events.append(dict(time=now, event='DRAIN_FOR_HEAD', job=head, held=sorted(held)))
            elif getattr(self, '_optional_job_selection', False):
                admitted.add(head)
            else:
                for name in self.ready:
                    if solve(admitted | {name}) is not None:
                        admitted.add(name)
        required_for_audit = set(admitted)
        optional = getattr(self, '_optional_job_selection', False) and not drain
        chosen = solve(admitted, objective=True, tie=True, optional=optional)
        if chosen is None:
            raise RuntimeError('INFEASIBLE_FINAL_ADMISSION')
        if search is not None and search.mode == 'LOCAL':
            from local_search import improve
            status = search.stats['local']
            cutoff = search.deadline - search.audit_reserve
            if not deferred:
                status['local_status'] = 'recovery_round'
            elif drain:
                status['local_status'] = 'queue_drain'
            elif time.perf_counter() >= cutoff:
                status['local_status'] = 'fo_used_search_budget'
            else:
                chosen = improve(self, s, chosen, search, cutoff, now, status)
        release_evidence = None
        if hasattr(self, 'refine_completion_bundle'):
            chosen, release_evidence = self.refine_completion_bundle(s, columns, chosen, now)
        independent = s.audit(chosen)
        accepted_domain = {geometry(c) for c in columns}
        if deferred:
            accepted_domain.update(search.eligible_keys)
        if (not independent['valid'] or any(
                geometry(c) not in accepted_domain for c in chosen)):
            raise RuntimeError(f'INVALID_FINAL_COMMIT: {independent}')
        selected = {c.job_id: c for c in chosen}
        if getattr(self, '_optional_job_selection', False):
            independent.update(required_jobs=sorted(required_for_audit),
                optional_jobs=sorted(set(jobs)-required_for_audit),
                optional_selected=sorted(set(selected)-required_for_audit),
                ready_before_commit=list(self.ready),optional_admission_enabled=optional)
        admitted = set(selected)
        budget_changes = {}
        for name in sorted(admitted - held, key=lambda n: self.ready.index(n)):
            column, job = selected[name], jobs[name]
            same_gpu = (job.current_config_id == column.config.config_id
                        and job.current_placement == column.placement.node_gpus)
            charged = 0. if same_gpu else self._delay(name, column)
            history = self._lifetime_history.get(name, {})
            spent = float(history.get('delay_charged', 0.))
            self.sequence += 1
            self.services[name] = dict(
                column=column, issued_time=now, productive_base=productive.get(name, 0.),
                sequence=self.sequence, delay_charged=spent+charged, geometry_changes=history.get('geometry_changes', 0),
                last_geometry_productive=(productive.get(name, 0.) if charged else
                    history.get('last_geometry_productive',productive.get(name,0.))))
            for key in ('tail_extra_budget', 'tail_recovery_used'):
                if key in history:
                    self.services[name][key] = history[key]
            if spent + charged > self.delay_limit(name, self.services[name]) + 1e-9:
                raise RuntimeError('LIFETIME_REISSUE_BUDGET_VIOLATION')
            budget_changes[name] = dict(before=None, charged_this_round=charged,
                                        change='INITIAL_STARTUP' if charged else 'CONTINUE_EXISTING_GPU')
            self.events.append(dict(time=now, event='ISSUE_SERVICE', job=name,
                                    sequence=self.sequence, delay_charged=charged))
        self.ready = [j for j in self.ready if j not in admitted]
        for name in held:
            service, column = self.services[name], selected[name]
            old = service['column']
            before = {key: service[key] for key in (
                'issued_time', 'productive_base', 'delay_charged',
                'last_geometry_productive', 'geometry_changes')}
            cost = 0.
            change = 'UNCHANGED'
            if gpu_geometry(old) != gpu_geometry(column):
                cost = self._delay(name, column)
                change = 'GPU_GEOMETRY'
                service['delay_charged'] += cost
                service['geometry_changes'] += 1
                service['last_geometry_productive'] = productive.get(name, 0.)
                self.events.append(dict(time=now, event='ADAPT_GEOMETRY', job=name,
                                        old_config=old.config.config_id, new_config=column.config.config_id,
                                        delay_cost=cost, delay_charged=service['delay_charged']))
            elif geometry(old) != geometry(column):
                change = 'AUX_QUOTA'
                self.events.append(dict(time=now, event='ADAPT_AUX', job=name,
                                        old_cpu=old.query.cpu_per_gpu, new_cpu=column.query.cpu_per_gpu,
                                        old_ram=old.query.ram_gb_per_gpu, new_ram=column.query.ram_gb_per_gpu))
            if service['delay_charged'] > self.delay_limit(name, service) + 1e-9:
                raise RuntimeError(f'INVALID_SERVICE_DELAY_LEDGER: {name}')
            service['column'] = column
            budget_changes[name] = dict(before=before, charged_this_round=cost,
                                        change=change)
        ledgers = {name: {key: service[key] for key in (
            'issued_time', 'productive_base', 'delay_charged',
            'last_geometry_productive', 'geometry_changes')}
                   for name, service in self.services.items()}
        for name, values in budget_changes.items():
            values['after'] = ledgers[name]
        independent.update(
            ready=list(self.ready), checkpoint_locks=sorted(locked),
            service_commitments=sorted(self.services), drain=drain,
            solver_trace=traces, service_delay_ledgers=ledgers,
            service_budget=budget_changes,
            adaptive_candidate_filters=filters, shared_eligibility=filters, serial_proposals=self.serial_proposals,
            incumbent_domain_membership=incumbent_membership,
            productive_service={j: productive.get(j, 0.) - v['productive_base']
                                for j, v in self.services.items()})
        if release_evidence is not None:
            independent['release_completion'] = release_evidence
        if search is not None:
            independent['search'] = search.stats
        return PolicyResult(
            'L2_ADAPTIVE', 'DRAIN_FOR_HEAD' if drain else 'OPTIMAL_ADAPTIVE_L2',
            selected, sorted(set(jobs) - admitted),
            sum(c.scheduling_utility for c in chosen),
            sum(c.query.effective_goodput for c in chosen),
            time.perf_counter() - start, audit=independent)




def contract(group, ordered, continuous, gpu, aux):
    return dict(version='CLEAN_AB_V1', group=group,
        objective=('SIA_FAIR_UTILITY_AND_ORIGINAL_RESTART_PROXY' if group=='A' else 'SUM_RECIPROCAL_CURRENT_COMPLETION_ETA'),
        cold_start=('ORIGINAL_SIA_FAIR_COLD_UTILITY' if group=='A' else 'COMMON_UNKNOWN_ETA_ZERO_PROP_BOOTSTRAP'),
        ordered=bool(ordered),continuous=bool(continuous),gpu_factor=gpu,aux_factor=aux,
        eligibility='ALL_STARTED_HISTORY_EVENT_PROFILE_PRODUCTIVE_LIFETIME',
        locks='EXACT_FULL_ALLOCATION',interval=60,service_quantum=300.,startup_budget=250.,
        secondary='MIN_ACTUAL_STARTUP_ON_PRIMARY_TIES',response='SHARED_ORIGINAL_ORACLE')


class CleanAdmission(TailRecoveryAdmission, CleanCore):
    # The shared scope is explicit and tested; no input queue truncation.
    MAX_ACTIVE_JOBS=20
    def is_prop(self,c):
        return (abs(c.query.cpu_per_gpu-c.proportional_cpu_per_gpu)<1e-8 and
                abs(c.query.ram_gb_per_gpu-c.proportional_ram_per_gpu)<1e-8)

    def service_expired(self,name,service,productive):
        if self.factors[2] or productive.get(name,0.)-service['productive_base']<self.quantum-1e-9:
            return False
        cost=self._resume_lower_bound.get(name,math.inf)
        if service['delay_charged']+cost>self.delay_limit(name,service)+1e-9:
            self._budget_hold.add(name)
            return False
        return True

    def shared_eligible(self,s,j,h,columns,productive,now):
        own=[c for c in columns if c.job_id==j.job_id]
        current=gpu_geometry(h['column']);legal=[];rejected=[]
        for c in own:
            reason=None
            same_live=(j.current_config_id==c.config.config_id and j.current_placement==c.placement.node_gpus)
            cost=0. if same_live else self._delay(j.job_id,c)
            if s.lock_active(j):
                if (c.config.config_id!=j.current_config_id or c.placement.node_gpus!=j.current_placement or
                    abs(c.query.cpu_per_gpu-j.current_cpu_per_gpu)>1e-9 or abs(c.query.ram_gb_per_gpu-j.current_ram_gb_per_gpu)>1e-9):
                    reason='REAL_CHECKPOINT_LOCK'
            elif gpu_geometry(c)!=current:
                if productive.get(j.job_id,0.)<=h['last_geometry_productive']+1e-9:
                    reason='NO_PRODUCTIVE_SERVICE_SINCE_GEOMETRY'
                elif self.profile_sources.get(j.job_id,{}).get(c.config.config_id) not in {'direct_profile','projected_profile'}:
                    reason='UNPROFILED_TARGET'
                elif not self.geometry_event_reasons:
                    reason='NO_SHARED_GEOMETRY_EVENT'
            if reason is None and h['delay_charged']+cost>self.delay_limit(j.job_id,h)+1e-9:
                reason='SERVICE_DELAY_BUDGET'
            if reason:rejected.append(dict(column_id=c.column_id,reason=reason))
            else:legal.append(c)
        # The exact original once-only tail exception is offered to every
        # history, not only held jobs; expiry itself cannot unlock candidates.
        if (not getattr(s,'recovery_proved_no_offer',False) and
                not h.get('tail_recovery_used',False) and len(s.job_map)==1 and
                (not self.ready or self.ready==[j.job_id]) and not s.lock_active(j) and
                j.current_config_id is not None and productive.get(j.job_id,0.)>h['last_geometry_productive']+1e-9):
            scores=getattr(s,'current_completion_scores',{})
            def eta(c):
                v=scores.get(c.column_id,{})
                e,r=v.get('estimated_completion'),v.get('raw_rate')
                return e if (e is not None and math.isfinite(e) and e>0 and
                    r is not None and math.isfinite(r) and r>0) else None
            inc=[eta(c) for c in legal if gpu_geometry(c)==current and eta(c) is not None]
            if inc:
                opts=[c for c in own if gpu_geometry(c)!=current and eta(c) is not None and eta(c)<min(inc)-1e-9
                    and self.profile_sources.get(j.job_id,{}).get(c.config.config_id) in {'direct_profile','projected_profile'}
                    and math.isfinite(self._delay(j.job_id,c)) and 0<=self._delay(j.job_id,c)<=self.delay_bound
                    and scores[c.column_id].get('delay')==self._delay(j.job_id,c)
                    and s.audit([c],require_locks=False)['valid']]
                if opts:
                    best=min(opts,key=lambda c:(eta(c),c.column_id))
                    if best not in legal:
                        cost=self._delay(j.job_id,best)
                        # Recovery eligibility belongs to one GPU geometry,
                        # not a quota/tie-break winner. Close that geometry over
                        # current, audited, cost-paying auxiliary candidates so
                        # the serial proportional proposal has the same permit.
                        permitted=[c for c in opts if gpu_geometry(c)==gpu_geometry(best)
                            and self._delay(j.job_id,c)==cost]
                        self._pending_tail[j.job_id]=dict(column_id=best.column_id,
                            column_ids=sorted(c.column_id for c in permitted),
                            eta_by_column={c.column_id:eta(c) for c in permitted},
                            gpu_geometry=gpu_geometry(best),
                            recovery_scope='SOLO_GPU_GEOMETRY_QUOTA_CLOSURE',
                            extra_budget=max(0.,h['delay_charged']+cost-self.delay_bound),actual_startup=cost,
                            eta_before=min(inc),eta_after=eta(best),time=now)
                        legal.extend(c for c in permitted if c not in legal)
        if h.get('tail_recovery_used',False):legal=[c for c in legal if gpu_geometry(c)==current]
        return legal,dict(rule='SHARED_FOR_HELD_AND_REQUEUED_HISTORY',
            eligible_geometries=sorted(str(geometry(c)) for c in legal),
            eligible_count=len(legal),rejected=rejected,
            rejection_counts=dict(Counter(x['reason'] for x in rejected)),
            delay_charged=h['delay_charged'], geometry_event_reasons=list(self.geometry_event_reasons))

    def _decide(self,s,columns,now,productive):
        self._lifetime_history=getattr(self,'_lifetime_history',{})
        self._lifetime_history.update(copy.deepcopy(self.services))
        self._resume_lower_bound={name:min((self._delay(name,c) for c in columns if c.job_id==name),default=math.inf) for name in s.job_map}
        self._budget_hold=set();self.serial_proposals=[]
        result=super()._decide(s,columns,now,productive)
        for name in sorted(self._budget_hold):
            self.events.append(dict(time=now,event='QUANTUM_HELD_NO_LIFETIME_RESUME_BUDGET',job=name))
        self._lifetime_history.update(copy.deepcopy(self.services))
        result.audit.update(budget_safety_holds=sorted(self._budget_hold),
            ordered=bool(self.factors[1]),continuous=bool(self.factors[2]),
            candidate_eligibility_independent_of_service_membership=True,
            nonempty_start_safety=True,individual_starvation_guarantee=False,
            serial_proposals=self.serial_proposals)
        result.policy='CLEAN_AB';result.status='DRAIN_FOR_HEAD' if result.audit['drain'] else 'OPTIMAL_CLEAN_AB'
        return result


class CleanPolicy(TailRecoveryPolicy):
    def factor_columns(self,jobs,specs,s):
        columns,proposals=super().factor_columns(jobs,specs,s)
        if self.clean_group=='A':
            columns=utility(self,jobs,specs,columns)
        self.exchange_columns=columns
        return columns,proposals

    def optimize(self,*args):
        answer=super().optimize(*args)
        effective=contract(self.clean_group,self.liveness.factors[1],
            self.liveness.factors[2],self.liveness.clean_gpu,self.liveness.clean_aux)
        self.last_decision['clean_contract']=effective
        self.last_decision['shared_contract'].update(objective=effective['objective'],
            admission='FIFO_HEAD_DRAIN' if self.liveness.factors[1] else 'OBJECTIVE_OPTIONAL_NONEMPTY',
            service='UNTIL_COMPLETION' if self.liveness.factors[2] else '300_PRODUCTIVE_SECONDS_WITH_BUDGET_SAFETY',
            eligibility=effective['eligibility'],cold_start=effective['cold_start'],
            startup_budget_scope='JOB_LIFETIME',nonpreemptive_completion=bool(self.liveness.factors[2]),
            p_used=self.clean_group=='A',lambda_n_used=self.clean_group=='A',net_progress=False)
        self.last_decision['probe']['contract'].update(objective=effective['objective'],
            p_used=self.clean_group=='A',lambda_n_used=self.clean_group=='A',net_progress=False)
        # Parent pre-score diagnostics are not this adapter's objective.
        self.last_decision.pop('net_progress_scores',None)
        return answer


def install(cluster,group='A',ordered=1,continuous=1,gpu='joint',aux='prop'):
    if group not in ('A','B') or gpu not in ('fixed','joint') or aux not in ('prop','opt'):
        raise ValueError('UNREGISTERED_CLEAN_FACTOR')
    if cluster.policy.liveness.seen:raise RuntimeError('INSTALL_BEFORE_FIRST_DECISION')
    cluster.policy.__class__=CleanPolicy
    cluster.policy.clean_group=group
    if group=='B':
        # Score the SAME complete universe before the quota factor is applied
        # in CleanCore. A shared positive normalizer alone does not change the
        # mathematical objective, but it also keeps numerical tie bands equal.
        cluster.policy.mode='u_jo'
    cluster.policy.liveness.__class__=CleanAdmission
    cluster.policy.liveness.factors=(0 if group=='A' else 1,int(ordered),int(continuous))
    cluster.policy.liveness.clean_gpu=gpu
    cluster.policy.liveness.clean_aux=aux
    return cluster
