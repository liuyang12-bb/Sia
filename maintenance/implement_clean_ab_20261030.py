from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'cc-joint-queue-ablation/sia-simulator/joint_prototype/queue_factorial.py'
DEST=ROOT/'cc-joint-clean-ab/sia-simulator/joint_prototype/clean_ablation.py'
def main():
 text=OLD.read_text(encoding='utf8')
 text=text[text.index('import copy,math,time'):text.index('class QueueFactorAdmission')]
 text=text.replace('class FactorCore(AdaptiveAdmission):','class CleanCore(AdaptiveAdmission):')
 text=text.replace("        if self.factors[1:] == (1, 1):\n            return super()._decide(s, columns, now, productive)\n",'')
 start=text.index('        filters = {}')
 end=text.index('        traces = []',start)
 text=text[:start]+'''        # A COMMON history filter, independent of whether service expired.
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
''' + text[end:]
 text=text.replace('        def solve(required, objective=False, tie=False, optional=False):', '''        def solve(required, objective=False, tie=False, optional=False, domain=None, serial_pass=False):
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
            columns_local = active_columns''')
 # Restrict lexical columns only inside solve; the outer current-round domain
 # remains intact for the independent final candidate-membership audit.
 start=text.index('            required = set(required)',text.index('        def solve('))
 end=text.index('\n        admitted = set(held)',start)
 section=text[start:end]
 import re
 section=re.sub(r'\bcolumns\b','columns_local',section)
 text=text[:start]+section+text[end:]
 text=text.replace("            adaptive_candidate_filters=filters,", "            adaptive_candidate_filters=filters, shared_eligibility=filters, serial_proposals=self.serial_proposals,")
 text += '''

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
        if (not h.get('tail_recovery_used',False) and len(s.job_map)==1 and
                (not self.ready or self.ready==[j.job_id]) and not s.lock_active(j) and
                j.current_config_id is not None and productive.get(j.job_id,0.)>h['last_geometry_productive']+1e-9):
            scores=getattr(s,'current_completion_scores',{})
            def eta(c):
                v=scores.get(c.column_id,{})
                return v.get('estimated_completion') if v.get('raw_rate') is not None else None
            inc=[eta(c) for c in legal if gpu_geometry(c)==current and eta(c) is not None]
            if inc:
                opts=[c for c in own if gpu_geometry(c)!=current and eta(c) is not None and eta(c)<min(inc)-1e-9
                    and self.profile_sources.get(j.job_id,{}).get(c.config.config_id) in {'direct_profile','projected_profile'}
                    and s.audit([c],require_locks=False)['valid']]
                if opts:
                    best=min(opts,key=lambda c:(eta(c),c.column_id))
                    if best not in legal:
                        cost=self._delay(j.job_id,best)
                        self._pending_tail[j.job_id]=dict(column_id=best.column_id,
                            extra_budget=max(0.,h['delay_charged']+cost-self.delay_bound),actual_startup=cost,
                            eta_before=min(inc),eta_after=eta(best),time=now)
                        legal.append(best)
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
        self.last_decision['clean_contract']=contract(self.clean_group,self.liveness.factors[1],
            self.liveness.factors[2],self.liveness.clean_gpu,self.liveness.clean_aux)
        return answer


def install(cluster,group='A',ordered=1,continuous=1,gpu='joint',aux='prop'):
    if group not in ('A','B') or gpu not in ('fixed','joint') or aux not in ('prop','opt'):
        raise ValueError('UNREGISTERED_CLEAN_FACTOR')
    if cluster.policy.liveness.seen:raise RuntimeError('INSTALL_BEFORE_FIRST_DECISION')
    cluster.policy.__class__=CleanPolicy
    cluster.policy.clean_group=group
    cluster.policy.liveness.__class__=CleanAdmission
    cluster.policy.liveness.factors=(0 if group=='A' else 1,int(ordered),int(continuous))
    cluster.policy.liveness.clean_gpu=gpu
    cluster.policy.liveness.clean_aux=aux
    return cluster
'''
 DEST.write_text(text,encoding='utf8')
 print(DEST)
if __name__=='__main__':main()
