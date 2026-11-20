"""Stage2B: bounded, preregistered <=3 job fixtures, never a cohort run."""
import copy
import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_joint_integration_coverage as fixtures
from joint_prototype.joint_profile import GpuConfig, JobSpec, NodeSpec, JointProfileOracle
from joint_prototype.joint_scheduler import JointScheduler


def evidence(name, value):
    root = os.environ.get('STAGE2B_EVIDENCE_DIR')
    if root:
        p=Path(root); p.mkdir(parents=True,exist_ok=True)
        with (p/(name+'.json')).open('x',encoding='utf-8') as f:
            json.dump(value,f,indent=2,default=str)


class LegacyAdmission:
    """Red-test control only: actual old solver, no liveness constraints."""
    def __init__(self, **kwargs):
        self.ready=[]; self.services={}; self.events=[]
    def decide(self,scheduler,columns,now,productive):
        return scheduler.solve('LEGACY_CONTROL',columns,('gpu','cpu','ram'))


def admission():
    try:
        from joint_prototype.neutral_liveness import L2Admission
        return L2Admission(quantum=300,delay_bound=250,interval=60)
    except ModuleNotFoundError:
        return LegacyAdmission()


def schedule(jobs, nodes=None):
    return JointScheduler(nodes or [NodeSpec('aws:0','aws',1,3,62.5)],jobs,
        JointProfileOracle(fixtures.MODELS,'fixed_reference',neutral_response=True),
        [1,2,3],[20.83,62.5],verify_exact=False,minimum_cpu_per_gpu=1,
        minimum_ram_gb_per_gpu=20.83,candidate_policy='b3_prop')


def spec(name, arrival=0, gpus=1, nodes=1):
    return JobSpec(name,'res18_1.py',(GpuConfig(f'aws:{nodes}:{gpus}','aws',gpus,nodes,1.),),arrival_time=arrival)


def current(job, selected, round_index=0, locked=False):
    c=selected.get(job.job_id)
    return replace(job,query_round=round_index,checkpoint_locked=locked and c is not None,
        current_config_id=c.config.config_id if c else None,
        current_nodes=c.placement.node_ids if c else (),
        current_placement=c.placement.node_gpus if c else (),
        current_cpu_per_gpu=c.query.cpu_per_gpu if c else None,
        current_ram_gb_per_gpu=c.query.ram_gb_per_gpu if c else None)


def negative(s):
    # Negative objective fixture, NOT falsified training Goodput.
    return [replace(c,scheduling_utility=-100.) for c in s.build_proportional_columns()]


class Stage2BNeutralTests(unittest.TestCase):
    def test_negative_columns_serve_real_fifo_not_job_id(self):
        a=admission(); s=schedule([spec('a',40),spec('z',20)])
        r=a.decide(s,negative(s),60,{})
        self.assertEqual(set(r.selected),{'z'})
        self.assertTrue(r.audit['valid'])

    def test_repeat_service_requeues_after_productive_not_delay(self):
        a=admission(); jobs=[spec('first'),spec('second',1)]; chosen={}; totals={j.job_id:0. for j in jobs}; log=[]
        for i in range(16):
            now=60*i
            active=[current(j,chosen,i) for j in jobs if j.arrival_time<=now]
            s=schedule(active); r=a.decide(s,negative(s),now,totals)
            chosen=r.selected
            log.append(dict(time=now,selected=sorted(chosen),productive=dict(totals),events=copy.deepcopy(a.events)))
            for name in chosen: totals[name]+=60
        evidence('repeat_state_machine',log)
        self.assertEqual(set(log[0]['selected']),{'first'})
        self.assertEqual(set(log[5]['selected']),{'second'})
        self.assertEqual(set(log[10]['selected']),{'first'})
        self.assertTrue(all(v>=300 for v in totals.values()))

    def test_large_gang_drains_without_backfill(self):
        nodes=[NodeSpec('aws:0','aws',2,6,125),NodeSpec('aws:1','aws',2,6,125)]
        jobs=[spec('small-a'),spec('small-b'),spec('gang',60,4,2)]
        a=admission(); chosen={}; totals={j.job_id:0. for j in jobs}; log=[]
        for i in range(12):
            s=schedule([current(j,chosen,i) for j in jobs if j.arrival_time<=60*i],nodes)
            r=a.decide(s,negative(s),60*i,totals); chosen=r.selected
            log.append(dict(time=60*i,selected=sorted(chosen),events=copy.deepcopy(a.events),audit=r.audit))
            for name in chosen: totals[name]+=60
        evidence('gang_drain',log)
        self.assertIn('DRAIN_FOR_HEAD',str(log))
        self.assertIn('gang',log[5]['selected'])
        self.assertTrue(all(v>=300 for v in totals.values()))

    def test_checkpoint_and_service_separate_full_geometry(self):
        a=admission(); job=spec('j'); s=schedule([job]); r=a.decide(s,negative(s),0,{})
        self.assertIn('j',r.selected)
        old=r.selected['j']
        for i,locked in enumerate((True,False),1):
            s=schedule([current(job,r.selected,i,locked)])
            r=a.decide(s,negative(s),i*60,{'j':0})
            self.assertEqual(r.selected['j'].placement,old.placement)
            self.assertEqual(r.selected['j'].query.cpu_per_gpu,old.query.cpu_per_gpu)
            self.assertEqual(r.selected['j'].candidate_kind=='locked',locked)
        self.assertIn('j',a.services)

    def test_no_candidates_explicit_stop(self):
        a=admission(); s=schedule([replace(spec('none'),configs=())])
        with self.assertRaisesRegex(RuntimeError,'NO_SOLO_FEASIBLE_CANDIDATE'):
            a.decide(s,[],0,{})

    def test_solver_failure_no_partial_transaction(self):
        a=admission(); s=schedule([spec('j')]); snapshot=copy.deepcopy(a.__dict__)
        # Fault injection only into the new solver API, never Goodput or job state.
        try:
            import joint_prototype.neutral_liveness as module
        except ModuleNotFoundError:
            self.fail('new solver failure path is absent')
        with patch.object(module,'milp',return_value=SimpleNamespace(x=None,status=1,success=False,message='timeout')):
            with self.assertRaisesRegex(RuntimeError,'SOLVER_UNDECIDED'):
                a.decide(s,negative(s),0,{})
        self.assertEqual(a.__dict__,snapshot)

    def test_atomic_invalid_solution_refused(self):
        import joint_prototype.neutral_liveness as module
        a=admission(); s=schedule([spec('one'),spec('two')])
        with patch.object(module,'milp',return_value=SimpleNamespace(x=[1.,1.],status=0,success=True,message='forged')):
            with self.assertRaisesRegex(RuntimeError,'INVALID_SOLVER_ALLOCATION'):
                a.decide(s,negative(s),0,{})
        self.assertEqual(a.services,{})

    def test_zero_execution_rate_explicit_stop(self):
        from joint_prototype.neutral_liveness import validate_productive_segments
        with self.assertRaisesRegex(RuntimeError,'ZERO_EXECUTION_RATE'):
            validate_productive_segments('bad',[dict(seconds=60,raw_rate_before_rho=0,effective_rate=0,rho=1)])
        validate_productive_segments('checkpoint',[])

    def test_expired_fixed_proposal_is_recomputed_before_factor_filter(self):
        from joint_prototype.neutral_liveness import NeutralL2Policy
        # Same-state candidate contract only, no fabricated training rollout.
        p=object.__new__(NeutralL2Policy); p.mode='u_fp'; p.p_fairness=-.5; p.lambda_n=1.1
        p.liveness=admission(); p.productive={'j':300.}
        a=GpuConfig('aws:1:1','aws',1,1,1.)
        b=GpuConfig('aws:1:2','aws',2,1,2.)
        job=JobSpec('j','res18_1.py',(a,b),current_config_id=a.config_id,current_nodes=('aws:0',),current_placement=(('aws:0',1),))
        s=schedule([job],[NodeSpec('aws:0','aws',4,12,250)])
        p.liveness.services={'j':dict(productive_base=0.)}
        info=SimpleNamespace(age=0,num_restarts=0)
        cols,proposal=p.factor_columns({'j':info},[job],s)
        self.assertEqual(proposal['j'],b.config_id)
        self.assertEqual({c.config.config_id for c in cols},{b.config_id})

    def test_infeasible_commitment_stops_without_mutation(self):
        a=admission(); job=spec('j'); s=schedule([job]); r=a.decide(s,negative(s),0,{})
        snapshot=copy.deepcopy(a.__dict__)
        bad=current(job,r.selected,1)
        s=schedule([bad],[NodeSpec('aws:0','aws',1,.5,62.5)])
        with self.assertRaises((RuntimeError,ValueError)):
            a.decide(s,negative(s),60,{'j':0.})
        self.assertEqual(a.__dict__,snapshot)

    def test_solver_failure_during_service_preserves_entire_committed_state(self):
        import joint_prototype.neutral_liveness as module
        a=admission(); job=spec('j'); s=schedule([job]); r=a.decide(s,negative(s),0,{})
        snapshot=copy.deepcopy(a.__dict__)
        s=schedule([current(job,r.selected,1,True)])
        with patch.object(module,'milp',return_value=SimpleNamespace(x=None,status=1,success=False,message='timeout')):
            with self.assertRaisesRegex(RuntimeError,'SOLVER_UNDECIDED'):
                a.decide(s,negative(s),60,{'j':0.})
        self.assertEqual(a.__dict__,snapshot)

    def test_delay_bound_and_fifo_wait_bound_independent_small_state_enumeration(self):
        # Exhaust the 8 corner combinations of per-job startup delays 0/250.
        # Single resource: no solver utility assumption enters the expected bound.
        import itertools
        log=[]
        for delays in itertools.product((0,250),repeat=3):
            a=admission(); jobs=[spec(str(i),i*17) for i in range(3)]
            totals={j.job_id:0. for j in jobs}; selected={}; remaining={}; first={}; previous=None
            for i in range(30):
                for name in selected:
                    delay=min(60,remaining[name]); remaining[name]-=delay
                    dt=60-delay; totals[name]+=dt
                    if dt>0 and name not in first: first[name]=(i-1)*60+delay
                now=i*60
                s=schedule([current(j,selected,i,remaining.get(j.job_id,0)>0) for j in jobs if j.arrival_time<=now])
                r=a.decide(s,negative(s),now,totals)
                for name in r.selected:
                    if name not in selected: remaining[name]=delays[int(name)]
                selected=r.selected
            self.assertEqual(set(first),{'0','1','2'})
            bound=60+4*(250+300+60)+250+60
            self.assertTrue(all(first[j.job_id]-j.arrival_time<=bound for j in jobs))
            self.assertTrue(all(v>=300 for v in totals.values()))
            log.append(dict(delays=delays,first=first,productive=totals,bound=bound,events=a.events))
        evidence('state_boundary_enumeration',log)

    def cluster(self,cell,rows,capacities=None,ngpus=None):
        from joint_prototype.neutral_liveness import NeutralL2Policy, NeutralL2Cluster
        f=fixtures.JointIntegrationCoverageTests(); f.setUp()
        c=f._cluster(rows,mode=cell,policy_class=NeutralL2Policy,capacities=capacities,ngpus=ngpus)
        # Same constructor/Job objects as real JointCluster, subclass adds the
        # execution observation hook before genuine optimize_policy only.
        c.__class__=NeutralL2Cluster
        return c

    def test_four_cells_same_state_candidate_inclusions(self):
        rows=[dict(name='bert-mid',time=17,application='bert',num_replicas=1,batch_size=384)]
        keys={}; contracts={}
        for mode in ('u_fp','u_fo','u_jp','u_jo'):
            c=self.cluster(mode,rows); c.step(60)
            keys[mode]={tuple(k) for k in c.policy.last_decision['factor_candidate_keys']}
            contracts[mode]=c.policy.last_decision['shared_contract']
        self.assertEqual(contracts['u_fp'],contracts['u_jo'])
        self.assertTrue(keys['u_fp']<=keys['u_fo']<=keys['u_jo'])
        self.assertTrue(keys['u_fp']<=keys['u_jp']<=keys['u_jo'])
        evidence('factor_contracts',dict(keys={k:sorted(v) for k,v in keys.items()},contracts=contracts))

    def test_unseeded_bert_real_chain_no_seed_or_optimize_mock(self):
        for mode in ('u_fp','u_fo','u_jp','u_jo'):
            c=self.cluster(mode,[dict(name='bert-mid',time=17,application='bert',num_replicas=1,batch_size=384)])
            for dt in (60,60,60,1): c.step(dt)
            j=c.jobs[0]
            self.assertEqual(j.first_allocation_time,60)
            self.assertEqual(j.first_productive_training_time,180)
            self.assertGreater(j.progress,0)
            self.assertEqual(j.num_restarts,0)
            self.assertTrue(all(e['independent_execution_audit']['valid'] for r in c.joint_round_logs for e in r['execution']))
            evidence('real_unseeded_'+mode,c.joint_round_logs)

    def test_three_real_jobs_repeated_neutral_service_30_round_cap(self):
        rows=[dict(name='bert-'+str(i),time=i*17,application='bert',num_replicas=1,batch_size=384) for i in range(3)]
        for mode in ('u_fp','u_fo','u_jp','u_jo'):
            c=self.cluster(mode,rows,
                capacities={t:dict(cpu_cores=1,ram_gb=62.5) for t in ('aws','dgx-ext','rtx')})
            for _ in range(30): c.step(60)
            self.assertTrue(all(j.progress>0 for j in c.jobs))
            self.assertTrue(all(j.execution_time>=300 for j in c.jobs))
            self.assertGreaterEqual(c.policy.liveness.sequence,6)
            self.assertTrue(all(r['decision_for_next_interval']['result']['audit']['valid'] for r in c.joint_round_logs))
            self.assertTrue(all(e['rho']==1 for r in c.joint_round_logs for e in r['execution']))
            evidence('real_three_'+mode,dict(rounds=c.joint_round_logs,
                final=[dict(job=j.name,progress=j.progress,training=j.execution_time,restarts=j.num_restarts,completion=j.completion_time) for j in c.jobs]))


if __name__=='__main__': unittest.main()
