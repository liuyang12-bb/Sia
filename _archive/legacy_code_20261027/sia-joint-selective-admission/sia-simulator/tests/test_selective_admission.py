"""Fails on historical greedy membership, then validates opt-in admission."""
import copy,unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from test_adaptive_service import spec,cfg,current,scheduler,columns,admission
from joint_prototype.joint_profile import NodeSpec
from joint_prototype.neutral_liveness import geometry
from joint_prototype.bundle_adaptation import BundleAdmission
try:
    from joint_prototype.selective_admission import SelectiveAdmission
except ModuleNotFoundError:
    SelectiveAdmission=BundleAdmission

def controller(jobs):
    a=SelectiveAdmission(300,250,60);old=admission(jobs,50)
    a.delay_costs=old.delay_costs;a.profile_sources=old.profile_sources
    return a
def node(g=2):return [NodeSpec('aws:0','aws',g,3.*g,62.5*g)]

class SelectiveTests(unittest.TestCase):
    def test_negative_tail_not_automatically_committed(self):
        jobs=[spec('head'),spec('tail',index=1)];a=controller(jobs);s=scheduler(jobs,node())
        r=a.decide(s,columns(s,negative=True),0,{})
        self.assertEqual(set(r.selected),{'head'})
        self.assertEqual(a.ready,['tail'])

    def test_fair_utility_can_prefer_larger_head_to_maximum_concurrency(self):
        jobs=[spec('head',[cfg('aws',1,1),cfg('aws',2,100)]),spec('tail',index=1)]
        a=controller(jobs);s=scheduler(jobs,node());r=a.decide(s,columns(s),0,{})
        self.assertEqual(set(r.selected),{'head'})
        self.assertEqual(r.selected['head'].config.gpus,2)

    def test_negative_tail_gets_fifo_service_after_expiry(self):
        jobs=[spec('head'),spec('tail',index=1)];a=controller(jobs);s=scheduler(jobs,node())
        first=a.decide(s,columns(s,negative=True),0,{})
        s=scheduler([current(j,first.selected) for j in jobs],node())
        r=a.decide(s,columns(s,negative=True),360,{'head':300})
        self.assertEqual(set(r.selected),{'tail'})
        self.assertEqual(a.services['tail']['productive_base'],0)

    def test_positive_optional_can_join_without_new_weight(self):
        jobs=[spec('head'),spec('tail',index=1)];a=controller(jobs);s=scheduler(jobs,node())
        r=a.decide(s,columns(s),0,{})
        self.assertEqual(set(r.selected),{'head','tail'})

    def test_locked_negative_service_remains_complete(self):
        jobs=[spec('a')];a=controller(jobs);s=scheduler(jobs,node())
        first=a.decide(s,columns(s,negative=True),0,{})
        s=scheduler([current(j,first.selected,locked=True) for j in jobs],node())
        r=a.decide(s,columns(s,negative=True),60,{})
        self.assertEqual(geometry(first.selected['a']),geometry(r.selected['a']))
        self.assertEqual(a.services['a']['issued_time'],0)

    def test_drain_for_large_head_disallows_late_small_admission(self):
        a_job=spec('a');b=spec('gang',[cfg('aws',2)],arrival=60,index=1)
        c=spec('late',arrival=60,index=2);a=controller([a_job,b,c])
        s=scheduler([a_job],node());first=a.decide(s,columns(s),0,{})
        js=[current(a_job,first.selected,locked=True),b,c];s=scheduler(js,node())
        r=a.decide(s,columns(s),60,{})
        self.assertTrue(r.audit['drain']);self.assertEqual(set(r.selected),{'a'})
        js=[current(a_job,r.selected),b,c];s=scheduler(js,node())
        r=a.decide(s,columns(s),360,{'a':300})
        self.assertEqual(set(r.selected),{'gang'})

    def test_no_solo_candidate_stops_explicitly(self):
        j=spec('head',[cfg('aws',2)]);a=controller([j])
        s=scheduler([j],[NodeSpec('aws:0','aws',2,1.,125.)])
        with self.assertRaisesRegex(RuntimeError,'NO_SOLO_FEASIBLE_CANDIDATE'):a.decide(s,columns(s),0,{})

    def test_solver_failure_does_not_commit_queue_or_optional_service(self):
        jobs=[spec('head'),spec('tail',index=1)];a=controller(jobs);s=scheduler(jobs,node())
        before=copy.deepcopy(a.__dict__)
        with patch('joint_prototype.adaptive_service.milp',return_value=SimpleNamespace(x=None,status=1,success=False,message='injected')):
            with self.assertRaisesRegex(RuntimeError,'SOLVER_UNDECIDED'):a.decide(s,columns(s),0,{})
        self.assertEqual(a.__dict__,before)

    def test_original_greedy_policy_is_preserved(self):
        jobs=[spec('head'),spec('tail',index=1)];a=BundleAdmission(300,250,60)
        old=admission(jobs,50);a.delay_costs=old.delay_costs;a.profile_sources=old.profile_sources
        s=scheduler(jobs,node());r=a.decide(s,columns(s,negative=True),0,{})
        self.assertEqual(set(r.selected),{'head','tail'})

    def test_exhaustive_three_job_optimum_matches_selected_objective(self):
        import itertools
        jobs=[spec('h',[cfg('aws',1,1),cfg('aws',2,10)]),spec('b',index=1),spec('c',index=2)]
        a=controller(jobs);s=scheduler(jobs,node());cs=columns(s);r=a.decide(s,cs,0,{})
        by={j.job_id:[c for c in cs if c.job_id==j.job_id] for j in jobs};best=-float('inf')
        for choice in itertools.product(by['h'],[None]+by['b'],[None]+by['c']):
            chosen=[c for c in choice if c is not None]
            if s.audit(chosen)['valid']:best=max(best,sum(c.scheduling_utility for c in chosen))
        self.assertAlmostEqual(best,r.scheduling_objective)

    def test_twenty_negative_jobs_receive_two_fifo_services(self):
        jobs=[spec('j%02d'%i,index=i) for i in range(20)]
        a=controller(jobs);productive={j.job_id:0. for j in jobs};last={};sequence=[]
        for tick in range(40):
            s=scheduler([current(j,last,round_index=tick) for j in jobs],node(4))
            r=a.decide(s,columns(s,negative=True),tick*360,productive)
            self.assertEqual(len(r.selected),1);name=next(iter(r.selected));sequence.append(name)
            productive[name]+=300;last=r.selected
        self.assertEqual(sequence,[j.job_id for j in jobs]*2)
