"""Red harness uses legacy behavior until the opt-in bundle class exists."""
import copy,unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from test_adaptive_service import spec,cfg,current,scheduler,columns,admission
from joint_prototype.joint_profile import NodeSpec
from joint_prototype.neutral_liveness import geometry
try:
    from joint_prototype.bundle_adaptation import BundleAdmission
except ModuleNotFoundError:
    from joint_prototype.scaled_adaptive import ScaledAdmission as BundleAdmission

def controller(jobs,delay=50.):
    a=BundleAdmission(300.,250.,60.)
    old=admission(jobs,delay)
    a.delay_costs=old.delay_costs;a.profile_sources=old.profile_sources
    return a

def swap_state():
    jobs=[spec('a',configs=[cfg('aws',rate=1.),cfg('dgx-ext',rate=10.)]),
          spec('b',configs=[cfg('aws',rate=2.),cfg('dgx-ext',rate=3.)],index=1)]
    nodes=[NodeSpec(t+':0',t,1,3.,62.5) for t in ['aws','dgx-ext']]
    a=controller(jobs);s=scheduler(jobs,nodes)
    cs=[c for c in columns(s) if c.config.gpu_type==('aws' if c.job_id=='a' else 'dgx-ext')]
    first=a.decide(s,cs,0.,{})
    return jobs,nodes,a,first

class BundleTests(unittest.TestCase):
    def test_empty_arrived_set_is_audited_idle(self):
        a=controller([]);s=scheduler([])
        r=a.decide(s,[],0.,{})
        self.assertEqual(r.selected,{})
        self.assertTrue(r.audit['valid'])
        self.assertEqual(r.audit['secondary_tie']['minimum_actual_delay'],0.)

    def test_coordinated_gpu_exchange_may_reduce_one_peer_rate(self):
        jobs,nodes,a,first=swap_state();s=scheduler([current(j,first.selected) for j in jobs],nodes)
        cs=columns(s)
        # Production-like restart factor is charged in primary utility, not bypassed.
        cs=[replace(c,scheduling_utility=1.1-round((c.query.effective_goodput*(1. if c.config.config_id==first.selected[c.job_id].config.config_id else .8))**-.5,2)) for c in cs]
        r=a.decide(s,cs,180.,{'a':130.,'b':130.})
        self.assertEqual(r.selected['a'].config.gpu_type,'dgx-ext')
        self.assertEqual(r.selected['b'].config.gpu_type,'aws')
        self.assertLess(r.selected['b'].query.effective_goodput,first.selected['b'].query.effective_goodput)
        self.assertTrue(r.audit['valid'])
        self.assertEqual(a.services['a']['productive_base'],0.)
        self.assertEqual(a.services['a']['delay_charged'],100.)

    def test_auxiliary_decrease_is_candidate_not_a_per_job_veto(self):
        j=spec('j',template='alexnet_1.py');a=controller([j]);s=scheduler([j],medium=True)
        cs=[c for c in columns(s) if c.query.cpu_per_gpu==6.]
        first=a.decide(s,cs,0.,{})
        s=scheduler([current(j,first.selected)],medium=True)
        legal,ev=a.eligible_held_columns(s,s.jobs[0],a.services['j'],columns(s),{'j':130.},180.)
        base=first.selected['j'].query.effective_goodput
        self.assertTrue(any(c.query.effective_goodput<base-1e-8 for c in legal))

    def test_checkpoint_lock_still_blocks_bundle_exchange(self):
        jobs,nodes,a,first=swap_state()
        s=scheduler([current(j,first.selected,locked=True) for j in jobs],nodes)
        r=a.decide(s,columns(s),60.,{})
        for j in jobs:self.assertEqual(geometry(r.selected[j.job_id]),geometry(first.selected[j.job_id]))

    def test_delay_budget_cannot_be_bypassed_by_global_gain(self):
        jobs,nodes,a,first=swap_state()
        for v in a.services.values():v['delay_charged']=250.
        s=scheduler([current(j,first.selected) for j in jobs],nodes)
        r=a.decide(s,columns(s),180.,{'a':130.,'b':130.})
        for j in jobs:self.assertEqual(r.selected[j.job_id].config.config_id,first.selected[j.job_id].config.config_id)

    def test_exact_primary_tie_prefers_no_actual_restart(self):
        j=spec(configs=[cfg('aws'),cfg('rtx')]);a=controller([j]);s=scheduler([j])
        first=a.decide(s,[c for c in columns(s) if c.config.gpu_type=='rtx'],0.,{})
        s=scheduler([current(j,first.selected)])
        r=a.decide(s,[replace(c,scheduling_utility=.1) for c in columns(s)],180.,{'j':130.})
        self.assertEqual(r.selected['j'].config.gpu_type,'rtx')
        self.assertEqual(a.services['j']['delay_charged'],50.)

    def test_unknown_profile_and_no_training_remain_excluded(self):
        jobs,nodes,a,first=swap_state();s=scheduler([current(j,first.selected) for j in jobs],nodes)
        for name in a.profile_sources:
            a.profile_sources[name]={k:'unprofiled_sia_fallback' for k in a.profile_sources[name]}
        r=a.decide(s,columns(s),180.,{'a':130.,'b':130.})
        for j in jobs:self.assertEqual(r.selected[j.job_id].config.config_id,first.selected[j.job_id].config.config_id)
        r=a.decide(s,columns(s),180.,{'a':0.,'b':0.})
        for j in jobs:self.assertEqual(r.selected[j.job_id].config.config_id,first.selected[j.job_id].config.config_id)

    def test_solver_failure_rolls_back_queue_and_budget(self):
        jobs,nodes,a,first=swap_state();s=scheduler([current(j,first.selected) for j in jobs],nodes)
        before=copy.deepcopy(a.__dict__)
        fail=SimpleNamespace(x=None,status=1,success=False,message='injected')
        with patch('joint_prototype.adaptive_service.milp',return_value=fail):
            with self.assertRaisesRegex(RuntimeError,'SOLVER_UNDECIDED'):a.decide(s,columns(s),180.,{'a':130.,'b':130.})
        self.assertEqual(a.__dict__,before)

    def test_negative_utility_does_not_restore_empty_absorption(self):
        j=spec();a=controller([j]);s=scheduler([j])
        r=a.decide(s,columns(s,negative=True),0.,{})
        self.assertEqual(set(r.selected),{'j'});self.assertTrue(r.audit['valid'])
