import unittest
import copy
from dataclasses import replace
from unittest.mock import patch
from test_adaptive_service import spec,columns,cfg,current
from test_clean_ablation import controller,state
from joint_prototype.joint_profile import NodeSpec
from joint_prototype.release_evaluation import ReleaseEvaluationMixin
from joint_prototype.neutral_liveness import geometry

class ReleaseIntegrationTests(unittest.TestCase):
    def test_opt_in_accounts_for_donor_before_atomic_commit(self):
        jobs=[spec('receiver'),spec('donor',index=1)]
        s=state(jobs,[NodeSpec('aws:0','aws',2,6.,125.)]);s.enforce_floor=False
        costs={'receiver':{3.:50.,4.:40.},'donor':{3.:200.,1.:500.}}
        cs=[];scores={}
        for c in columns(s):
            if c.query.cpu_per_gpu not in costs[c.job_id] or c.query.ram_gb_per_gpu!=62.5:continue
            eta=costs[c.job_id][c.query.cpu_per_gpu]
            cs.append(replace(c,scheduling_utility=1./eta))
            scores[c.column_id]=dict(raw_rate=100./eta,rho=c.query.rho,
                remaining_work=100.,delay=0.,estimated_completion=eta)
        s.current_completion_scores=scores
        x=controller(jobs,gpu='joint',aux='opt')
        x.__class__=type('TestRelease',(ReleaseEvaluationMixin,x.__class__),{})
        r=x.decide(s,cs,0.,{})
        self.assertEqual(r.selected['receiver'].query.cpu_per_gpu,3.)
        self.assertEqual(r.selected['donor'].query.cpu_per_gpu,3.)
        self.assertTrue(r.audit['valid'])
        ev=r.audit['release_completion']
        self.assertAlmostEqual(ev['baseline']['sum_completion'],276.)
        self.assertAlmostEqual(ev['selected']['sum_completion'],250.)
        self.assertAlmostEqual(ev['receiver_gain']-ev['donor_loss'],26.)

    def test_fixed_fo_cannot_escape_its_frozen_gpu_intent(self):
        jobs=[spec('a',configs=[cfg('aws'),cfg('dgx-ext')])]
        nodes=[NodeSpec(t+':0',t,1,6.,125.) for t in ('aws','dgx-ext')]
        s=state(jobs,nodes);s.enforce_floor=False
        cs=[];scores={}
        # Proportional proposal: DGX50 beats AWS100. AWS nonproportional10
        # would beat every DGX candidate but is outside serial GPU intent.
        etas={('aws',6.):100.,('aws',3.):10.,
              ('dgx-ext',6.):50.,('dgx-ext',3.):30.}
        for c in columns(s):
            key=(c.config.gpu_type,c.query.cpu_per_gpu)
            if key not in etas or c.query.ram_gb_per_gpu!=125.:continue
            eta=etas[key]
            cs.append(replace(c,scheduling_utility=1./eta))
            scores[c.column_id]=dict(raw_rate=100./(eta*c.query.rho),
                rho=c.query.rho,remaining_work=100.,delay=0.,estimated_completion=eta)
        s.current_completion_scores=scores
        x=controller(jobs,gpu='fixed',aux='opt')
        x.__class__=type('TestReleaseFixed',(ReleaseEvaluationMixin,x.__class__),{})
        r=x.decide(s,cs,0.,{})
        self.assertTrue(r.audit['valid'])
        self.assertEqual(r.selected['a'].config.config_id,'dgx-ext:1:1')
        self.assertEqual(r.audit['serial_proposals'][-1]['fixed'],{'a':'dgx-ext:1:1'})
        ev=r.audit['release_completion']
        self.assertEqual(ev['fo_fixed_gpu'],{'a':'dgx-ext:1:1'})
        self.assertTrue(ev['selected_validated'])

    def test_evaluator_failure_does_not_commit_any_ledger(self):
        jobs=[spec('a')];s=state(jobs)
        cs=[c for c in columns(s) if c.query.cpu_per_gpu==3.
            and c.query.ram_gb_per_gpu==62.5]
        s.current_completion_scores={c.column_id:dict(raw_rate=1.,rho=c.query.rho,
            remaining_work=100.,delay=0.,estimated_completion=100./c.query.rho) for c in cs}
        x=controller(jobs,gpu='joint',aux='opt')
        x.__class__=type('TestReleaseAtomic',(ReleaseEvaluationMixin,x.__class__),{})
        before=copy.deepcopy(x.__dict__)
        with patch('joint_prototype.release_evaluation.evaluate_plan',
                   side_effect=RuntimeError('INJECTED_RELEASE_EVALUATOR_FAILURE')):
            with self.assertRaisesRegex(RuntimeError,'INJECTED_RELEASE_EVALUATOR_FAILURE'):
                x.decide(s,cs,0.,{})
        self.assertEqual(x.__dict__,before)
        self.assertEqual(x.services,{})
        self.assertEqual(x.ready,[])

    def test_checkpoint_keeps_current_full_bundle_in_new_mode(self):
        jobs=[spec('a'),spec('b',arrival=60.,index=1)]
        s=state(jobs[:1]);s.enforce_floor=False
        cs=[c for c in columns(s) if c.query.cpu_per_gpu==3.
            and c.query.ram_gb_per_gpu==62.5]
        s.current_completion_scores={c.column_id:dict(raw_rate=1.,rho=c.query.rho,
            remaining_work=100.,delay=50.,estimated_completion=150.) for c in cs}
        x=controller(jobs,gpu='joint',aux='opt')
        x.__class__=type('TestReleaseCheckpoint',(ReleaseEvaluationMixin,x.__class__),{})
        first=x.decide(s,cs,0.,{})
        locked_a=current(jobs[0],first.selected,locked=True)
        next_s=state([locked_a,jobs[1]]);next_s.enforce_floor=False
        next_cols=columns(next_s)
        next_s.current_completion_scores={c.column_id:dict(raw_rate=1.,rho=c.query.rho,
            remaining_work=100.,delay=50.,estimated_completion=150.) for c in next_cols}
        result=x.decide(next_s,next_cols,60.,{'a':0.})
        self.assertTrue(result.audit['valid'])
        self.assertEqual(set(result.selected),{'a'})
        self.assertEqual(geometry(result.selected['a']),geometry(first.selected['a']))
        self.assertEqual(result.selected['a'].candidate_kind,'locked')
        self.assertIn('a',result.audit['checkpoint_locks'])
        self.assertEqual(x.services['a']['delay_charged'],50.)

if __name__=='__main__':unittest.main()
