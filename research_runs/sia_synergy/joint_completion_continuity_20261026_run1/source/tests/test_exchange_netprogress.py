"""Analytic response witnesses; these are declared synthetic, not measured."""
import math
import unittest
from dataclasses import replace
from pathlib import Path
from test_adaptive_service import cfg, spec, current, admission
from joint_prototype.joint_profile import JointProfileOracle, NodeSpec
from joint_prototype.neutral_liveness import FactorScheduler, geometry
from joint_prototype.bundle_adaptation import BundleAdmission
from joint_prototype.small_mechanism import fixed_set_solve

ROOT=Path(__file__).resolve().parents[3]
try:
    from joint_prototype.exchange_netprogress import fair_utility, window_utility, configure_scheduler
except ModuleNotFoundError:
    # Red phase: evaluate the previous algorithm, not a fake replacement.
    def fair_utility(rate,p=-.5,lambda_n=1.1):return lambda_n-round(max(rate,1e-8)**p,2)
    def window_utility(rate,delay,horizon=300,p=-.5,lambda_n=1.1):return fair_utility(rate,p,lambda_n)
    def configure_scheduler(s,allow_exchange):return s

class AnalyticOracle(JointProfileOracle):
    def __init__(self,kind='cpu'):
        super().__init__(ROOT/'research_sources/Synergy/simulator/models','fixed_reference',neutral_response=True)
        self.kind=kind
    def query(self,job,config,cpu,ram,prop):
        q=super().query(job,config,cpu,ram,prop)
        if self.kind=='cpu':
            rate=({1:.4,2:.7,3:1.,4:1.5,6:1.5} if job.job_id=='receiver' else
                  {1:.7,2:.95,3:1.,4:1.,6:1.}).get(cpu,1.)
        elif self.kind=='ram':
            rate=({20.83:.3,62.5:.5,125:1.,187.5:1.5,250:1.5} if job.job_id=='receiver' else
                  {20.83:.7,62.5:.95,125:1.,187.5:1.,250:1.})[ram]
        else:rate=1.
        return replace(q,rho=rate,raw_rho=rate,effective_goodput=config.sia_goodput*rate,
                       evidence='DECLARED_ANALYTIC_TEST_RESPONSE')

def witness(kind='cpu',exchange=False):
    jobs=[spec('receiver'),spec('donor',index=1)]
    nodes=[NodeSpec('aws:0','aws',2,6.,250.)]
    s=FactorScheduler(nodes,jobs,AnalyticOracle(kind),[1.,2.,3.,4.,6.],
        [20.83,62.5,125.,187.5,250.],verify_exact=False,
        minimum_cpu_per_gpu=1.,minimum_ram_gb_per_gpu=20.83,candidate_policy='b3')
    return jobs,configure_scheduler(s,exchange)

class ExchangeNetTests(unittest.TestCase):
    def test_cpu_exchange_enters_candidates_and_improves_existing_fairness(self):
        jobs,s=witness(exchange=True)
        cs=[replace(c,scheduling_utility=fair_utility(c.query.effective_goodput)) for c in s.build_joint_columns()]
        r=fixed_set_solve(s,cs,[j.job_id for j in jobs])
        self.assertEqual(r.selected['receiver'].query.cpu_per_gpu,4.)
        self.assertEqual(r.selected['donor'].query.cpu_per_gpu,2.)
        self.assertAlmostEqual(sum(c.query.effective_goodput for c in r.selected.values()),2.45)
        self.assertTrue(r.audit['valid'])

    def test_ram_exchange_enters_candidates(self):
        jobs,s=witness('ram',True)
        cs=[replace(c,scheduling_utility=fair_utility(c.query.effective_goodput)) for c in s.build_joint_columns()]
        r=fixed_set_solve(s,cs,[j.job_id for j in jobs])
        self.assertEqual(r.selected['receiver'].query.ram_gb_per_gpu,187.5)
        self.assertEqual(r.selected['donor'].query.ram_gb_per_gpu,62.5)

    def test_historical_mode_retains_floor(self):
        _,s=witness(exchange=False)
        self.assertTrue(all(c.query.effective_goodput>=1 for c in s.build_joint_columns()))

    def test_real_delay_can_reverse_greedy_speed_ranking(self):
        self.assertGreater(window_utility(1.,0),window_utility(1.1,120.))

    def test_startup_included_for_new_job_and_not_multiplied_by_rho(self):
        self.assertEqual(window_utility(2.,150.),fair_utility(1.))

    def test_invalid_delay_rejected(self):
        for d in [-1.,float('nan')]:
            with self.assertRaises(ValueError):window_utility(1.,d)

    def test_exact_locks_remain_mandatory_with_exchange(self):
        jobs,s=witness(exchange=True)
        cs=[replace(c,scheduling_utility=fair_utility(c.query.effective_goodput)) for c in s.build_joint_columns()]
        a=BundleAdmission(300.,250.,60.)
        base=admission(jobs,50.);a.delay_costs=base.delay_costs;a.profile_sources=base.profile_sources
        first=a.decide(s,cs,0.,{})
        locked=[current(j,first.selected,locked=True) for j in jobs]
        s.jobs=locked;s.job_map={j.job_id:j for j in locked}
        r=a.decide(s,[replace(c,scheduling_utility=-100.) for c in s.build_joint_columns()],60.,{})
        for j in jobs:self.assertEqual(geometry(r.selected[j.job_id]),geometry(first.selected[j.job_id]))

    def test_capacity_audit_still_rejects_overallocation(self):
        _,s=witness(exchange=True)
        cs=[c for c in s.build_joint_columns() if c.query.cpu_per_gpu==4.]
        self.assertFalse(s.audit([next(c for c in cs if c.job_id==j) for j in ['receiver','donor']])['valid'])

if __name__=='__main__':unittest.main()
