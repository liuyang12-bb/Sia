import contextlib,io,json,math,unittest
from pathlib import Path
import pandas as pd
from joint_prototype.full160 import make_cluster
from joint_prototype.bundle_adaptation import install as install_old
from joint_prototype.exchange_netprogress import install,window_utility
from joint_prototype.execution_audit import audit_execution

ROOT=Path(__file__).resolve().parents[3]
class ExchangeIntegration(unittest.TestCase):
    def build(self):
        p=json.loads((ROOT/'research_runs/sia_synergy/small_mechanism_20261025/preregistration.json').read_text())
        return make_cluster(json.loads(Path(p['config']).read_text()),
            pd.DataFrame([dict(name='bert-mid',time=17,application='bert',num_replicas=1,batch_size=384)]),
            p['registry'],p['models'],'joint_o','medium','half',False)

    def test_both_flags_off_match_actual_g_decisions_progress_restarts(self):
        with contextlib.redirect_stdout(io.StringIO()):
            old=install_old(self.build());new=install(self.build(),False,False)
            for _ in range(8):
                old.step(60);new.step(60)
                self.assertEqual(old.allocations,new.allocations)
                self.assertEqual(old.jobs[0].progress,new.jobs[0].progress)
                self.assertEqual(old.jobs[0].num_restarts,new.jobs[0].num_restarts)
                self.assertEqual(old.policy.current_resources,new.policy.current_resources)

    def test_unseeded_full_schedule_and_primary_scores_no_proxy_double_charge(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c=install(self.build(),True,True);productive=0;locks=0
            for _ in range(10):
                c.step(60);d=c.policy.last_decision
                self.assertTrue(d['result']['audit']['valid'])
                self.assertTrue(d['shared_contract']['admission_unchanged'])
                for v in d['net_progress_scores']:
                    self.assertAlmostEqual(v['utility'],window_utility(v['normalized_rate_before_delay'],v['candidate_delay']))
                for e in c.joint_round_logs[-1]['execution']:
                    self.assertTrue(audit_execution(e)['valid']);productive+=e['training_seconds']
                    locks+=e['checkpoint_delay_consumed']>0
                    available=max(0.,e['interval_end']-max(e['interval_start'],e['arrival_time']))
                    if e['checkpoint_delay_consumed']>=available>0:
                        self.assertAlmostEqual(e['progress_after'],e['progress_before'])
                for v in d['result']['audit']['service_delay_ledgers'].values():self.assertLessEqual(v['delay_charged'],250)
            self.assertGreater(productive,0);self.assertGreater(locks,0)

if __name__=='__main__':unittest.main()
