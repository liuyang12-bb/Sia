import gzip,json,unittest
from pathlib import Path
import pandas as pd
from joint_prototype.full160 import make_cluster
from joint_prototype.bundle_adaptation import install

class LegacyTrajectory(unittest.TestCase):
    def test_unchanged_bundle_first_eight_rounds_match_frozen_allocation_and_execution(self):
        root=Path(__file__).resolve().parents[3]
        old=root/'research_runs/sia_synergy/joint_budgeted_adaptation_20261025/attempt3'
        p=json.loads((old/'preregistration.json').read_text())
        c=install(make_cluster(json.loads(Path(p['config']).read_text()),pd.read_csv(p['workloads']['8']),
            p['registry'],p['models'],'joint_o','neutral','main',True))
        with gzip.open(old/'n8_neutral_joint_g/rounds.jsonl.gz','rt') as f:
            for _,line in zip(range(8),f):
                expected=json.loads(line);c.step(60);r=c.joint_round_logs[-1]
                d=r['decision_for_next_interval']['result'];e=expected['decision_for_next_interval']['result']
                self.assertEqual(d['selected'],e['selected'])
                for actual,prior in zip(r['execution'],expected['execution']):
                    for key in ['job_id','progress_before','progress_after','rho','training_seconds','checkpoint_delay_consumed','completion_time']:
                        self.assertEqual(actual[key],prior[key],key)
