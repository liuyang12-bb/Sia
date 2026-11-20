"""Frozen Stage2C fixtures: real unseeded apps, at most 10 x 60s each."""
import copy
import json
import os
from pathlib import Path
import unittest
from dataclasses import replace

import pandas as pd
from joint_prototype.joint_profile import JobSpec, GpuConfig
from joint_prototype.r0_liveness import R0Oracle, R0Policy, R0Cluster, make_cluster

ROOT=Path(__file__).resolve().parents[3]
REG=ROOT/'research_runs/sia_synergy/stage2b_neutral/g0_verified/registry.json'
MODELS=ROOT/'research_sources/Synergy/simulator/models'
CONFIG=ROOT/'research_runs/sia_synergy/b3_multiworkload_medium/followup_w10_late_6000/frozen_config_6000.json'
APPS={'bert':384,'cifar10':4096,'deepspeech2':320,'imagenet':1600,'yolov3':64}

def oracle(response='medium'):
    return R0Oracle(REG,MODELS,{a:a for a in APPS},response)

def evidence(name,data):
    if os.environ.get('STAGE2C_EVIDENCE_DIR'):
        p=Path(os.environ['STAGE2C_EVIDENCE_DIR']); p.mkdir(parents=True,exist_ok=True)
        with (p/(name+'.json')).open('x',encoding='utf-8') as f: json.dump(data,f,default=str)

class Stage2CR0Tests(unittest.TestCase):
    def test_registry_full_domain_reference_identity_and_fixed_phi(self):
        o=oracle(); self.assertEqual(len(o.registry),55)
        for row in o.registry.values():
            app=row['application']; typ,nn,g=row['config_id'].split(':')
            cfg=GpuConfig(row['config_id'],typ,int(g),int(nn),1.,row['phi_c'],row['phi_m'])
            job=JobSpec(app,row['template'],(cfg,))
            a=o.query(job,cfg,3,62.5,(1,125)); self.assertEqual(a.rho,1)
            b=o.query(job,cfg,1,125,(1,125))
            changed=replace(cfg,sia_goodput=100)
            other=replace(job,configs=(changed,),observed_progress=123,query_round=9)
            c=o.query(other,changed,1,125,(1,125))
            self.assertEqual(b.rho,c.rho)
            self.assertEqual(c.effective_goodput,100*b.rho)
            wrong=replace(cfg,cpu_demand_scale=cfg.cpu_demand_scale+.01)
            with self.assertRaisesRegex(ValueError,'REGISTRY_SCALE_MISMATCH'):
                o.query(replace(job,configs=(wrong,)),wrong,1,125,(1,125))

    def test_missing_registry_entry_not_default_one(self):
        o=oracle()
        with self.assertRaisesRegex(ValueError,'UNCOVERED_R0'):
            o.reference('bert','aws:2:8')

    def test_new_scope_does_not_change_old_neutral_scope(self):
        from joint_prototype.neutral_liveness import L2Admission
        from joint_prototype.r0_liveness import R0Admission
        self.assertEqual(L2Admission.MAX_ACTIVE_JOBS,3)
        self.assertEqual(R0Admission.MAX_ACTIVE_JOBS,20)
        self.assertTrue(L2Admission.REQUIRE_NEUTRAL)
        self.assertFalse(R0Admission.REQUIRE_NEUTRAL)

    def test_all_five_apps_real_unseeded_medium_10_rounds(self):
        from run_stage2c_cell import Accumulator
        config=json.loads(CONFIG.read_text(encoding='utf-8'))
        for app,batch in APPS.items():
            rows=pd.DataFrame([dict(name=app,time=17,application=app,num_replicas=1,batch_size=batch)])
            c=make_cluster(config,rows,REG,MODELS,'u_jo','medium',seed=False)
            accumulator=Accumulator(c)
            self.assertTrue(all(not p for p in c.jobs[0].profiles.values()))
            records=[]
            for _ in range(10):
                if c.all_complete(): break
                c.step(60); records.append(copy.deepcopy(c.joint_round_logs[-1]))
                accumulator.consume(c,records[-1])
            j=c.jobs[0]
            self.assertGreater(j.progress,0,app)
            self.assertGreater(j.execution_time,0,app)
            segments=[s for r in records for e in r['execution'] for s in e['training_segments']]
            self.assertAlmostEqual(j.execution_time,sum(s['seconds'] for s in segments))
            self.assertTrue(all(abs(s['effective_rate']-s['raw_rate_before_rho']*s['rho'])<1e-7 for s in segments))
            self.assertTrue(all(e['independent_execution_audit']['valid'] for r in records for e in r['execution']))
            self.assertTrue(all(r['decision_for_next_interval']['result']['audit']['valid'] for r in records))
            evidence('unseeded_'+app,records)
            summary=accumulator.summary(c,{'rmst_tau_seconds':100})
            self.assertEqual(summary['violations'],0)
            self.assertGreater(summary['progress_floor_checks'],0)

    def test_medium_solver_failure_preserves_service_transaction(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import joint_prototype.neutral_liveness as module
        config=json.loads(CONFIG.read_text(encoding='utf-8'))
        rows=pd.DataFrame([dict(name='bert',time=0,application='bert',num_replicas=1,batch_size=384)])
        c=make_cluster(config,rows,REG,MODELS,'u_jo','medium',seed=False); c.step(60)
        state=copy.deepcopy(c.policy.liveness.__dict__); allocations=copy.deepcopy(c.allocations)
        with patch.object(module,'milp',return_value=SimpleNamespace(x=None,status=1,success=False,message='timeout')):
            with self.assertRaisesRegex(RuntimeError,'SOLVER_UNDECIDED'): c.optimize_policy()
        self.assertEqual(c.policy.liveness.__dict__,state)
        self.assertEqual(c.allocations,allocations)

    def test_medium_same_state_four_cell_inclusion_and_shared_contract(self):
        config=json.loads(CONFIG.read_text(encoding='utf-8')); keys={}; contracts={}
        for cell in ('u_fp','u_fo','u_jp','u_jo'):
            rows=pd.DataFrame([dict(name='bert',time=17,application='bert',num_replicas=1,batch_size=384)])
            c=make_cluster(config,rows,REG,MODELS,cell,'medium',seed=False); c.step(60)
            d=c.policy.last_decision
            keys[cell]={tuple(k) for k in d['factor_candidate_keys']}; contracts[cell]=d['shared_contract']
        self.assertTrue(keys['u_fp']<=keys['u_fo']<=keys['u_jo'])
        self.assertTrue(keys['u_fp']<=keys['u_jp']<=keys['u_jo'])
        self.assertTrue(all(v==contracts['u_fp'] for v in contracts.values()))
        evidence('contracts',dict(keys={k:sorted(v) for k,v in keys.items()},contracts=contracts))
