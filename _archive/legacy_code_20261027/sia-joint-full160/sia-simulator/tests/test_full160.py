"""Predeclared adapters, scale and bounded real-chain acceptance, not JCT tuning."""
import copy,json,os,unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from joint_prototype.full160 import make_cluster, FullAdmission, capacities
from joint_prototype.r0_liveness import R0Admission
from joint_prototype.neutral_liveness import L2Admission
from sia import SiaPolicy
from test_stage2c_r0 import REG,MODELS,CONFIG
from test_stage2b_neutral import schedule,spec,current,negative
from joint_prototype.joint_profile import NodeSpec

def config(): return json.loads(CONFIG.read_text())
def cluster(method='sia_p',capacity='main',response='neutral',rows=None,seed=True):
    if rows is None: rows=[dict(name='bert-mid',time=17,application='bert',num_replicas=1,batch_size=384)]
    return make_cluster(config(),pd.DataFrame(rows),REG,MODELS,method,response,capacity,seed)
def evidence(name,data):
    if os.environ.get('FULL160_EVIDENCE'):
        p=Path(os.environ['FULL160_EVIDENCE']);p.mkdir(parents=True,exist_ok=True)
        with (p/(name+'.json')).open('x') as f: json.dump(data,f,default=str)

class Full160Tests(unittest.TestCase):
    def test_sia_p_calls_actual_official_optimize_without_l2(self):
        original=SiaPolicy.optimize; calls=[]
        def wrapped(self,*args): calls.append(True); return original(self,*args)
        c=cluster()
        with patch.object(SiaPolicy,'optimize',wrapped): c.step(60)
        self.assertEqual(len(calls),1);self.assertFalse(hasattr(c.policy,'liveness'))
        self.assertEqual(c.allocations,c.policy.last_decision['original_sia_allocations'])

    def test_main_medium_proportional_identity_and_all_capacity_floors(self):
        for cap in ('half','main','double'):
            c=cluster(capacity=cap,response='medium');c.step(60)
            for a in c.policy.current_resources.values():
                self.assertEqual(a['cpu_per_gpu'],{'half':1.5,'main':3.,'double':6.}[cap])
                self.assertEqual(a['ram_gb_per_gpu'],62.5)
                if cap=='main': self.assertEqual(a['rho'],1.)
            self.assertTrue(c.policy.last_decision['result']['audit']['valid'])

    def test_tune_real_unseeded_chain_gpu_intent_and_locks(self):
        c=cluster('sia_tune',response='medium',seed=False); records=[]
        for _ in range(10):
            c.step(60); records.append(copy.deepcopy(c.joint_round_logs[-1]))
            self.assertEqual(c.allocations,c.policy.last_decision['original_sia_allocations'])
            self.assertTrue(c.policy.last_decision['result']['audit']['valid'])
        self.assertGreater(c.jobs[0].progress,0);evidence('tune_unseeded',records)

    def test_tune_failure_atomic_proportional_fallback_no_gpu_changes(self):
        from joint_prototype.synergy_adapter import SynergyTuneAdapter
        c=cluster('sia_tune')
        with patch.object(SynergyTuneAdapter,'run',side_effect=RuntimeError('injected failure')): c.step(60)
        self.assertEqual(c.policy.last_decision['result']['status'],'FALLBACK_PROPORTIONAL')
        self.assertEqual(c.allocations,c.policy.last_decision['original_sia_allocations'])
        self.assertTrue(c.policy.last_decision['result']['audit']['valid'])

    def test_full160_queue_no_truncation_and_old_guards_preserved(self):
        self.assertEqual(L2Admission.MAX_ACTIVE_JOBS,3);self.assertEqual(R0Admission.MAX_ACTIVE_JOBS,20)
        a=FullAdmission(300,250,60);jobs=[spec(f'j{i:03}',0) for i in range(160)]
        s=schedule(jobs,[NodeSpec('aws:0','aws',1,3,62.5)])
        r=a.decide(s,negative(s),0,{})
        self.assertEqual(len(a.seen),160);self.assertEqual(len(a.ready),159)
        self.assertEqual(set(r.selected),{'j000'})
        s=schedule([current(j,r.selected,1,True) for j in jobs],[NodeSpec('aws:0','aws',1,3,62.5)])
        r2=a.decide(s,negative(s),60,{'j000':0})
        self.assertEqual(set(r2.selected),{'j000'});self.assertEqual(len(a.ready),159)
        self.assertTrue(r2.audit['valid'])
        s=schedule([current(j,r2.selected,2,False) for j in jobs],[NodeSpec('aws:0','aws',1,3,62.5)])
        r3=a.decide(s,negative(s),360,{'j000':300})
        self.assertEqual(set(r3.selected),{'j001'});self.assertEqual(a.ready[-1],'j000')
        evidence('queue160',dict(seen=len(a.seen),ready=a.ready,services=list(a.services),audit=r3.audit))

    def test_real160_active_job_infos_no_filter(self):
        rows=[dict(name=f'bert-{i}',time=0,application='bert',num_replicas=1,batch_size=384) for i in range(160)]
        c=cluster('joint_p',rows=rows,seed=False); c.step(60)
        self.assertEqual(len(c.policy.liveness.seen),160)
        self.assertEqual(len(c.policy.last_decision['candidate_summary']),160)
        self.assertEqual(set(c.policy.liveness.ready)|set(c.policy.liveness.services),{j.name for j in c.jobs})
        self.assertTrue(c.policy.last_decision['result']['audit']['valid'])

    def test_same_seed_and_execution_inputs_all_methods(self):
        states=[]
        for method in ('sia_p','sia_tune','joint_p','joint_o'):
            c=cluster(method,response='medium')
            j=c.jobs[0]
            states.append(str((j.profiles,j.target_batch_size,j.submission_time)))
            self.assertEqual(c.policy.oracle.response_exponent,.5)
            self.assertEqual(c.policy.node_capacities,capacities('main'))
        self.assertTrue(all(s==states[0] for s in states))

    def test_metrics_actual_short_chain_all_methods_and_responses(self):
        from run_full160_cell import FullMetrics
        for method in ('sia_p','sia_tune','joint_p','joint_o'):
            for response in ('neutral','medium'):
                c=cluster(method,response=response);m=FullMetrics(c)
                for _ in range(5):c.step(60);m.consume(c,c.joint_round_logs[-1])
                result=m.summary(c,{'rmst_tau_seconds':200})
                self.assertEqual(result['violations'],0)
                self.assertGreater(result['execution_audits'],0)
                self.assertTrue(c.policy.last_decision['result']['audit']['valid'])

    def test_checkpoint_observation_is_before_new_decision(self):
        from run_full160_cell import FullMetrics
        c=cluster('sia_p',seed=False);m=FullMetrics(c);evidence_rows=[]
        for dt in (60,60,60,60,60,60,60,60,60,60):
            c.step(dt);r=c.joint_round_logs[-1];m.consume(c,r)
            e=r['execution'][0]
            self.assertEqual(e['checkpoint_remaining_before_decision'],c.checkpoint_before_decision[c.jobs[0].name])
            evidence_rows.append(dict(before=e['checkpoint_remaining_before_decision'],after=c.jobs[0].rescale_time,training=e['training_seconds']))
        self.assertTrue(any(x['before']==0 and x['after']>0 for x in evidence_rows))
        self.assertTrue(any(x['before']>0 for x in evidence_rows))
        evidence('checkpoint_decision_boundary',evidence_rows)

if __name__=='__main__': unittest.main()
