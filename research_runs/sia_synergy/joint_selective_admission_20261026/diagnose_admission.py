"""Four predeclared structural states, no training simulation or JCT inference."""
import copy,gzip,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import LinearConstraint
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
OLD=ROOT/'research_runs/sia_synergy/joint_budgeted_adaptation_20261025/attempt3'
p=json.loads((OLD/'preregistration.json').read_text())
sys.path.insert(0,str(ROOT/'sia-joint-selective-admission/sia-simulator'))
sys.path.insert(0,str(ROOT/'research_runs/sia_synergy/joint_budgeted_adaptation_20261025'))
from diagnose import job,column
from joint_prototype.bundle_adaptation import BundleAdmission
from joint_prototype.neutral_liveness import FactorScheduler,geometry
from joint_prototype.joint_profile import NodeSpec
from joint_prototype.r0_liveness import R0Oracle

def main():
    criteria=['FIRST_DECISION','FIRST_SERVICE_EXPIRY','FIRST_BERT54_RTX1_TRAINED','FIRST_DRAIN']
    found={};costs=json.loads((OLD/'n20_medium_joint_g/start.json').read_text())['actual_delay_costs']
    with gzip.open(OLD/'n20_medium_joint_g/rounds.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);d=r['decision_for_next_interval'];pb=d['probe'];b=pb['l2_before']
            expired=any(b['productive'][j]-v['productive_base']>=300 for j,v in b['services'].items() if j in b['productive'])
            bert=next((j for j in pb['specs'] if j['job_id']=='bert-54'),None)
            conditions=[True,expired,bert is not None and bert['current_config_id']=='rtx:1:1' and not bert['checkpoint_locked'] and b['productive'].get('bert-54',0)>0,d['result']['audit']['drain']]
            for criterion,match in zip(criteria,conditions):
                if match and criterion not in found:found[criterion]=r
            if len(found)==4:break
    results=[]
    for criterion in criteria:
        if criterion not in found:results.append(dict(criterion=criterion,status='NO_MATCH'));continue
        r=found[criterion];d=r['decision_for_next_interval'];pb=d['probe'];b=pb['l2_before'];specs=[job(v) for v in pb['specs']]
        oracle=R0Oracle(p['registry'],p['models'],{j.job_id:j.job_id.split('-')[0] for j in specs},'medium')
        cfg=json.loads(Path(p['config']).read_text());nodes=[NodeSpec(t+':0',t,g,g*3.,g*62.5) for t,g in [('aws',4),('dgx-ext',8),('rtx',8)]]
        s=FactorScheduler(nodes,specs,oracle,cfg['response']['aux_cpu_points'],cfg['response']['aux_ram_points'],allocation_bonus=0,minimum_cpu_per_gpu=1.,minimum_ram_gb_per_gpu=20.83,verify_exact=False,candidate_policy='b3')
        cs=[column(c) for c in pb['joint_columns']]
        a=BundleAdmission(300,250,60);a.delay_costs=costs;a.profile_sources={j:{c:v['goodput_source'] for c,v in rows.items()} for j,rows in pb['provenance'].items()}
        a.ready=list(b['ready']);a.services=copy.deepcopy(b['services'])
        for v in a.services.values():v['column']=column(v['column'])
        a.seen=set(a.ready)|set(a.services);a.sequence=max([v['sequence'] for v in a.services.values()]+[0])
        ready=list(a.ready)+[j.job_id for j in sorted(specs,key=lambda j:(j.arrival_time,j.source_row_index,j.job_id)) if j.job_id not in a.seen]
        expired=[j for j,v in a.services.items() if j in s.job_map and b['productive'][j]-v['productive_base']>=300-1e-9]
        ready+=[j for j in sorted(expired,key=lambda j:a.services[j]['sequence'])]
        held={j for j in a.services if j in s.job_map and j not in expired}
        old=a.decide(s,cs,r['interval_end'],b['productive'])
        assert abs(old.scheduling_objective-d['result']['scheduling_objective'])<1e-7,'RECONSTRUCTION_MISMATCH'
        rejected={v['column_id'] for ev in old.audit['adaptive_candidate_filters'].values() for v in ev['rejected']}
        cs=sorted([c for c in cs if c.column_id not in rejected],key=geometry)
        required=held|({ready[0]} if ready and not old.audit['drain'] else set())
        optional=set(ready[1:]) if ready and not old.audit['drain'] else set()
        matrix,upper=s._constraint_matrix(cs,('gpu','cpu','ram'));lower=np.full(len(upper),-np.inf)
        for i,j in enumerate(s.jobs):lower[i]=float(j.job_id in required);upper[i]=float(j.job_id in required|optional)
        a._diagnostic_job_order=[j.job_id for j in s.jobs];a.numerical_solver_retries=[]
        if required:
            res=a._milp(-np.array([c.scheduling_utility for c in cs]),cs,[LinearConstraint(matrix,lower,upper)])
            assert res.status==0 and res.success
            selected=[c for c,x in zip(cs,res.x) if x>.5];assert s.audit(selected)['valid']
        else:selected=[]
        def short(v):return dict(job=v.job_id,config=v.config.config_id,cpu=v.query.cpu_per_gpu,ram=v.query.ram_gb_per_gpu,raw=v.query.raw_sia_goodput,rho=v.query.rho,utility=v.scheduling_utility)
        value=sum(c.scheduling_utility for c in selected)
        assert value>=old.scheduling_objective-1e-7
        results.append(dict(criterion=criterion,status='SOLVED',time=r['interval_end'],held=sorted(held),required=sorted(required),optional=sorted(optional),
            old=dict(objective=old.scheduling_objective,count=len(old.selected),selected=[short(c) for c in old.selected.values()]),
            selective=dict(objective=value,count=len(selected),selected=[short(c) for c in selected]),
            interpretation='same current online candidates, locks, held services, quotas, objective; only uncommitted admissions optional; not JCT evidence'))
        print(criterion,r['interval_end'],len(old.selected),old.scheduling_objective,'=>',len(selected),value,flush=True)
    with (OUT/'same_state_admission.json').open('x') as f:json.dump(results,f,indent=2)
if __name__=='__main__':main()
