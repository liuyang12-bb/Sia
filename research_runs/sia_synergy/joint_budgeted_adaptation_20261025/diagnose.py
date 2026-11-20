"""At most six structurally selected old states, fixed-set exact MIP counterfactual."""
import copy,gzip,hashlib,json,sys
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'joint_r_scale_check_20261025'
p=json.loads((OLD/'preregistration_v2.json').read_text());sys.path.insert(0,str(ROOT/'sia-joint-budgeted-adaptation/sia-simulator'))
from joint_prototype.joint_profile import JobSpec,GpuConfig,NodeSpec,ProfileQuery
from joint_prototype.joint_scheduler import JointColumn,Placement
from joint_prototype.r0_liveness import R0Oracle
from joint_prototype.neutral_liveness import FactorScheduler,geometry
from joint_prototype.scaled_adaptive import ScaledAdmission
from joint_prototype.small_mechanism import fixed_set_solve

def column(v):
    v=copy.deepcopy(v);v['config']=GpuConfig(**v['config']);v['placement']=Placement(tuple(map(tuple,v['placement']['node_gpus'])))
    v['query']=ProfileQuery(**v['query']);v['node_usage']=tuple(map(tuple,v['node_usage']))
    return JointColumn(**v)
def job(v):
    v=copy.deepcopy(v);v['configs']=tuple(GpuConfig(**c) for c in v['configs']);v['current_nodes']=tuple(v['current_nodes']);v['current_placement']=tuple(map(tuple,v['current_placement']))
    return JobSpec(**v)
def main():
    for path,h in p['hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()==h,path
    plan=[(20,'medium','bert-54',60),(20,'medium','bert-54',120),(20,'medium','bert-54',240),
          (20,'medium','bert-99',60),(8,'medium','bert-28',60),(20,'neutral','bert-54',60)]
    states=[]
    for n,response,target,done_min in plan:
        cid=f'n{n}_{response}_joint_r';costs=json.loads((OLD/cid/'start.json').read_text())['actual_delay_costs']
        chosen=None
        with gzip.open(OLD/cid/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);d=r['decision_for_next_interval'];pb=d['probe'];before=pb['l2_before'];service=before['services'].get(target)
                if not service:continue
                spec=next((v for v in pb['specs'] if v['job_id']==target),None)
                if spec is None:continue
                done=before['productive'][target]-service['productive_base']
                if spec['current_config_id']=='rtx:1:1' and not spec['checkpoint_locked'] and done>=done_min and done<300:
                    chosen=r;break
        if chosen is None:
            states.append(dict(cell=cid,target=target,productive_min=done_min,status='NO_MATCH'));continue
        r=chosen;d=r['decision_for_next_interval'];pb=d['probe'];before=pb['l2_before'];specs=[job(v) for v in pb['specs']]
        oracle=R0Oracle(p['registry'],p['models'],{j.job_id:j.job_id.split('-')[0] for j in specs},response)
        nodes=[NodeSpec(t+':0',t,g,g*3.,g*62.5) for t,g in [('aws',4),('dgx-ext',8),('rtx',8)]]
        cfg=json.loads(Path(p['config']).read_text())
        s=FactorScheduler(nodes,specs,oracle,cfg['response']['aux_cpu_points'],cfg['response']['aux_ram_points'],allocation_bonus=0,minimum_cpu_per_gpu=1.,minimum_ram_gb_per_gpu=20.83,verify_exact=False,candidate_policy='b3')
        allcols=[column(v) for v in pb['joint_columns']]
        a=ScaledAdmission(300.,250.,60.);a.delay_costs=costs;a.profile_sources={j:{c:v['goodput_source'] for c,v in rows.items()} for j,rows in pb['provenance'].items()}
        actual_cols=[];relaxed_cols=[];global_cols=[];filter_evidence={}
        held=set()
        for j,service in before['services'].items():
            if j not in s.job_map:continue
            if before['productive'][j]-service['productive_base']>=300.-1e-9:continue
            held.add(j)
            service=copy.deepcopy(service);service['column']=column(service['column']);js=s.job_map[j]
            legal,ev=a.eligible_held_columns(s,js,service,allcols,before['productive'],r['interval_end'])
            allowed={v.column_id for v in legal};current=next(v for v in legal if geometry(v)==geometry(s.refresh_column(service['column'])))
            reclaimed=[]
            rejected={v['column_id']:v['reason'] for v in ev['rejected']}
            for c in allcols:
                if c.job_id==j and rejected.get(c.column_id)=='NO_REMAINING_SERVICE_PAYBACK' and a._rate(c)>a._rate(current)+1e-12:
                    reclaimed.append(c)
            actual_cols+=legal;relaxed_cols+=legal+reclaimed
            # Global resource exchanges may slow one peer while improving the
            # original cost-aware fairness objective for the whole bundle.
            global_cols+=legal+[c for c in allcols if c.job_id==j and
                rejected.get(c.column_id) in {'NO_REMAINING_SERVICE_PAYBACK','AUX_REDUCES_CURRENT_RATE'}]
            filter_evidence[j]=dict(current_config=current.config.config_id,current_rate=a._rate(current),remaining=ev['productive_remaining'],
                reclaimed=[dict(config=c.config.config_id,cpu=c.query.cpu_per_gpu,ram=c.query.ram_gb_per_gpu,rate=a._rate(c),utility=c.scheduling_utility,delay=a._delay(j,c)) for c in reclaimed])
        required=set(d['result']['selected'])
        # Snapshot may include a just-expired service; only keep exactly the same served set.
        actual_cols += [c for c in allcols if c.job_id not in held]
        relaxed_cols += [c for c in allcols if c.job_id not in held]
        global_cols += [c for c in allcols if c.job_id not in held]
        old=fixed_set_solve(s,actual_cols,required);new=fixed_set_solve(s,relaxed_cols,required)
        global_result=fixed_set_solve(s,global_cols,required)
        assert abs(old.scheduling_objective-d['result']['scheduling_objective'])<1e-7,('RECONSTRUCTION_OBJECTIVE_MISMATCH',r['interval_end'])
        states.append(dict(cell=cid,target=target,productive_min=done_min,time=r['interval_end'],required=sorted(required),
            old=old.to_dict(),relaxed=new.to_dict(),global_bundle=global_result.to_dict(),filters=filter_evidence,
            interpretation='Same online candidates, required jobs, objective, locks and capacity; only short-payback eligibility relaxed; not a JCT estimate'))
    with (OUT/'same_state_bundle_diagnosis.json').open('x',encoding='utf-8') as f:json.dump(states,f,indent=2)
    for v in states:
        print(v['cell'],v['target'],v.get('time'),v.get('status','SOLVED'),(v.get('old') or {}).get('scheduling_objective'),(v.get('relaxed') or {}).get('scheduling_objective'),(v.get('global_bundle') or {}).get('scheduling_objective'),flush=True)
if __name__=='__main__':main()
