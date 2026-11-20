"""Read existing logs only. No simulator import, candidate solve or training run."""
import ast
import collections
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / 'rechecked'
OUT.mkdir(exist_ok=False)
ROOT = BASE.parents[2]
SRC = ROOT / 'cc-joint-recovery-equivalent-v1'
RUN = BASE.parent / 'cc_full_fo_event_gate_experiment_20261101_run1'

def read(p):
    return json.loads(p.read_text(encoding='utf8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()

def save(name, value):
    with (OUT/name).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

pre = read(RUN/'preregistration.json')
integrity = dict(source_count=len(pre['source_hashes']),
    source_mismatches=[p for p,h in pre['source_hashes'].items() if sha(SRC/p)!=h],
    input_mismatches=[p for p,h in pre['input_hashes'].items() if sha(Path(p))!=h],
    historical_artifact_mismatches=[p for p,h in read(RUN/'delivery_hashes.json').items()
        if sha(RUN/p)!=h])
assert not any(integrity[k] for k in ('source_mismatches','input_mismatches','historical_artifact_mismatches'))
save('integrity.json', integrity)

data = {}
raw_input_hashes = {}
for method in ('FO','FULL'):
    path=RUN/'cells'/('proposed_6job_medium_'+method)/'rounds.jsonl.gz'
    raw_input_hashes[str(path)]=sha(path)
    with gzip.open(path,'rt',encoding='utf8') as f:
        rows=[json.loads(line) for line in f]
    serial=[]; refs=[]; timelines=[]; reasons=collections.Counter()
    job_round_reasons=collections.Counter(); event_times=[]; wait=0.
    no_event=0; previous={}; singleton_rounds=0; seen=set(); repeated=[]
    for index,r in enumerate(rows,1):
        d=r['decision_for_next_interval']; a=d['result']['audit']; sel=d['result']['selected']
        t=r['interval_end']; wait+=sum(e['waiting_seconds'] for e in r['execution'] if r['interval_start']>=360)
        if d['round'] in seen:
            repeated.append(dict(log_line=index,time=t,retained_decision_round=d['round']))
            continue
        seen.add(d['round'])
        if a.get('geometry_event_reasons'):event_times.append(dict(time=t,reasons=a['geometry_event_reasons']))
        elif sel:no_event+=1
        allowed={}
        for j,v in a.get('shared_eligibility',{}).items():
            reasons.update(v.get('rejection_counts',{}))
            job_round_reasons.update(k for k,vv in v.get('rejection_counts',{}).items() if vv)
            allowed[j]=sorted({ast.literal_eval(s)[1] for s in v.get('eligible_geometries',[])})
        if len(sel)==6 and all(len(allowed.get(j,[]))==1 for j in sel):singleton_rounds+=1
        for p in a.get('serial_proposals',[]):
            before=sum(x['utility'] for x in p['proposal']);after=sum(x['utility'] for x in p['selected'])
            serial.append(dict(time=t,proposal=before,aux_optimized=after,
                improvement_pct=100*(after/before-1) if before>0 else None,
                changed_jobs=[x['job'] for x in p['selected'] if any(
                    v['job']==x['job'] and (abs(x['cpu']-v['cpu'])>1e-8 or abs(x['ram']-v['ram'])>1e-8)
                    for v in p['proposal'])]))
        configs={j:c['config']['config_id'] for j,c in sel.items()}
        changes={j:dict(before=previous.get(j),after=configs.get(j)) for j in previous.keys()|configs.keys()
            if previous.get(j)!=configs.get(j)}
        if changes:timelines.append(dict(time=t,changes=changes,event_reasons=a.get('geometry_event_reasons',[])))
        previous=configs
        if t in (240,420,1200,1740,1920,2400,3420,3600):
            nxt=rows[index] if index<len(rows) else None
            execution={e['job_id']:{k:e.get(k) for k in (
                'training_seconds','checkpoint_delay_consumed','progress_delta','raw_goodput','rho','effective_goodput')}
                for e in nxt['execution']} if nxt else {}
            compact={j:dict(config=c['config']['config_id'],gpus=c['config']['gpus'],
                cpu_per_gpu=c['cpu_per_gpu'],ram_per_gpu=c['ram_gb_per_gpu'],rho=c['rho'],
                clipped=c['source_coordinates']['clipped'],
                progress=r['job_end_state'][j]['progress'],
                allowed_configs=allowed.get(j),rejected_counts=a.get('shared_eligibility',{}).get(j,{}).get('rejection_counts',{}),
                next_actual_execution=execution.get(j)) for j,c in sel.items()}
            refs.append(dict(log_line=index,decision_time=t,executed_interval_start=t,
                selected=compact,node_totals=a['node_totals'],ready=a['ready'],
                service_commitments=a['service_commitments'],geometry_events=a.get('geometry_event_reasons',[]),
                serial_proposals=a.get('serial_proposals',[])))
    ups=[x['improvement_pct'] for x in serial if x['improvement_pct'] is not None]
    data[method]=dict(log=str(path),rounds=len(rows),distinct_decisions=len(seen),retained_decisions_excluded=repeated,
        selected_states=refs,configuration_timeline=timelines,
        geometry_events=event_times,no_geometry_event_rounds=no_event,all6_only1config_rounds=singleton_rounds,
        rejection_column_counts=dict(reasons),rejection_job_round_counts=dict(job_round_reasons),
        wait_after_360=wait,serial_objective_comparison=serial,
        serial_aux_summary=dict(evaluations=len(serial),positive=sum(x>1e-6 for x in ups),
            median_pct=statistics.median(ups) if ups else None,
            max_pct=max(ups) if ups else None,
            scope='same-state own proportional proposal vs aux optimum; surrogate objective, NOT JCT improvement'))
save('log_diagnosis.json',data)

jobs=read(RUN/'per_job.json');a=jobs['proposed_6job_medium_FO'];b=jobs['proposed_6job_medium_FULL']
diff={j:a[j]['jct']-b[j]['jct'] for j in a}
gain=sum(max(v,0) for v in diff.values());loss=-sum(min(v,0) for v in diff.values())
steady=read(RUN/'steady_window_realization.json')
phase=steady['per_job_progress_gain'];gross=sum(max(v,0) for v in phase.values());lost=-sum(min(v,0) for v in phase.values())
gate=read(RUN/'event_gate_result.json')
derived=dict(per_job_jct_saved=diff,gross_saved=gain,donor_losses=loss,net_saved=gain-loss,
    gross_cancelled_pct=100*loss/gain,first_release_difference=1895-1735,
    wait_after_all_admitted={m:v['wait_after_360'] for m,v in data.items()},
    window=dict(start=420,end=1740,per_job_progress_gain=phase,gross_gain=gross,gross_loss=lost,
        net=gross-lost,net_vs_fo_progress_pct=100*(gross-lost)/sum(x['progress'] for x in steady['window']['FO'].values()),
        twin_0_1_net=phase['cifar10-0']+phase['cifar10-1']),
    same_state_gate_objective=[dict(owner=x['evidence_owner'],time=x['event_time'],
        increase_pct=100*(x['event_objectives']['full_objective']/x['event_objectives']['fo_objective']-1)) for x in gate['witnesses']],
    evidence_limits=['No new counterfactual solve or simulator run',
        'FO/FULL actual trajectories differ; shared wall times are not shared job states',
        'Rejected columns do not prove a missed beneficial legal allocation',
        'No exact optimal completion schedule calculated; 4.81% is not an upper bound',
        'Resource-release area is a total consistency identity, not a single-event causal estimate'])
assert math.isclose(derived['net_saved'],836)
save('attribution.json',derived)
save('inputs_receipt.json',dict(existing_run=str(RUN),source=str(SRC),raw_log_hashes=raw_input_hashes,
    preregistration_sha256=sha(RUN/'preregistration.json'),source_snapshot_sha256=sha(RUN/'source_snapshot.zip'),
    scope='READ_ONLY_EXISTING_LOG_DIAGNOSIS_NO_SIMULATION_NO_SOLVER_NO_ALGORITHM_CHANGE'))
print(json.dumps(dict(integrity=integrity,derived=derived,serial={m:v['serial_aux_summary'] for m,v in data.items()},
    domain={m:{k:v[k] for k in ('no_geometry_event_rounds','all6_only1config_rounds','rejection_job_round_counts')} for m,v in data.items()}),ensure_ascii=False,indent=2))
