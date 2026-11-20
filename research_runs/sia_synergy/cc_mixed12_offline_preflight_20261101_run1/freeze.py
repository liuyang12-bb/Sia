import csv,hashlib,json,platform,sys,time
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-recovery-equivalent-v1'
P=O.parent/'cc_full_fo_event_gate_experiment_20261101_run1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
p=json.loads((P/'preregistration.json').read_text(encoding='utf8'))
assert all(sha(T/n)==h for n,h in p['source_hashes'].items())
src=R/'official-artifacts/sia-simulator/workloads/saturn/workload-1.csv'
rows=list(csv.DictReader(src.open(encoding='utf8')));assert len(rows)==160
chosen={a:[(i,r) for i,r in enumerate(rows) if r['application']==a][:n] for a,n in [('cifar10',6),('deepspeech2',4),('bert',2)]}
iters={a:iter(rs) for a,rs in chosen.items()};order=['cifar10','deepspeech2','bert','cifar10','deepspeech2','cifar10']*2
cohort=[]
for slot,a in enumerate(order):
 i,r=next(iters[a]);r=dict(r);r.update(original_arrival=r['time'],original_source_row_index=i,source_line=i+2,source_row_index=slot,time=0 if slot<6 else 600)
 cohort.append(r)
(O/'inputs').mkdir(exist_ok=False)
with (O/'inputs/workload.csv').open('x',encoding='utf8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(cohort[0]));w.writeheader();w.writerows(cohort)
config=json.loads((P/'inputs/proposed_cluster.json').read_text(encoding='utf8'))
config['runtime_warning']='Use existing mismatch_adapter.build; it restores these exact capacities before the first decision. This phase is offline-only.'
save('inputs/config.json',config)
save('CURRENT_TASK.json',dict(scope='OFFLINE_MIXED12_PREFLIGHT_ONLY',training_simulation_authorized=False,source=str(T),protected_parent=str(P),
 resume='Read preregistration.json and any result/failure. No Job.step, Cluster.step, future training or full-run command may be invoked. Do not tune inputs after seeing outputs.'))
save('preregistration.json',dict(version='MIXED12_STRUCTURAL_OFFLINE_V1',frozen_unix=time.time(),
 selection='workload-1 first6CIFAR, first4DeepSpeech2, first2BERT in original source order. No output/JCT selection.',
 workload=cohort,arrival_rule='Two fixed waves at0 and600 seconds; each3CIFAR+2DS+1BERT; stable order CF,DS,BERT,CF,DS,CF.',
 capacities=config['cluster'],response=config['response'],source_hashes=p['source_hashes'],source_snapshot_sha256=p['source_snapshot_sha256'],
 input_hashes={str(q):sha(q) for q in [src,O/'inputs/workload.csv',O/'inputs/config.json',T/'inputs/registry.json']},
 environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform()),
 guard=dict(memory_gib=2,disk_gib=5,Job_step_forbidden=True,Cluster_step_forbidden=True),
 reference_states=[dict(id='E0',fraction=0.),dict(id='E25',fraction=.25)],
 reference_definition='Offline reference only: seed performance profile(1,8); use source application gradient at floor(fraction*max_epochs), init batch; current Sia adaptive batch and predict_job; remaining original declared work. Not true cold online state, not new initialization rule.',
 resource_subproblem='Each wave independent six-task mandatory service envelope; same original columns, node caps, objective sum1/(startup+remaining_work/(raw*rho)). FO chooses GPU at proportional points then optimizes aux; FULL jointly chooses all. No full-controller or ongoing checkpoint-history claim.',
 stages=['all72job-config response coverage','same-state six-task serial vs joint resource optimization for both fixed reference epochs','fixed-selected-bundle physical rate hold at360 and600 seconds; no Job.step or progress feedback','one-shot actual cold controller and separate offline-reference controller on firstwave only'],
 solver=dict(time_limit_seconds=30,mip_rel_gap=0.,scope='static audit optimizer; primary objective certificate required; no claim its tie outcome equals live controller'),
 gates=dict(domain='all12jobs/all6AWS-RTX1-2-4 configs, no missing registry/rate; physical prop feasibility; clipped points disclosed',
 joint='same-state FULL positive primary objective gap beyond numerical tolerance vs optimized FO, changed GPU config, valid capacities; include donor losses in total fraction-of-work progress and sumETA',
 persistence='selected static bundle positive net capped fraction-of-work gain at360 AND600 seconds after source startup delays, plus predicted completion/release availability; conditional model evidence only',
 support='list numerator and denominator clipping, especially every changed task; preserve E-legacy, no clipped unsupported donor described as measured',
 online='cold reachability, real co-residency, checkpoint history and actual persistence remain UNVERIFIED without training; no complete JCT claim'),
 no_adaptive_search=True,label='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE'))
print('FROZEN',sha(O/'preregistration.json'))
print(json.dumps(cohort,ensure_ascii=False,indent=2))
