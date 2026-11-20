"""Offline structural screen only. No Job.step, Cluster or optimizer calls.
The diagnostic sum of normalized speeds is NOT the FO/FULL completion objective.
"""
import gzip, hashlib, itertools, json, sys
from dataclasses import asdict, replace
from pathlib import Path

O=Path(__file__).resolve().parent; R=O.parents[2]
T=R/'cc-joint-residual-capacity-v1'
H=O.parent/'cc_residual_capacity_20261101_run1'
sys.path[:0]=[str(T),str(T/'sia-simulator')]
from joint_prototype.joint_profile import JobSpec,GpuConfig,NodeSpec
from joint_prototype.joint_scheduler import JointScheduler,Placement
from joint_prototype.r0_liveness import R0Oracle
from residual_capacity import resource_points

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(name,x):
    with (O/name).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)

# Fixed structural interventions, declared before any evaluation. No sweep.
SCENARIOS={
 'S0_symmetric':{'aws':[6.,125.],'dgx-ext':[6.,125.]},
 'S1_cpu_mismatch':{'aws':[9.,125.],'dgx-ext':[3.,125.]},
 'S2_ram_mismatch':{'aws':[6.,187.5],'dgx-ext':[6.,62.5]},
 'S3_crossed':{'aws':[9.,62.5],'dgx-ext':[3.,187.5]},
}
save('screen_plan.json',dict(scenarios=SCENARIOS,state_rule='420s, same first CIFAR/BERT pair as prior candidate evidence; not chosen by new JCT',
    scope='both jobs mandatory; no locks or ongoing service commitments in hypothetical fresh placement; diagnostic normalized speed sum only',
    evaluations='all four scenarios; all legal GPU configs and current trusted resource points; no simulated training',
    limitations='state learned in old symmetric FULL; not a fresh startup forecast; not actual FO or FULL policy selection; no JCT claim'))
hashes=json.loads((H/'source_hashes.json').read_text())
assert all(sha(T/p)==h for p,h in hashes.items())
with gzip.open(H/'cells/NEW_FULL/rounds.jsonl.gz','rt',encoding='utf8') as f:
    row=next(json.loads(l) for l in f if json.loads(l)['interval_end']==420)
names=('cifar10-0','bert-28');jobs=[]
for z in row['decision_for_next_interval']['probe']['specs']:
    if z['job_id'] not in names:continue
    z=dict(z);z['configs']=tuple(GpuConfig(**c) for c in z['configs'])
    for k in ('current_nodes','current_placement'):z[k]=tuple(tuple(v) if isinstance(v,list) else v for v in z[k])
    jobs.append(replace(JobSpec(**z),checkpoint_locked=False,current_config_id=None,current_nodes=(),current_placement=(),current_cpu_per_gpu=None,current_ram_gb_per_gpu=None))
assert tuple(j.job_id for j in jobs)==names
oracle=R0Oracle(T/'inputs/registry.json',T/'inputs/models',dict(zip(names,('cifar10','bert'))),'medium')
save('state_receipt.json',dict(source=str(H/'cells/NEW_FULL/rounds.jsonl.gz'),sha256=sha(H/'cells/NEW_FULL/rounds.jsonl.gz'),time=420,specs=[asdict(j) for j in jobs],
    deliberate_hypothetical_change='remove incumbent geometry for structural fresh-placement comparison; no claim of safe live switch'))

def slim(c):return dict(job=c['job'],config=c['config'],gpu=c['gpu'],cpu=c['cpu'],ram=c['ram'],rho=c['rho'],speed=c['speed'],num_clipped=c['num_clipped'],den_clipped=c['den_clipped'],requested=c['requested'],reference=c['reference'])
results=[];coverage=[]
for sid,caps in SCENARIOS.items():
    nodes=[NodeSpec(t+':0',t,2,*cap) for t,cap in caps.items()]
    s=JointScheduler(nodes,jobs,oracle,(1.,2.,3.,4.,6.,8.),(20.83,41.66,62.5,125.),enforce_proportional_progress_floor=False,minimum_cpu_per_gpu=1.,minimum_ram_gb_per_gpu=20.83)
    allcols={};propcols={}
    for j in jobs:
        cs=[];ps=[]
        for cfg in j.configs:
            n=s.node_map[cfg.gpu_type+':0'];g=cfg.gpus
            p=Placement(((n.node_id,g),));prop=(n.cpu_cores/n.gpus,n.ram_gb/n.gpus)
            cpu,ram=resource_points(s,j,cfg,p)
            pts=set(itertools.product(cpu,ram));pts.add(prop)
            count=0;unclipped=0
            for c,m in sorted(pts):
                if c*g>n.cpu_cores+1e-9 or m*g>n.ram_gb+1e-9:continue
                q=oracle.query(j,cfg,c,m,prop)
                z=dict(job=j.job_id,config=cfg.config_id,node=n.node_id,gpu=g,cpu=c,ram=m,rho=q.rho,speed=q.effective_goodput,
                    num_clipped=q.source_coordinate_clipped,den_clipped=q.baseline_coordinate_clipped,
                    requested=[q.requested_source_cpu_coordinate,q.requested_source_ram_coordinate],reference=[q.requested_baseline_source_cpu_coordinate,q.requested_baseline_source_ram_coordinate])
                cs.append(z);count+=1;unclipped+=not(q.source_coordinate_clipped or q.baseline_coordinate_clipped)
                if (c,m)==prop:ps.append(z)
            coverage.append(dict(scenario=sid,job=j.job_id,config=cfg.config_id,feasible_points=count,fully_supported_points=unclipped))
        allcols[j.job_id]=cs;propcols[j.job_id]=ps
    def feasible(a,b):
        if a['node']!=b['node']:return True
        n=s.node_map[a['node']]
        return a['gpu']+b['gpu']<=n.gpus and a['gpu']*a['cpu']+b['gpu']*b['cpu']<=n.cpu_cores+1e-9 and a['gpu']*a['ram']+b['gpu']*b['ram']<=n.ram_gb+1e-9
    def best(domain,fixed=None):
        winner=None;value=-1.;pairs=0
        for a,b in itertools.product(*(domain[j] for j in names)):
            if fixed and (a['config'],b['config'])!=fixed:continue
            if not feasible(a,b):continue
            pairs+=1;v=a['speed']+b['speed']
            if v>value+1e-12:value=v;winner=[a,b]
        return dict(value=value,allocation=[slim(c) for c in winner] if winner else None,feasible_pairs=pairs)
    prop=best(propcols);fixed=tuple(c['config'] for c in prop['allocation'])
    serial=best(allcols,fixed);joint=best(allcols)
    results.append(dict(scenario=sid,proportional_proposal=prop,fixed_gpu_aux_optimum=serial,joint_optimum=joint,
        diagnostic_joint_gain_pct=100*(joint['value']/serial['value']-1),
        warning='DIAGNOSTIC OBJECTIVE, NOT ACTUAL FO/FULL OR JCT; equal per-job normalized speed sum; no costs'))
save('offline_results.json',results);save('coverage.json',coverage)
assert all(sha(T/p)==h for p,h in hashes.items())
save('integrity.json',dict(source_files=len(hashes),source_unchanged=True,source_hashes_sha256=sha(H/'source_hashes.json'),registry_sha256=sha(T/'inputs/registry.json'),workload_sha256=sha(H/'inputs/workload.csv')))
for r in results:print(r['scenario'],round(r['diagnostic_joint_gain_pct'],4),r['fixed_gpu_aux_optimum']['allocation'],r['joint_optimum']['allocation'])
