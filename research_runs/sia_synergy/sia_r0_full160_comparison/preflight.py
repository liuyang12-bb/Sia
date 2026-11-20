"""Freeze data and read-only whole-domain response coverage, no simulation."""
import csv,hashlib,json,math,subprocess,sys,importlib.metadata
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];WORK=ROOT/'sia-joint-full160/sia-simulator'
sys.path.insert(0,str(WORK))
from joint_prototype.full160 import capacities,METHODS
from joint_prototype.r0_liveness import R0Oracle,APPLICATION_TEMPLATES
from joint_prototype.joint_profile import GpuConfig,JobSpec
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(p,data):
    with p.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2)
data=OUT/'inputs';data.mkdir(exist_ok=False)
base=ROOT/'research_runs/sia_synergy/stage2c_r0_w10late/attempt1/frozen_config_6000.json'
old=json.loads(base.read_text());config={k:old[k] for k in ('cluster','response','simulation')}
config['cluster']={k:old['cluster'][k] for k in ('nnodes','ngpus_per_node','max_physical_nodes')}
config['cluster']['capacity_scenarios']={k:capacities(k) for k in ('main','half','double')}
config['response'].pop('template_map');config['response']['response_settings']={'neutral':{'exponent':0.},'medium':{'exponent':.5}}
config.update(resource_model_type='synthetic_cross_hardware_transfer',warning='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE',seed_profiles=[1,8],reference_config_hash=sha(base))
save(data/'config.json',config)
reg=ROOT/'research_runs/sia_synergy/stage2b_neutral/g0_verified/registry.json';(data/'registry.json').write_bytes(reg.read_bytes())
models=ROOT/'research_sources/Synergy/simulator/models';o=R0Oracle(reg,models,{a:a for a in APPLICATION_TEMPLATES},'medium')
traces={};hashes={str(reg):sha(reg),str(base):sha(base)}
for i in range(1,11):
    src=ROOT/f'official-artifacts/sia-simulator/workloads/saturn/workload-{i}.csv'
    with src.open(newline='') as f:r=list(csv.DictReader(f))
    assert len(r)==160 and len({x['name'] for x in r})==160
    assert {x['application'] for x in r}<=set(APPLICATION_TEMPLATES)
    arrival=[float(x['time']) for x in r];assert arrival==sorted(arrival)
    new=[dict(x,time=str(float(x['time'])-arrival[0]),source_row_index=str(j)) for j,x in enumerate(r)]
    dest=data/f'workload-{i}.csv'
    with dest.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(r[0])+['source_row_index']);w.writeheader();w.writerows(new)
    assert all(all(a[k]==b[k] for k in a if k!='time') for a,b in zip(r,new))
    traces[str(i)]=dict(source=str(src),source_hash=sha(src),file=str(dest),hash=sha(dest),rows=160,
        arrival_offset=arrival[0],arrival_span=arrival[-1]-arrival[0],rmst_tau=360000-(arrival[-1]-arrival[0]),
        application_counts={a:sum(x['application']==a for x in r) for a in APPLICATION_TEMPLATES})
    hashes[str(src)]=sha(src)
coverage=[]
for row in o.registry.values():
    a=row['application'];t,n,g=row['config_id'].split(':');g=int(g)
    cfg=GpuConfig(row['config_id'],t,g,int(n),1.,row['phi_c'],row['phi_m']);j=JobSpec(a,row['template'],(cfg,))
    for level in ('main','half','double'):
        cap=capacities(level)[t];ng=config['cluster']['ngpus_per_node'][t];pc=cap['cpu_cores']/ng;pm=cap['ram_gb']/ng
        assert pc>=1 and pm>=20.83 and g*pc<=cap['cpu_cores'] and g*pm<=cap['ram_gb']
        template=o.template(row['template']);demand=(max(1,template.cpu_demand*row['phi_c']),max(20.83,template.memory_demand_gb*row['phi_m']))
        points={(c,m) for c in config['response']['aux_cpu_points']+[pc] for m in config['response']['aux_ram_points']+[pm]}
        points.add(demand)
        for c,m in sorted(points):
            q=o.query(j,cfg,c,m,(pc,pm));assert math.isfinite(q.rho) and q.rho>0
            coverage.append(dict(application=a,config=cfg.config_id,capacity=level,cpu=c,ram=m,proportional=(c,m)==(pc,pm),
                feasible=g*c<=cap['cpu_cores']+1e-9 and g*m<=cap['ram_gb']+1e-9,rho=q.rho,
                numerator_cpu_clipped=q.requested_source_cpu_coordinate!=q.source_cpu_coordinate,
                numerator_ram_clipped=q.requested_source_ram_coordinate!=q.source_ram_coordinate,
                denominator_cpu_clipped=q.requested_baseline_source_cpu_coordinate!=q.baseline_source_cpu_coordinate,
                denominator_ram_clipped=q.requested_baseline_source_ram_coordinate!=q.baseline_source_ram_coordinate))
        for src in row['source_rows']: pass
for p in models.glob('*.py'):hashes[str(p)]=sha(p)
for p in (ROOT/'official-artifacts/sia-simulator').rglob('*.csv'):hashes[str(p)]=sha(p)
for p in (WORK/'traces').rglob('*'):
    if p.is_file():hashes[str(p)]=sha(p)
for p in data.iterdir():hashes[str(p)]=sha(p)
save(OUT/'data_preflight.json',dict(traces=traces,registry_entries=len(o.registry),coverage=coverage,input_hashes=hashes,
    proportional_all_165_feasible=True,missing=[]))
print(json.dumps(dict(traces=len(traces),jobs=sum(t['rows'] for t in traces.values()),coverage=len(coverage))))
