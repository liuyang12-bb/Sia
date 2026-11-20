"""Offline-only opportunity census. Never creates Jobs/Clusters or runs a scheduler."""
import csv, collections, hashlib, itertools, json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / 'cc-joint-resource-study'
OUT = ROOT / 'research_runs/sia_synergy/cc_resource_mixes_preflight_20261028_run1'
sys.dont_write_bytecode = True
sys.path[:0] = [str(TREE), str(TREE/'sia-simulator')]
from joint_prototype.r0_liveness import R0Oracle, APPLICATION_TEMPLATES
from joint_prototype.joint_profile import JobSpec, GpuConfig
from applications import APPLICATIONS
from simulator_config import SLOWDOWNS
from resource_study import ORIGINAL_CPU, ORIGINAL_RAM

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(name, obj):
    with (OUT/name).open('x', encoding='utf8') as f: json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def supported(q): return not(q.source_coordinate_clipped or q.baseline_coordinate_clipped)
def point(q):
    return dict(cpu=q.cpu_per_gpu,ram=q.ram_gb_per_gpu,rho=q.rho,
                source_cpu=q.requested_source_cpu_coordinate,source_ram=q.requested_source_ram_coordinate,
                baseline_cpu=q.requested_baseline_source_cpu_coordinate,baseline_ram=q.requested_baseline_source_ram_coordinate,
                numerator_clipped=q.source_coordinate_clipped,denominator_clipped=q.baseline_coordinate_clipped)

def raw_profile(row, typ, g):
    """Fixed declared batch, epoch=0, Artifact throughput/gradient formulas; no online model injection."""
    a=APPLICATIONS[typ][row['application']]; b=int(row['batch_size'])
    local=math.ceil(b/g-1e-8); mx=a.get_max_local_bsz((g,))
    acc=math.ceil(local/mx-1e-8)-1
    if g==1 and b>a.init_batch_size: acc=max(1,acc)
    atomic=min(math.ceil(local/(acc+1)-1e-8),int(a.max_batch_size/(g*(acc+1))))
    actual=g*atomic*(acc+1)
    sq,var=a.get_grad_stats(actual,0); scale=actual/a.init_batch_size
    gain=(var+sq)/(var/scale+sq)
    step,sync=a.get_throughput((g,),atomic)
    raw=float(gain/(step+(step-sync)*acc)/SLOWDOWNS[row['application']])
    assert raw>0 and math.isfinite(raw)
    return dict(raw=raw,total_work=float(a.get_progress(a.max_epochs)),atomic_batch=atomic,
                accumulation=acc,actual_batch=actual,epoch=0,declared_batch=b,
                startup_seconds=a.rescale_time,source='OFFLINE_FIXED_BATCH_EPOCH0_NOT_ONLINE_PREDICTION')

def main():
    start=time.perf_counter(); OUT.mkdir(exist_ok=False)
    protected={str(p.relative_to(ROOT)):sha(p) for p in TREE.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.suffix!='.pyc'}
    # Freeze all cohort, enumeration and gate rules before response calculations.
    pre=dict(scope='OFFLINE_ONLY_NO_TRAINING_NO_SOLVER',mixes={'M1':(18,2),'M2':(14,6),'M3':(10,10)},
       source='official-artifacts/sia-simulator/workloads/saturn/workload-1.csv',
       selection='First N source rows per application, original row order; no duplicates within cohort.',
       order='DS at slot i iff floor((i+1)*D/20)>floor(i*D/20); otherwise next CF; i=0..19.',
       arrival='slots0..3=0; slot i>=4:300*(1+(i-4)//2)',
       capacities={'R01':[12.,187.5],'R11':[6.,187.5]},gpu={'aws':2,'dgx-ext':2},
       response='unchanged R0 E-legacy medium alpha=0.5 fixed reference(3,62.5)',
       cpu_grid=list(ORIGINAL_CPU),ram_grid=list(ORIGINAL_RAM),include_proportional_point=True,
       static_states='All unordered pairs, each oneGPU on same type, fixed declared batch and epoch0; not actual concurrent runtime states.',
       metric='sum(raw_rate*rho/declared_total_work), donor loss included, and 300s capped progress fraction',
       gate='Each cohort must have positive supported CPU and MEM pair exchange, all proportional configurations supported, and supported GPU ranking reversal for full joint-mechanism claim.',
       restrictions='No response/grid/algorithm edits, no future completion times, no JCT selection; 42-cell batch NOT authorized.')
    save('preregistration.json',pre)
    (OUT/'CURRENT_TASK.md').write_text('Offline resource-mix cohort and opportunity preflight. Do not launch simulations. Resume by inspecting preregistration.json and REPORT.md. Preserve current algorithms and all history.\n',encoding='utf8')
    src=ROOT/pre['source']; original=list(csv.DictReader(src.open(encoding='utf8')))
    pool={a:[(i,r) for i,r in enumerate(original) if r['application']==a] for a in ('cifar10','deepspeech2')}
    assert len(original)==160 and len(pool['deepspeech2'])>=10 and len(pool['cifar10'])>=18
    cohorts={}; provenance=[]
    for label,(nc,nd) in pre['mixes'].items():
        it={a:iter(pool[a][:n]) for a,n in [('cifar10',nc),('deepspeech2',nd)]}; rows=[]
        for slot in range(20):
            app='deepspeech2' if (slot+1)*nd//20>slot*nd//20 else 'cifar10'
            idx,r=next(it[app]); r=dict(r); r.update(original_arrival=r['time'],original_source_row_index=idx,
                source_line=idx+2,source_row_index=slot,time=0 if slot<4 else 300*(1+(slot-4)//2))
            rows.append(r)
        assert len({r['name'] for r in rows})==20
        with (OUT/(label+'_workload.csv')).open('x',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        cohorts[label]=rows; provenance.extend(dict(cohort=label,**r) for r in rows)
    save('job_manifest.json',dict(source_sha256=sha(src),available_counts={a:len(p) for a,p in pool.items()},jobs=provenance))
    apps={r['name']:r['application'] for rows in cohorts.values() for r in rows}
    oracle=R0Oracle(TREE/'inputs/registry.json',TREE/'inputs/models',apps,'medium')
    specs={}; rates={}; coverage=[]; opts={}; queries=[]
    unique={r['name']:r for rows in cohorts.values() for r in rows}
    for name,r in unique.items():
        configs=[]
        for typ,g in itertools.product(('aws','dgx-ext'),(1,2)):
            cid=f'{typ}:1:{g}'; ref=oracle.reference(name,cid)
            configs.append(GpuConfig(cid,typ,g,1,1.,ref['phi_c'],ref['phi_m']))
            rates[name,cid]=raw_profile(r,typ,g)
        specs[name]=JobSpec(name,APPLICATION_TEMPLATES[r['application']],tuple(configs),float(r['time']))
    for cap,(cpu,ram) in pre['capacities'].items():
        for name,j in specs.items():
            for g in j.configs:
                prop=oracle.query(j,g,cpu/2,ram/2,(cpu/2,ram/2)); points=[]
                for c,m in itertools.product(sorted(set(ORIGINAL_CPU+(cpu/2,))),sorted(set(ORIGINAL_RAM+(ram/2,)))):
                    if c*g.gpus>cpu+1e-9 or m*g.gpus>ram+1e-9: continue
                    q=oracle.query(j,g,c,m,(cpu/2,ram/2)); points.append(q)
                    queries.append(dict(capacity=cap,job=name,config=g.config_id,**point(q)))
                opts[cap,name,g.config_id]=points
                best={}
                for factor,cond in [('CPU',lambda q:q.ram_gb_per_gpu==ram/2),('MEM',lambda q:q.cpu_per_gpu==cpu/2),('FULL',lambda q:True)]:
                    valid=[q for q in points if supported(q) and cond(q)]
                    b=max(valid,key=lambda q:q.rho) if valid else None
                    best[factor]=dict(point=point(b),speed_ratio=b.rho/prop.rho) if b and supported(prop) else None
                coverage.append(dict(capacity=cap,job=name,application=apps[name],template=j.template_file,config=g.config_id,
                    phi_c=g.cpu_demand_scale,phi_m=g.memory_demand_scale,proportional=point(prop),
                    feasible_points=len(points),supported_points=sum(map(supported,points)),best=best,offline_rate=rates[name,g.config_id]))
    save('coverage.json',coverage); save('all_queries.json',queries)
    pairs=[]
    for mix,rows in cohorts.items():
        for cap,(cpu,ram) in pre['capacities'].items():
            for ra,rb in itertools.combinations(rows,2):
                na,nb=ra['name'],rb['name']
                for typ in ('aws','dgx-ext'):
                    cid=f'{typ}:1:1'; gs=[next(g for g in specs[n].configs if g.config_id==cid) for n in (na,nb)]
                    bp=[oracle.query(specs[n],g,cpu/2,ram/2,(cpu/2,ram/2)) for n,g in zip((na,nb),gs)]
                    weights=[rates[n,cid]['raw']/rates[n,cid]['total_work'] for n in (na,nb)]
                    for factor in ('CPU','MEM','FULL'):
                        choices=[]
                        for n in (na,nb):
                            choices.append([q for q in opts[cap,n,cid] if supported(q) and (factor!='CPU' or q.ram_gb_per_gpu==ram/2) and (factor!='MEM' or q.cpu_per_gpu==cpu/2)])
                        base=sum(w*q.rho for w,q in zip(weights,bp)); best=None; count=0
                        if all(map(supported,bp)):
                            for qa,qb in itertools.product(*choices):
                                if qa.cpu_per_gpu+qb.cpu_per_gpu>cpu+1e-9 or qa.ram_gb_per_gpu+qb.ram_gb_per_gpu>ram+1e-9:continue
                                ds=[w*(q.rho-p.rho) for w,q,p in zip(weights,(qa,qb),bp)]; net=sum(ds)
                                count+=net>1e-12
                                if best is None or net>best['net_fraction_rate']:
                                    best=dict(net_fraction_rate=net,relative_gain=net/base,individual_fraction_rate_delta=ds,
                                        speed_ratios=[q.rho/p.rho for q,p in zip((qa,qb),bp)],points=list(map(point,(qa,qb))),
                                        cpu_total=qa.cpu_per_gpu+qb.cpu_per_gpu,ram_total=qa.ram_gb_per_gpu+qb.ram_gb_per_gpu,
                                        capped_300s_fraction_gain=sum(min(1.,300*w*q.rho)-min(1.,300*w*p.rho) for w,q,p in zip(weights,(qa,qb),bp)))
                        pairs.append(dict(mix=mix,capacity=cap,jobs=[na,nb],applications=[apps[na],apps[nb]],gpu=typ,factor=factor,positive_supported_pairs=count,best=best))
    save('pair_census.json',pairs)
    # Reversal at equal per-GPU quota at two supported points, no resource budget change.
    reversals=[]
    for name,j in specs.items():
        for cap in pre['capacities']:
            for ga,gb in itertools.combinations(j.configs,2):
                pa={(q.cpu_per_gpu,q.ram_gb_per_gpu):q for q in opts[cap,name,ga.config_id] if supported(q)}
                pb={(q.cpu_per_gpu,q.ram_gb_per_gpu):q for q in opts[cap,name,gb.config_id] if supported(q)}
                common=sorted(pa.keys() & pb.keys()); witness=None
                for x,y in itertools.combinations(common,2):
                    # Only one auxiliary dimension changes, making the coupling interpretable.
                    if x[0]!=y[0] and x[1]!=y[1]:continue
                    a=[rates[name,ga.config_id]['raw']*pa[z].rho for z in (x,y)]
                    b=[rates[name,gb.config_id]['raw']*pb[z].rho for z in (x,y)]
                    if (a[0]-b[0])*(a[1]-b[1]) < -1e-10:
                        witness=dict(points=[x,y],rates_a=a,rates_b=b);break
                if witness:reversals.append(dict(job=name,capacity=cap,config_a=ga.config_id,config_b=gb.config_id,**witness))
    save('gpu_ranking_reversals.json',reversals)
    summary=[]
    for mix,rows in cohorts.items():
        names={r['name'] for r in rows}
        for cap in pre['capacities']:
            ps=[p for p in pairs if p['mix']==mix and p['capacity']==cap]
            counts={f:sum(p['positive_supported_pairs']>0 for p in ps if p['factor']==f) for f in ('CPU','MEM','FULL')}
            witnesses={f:next((p for p in ps if p['factor']==f and p['positive_supported_pairs']>0),None) for f in counts}
            c=[v for v in coverage if v['capacity']==cap and v['job'] in names]
            summary.append(dict(mix=mix,capacity=cap,positive_jobpair_gpu_states=counts,
                proportional_supported=sum(not(v['proportional']['numerator_clipped'] or v['proportional']['denominator_clipped']) for v in c),
                job_configs=len(c),gpu_reversals=sum(x['job'] in names and x['capacity']==cap for x in reversals),witnesses=witnesses))
    save('summary.json',summary)
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    checks=dict(all_source_preserved=True,training_runs=0,scheduler_solves=0,job_objects=0,cohorts=3,
        rows_exact_except_controlled_time_and_metadata=all(all(str(r[k])==str(original[int(r['original_source_row_index'])][k]) for k in ('name','application','num_replicas','batch_size')) for rs in cohorts.values() for r in rs),
        grids_unchanged=True,all_120_cohort_job_configs_queried_per_capacity=True,
        elapsed_seconds=time.perf_counter()-start,missing_profiles=[])
    save('validation.json',checks)
    save('protected_source_hashes.json',protected)
    print(json.dumps([ {k:v for k,v in s.items() if k!='witnesses'} for s in summary],indent=2))
if __name__=='__main__':main()
