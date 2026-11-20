"""Static queries, capacity arithmetic and <=8 resource-only offline MILPs.

This program never imports Job/Cluster/full160 nor calls any training step or
scheduler optimize. Existing response code is called without modifying it.
"""
import argparse,collections,csv,datetime,gzip,hashlib,importlib.metadata as md,itertools,json,math,statistics,subprocess,sys,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
BRIDGE=ROOT/'research_runs/sia_synergy/b3_version_bridge_20261026'
P=json.loads((BRIDGE/'preregistration.json').read_text(encoding='utf-8'))
TREE=Path(P['tree']);sys.path.insert(0,str(TREE/'sia-simulator'))
from joint_prototype.joint_profile import JointProfileOracle,GpuConfig,JobSpec,CPU_COORDS,RAM_COORDS

CONFIG=json.loads(Path(P['config']).read_text(encoding='utf-8'))
REG_ROWS=json.loads(Path(P['registry']).read_text(encoding='utf-8'));REG={r['key']:r for r in REG_ROWS}
TYPES={'aws':4,'dgx-ext':8,'rtx':8};APPS=('bert','cifar10','deepspeech2')
GRID=CONFIG['response'];MAPPING=GRID['application_template_rule']
ORACLES={r:JointProfileOracle(Path(P['models']),'fixed_reference',(3.,62.5),r=='neutral',0. if r=='neutral' else .5) for r in ['neutral','medium']}
QUERIES=0

def sha(f):
    h=hashlib.sha256()
    with Path(f).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest().upper()
def save(name,data):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False)
def readcsv(f):
    with Path(f).open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))
def caps(level):return CONFIG['cluster']['capacity_scenarios'][level]
def prop(level):return (3. if level=='main' else 1.5,62.5)
def close(a,b):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9)
def clip_flags(q):
    def side(v,lo,hi):return 'low' if v<lo else ('high' if v>hi else 'inside')
    return dict(numerator_cpu=side(q.requested_source_cpu_coordinate,1.,24.),
        numerator_ram=side(q.requested_source_ram_coordinate,20.83,250.),
        denominator_cpu=side(q.requested_baseline_source_cpu_coordinate,1.,24.),
        denominator_ram=side(q.requested_baseline_source_ram_coordinate,20.83,250.))
def inside(q):return all(v=='inside' for v in clip_flags(q).values())
def query(app,cid,cpu,ram,level='half',response='medium',job_id=None,raw=1.):
    global QUERIES
    r=REG[app+'/'+cid];typ,n,g=cid.split(':')
    c=GpuConfig(cid,typ,int(g),int(n),float(raw),r['phi_c'],r['phi_m'])
    j=JobSpec(job_id or 'OFFLINE_RESPONSE_UNIT_'+app,MAPPING[app],(c,))
    q=ORACLES[response].query(j,c,cpu,ram,prop(level))
    # Independent interpolation: source code's _bilinear is not used here.
    matrix=ORACLES[response].template(MAPPING[app]).matrix
    def interpolate(cu,me):return float(np.interp(cu,CPU_COORDS,[np.interp(me,RAM_COORDS,row) for row in matrix]))
    num=interpolate(cpu/r['phi_c'],ram/r['phi_m']);den=interpolate(3./r['phi_c'],62.5/r['phi_m'])
    expected=1. if response=='neutral' else math.sqrt(num/den)
    assert close(q.rho,expected) and close(q.effective_goodput,float(raw)*expected)
    assert q.profile_baseline_cpu==3. and q.profile_baseline_ram_gb==62.5
    QUERIES+=1;return q
def brief(q):
    return dict(cpu=q.cpu_per_gpu,ram=q.ram_gb_per_gpu,rho=q.rho,
        response_raw=q.response_raw,response_baseline=q.response_baseline,
        source_cpu=q.requested_source_cpu_coordinate,source_ram=q.requested_source_ram_coordinate,
        baseline_cpu=q.requested_baseline_source_cpu_coordinate,baseline_ram=q.requested_baseline_source_ram_coordinate,
        used_cpu=q.source_cpu_coordinate,used_ram=q.source_ram_coordinate,clipping=clip_flags(q))

def verify_sources():
    files={};checked=0
    for name,h in P['hashes'].items():assert sha(name)==h,name;checked+=1
    expected={f'{a}/{t}:1:{g}' for a in MAPPING for t,gs in [('aws',[1,2,4]),('dgx-ext',[1,2,4,8]),('rtx',[1,2,4,8])] for g in gs}
    assert set(REG)==expected and len(REG_ROWS)==55
    assert sha(P['registry'])=='78F676E7695CBE09AA9A3DB71F539FE2A2CEF1DBC243E57CC064E4E294DCF1C9'
    cache={};sources=[]
    for r in REG_ROWS:
        for item in r['source_rows']:
            name=item['file'];files[name]=item['sha256']
            if name not in cache:assert sha(name)==item['sha256'];cache[name]=readcsv(name)
            row=cache[name][item['line']-2]
            assert close(float(row['local_bsz']),item['microbatch']) and close(float(row['step_time']),item['step_time'])
        step=statistics.median(s['step_time'] for s in r['source_rows'])
        g=int(r['config_id'].split(':')[-1])
        assert close(step,r['step_time']) and close(r['T0'],r['b_ref']*g/step)
        assert close(r['per_gpu_T0'],r['T0']/g)
        assert close(r['phi_c'],r['per_gpu_T0']/r['denominator']) and close(r['phi_m'],math.sqrt(r['phi_c']))
        sources.append({k:r[k] for k in ['key','b_ref','aggregation','source_rows','T0','denominator','phi_c','phi_m','template','reference_state']})
    for a in MAPPING:
        rows=[r for r in REG_ROWS if r['application']==a]
        assert len(rows)==11 and all(close(r['denominator'],min(x['per_gpu_T0'] for x in rows)) for r in rows)
    for a,t in MAPPING.items():
        f=Path(P['models'])/t;files[str(f)]=sha(f)
    save('source_validation.json',dict(valid=True,bridge_hashes_checked=checked,registry_entries=len(REG),source_files=files,source_rows=sources))
    return files

def domain():
    rows=[];points=[]
    for a in APPS:
      for cid in sorted(r['config_id'] for r in REG_ROWS if r['application']==a):
        typ,n,g=cid.split(':');g=int(g)
        main=query(a,cid,3.,62.5,'main');half=query(a,cid,1.5,62.5,'half')
        assert close(main.rho,1.)
        both={}
        for level,baseline in [('main',main),('half',half)]:
            cpu_set=sorted(set(GRID['aux_cpu_points']+[prop(level)[0]]));ram_set=sorted(set(GRID['aux_ram_points']+[62.5]))
            feasible=[];eligible=[]
            for cpu,ram in itertools.product(cpu_set,ram_set):
                q=query(a,cid,cpu,ram,level);fits=g*cpu<=caps(level)[typ]['cpu_cores']+1e-9 and g*ram<=caps(level)[typ]['ram_gb']+1e-9
                legal=fits and q.rho+1e-9>=baseline.rho
                point=dict(application=a,config=cid,capacity=level,physical_fits=fits,progress_floor_eligible=legal,**brief(q))
                points.append(point)
                if fits:feasible.append(q)
                if legal:eligible.append(q)
            assert eligible
            def best(vs):return max(vs,key=lambda q:(q.rho,-q.cpu_per_gpu,-q.ram_gb_per_gpu))
            cpu_only=[q for q in eligible if q.ram_gb_per_gpu==62.5]
            ram_only=[q for q in eligible if q.cpu_per_gpu==prop(level)[0]]
            cpu_best=best(cpu_only);ram_best=best(ram_only);joint_best=best(eligible)
            donors=[q for q in cpu_only if q.cpu_per_gpu<prop(level)[0] and close(q.rho,baseline.rho)]
            interior=[q for q in eligible if inside(q) and inside(baseline)]
            cpu_interior=[q for q in cpu_only if inside(q) and inside(baseline)]
            both[level]=dict(proportional=brief(baseline),proportional_feasible=g*prop(level)[0]<=caps(level)[typ]['cpu_cores']+1e-9 and g*62.5<=caps(level)[typ]['ram_gb']+1e-9,
                physical_point_count=len(feasible),eligible_point_count=len(eligible),cpu_best=brief(cpu_best),ram_best=brief(ram_best),joint_best=brief(joint_best),
                cpu_gain=cpu_best.rho/baseline.rho-1,ram_gain=ram_best.rho/baseline.rho-1,joint_gain=joint_best.rho/baseline.rho-1,
                cpu_no_loss_donors=[brief(q) for q in donors],interior_eligible_point_count=len(interior),
                interior_cpu_gain=(best(cpu_interior).rho/baseline.rho-1) if cpu_interior else None)
        rows.append(dict(application=a,template=MAPPING[a],config=cid,gpu_type=typ,gpus=g,phi_c=REG[a+'/'+cid]['phi_c'],phi_m=REG[a+'/'+cid]['phi_m'],
            main=both['main'],half=both['half'],half_relative_to_main=half.rho/main.rho,
            note='Physical R0 domain; not an assertion that every job has every config online at every round. Unit raw Goodput only isolates rho; no absolute execution rate inferred.'))
    save('configuration_coverage.json',rows);save('all_response_points.json',points)
    return rows,points

def state_sources():
    result={}
    for ds in ['original20','holdout20']:
        folder=Path(next(c['folder'] for c in P['reference_cells'] if c['dataset']==ds and c['response']=='medium' and c['method']=='joint_g')) if ds=='original20' else BRIDGE/'holdout20_medium_joint_g'
        result[ds]=folder
    return result

def scan_logs():
    states=[];observed={};loghash={}
    for ds,folder in state_sources().items():
        file=folder/'rounds.jsonl.gz';loghash[str(file)]=sha(file);first=None;warm=None;last_round=None
        with gzip.open(file,'rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);d=r['decision_for_next_interval']
                if d['round']==last_round:continue
                last_round=d['round'];pb=d['probe'];specs=pb['specs']
                for j in specs:
                    for c in j['configs']:
                        observed.setdefault((ds,j['job_id'],c['config_id']),dict(dataset=ds,job=j['job_id'],application=j['job_id'].rsplit('-',1)[0],config=c['config_id'],first_observed_time=r['interval_end'],source=pb['provenance'][j['job_id']][c['config_id']]['goodput_source'],normalized_online_goodput=c['sia_goodput']))
                if first is None:first=r
                productive=pb['l2_before']['productive']
                present={j['job_id'].rsplit('-',1)[0] for j in specs if productive.get(j['job_id'],0.)>0}
                if warm is None and set(APPS)<=present and not any(j['checkpoint_locked'] for j in specs):warm=r
        for label,r in [('FIRST_DECISION',first),('FIRST_WARM_NO_CHECKPOINT',warm)]:
            states.append(dict(dataset=ds,criterion=label,status='FOUND' if r else 'NO_MATCH',round=r))
    save('selected_state_inputs.json',states)
    return states,list(observed.values()),loghash

def task_coverage(observed,rows):
    lookup={(r['application'],r['config']):r for r in rows};records=[];tasks=[]
    for ds in ['original20','holdout20']:
        for job in readcsv(P['datasets'][ds]):
            seen=[v for v in observed if v['dataset']==ds and v['job']==job['name']]
            assert seen,job
            cfgs=[lookup[job['application'],v['config']] for v in seen]
            t=dict(dataset=ds,job=job['name'],application=job['application'],source_row_index=int(job['source_row_index']),original_arrival=float(job['original_arrival']),arrival=float(job['time']),gpu_demand=int(job['num_replicas']),batch=int(job['batch_size']),
                observed_configs=len(cfgs),slowdown_configs=sum(r['half_relative_to_main']<1-1e-9 for r in cfgs),cpu_recoverable_configs=sum(r['half']['cpu_gain']>1e-9 for r in cfgs),
                max_cpu_gain=max(r['half']['cpu_gain'] for r in cfgs),no_loss_donor_configs=sum(bool(r['half']['cpu_no_loss_donors']) for r in cfgs))
            tasks.append(t)
            for v in seen:
                r=lookup[job['application'],v['config']]
                records.append(dict(**v,half_rho=r['half']['proportional']['rho'],main_rho=r['main']['proportional']['rho'],half_proportional=r['half']['proportional'],
                    feasible_cpu_alternative=r['half']['cpu_best'],cpu_gain=r['half']['cpu_gain'],feasible_joint_alternative=r['half']['joint_best'],joint_gain=r['half']['joint_gain']))
    save('task_coverage.json',tasks);save('task_configuration_evidence.json',records)
    return tasks

def options(job,level,mode):
    a=job['job_id'].rsplit('-',1)[0];cid=job['config']['config_id'];g=job['config']['gpus'];typ=job['config']['gpu_type'];base=query(a,cid,*prop(level),level)
    cpus=sorted(set(GRID['aux_cpu_points']+[prop(level)[0]]));rams=[62.5] if mode=='cpu_only' else sorted(set(GRID['aux_ram_points']+[62.5]))
    result=[]
    for cpu,ram in itertools.product(cpus,rams):
        if cpu*g>caps(level)[typ]['cpu_cores']+1e-9 or ram*g>caps(level)[typ]['ram_gb']+1e-9:continue
        q=query(a,cid,cpu,ram,level)
        if q.rho+1e-9<base.rho:continue
        result.append(dict(cpu=cpu,ram=ram,rho=q.rho,relative_speed=q.rho/base.rho,baseline_rho=base.rho,
            clipping=clip_flags(q),baseline_clipping=clip_flags(base),inside=inside(q) and inside(base)))
    return result

def cpu_dp(jobs,level):
    result={};score=0.
    for typ in TYPES:
        subset=[j for j in jobs if j['config']['gpu_type']==typ];capacity=int(round(caps(level)[typ]['cpu_cores']*2))
        dp={0:(0.,[])}
        for j in subset:
            nxt={}
            for used,(value,choices) in dp.items():
                for q in options(j,level,'cpu_only'):
                    cost=int(round(q['cpu']*j['config']['gpus']*2));u=used+cost
                    if u>capacity:continue
                    v=value+q['relative_speed']
                    if u not in nxt or v>nxt[u][0]+1e-10:nxt[u]=(v,choices+[(j['job_id'],q)])
            dp=nxt
        assert dp
        used,(value,choices)=max(dp.items(),key=lambda kv:(kv[1][0],-kv[0]));score+=value;result.update(choices)
    return result,score,dict(method='EXACT_HALF_CPU_UNIT_DYNAMIC_PROGRAMMING',optimal=True)

def joint_mip(jobs,level):
    groups={}
    for j in jobs:groups.setdefault((j['config']['gpu_type'],j['job_id'].rsplit('-',1)[0],j['config']['config_id']),[]).append(j)
    keys=sorted(groups);cols=[]
    for key in keys:
        for q in options(groups[key][0],level,'joint'):cols.append((key,q))
    A=[];lb=[];ub=[]
    for k in keys:A.append([float(key==k) for key,q in cols]);lb.append(len(groups[k]));ub.append(len(groups[k]))
    for typ in TYPES:
        for resource,capacity in [('cpu',caps(level)[typ]['cpu_cores']),('ram',caps(level)[typ]['ram_gb'])]:
            A.append([groups[k][0]['config']['gpus']*q[resource] if k[0]==typ else 0. for k,q in cols]);lb.append(-np.inf);ub.append(capacity)
    A=np.asarray(A);lb=np.asarray(lb);ub=np.asarray(ub);cost=-np.asarray([q['relative_speed'] for k,q in cols])
    tick=time.perf_counter();res=milp(c=cost,integrality=np.ones(len(cols)),bounds=Bounds(0,np.asarray([len(groups[k]) for k,q in cols])),
        constraints=LinearConstraint(A,lb,ub),options={'time_limit':30.,'mip_rel_gap':0.})
    diag=dict(method='OFFLINE_GROUPED_AUXILIARY_RESOURCE_MILP',status=int(res.status),message=str(res.message),seconds=time.perf_counter()-tick,
        columns=len(cols),groups=len(keys),objective='sum individual rho / own proportional rho; all jobs mandatory; no GPU selection or training simulation',optimal=bool(res.success and res.status==0))
    if res.x is None:return None,None,diag
    x=np.asarray(res.x);y=np.round(x)
    if np.max(abs(x-y))>1e-6 or np.any(A@x<lb-1e-7) or np.any(A@x>ub+1e-7) or np.any(A@y<lb-1e-7) or np.any(A@y>ub+1e-7):
        diag['validation_failure']='INVALID_RAW_OR_ROUNDED_VECTOR';return None,None,diag
    selected={};cursors=collections.Counter()
    for (k,q),count in zip(cols,y.astype(int)):
        for _ in range(count):
            j=groups[k][cursors[k]];cursors[k]+=1;selected[j['job_id']]=q
    assert len(selected)==len(jobs)
    return selected,sum(q['relative_speed'] for q in selected.values()),diag

def verify_witness(jobs,selected,level):
    usage={t:dict(gpu=0,cpu=0.,ram=0.) for t in TYPES};changes=[]
    for j in jobs:
        name=j['job_id'];q=selected[name];typ=j['config']['gpu_type'];g=j['config']['gpus'];a=name.rsplit('-',1)[0]
        original=query(a,j['config']['config_id'],*prop(level),level,job_id=name)
        actual=query(a,j['config']['config_id'],q['cpu'],q['ram'],level,job_id=name)
        assert close(actual.rho,q['rho']) and actual.rho+1e-9>=original.rho
        u=usage[typ];u['gpu']+=g;u['cpu']+=g*q['cpu'];u['ram']+=g*q['ram']
        changes.append(dict(job=name,application=a,config=j['config']['config_id'],gpus=g,placement=j['placement'],
            baseline=brief(original),allocation=brief(actual),relative_speed=actual.rho/original.rho,
            cpu_delta=g*(q['cpu']-prop(level)[0]),ram_delta=g*(q['ram']-62.5),inside=inside(actual) and inside(original)))
    for t,u in usage.items():assert u['gpu']<=TYPES[t] and u['cpu']<=caps(level)[t]['cpu_cores']+1e-7 and u['ram']<=caps(level)[t]['ram_gb']+1e-7
    return dict(valid=True,usage=usage,job_changes=changes,improved_jobs=sum(c['relative_speed']>1+1e-9 for c in changes),
        no_loss_cpu_donors=sum(c['cpu_delta']<0 and close(c['relative_speed'],1) for c in changes),
        interior_improved_jobs=sum(c['relative_speed']>1+1e-9 and c['inside'] for c in changes),
        clipped_improved_jobs=sum(c['relative_speed']>1+1e-9 and not c['inside'] for c in changes),
        cpu_saved_without_slowdown=-sum(c['cpu_delta'] for c in changes if c['cpu_delta']<0),
        extra_cpu_consumed=sum(c['cpu_delta'] for c in changes if c['cpu_delta']>0))

def state_analysis(states):
    results=[]
    for record in states:
        if record['status']!='FOUND':results.append({k:v for k,v in record.items() if k!='round'});continue
        r=record['round'];d=r['decision_for_next_interval'];jobs=[dict(v,job_id=j) for j,v in sorted(d['result']['selected'].items())]
        for level in ['main','half']:
            baseline={j['job_id']:dict(cpu=prop(level)[0],ram=62.5,rho=query(j['job_id'].rsplit('-',1)[0],j['config']['config_id'],*prop(level),level).rho) for j in jobs}
            base=verify_witness(jobs,baseline,level)
            for mode in ['cpu_only','joint']:
                chosen,score,diag=cpu_dp(jobs,level) if mode=='cpu_only' else joint_mip(jobs,level)
                entry=dict(dataset=record['dataset'],criterion=record['criterion'],time=r['interval_end'],capacity=level,mode=mode,jobs=len(jobs),baseline=base,diagnostics=diag)
                if chosen is not None:
                    witness=verify_witness(jobs,chosen,level);assert score>=len(jobs)-1e-7
                    entry.update(relative_speed_sum=score,mean_relative_speed=score/len(jobs),witness=witness)
                else:entry['status']='NO_VALID_WITNESS'
                results.append(entry)
    save('same_geometry_resource_witnesses.json',results);return results

def summary(rows,points,tasks,witnesses):
    config_counts={}
    for a in APPS:
        rs=[r for r in rows if r['application']==a];hr=[r for r in rs if r['half_relative_to_main']<1-1e-9]
        config_counts[a]=dict(configs=len(rs),half_slowdown=len(hr),half_rho_range=[min(r['half_relative_to_main'] for r in rs),max(r['half_relative_to_main'] for r in rs)],
            cpu_gain_configs=sum(r['half']['cpu_gain']>1e-9 for r in rs),ram_gain_configs=sum(r['half']['ram_gain']>1e-9 for r in rs),
            no_loss_cpu_donor_configs=sum(bool(r['half']['cpu_no_loss_donors']) for r in rs),
            cpu_interior_gain_configs=sum(r['half']['interior_cpu_gain'] is not None and r['half']['interior_cpu_gain']>1e-9 for r in rs),
            half_proportional_numerator_cpu_clipped=sum(r['half']['proportional']['clipping']['numerator_cpu']!='inside' for r in rs),
            denominator_cpu_clipped=sum(r['half']['proportional']['clipping']['denominator_cpu']!='inside' for r in rs))
    point_counts={}
    for level in ['main','half']:
        rs=[r for r in points if r['capacity']==level];feasible=[r for r in rs if r['physical_fits']];eligible=[r for r in rs if r['progress_floor_eligible']]
        point_counts[level]=dict(queried=len(rs),physical=len(feasible),eligible=len(eligible),
            clipping_by_physical_point={k:dict(collections.Counter(r['clipping'][k] for r in feasible)) for k in ['numerator_cpu','numerator_ram','denominator_cpu','denominator_ram']},
            clipping_by_eligible_point={k:dict(collections.Counter(r['clipping'][k] for r in eligible)) for k in ['numerator_cpu','numerator_ram','denominator_cpu','denominator_ram']})
    sums=dict(application_config_counts=config_counts,point_counts=point_counts,
        tasks={ds:dict(jobs=sum(t['dataset']==ds for t in tasks),cpu_slowdown_possible_jobs=sum(t['dataset']==ds and t['slowdown_configs']>0 for t in tasks),cpu_recoverable_jobs=sum(t['dataset']==ds and t['cpu_recoverable_configs']>0 for t in tasks),observed_job_configs=sum(t['observed_configs'] for t in tasks if t['dataset']==ds)) for ds in ['original20','holdout20']},
        physical_proportional_feasible=all(r[l]['proportional_feasible'] for r in rows for l in ['main','half']),
        static_scenarios=[dict(dataset=w['dataset'],criterion=w['criterion'],capacity=w['capacity'],mode=w['mode'],time=w['time'],
            valid=w.get('witness',{}).get('valid',False),optimal=w['diagnostics']['optimal'],mean_relative_speed=w.get('mean_relative_speed'),
            improved=w.get('witness',{}).get('improved_jobs'),interior_improved=w.get('witness',{}).get('interior_improved_jobs'),clipped_improved=w.get('witness',{}).get('clipped_improved_jobs')) for w in witnesses if 'capacity' in w])
    save('summary.json',sums);return sums

def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(exist_ok=False);OUT=args.output.resolve()
    tick=time.perf_counter();started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    save('preregistration.json',dict(scope='OFFLINE_ONLY',script_sha256=sha(Path(__file__)),bridge_manifest_sha256=sha(BRIDGE/'preregistration.json'),
        datasets=['original20','holdout20'],capacities=['main','half'],states=['FIRST_DECISION','FIRST_WARM_NO_CHECKPOINT'],
        response='fixed R0/E-legacy Medium .5; Neutral .0 only identity checks',baseline_reference=[3.,62.5],
        static_models=['CPU-only exact DP','CPU+RAM grouped integer model'],maximum_milp_calls=8,milp_time_limit=30,
        changes_forbidden='model, mapping, phi, clipping, GPU/RAM capacities, workload, execution or scheduler source',
        interpretation='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE'))
    assert not any(m in sys.modules for m in ['job','cluster','joint_prototype.full160'])
    files=verify_sources();rows,points=domain();states,observed,loghash=scan_logs();tasks=task_coverage(observed,rows);witnesses=state_analysis(states);s=summary(rows,points,tasks,witnesses)
    for response in ORACLES:
        for a in APPS:
            for r in rows:
                if r['application']!=a:continue
                a1=query(a,r['config'],1.5,62.5,'half',response);a2=query(a,r['config'],1.5,62.5,'half',response,raw=2.)
                assert a1.rho==a2.rho
                if response=='neutral':assert a1.rho==1.
    assert not any(m in sys.modules for m in ['job','cluster','joint_prototype.full160'])
    for name,h in P['hashes'].items():assert sha(name)==h
    save('run_manifest.json',dict(scope='OFFLINE_ONLY_NO_SIMULATION',started_utc=started,seconds=time.perf_counter()-tick,
        command=sys.argv,tree=str(TREE),base_commit=P['base_commit'],branch=P['branch'],source_snapshot_sha256=sha(BRIDGE/'execution_source.zip'),
        bridge_manifest_sha256=sha(BRIDGE/'preregistration.json'),script_sha256=sha(Path(__file__)),
        config_sha256=sha(P['config']),registry_sha256=sha(P['registry']),models=files,logs=loghash,
        workload_hashes={k:sha(v) for k,v in P['datasets'].items() if k in ['original20','holdout20']},query_count=QUERIES,
        normalization='fixed_reference 3 CPU,62.5 GB; no capacity renormalization',alpha=.5,profiles_and_phi_unchanged=True,
        training_simulator_imported=False,cluster_or_job_constructed=False,simulation_step_calls=0,formal104='PAUSED',
        git_status=subprocess.check_output(['git','-C',str(TREE),'status','--short','--branch'],text=True),
        note='Resource-only relative speed and no-loss feasibility witnesses. Not JCT, online decision reproduction, exact joint scheduler objective, or target-hardware measurements.'))
    print(json.dumps(s,ensure_ascii=True,indent=2));print('OFFLINE_COMPLETE',QUERIES,time.perf_counter()-tick)
if __name__=='__main__':main()
