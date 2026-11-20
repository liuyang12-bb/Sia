"""Frozen legacy CC-Joint, twelve real jobs, eight preregistered cells."""
from pathlib import Path
import argparse, collections, csv, dataclasses, hashlib, importlib.metadata
import json, math, shutil, subprocess, sys, time, traceback, gzip

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
PARENT=OUT.parent/'joint_completion_continuity_20261026_run2'
SOURCE=OUT/'source'
LIVE=ROOT/'sia-joint-cc-12job/sia-simulator'
METHODS=['SIA_P','SIA_ECT_AUX','JOINT_CC_P','JOINT_CC']
RUNS=[(r,m) for r in ['neutral','medium'] for m in METHODS]
IDS=[0,1,2,3,4,5,8,9,12,16,21,30]
WARNING='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def default(v):
    if dataclasses.is_dataclass(v):return dataclasses.asdict(v)
    if hasattr(v,'item'):return v.item()
    raise TypeError(type(v).__name__)
def save(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False,default=default)
def gate():
    import psutil
    v=dict(memory_gib=psutil.virtual_memory().available/2**30,disk_gib=psutil.disk_usage(str(OUT)).free/2**30)
    if v['memory_gib']<2 or v['disk_gib']<5:raise RuntimeError('SAFETY_GATE:'+str(v))
    return v
def files(directory):return {f.relative_to(directory).as_posix():sha(f) for f in directory.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc'}
def verify():
    p=json.loads((OUT/'prepared.json').read_text())
    for directory,key in [(SOURCE,'source_hashes'),(OUT,'input_hashes')]:
        for n,h in p[key].items():assert sha(directory/n)==h,n
    if LIVE.exists():
        for n,h in p['source_hashes'].items():assert sha(LIVE/n)==h,n
    return p
def load():
    p=json.loads((OUT/'preregistration.json').read_text());verify()
    for n,h in p['runner_hashes'].items():assert sha(OUT/n)==h,n
    for pkg,v in p['packages'].items():assert importlib.metadata.version(pkg)==v,pkg
    return p

def prepare():
    parent=json.loads((PARENT/'preregistration.json').read_text())
    live=ROOT/'sia-joint-completion-continuity/sia-simulator'
    for n,h in parent['source_hashes'].items():assert sha(live/n)==h,n
    shutil.copytree(PARENT/'source',SOURCE)
    for n,h in parent['source_hashes'].items():assert sha(SOURCE/n)==h,n
    shutil.copytree(PARENT/'inputs',OUT/'inputs')
    # All inherited inputs are copied byte-for-byte; only workload is a new
    # deterministic extraction, never a duplicated or shortened training job.
    old=OUT/'inputs/workload.csv'
    old.rename(OUT/'inputs/parent_six_job_workload.csv')
    trace=ROOT/'official-artifacts/sia-simulator/workloads/saturn/workload-1.csv'
    shutil.copy2(trace,OUT/'inputs/original_workload-1.csv')
    with trace.open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    selected=[(i,r) for i,r in enumerate(rows) if int(r['name'].rsplit('-',1)[1]) in IDS]
    assert len(selected)==12
    assert [r['name'] for _,r in selected]==[f'cifar10-{i}' for i in range(6)]+[f'deepspeech2-{i}' for i in [8,9,12,16,21,30]]
    names=list(rows[0])+['original_arrival','source_row_index']
    with old.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        for i,r in selected:w.writerow(dict(r,time=0.,original_arrival=r['time'],source_row_index=i))
    shutil.copy2(PARENT/'experiment.py',OUT/'baseline_runner.py')
    save(OUT/'prepared.json',dict(parent_preregistration_sha256=sha(PARENT/'preregistration.json'),
        parent_source_fingerprint=parent['source_fingerprint'],source_hashes=parent['source_hashes'],
        input_hashes={f'inputs/{n}':h for n,h in files(OUT/'inputs').items()},
        baseline_runner_sha256=sha(OUT/'baseline_runner.py'),jobs=[dict(r,source_row_index=i) for i,r in selected],
        selection='Keep original6 and add next3 per application in original source order',warning=WARNING))
    print('PREPARED',len(parent['source_hashes']),len(selected),flush=True)

def construct(method,response):
    sys.path.insert(0,str(LIVE))
    import pandas as pd
    from joint_prototype.full160 import make_cluster
    from joint_prototype.small_mechanism import ProbeSia
    from joint_prototype.physical_domain import restrict_single_node_domain
    from joint_prototype.completion_time import install_aux
    from joint_prototype.completion_continuity import install
    cfg=json.loads((OUT/'inputs/config_symmetric.json').read_text())
    c=make_cluster(cfg,pd.read_csv(OUT/'inputs/workload.csv'),OUT/'inputs/registry.json',OUT/'inputs/models',
        'sia_p' if method.startswith('SIA') else 'joint_o',response,'main',True)
    c.policy.node_capacities={t:dict(cpu_cores=12.,ram_gb=250.) for t in ['aws','dgx-ext']}
    restrict_single_node_domain(c.policy)
    c.policy.aux_cpu_points=tuple(sorted(set(c.policy.aux_cpu_points)|{8.}))
    if method=='SIA_P':c.policy.__class__=ProbeSia
    elif method=='SIA_ECT_AUX':install_aux(c)
    else:install(c,proportional=method=='JOINT_CC_P')
    return c

def preflight():
    p=verify();sys.path.insert(0,str(SOURCE))
    from joint_prototype.joint_profile import JobSpec,GpuConfig
    from joint_prototype.r0_liveness import R0Oracle,APPLICATION_TEMPLATES
    apps={r['name']:r['application'] for r in p['jobs']}
    oracle=R0Oracle(OUT/'inputs/registry.json',OUT/'inputs/models',apps,'medium')
    cpus=[1.,2.,3.,4.,6.,8.,9.,12.,24.];rams=[20.83,62.5,125.,187.5,250.]
    rows=[];queries={};clip=collections.Counter()
    for j,a in apps.items():
      for typ in ['aws','dgx-ext']:
       for g in [1,2]:
        cid=f'{typ}:1:{g}';ref=oracle.reference(j,cid)
        config=GpuConfig(cid,typ,g,1,1.,ref['phi_c'],ref['phi_m'])
        spec=JobSpec(j,APPLICATION_TEMPLATES[a],(config,))
        prop=oracle.query(spec,config,6.,125.,(6.,125.));points=[]
        for cpu in cpus:
         for ram in rams:
          if cpu*g>12+1e-9 or ram*g>250+1e-9:continue
          q=oracle.query(spec,config,cpu,ram,(6.,125.))
          assert math.isfinite(q.rho) and q.rho>0
          points.append(q.to_dict());clip['queries']+=1
          clip['numerator_cpu']+=not 1<=q.requested_source_cpu_coordinate<=24
          clip['numerator_ram']+=not 20.83<=q.requested_source_ram_coordinate<=250
          clip['denominator_cpu']+=not 1<=q.requested_baseline_source_cpu_coordinate<=24
          clip['denominator_ram']+=not 20.83<=q.requested_baseline_source_ram_coordinate<=250
        supported=[q for q in points if not q['source_coordinate_clipped'] and not q['baseline_coordinate_clipped']]
        queries[j,cid]=points
        rows.append(dict(job=j,application=a,config=cid,phi_c=ref['phi_c'],phi_m=ref['phi_m'],
          template=ref['template'],proportional=prop.to_dict(),feasible_points=len(points),supported_points=len(supported),
          maximum_feasible_ratio=max(q['rho']/prop.rho for q in points),
          maximum_supported_ratio=max([q['rho']/prop.rho for q in supported],default=None),points=points))
    # Pair witness uses all template/CPU points at fixed one-GPU geometry;
    # normalize each job's rate to its own proportional rate. This is an
    # opportunity screen, not comparable absolute application work or JCT.
    pairs=[]
    ja=next(j for j,a in apps.items() if a=='cifar10');jb=next(j for j,a in apps.items() if a=='deepspeech2')
    for typ in ['aws','dgx-ext']:
      cid=f'{typ}:1:1';aa=queries[ja,cid];bb=queries[jb,cid]
      pa=next(q for q in aa if q['cpu_per_gpu']==6 and q['ram_gb_per_gpu']==125)
      pb=next(q for q in bb if q['cpu_per_gpu']==6 and q['ram_gb_per_gpu']==125)
      legal=[]
      for x in aa:
       for y in bb:
        if x['source_coordinate_clipped'] or x['baseline_coordinate_clipped'] or y['source_coordinate_clipped'] or y['baseline_coordinate_clipped']:continue
        if x['cpu_per_gpu']+y['cpu_per_gpu']>12 or x['ram_gb_per_gpu']+y['ram_gb_per_gpu']>250:continue
        legal.append(dict(cpu=[x['cpu_per_gpu'],y['cpu_per_gpu']],ram=[x['ram_gb_per_gpu'],y['ram_gb_per_gpu']],
          ratio=[x['rho']/pa['rho'],y['rho']/pb['rho']],score=x['rho']/pa['rho']+y['rho']/pb['rho']))
      best=max(legal,key=lambda v:v['score'])
      pairs.append(dict(gpu_type=typ,jobs=[ja,jb],fully_supported=True,enumerated=len(legal),best=best))
    assert len(rows)==48 and all(r['feasible_points'] for r in rows)
    assert any(p['best']['score']>2+1e-9 for p in pairs),'No supported pair complementarity'
    save(OUT/'coverage_preflight.json',dict(valid=True,training_run=False,coverage=rows,pair_witnesses=pairs,
        query_clipping_counts=dict(clip),warning=WARNING,
        interpretation='CPU/RAM speed opportunity exists under same budget; normalized pair gains do not prove global JCT benefit. Domain includes known legacy clipping, explicitly retained.'))
    print('PREFLIGHT_PASS',[(r['gpu_type'],r['best']) for r in pairs],flush=True)

def test():
    verify();t=time.time()
    assert LIVE.is_dir()
    cmd=[sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests','-p','test_*.py','-v']
    with (OUT/'tests_relocated.log').open('x',encoding='utf-8') as f:r=subprocess.run(cmd,cwd=LIVE,stdout=f,stderr=subprocess.STDOUT)
    save(OUT/'tests_relocated.json',dict(command=cmd,cwd=str(LIVE),exit_code=r.returncode,seconds=time.time()-t,log_sha256=sha(OUT/'tests_relocated.log')))
    assert r.returncode==0
    print('TESTS_PASS',flush=True)

def regression():
    verify();results={};commands=[];t=time.time()
    for name,sim in [('official',ROOT/'official-artifacts/sia-simulator'),('legacy_cc',SOURCE)]:
      cmd=[sys.executable,'-X','utf8','-B',str(SOURCE/'check_stage1_legacy_regression.py'),'--probe',str(sim)]
      commands.append(cmd)
      with (OUT/(name+'_sia_bounded.log')).open('x',encoding='utf-8') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
      assert r.returncode==0
      lines=(OUT/(name+'_sia_bounded.log')).read_text(encoding='utf-8').splitlines()
      results[name]=json.loads(next(l[len('PROBE_JSON='):] for l in lines if l.startswith('PROBE_JSON=')))
    save(OUT/'official_regression.json',dict(exact_match=results['official']==results['legacy_cc'],states=results,commands=commands,seconds=time.time()-t))
    assert results['official']==results['legacy_cc']
    print('OFFICIAL_REGRESSION_PASS',flush=True)

def freeze():
    p=verify();assert json.loads((OUT/'coverage_preflight.json').read_text())['valid']
    assert json.loads((OUT/'tests_relocated.json').read_text())['exit_code']==0
    assert json.loads((OUT/'official_regression.json').read_text())['exact_match']
    resource=gate()
    with (OUT/'dependencies.lock.txt').open('x',encoding='utf-8') as f:subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
    save(OUT/'preregistration.json',dict(status='FROZEN',created_unix=time.time(),runs=RUNS,methods=METHODS,
      jobs=12,interval=60,max_rounds=6000,seed_profiles=[1,8],batch_adaptation=True,
      source_fingerprint=p['parent_source_fingerprint'],source_hashes=p['source_hashes'],input_hashes=p['input_hashes'],
      algorithm_changes=False,alias={'CC-Joint':'JOINT_CC','CC-Joint-P':'JOINT_CC_P'},
      selection=p['selection'],arrival='controlled all-zero, original arrivals retained',
      nodes={t:dict(gpus=2,cpu=12,ram=250) for t in ['aws','dgx-ext']},
      responses={'neutral':0.,'medium':.5},normalization='fixed_reference_3CPU_62.5GB_R0_E-legacy',
      resource_gate=resource,runner_hashes={f.name:sha(f) for f in [Path(__file__),OUT/'baseline_runner.py',OUT/'audit.py']},
      packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','cvxpy','cylp','psutil']},
      endpoint='Complete or6000round censor; no extension/imputation/outcome-based rerun',
      primary='Full paired mean JCT only if all12 complete; otherwise completion+RMST',
      comparisons=['JOINT_CC vs SIA_P','JOINT_CC vs SIA_ECT_AUX','JOINT_CC vs JOINT_CC_P'],
      warning=WARNING))
    print('FROZEN',resource,flush=True)

def validate():
    load();gate();rows=[]
    for response,method in RUNS:
      c=construct(method,response)
      assert len(c.jobs)==12 and len(c.policy.config_to_idx_map)==4
      assert all(j.enable_bsz_tuning and j.submission_time==0 for j in c.jobs)
      assert c.policy.oracle.response_exponent==(0 if response=='neutral' else .5)
      rows.append(dict(response=response,method=method,job_names=[j.name for j in c.jobs],
        capacities=c.policy.node_capacities,gpu_domain=list(c.policy.config_to_idx_map)))
    save(OUT/'constructor_acceptance.json',dict(valid=True,training_run=False,cells=rows))
    print('CONSTRUCTORS_PASS',flush=True)

def cell(method,response):
    p=load();resources=gate();directory=OUT/'cells'/response/method;directory.mkdir(parents=True,exist_ok=False)
    save(directory/'start.json',dict(resources=resources,response=response,method=method,command=sys.argv,
        preregistration_sha256=sha(OUT/'preregistration.json'),start_unix=time.time()))
    import baseline_runner
    t=time.perf_counter();c=construct(method,response);m=baseline_runner.metrics_class()(c)
    try:
      with gzip.open(directory/'rounds.jsonl.gz','xt',encoding='utf-8') as f:
       for i in range(p['max_rounds']):
        if c.all_complete():break
        tick=time.perf_counter();c.step(60);r=c.joint_round_logs[-1];r['round_wall_seconds']=time.perf_counter()-tick
        r['job_end_state']={j.name:dict(restarts=j.num_restarts or 0,migrations=j.num_migrations or 0,
            wasted_gpu_seconds=j.wasted_gpu_seconds,completion=j.completion_time) for j in c.jobs}
        f.write(json.dumps(r,default=default,separators=(',',':'))+'\n');f.flush();m.consume(c,r)
        c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
        if (i+1)%100==0:print('PROGRESS',response,method,i+1,sum(j.completion_time is not None for j in c.jobs),flush=True)
      s=m.summary(c,dict(rmst_tau_seconds=360000))
      s.update(method=method,response=response,total_wall_seconds=time.perf_counter()-t,
        termination='ALL_COMPLETE' if c.all_complete() else 'ROUND_LIMIT')
      save(directory/'summary.json',s);print('RESULT',response,method,s['completed'],s['mean_jct'],flush=True)
    except Exception as exc:
      save(directory/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=c.current_time,
        last_decision=c.policy.last_decision,allocations=c.allocations,
        solver_diagnostics=getattr(exc,'solver_diagnostics',None)))
      raise

def batch():
    load();assert json.loads((OUT/'constructor_acceptance.json').read_text())['valid']
    with (OUT/'execution_manifest.jsonl').open('x',encoding='utf-8') as f:
      for response,method in RUNS:
        try:resources=gate()
        except RuntimeError as exc:
          f.write(json.dumps(dict(response=response,method=method,status='SAFETY_STOP',error=str(exc)))+'\n');f.flush();return
        cmd=[sys.executable,'-X','utf8','-B',str(Path(__file__)),'cell','--method',method,'--response',response]
        t=time.time()
        with (OUT/(response+'_'+method+'.log')).open('x',encoding='utf-8') as log:r=subprocess.run(cmd,cwd=LIVE,stdout=log,stderr=subprocess.STDOUT)
        entry=dict(response=response,method=method,exit_code=r.returncode,wall_seconds=time.time()-t,resources=resources,command=cmd)
        f.write(json.dumps(entry)+'\n');f.flush();print(entry,flush=True)
        if r.returncode:print('EXECUTION_FAILURE_STOP',flush=True);return
    print('EIGHT_CELL_BATCH_FINISHED',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','relocate','preflight','test','regression','freeze','validate','cell','batch'])
    ap.add_argument('--method',choices=METHODS);ap.add_argument('--response',choices=['neutral','medium']);a=ap.parse_args()
    {'prepare':prepare,'relocate':lambda:shutil.copytree(SOURCE,LIVE),'preflight':preflight,'test':test,'regression':regression,'freeze':freeze,'validate':validate,'batch':batch,
      'cell':lambda:cell(a.method,a.response)}[a.mode]()
