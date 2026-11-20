"""Single preregistered six-job mechanism revision with component controls."""
from pathlib import Path
import argparse, collections, dataclasses, gzip, hashlib, importlib.metadata
import json, math, shutil, subprocess, sys, time, traceback, zipfile

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
PREVIOUS=OUT.parent/'joint_ect_six_job_20261026_run1'
LIVE=ROOT/'sia-joint-completion-continuity/sia-simulator'
METHODS=['SIA_P','SIA_ECT_AUX','LEGACY_ECT','SCORE_ONLY','CONTINUITY_ONLY','JOINT_CC_P','JOINT_CC']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()
def default(v):
    if dataclasses.is_dataclass(v):return dataclasses.asdict(v)
    if hasattr(v,'item'):return v.item()
    raise TypeError(type(v).__name__)
def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,default=default)
def gate():
    import psutil
    v=dict(memory_gib=psutil.virtual_memory().available/2**30,disk_gib=psutil.disk_usage(str(OUT)).free/2**30)
    if v['memory_gib']<2:raise RuntimeError('MEMORY_GATE_2GiB:'+str(v))
    if v['disk_gib']<5:raise RuntimeError('DISK_GATE_5GiB:'+str(v))
    return v
def load():
    p=json.loads((OUT/'preregistration.json').read_text(encoding='utf-8'))
    for rel,h in p['source_hashes'].items():assert sha(OUT/'source'/rel)==h,rel
    for rel,h in p['input_hashes'].items():assert sha(OUT/rel)==h,rel
    for rel,h in p['runner_hashes'].items():assert sha(OUT/rel)==h,rel
    for pkg,v in p['packages'].items():assert importlib.metadata.version(pkg)==v,pkg
    return p

def freeze():
    resources=gate()
    old=json.loads((OUT/'source_changes.json').read_text(encoding='utf-8'))
    for rel,h in old['source_hashes'].items():assert sha(LIVE/rel)==h,rel
    assert json.loads((OUT/'full_suite.json').read_text())['exit_code']==0
    assert json.loads((OUT/'full_final.json').read_text())['exit_code']==0
    assert json.loads((OUT/'official_regression.json').read_text())['exact_match']
    shutil.copytree(LIVE,OUT/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    parent=json.loads((PREVIOUS/'preregistration.json').read_text(encoding='utf-8'))
    for rel,h in parent['input_hashes'].items():assert sha(OUT/rel)==h,rel
    p=dict(status='PREREGISTERED',created_unix=time.time(),methods=METHODS,max_rounds=6000,interval=60,
      seed_profiles=[1,8],batch_adaptation=True,response='medium',alpha=.5,
      nodes={t:dict(gpus=2,cpu=12.,ram=250.) for t in ['aws','dgx-ext']},
      allocation_floor=False,source_fingerprint=old['source_fingerprint'],
      source_version='Completion-priority and event-gated completion-continuity V1; historical modes retained',
      policies=dict(SIA_P='Official Sia GPU objective/admission/placement plus proportional execution',
        SIA_ECT_AUX='Same official Sia GPU intent, exact placement and set; ECT auxiliary optimization, no L2',
        LEGACY_ECT='Historical ECT objective and periodic service',
        SCORE_ONLY='Reciprocal ETA score, historical periodic service',
        CONTINUITY_ONLY='Historical ECT score, completion-continuity service',
        JOINT_CC_P='New score and continuity; proportional auxiliary quota',
        JOINT_CC='New score and continuity; jointly optimized auxiliary quota'),
      comparisons=['JOINT_CC vs SIA_P','JOINT_CC vs SIA_ECT_AUX','JOINT_CC vs JOINT_CC_P',
        'LEGACY_ECT / SCORE_ONLY / CONTINUITY_ONLY / JOINT_CC component matrix'],
      interpretation='Joint vs Sia includes goal/admission/placement; component controls are diagnostic, no tuning',
      known_data_gaps='CIFAR DGX2 denominator clipping, other numerator clipping. Retain legacy response and report all usage.',
      endpoint='All six complete or fixed 6000-round censor. No extension, JCT imputation, or outcome-based repeat.',
      primary='Mean JCT only for complete paired runs; all job JCTs, completion rate and RMST otherwise',
      warning='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE',
      resource_gate=resources,packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','cvxpy','cylp','psutil']},
      source_hashes={str(f.relative_to(OUT/'source')):sha(f) for f in (OUT/'source').rglob('*') if f.is_file()},
      input_hashes={str(f.relative_to(OUT)):sha(f) for f in (OUT/'inputs').rglob('*') if f.is_file()},
      runner_hashes={f.name:sha(f) for f in [Path(__file__),OUT/'audit_results.py']},
      algorithm=dict(priority='1/(actual_delay+remaining_work/current_predicted_rate)',
        continuity='nonpreemptive_until_actual_completion',
        geometry_events=['arrival','completion','first_valid_current_model'],
        lifetime_startup_budget_seconds=250,unknown_eta='proportional_bootstrap_FIFO'),
      historical_preregistration_sha256=sha(PREVIOUS/'preregistration.json'))
    with (OUT/'dependencies.lock.txt').open('x',encoding='utf-8') as f:subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
    save(OUT/'preregistration.json',p)
    print('FROZEN',resources,flush=True)

def construct(method):
    sys.path.insert(0,str(OUT/'source'))
    import pandas as pd
    from joint_prototype.full160 import make_cluster
    from joint_prototype.small_mechanism import ProbeSia
    from joint_prototype.physical_domain import restrict_single_node_domain
    from joint_prototype.completion_time import install,install_aux
    from joint_prototype.completion_continuity import install as install_cc
    config=json.loads((OUT/'inputs/config_symmetric.json').read_text())
    c=make_cluster(config,pd.read_csv(OUT/'inputs/workload.csv'),OUT/'inputs/registry.json',OUT/'inputs/models',
        'sia_p' if method.startswith('SIA') else 'joint_o','medium','main',True)
    c.policy.node_capacities={t:dict(cpu_cores=12.,ram_gb=250.) for t in ['aws','dgx-ext']}
    restrict_single_node_domain(c.policy)
    c.policy.aux_cpu_points=tuple(sorted(set(c.policy.aux_cpu_points)|{8.}))
    if method=='SIA_P':c.policy.__class__=ProbeSia
    elif method=='SIA_ECT_AUX':install_aux(c)
    elif method=='LEGACY_ECT':install(c)
    else:install_cc(c,proportional=method=='JOINT_CC_P',
        new_score=method!='CONTINUITY_ONLY',continuity=method!='SCORE_ONLY')
    return c

def metrics_class():
    from run_stage2c_cell import Accumulator,independent_rho
    from joint_prototype.execution_audit import audit_execution
    class Metrics(Accumulator):
      def __init__(self,c):
        super().__init__(c);self.clip=collections.Counter();self.selected=0;self.fallback=collections.Counter();self.below_floor=0;self.lock_checks=0
      def consume(self,c,r):
        d=r['decision_for_next_interval'];res=d['result']
        if d['round']!=getattr(self,'last_decision_round',None):
          self.last_decision_round=d['round'];assert res['audit']['valid']
          self.status[res['status']]+=1;self.solve.append(res['solve_seconds'])
          self.events.update(e['event'] for e in d.get('liveness_events',[]))
          if res['status'].startswith('FALLBACK_'):self.fallback[res['status']]+=1
          totals={t+':0':[0.,0.,0.] for t in c.ngpus_per_node}
          specs={s['job_id']:s for s in d['probe']['specs']}
          for jid,col in res['selected'].items():
            cfg=col['config'];rho=independent_rho(c.policy.oracle,jid,cfg['config_id'],col['cpu_per_gpu'],col['ram_gb_per_gpu'])
            assert math.isclose(rho,col['rho'],rel_tol=1e-12)
            if c.policy.mode in ['b1','u_jp']:assert (col['cpu_per_gpu'],col['ram_gb_per_gpu'])==(6.,125.)
            self.below_floor+=rho+1e-9<independent_rho(c.policy.oracle,jid,cfg['config_id'],6.,125.)
            row=c.policy.oracle.reference(jid,cfg['config_id']);self.selected+=1
            assert cfg['cpu_demand_scale']==row['phi_c'] and cfg['memory_demand_scale']==row['phi_m']
            for label,cpu,ram in [('numerator',col['cpu_per_gpu'],col['ram_gb_per_gpu']),('denominator',3.,62.5)]:
              self.clip[label+'_cpu']+=not 1<=cpu/row['phi_c']<=24
              self.clip[label+'_ram']+=not 20.83<=ram/row['phi_m']<=250
            s=specs[jid]
            if s['checkpoint_locked'] and s['current_config_id']:
              assert cfg['config_id']==s['current_config_id'] and col['placement']==dict(s['current_placement'])
              assert col['cpu_per_gpu']==s['current_cpu_per_gpu'] and col['ram_gb_per_gpu']==s['current_ram_gb_per_gpu'];self.lock_checks+=1
            for node,g in col['placement'].items():
              totals[node][0]+=g;totals[node][1]+=g*col['cpu_per_gpu'];totals[node][2]+=g*col['ram_gb_per_gpu']
          for node,v in totals.items():
            assert all(x<=y+1e-7 for x,y in zip(v,[2,12,250]))
            self.idle_saturated+=v[1]>=12-1e-8 and v[0]<2
            for k,x in zip(['gpu','cpu','ram'],v):self.node_peaks[node][k]=max(self.node_peaks[node][k],x)
          if d.get('strict_full_intent'):
            assert c.policy.canonical(c.allocations,c.allocations)==c.policy.canonical(d['original_sia_allocations'],c.allocations)
        for e in r['execution']:assert audit_execution(e)['valid']
        # This parent invocation integrates execution only. Its old no-slowdown
        # decision check is replaced above; donor slowdown is intentional here.
        super().consume(c,r)
      def summary(self,c,p):
        out=super().summary(c,p)
        out.update(clipping_counts=dict(self.clip),selected_columns=self.selected,
          below_proportional_selected=self.below_floor,checkpoint_lock_checks=self.lock_checks,
          fallback_status=dict(self.fallback),fallback_count=sum(self.fallback.values()))
        return out
    return Metrics

def validate():
    load();gate();rows=[]
    for method in METHODS:
      c=construct(method)
      assert len(c.jobs)==6 and len(c.policy.config_to_idx_map)==4
      assert set(c.ngpus_per_node)=={'aws','dgx-ext'} and all(v==2 for v in c.ngpus_per_node.values())
      assert all(v==dict(cpu_cores=12.,ram_gb=250.) for v in c.policy.node_capacities.values())
      assert all(j.enable_bsz_tuning for j in c.jobs)
      rows.append(dict(method=method,config_domain=list(c.policy.config_to_idx_map),
        jobs=[dict(name=j.name,arrival=j.submission_time,epoch=j.epoch,progress=j.progress) for j in c.jobs],
        node_capacities=c.policy.node_capacities))
    save(OUT/'constructor_acceptance.json',dict(valid=True,training_run=False,rows=rows))
    print('CONSTRUCTOR_ACCEPTANCE_PASS',flush=True)

def cell(method):
    p=load();resources=gate();directory=OUT/'cells'/method;directory.mkdir(parents=True,exist_ok=False)
    save(directory/'start.json',dict(method=method,resources=resources,command=sys.argv,
        preregistration_sha256=sha(OUT/'preregistration.json'),start_unix=time.time()))
    started=time.perf_counter();c=construct(method);m=metrics_class()(c)
    try:
      with gzip.open(directory/'rounds.jsonl.gz','xt',encoding='utf-8') as f:
        for i in range(p['max_rounds']):
          if c.all_complete():break
          tick=time.perf_counter();c.step(60);r=c.joint_round_logs[-1];r['round_wall_seconds']=time.perf_counter()-tick
          r['job_end_state']={j.name:dict(restarts=j.num_restarts or 0,migrations=j.num_migrations or 0,
            wasted_gpu_seconds=j.wasted_gpu_seconds,completion=j.completion_time) for j in c.jobs}
          f.write(json.dumps(r,default=default,separators=(',',':'))+'\n');f.flush();m.consume(c,r)
          c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
          if (i+1)%100==0:print('PROGRESS',method,i+1,sum(j.completion_time is not None for j in c.jobs),flush=True)
      s=m.summary(c,dict(rmst_tau_seconds=p['max_rounds']*60))
      s.update(method=method,response='medium',total_wall_seconds=time.perf_counter()-started,
        termination='ALL_COMPLETE' if c.all_complete() else 'ROUND_LIMIT')
      save(directory/'summary.json',s);print('RESULT',method,s['completed'],s['mean_jct'],flush=True)
    except Exception as exc:
      save(directory/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=c.current_time,
        last_decision=c.policy.last_decision,last_failure=getattr(c.policy,'last_failure',None),allocations=c.allocations,
        solver_diagnostics=getattr(exc,'solver_diagnostics',None)))
      raise

def batch():
    load()
    with (OUT/'execution_manifest.jsonl').open('x',encoding='utf-8') as f:
      for method in METHODS:
        try:resources=gate()
        except RuntimeError as exc:
          f.write(json.dumps(dict(method=method,status='SAFETY_STOP',reason=str(exc)))+'\n');f.flush();return
        tick=time.time()
        with (OUT/(method+'.log')).open('x',encoding='utf-8') as log:
          result=subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__)),'cell','--method',method],
            cwd=OUT/'source',stdout=log,stderr=subprocess.STDOUT)
        entry=dict(method=method,exit_code=result.returncode,wall_seconds=time.time()-tick,resources=resources)
        f.write(json.dumps(entry)+'\n');f.flush();print(entry,flush=True)
        if result.returncode:
          print('EXECUTION_FAILURE_STOP_PRESERVE_CELL',method,flush=True);return
    print('BATCH_FINISHED',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['freeze','validate','batch','cell']);a.add_argument('--method',choices=METHODS)
    v=a.parse_args();{'freeze':freeze,'validate':validate,'batch':batch,'cell':lambda:cell(v.method)}[v.mode]()
