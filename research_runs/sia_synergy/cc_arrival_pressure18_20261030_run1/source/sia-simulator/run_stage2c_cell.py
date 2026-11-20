"""Exactly one frozen w10_late cell; bounded, streaming, no overwrite/resume."""
import argparse
import collections
import gzip
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from joint_prototype.r0_liveness import make_cluster
from joint_prototype.joint_profile import CPU_COORDS, RAM_COORDS
from run_cpu_bottleneck_smoke import json_default


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(p,data):
    with Path(p).open('x',encoding='utf-8') as f: json.dump(data,f,indent=2,default=json_default)


def independent_rho(oracle,job,config,cpu,ram):
    row=oracle.reference(job,config); matrix=oracle.template(row['template']).matrix
    def interpolate(c,m):
        # Independent two 1-D interpolations, including legacy endpoint clipping.
        values=[np.interp(m,RAM_COORDS,r) for r in matrix]
        return float(np.interp(c,CPU_COORDS,values))
    raw=interpolate(cpu/row['phi_c'],ram/row['phi_m'])
    baseline=interpolate(3/row['phi_c'],62.5/row['phi_m'])
    return (raw/baseline)**oracle.response_exponent


class Accumulator:
    def __init__(self,cluster):
        self.rounds=0; self.wall_start=time.perf_counter(); self.solve=[]
        self.status=collections.Counter(); self.events=collections.Counter(); self.rhos=[]
        self.rows={j.name:dict(wait=0.,checkpoint=0.,training=0.,gpu_hours=0.,cpu_core_hours=0.,ram_gb_hours=0.,gpu_changes=0) for j in cluster.jobs}
        self.previous={}; self.node_hours=collections.defaultdict(lambda:dict(gpu=0.,cpu=0.,ram=0.))
        self.node_peaks=collections.defaultdict(lambda:dict(gpu=0.,cpu=0.,ram=0.))
        self.idle_saturated=0; self.audit_count=0; self.floor_checks=0

    def consume(self,c,r):
        self.rounds+=1
        d=r['decision_for_next_interval']; result=d.get('result',{}); audit=result.get('audit',{})
        # No active jobs: Cluster legitimately skips optimize; last_decision may
        # still describe previous boundary. Only count new decision round IDs.
        fresh=d.get('round')!=getattr(self,'last_decision_round',None)
        if fresh:
            self.last_decision_round=d.get('round')
            if not audit.get('valid'): raise RuntimeError('ROUND_AUDIT_INVALID')
            self.status[result['status']]+=1; self.solve.append(result['solve_seconds'])
            self.events.update(e['event'] for e in d.get('liveness_events',[]))
            for name,col in result.get('selected',{}).items():
                q=col; cfg=col['config']; typ=cfg['gpu_type']
                cap=c.policy.node_capacities[typ]; count=c.ngpus_per_node[typ]
                prop_cpu=max(cap['cpu_cores']/count,c.policy.minimum_cpu_per_gpu)
                prop_ram=max(cap['ram_gb']/count,c.policy.minimum_ram_gb_per_gpu)
                expected=independent_rho(c.policy.oracle,name,cfg['config_id'],q['cpu_per_gpu'],q['ram_gb_per_gpu'])
                floor=independent_rho(c.policy.oracle,name,cfg['config_id'],prop_cpu,prop_ram)
                if not math.isclose(q['rho'],expected,rel_tol=1e-12): raise RuntimeError('PREDICTED_RHO_MISMATCH')
                if q['rho']+1e-9<floor: raise RuntimeError('PROGRESS_FLOOR')
                self.floor_checks+=1
                if c.policy.mode.endswith('p') and (not math.isclose(q['cpu_per_gpu'],prop_cpu) or not math.isclose(q['ram_gb_per_gpu'],prop_ram)):
                    raise RuntimeError('PROPORTIONAL_CONTRACT')
            for node,v in audit.get('node_totals',{}).items():
                typ=node.rsplit(':',1)[0]; cap=c.policy.node_capacities[typ]; ng=c.ngpus_per_node[typ]
                if v['gpus']>ng+1e-7 or v['cpu']>cap['cpu_cores']+1e-7 or v['ram_gb']>cap['ram_gb']+1e-7: raise RuntimeError('NODE_CAPACITY')
                self.idle_saturated+=int(v['cpu']>=cap['cpu_cores']-1e-7 and v['gpus']<ng)
                for key,raw in [('gpu','gpus'),('cpu','cpu'),('ram','ram_gb')]: self.node_peaks[node][key]=max(self.node_peaks[node][key],v[raw])
        for e in r['execution']:
            name=e['job_id']; out=self.rows[name]
            if not e['independent_execution_audit']['valid']: raise RuntimeError('EXECUTION_AUDIT')
            self.audit_count+=1
            out['wait']+=e['waiting_seconds']; out['checkpoint']+=e['checkpoint_delay_consumed']; out['training']+=e['training_seconds']
            a=e['resource_allocation']
            if not a: continue
            expected=independent_rho(c.policy.oracle,name,a['config_id'],a['cpu_per_gpu'],a['ram_gb_per_gpu'])
            if not math.isclose(e['rho'],expected,rel_tol=1e-12): raise RuntimeError('EXECUTED_RHO_MISMATCH')
            self.rhos.append(e['rho'])
            for s in e['training_segments']:
                if not math.isclose(s['rho'],expected,rel_tol=1e-12): raise RuntimeError('SEGMENT_RHO_MISMATCH')
                if not math.isclose(s['effective_rate'],s['raw_rate_before_rho']*expected,rel_tol=1e-12): raise RuntimeError('ONCE_ONLY')
            if a['config_id']!=self.previous.get(name,a['config_id']): out['gpu_changes']+=1
            self.previous[name]=a['config_id']
            dt=e['allocated_seconds']/3600
            out['gpu_hours']+=a['gpus']*dt
            for u in a['node_usage']:
                out['cpu_core_hours']+=u['cpu']*dt; out['ram_gb_hours']+=u['ram_gb']*dt
                nh=self.node_hours[u['node_id']]
                for key,raw in [('gpu','gpus'),('cpu','cpu'),('ram','ram_gb')]: nh[key]+=u[raw]*dt

    def summary(self,c,prereg):
        jobs={}; tau=prereg['rmst_tau_seconds']; complete=c.all_complete()
        for j in c.jobs:
            row=dict(self.rows[j.name]); done=j.completion_time is not None
            row.update(arrival=j.submission_time,completed=done,completion=j.completion_time,
                jct=j.completion_time-j.submission_time if done else None,
                censor_time=None if done else c.current_time-j.submission_time,
                progress=float(j.progress),epoch=int(j.epoch),restarts=j.num_restarts or 0,
                migrations=j.num_migrations or 0,wasted_gpu_seconds=j.wasted_gpu_seconds,
                first_allocation=j.first_allocation_time,first_training=j.first_productive_training_time)
            observed=(j.completion_time if done else c.current_time)-j.submission_time
            if not math.isclose(row['wait']+row['checkpoint']+row['training'],observed,abs_tol=1e-6): raise RuntimeError('TOTAL_EXCLUSIVE_TIME')
            row['restricted_jct']=min(row['jct'] if done else row['censor_time'],tau)
            jobs[j.name]=row
        vals=[j['jct'] for j in jobs.values() if j['completed']]
        return dict(status='COMPLETE' if complete else 'CENSORED_ROUND_LIMIT',rounds=self.rounds,
            completed=len(vals),total=len(jobs),completion_rate=len(vals)/len(jobs),
            mean_jct=float(np.mean(vals)) if complete else None,median_jct=float(np.median(vals)) if complete else None,
            p95_jct=float(np.percentile(vals,95)) if complete else None,p99_jct=float(np.percentile(vals,99)) if complete else None,
            completed_only_mean=float(np.mean(vals)) if vals else None,
            makespan=max(j['completion'] for j in jobs.values())-min(j['arrival'] for j in jobs.values()) if complete else None,
            restricted_mean_jct=float(np.mean([j['restricted_jct'] for j in jobs.values()])),rmst_tau=tau,
            simulated_seconds=c.current_time,jobs=jobs,
            totals={key:sum(j[key] for j in jobs.values()) for key in ('wait','checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours','restarts','wasted_gpu_seconds')},
            node_resource_hours=dict(self.node_hours),node_peaks=dict(self.node_peaks),
            cpu_saturated_gpu_idle_node_round_events=self.idle_saturated,
            solver_status=dict(self.status),liveness_events=dict(self.events),fallback_count=0,
            solve_total_seconds=sum(self.solve),solve_p99_seconds=float(np.percentile(self.solve,99)) if self.solve else 0,
            rho_range=[min(self.rhos),max(self.rhos)] if self.rhos else None,
            execution_audits=self.audit_count,progress_floor_checks=self.floor_checks,violations=0,
            wall_seconds=time.perf_counter()-self.wall_start)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); ap.add_argument('--cell',required=True); ap.add_argument('--response',required=True); ap.add_argument('--output',required=True)
    args=ap.parse_args(); manifest=Path(args.manifest).resolve(); p=json.loads(manifest.read_text(encoding='utf-8'))
    if [args.response,args.cell] not in p['runs']: raise ValueError('UNREGISTERED_CELL')
    for path,digest in p['input_hashes'].items():
        if sha(path)!=digest: raise RuntimeError(f'INPUT_HASH {path}')
    work=Path(__file__).resolve().parent
    if subprocess.check_output(['git','-C',str(work),'rev-parse','HEAD']).decode().strip()!=p['commit']: raise RuntimeError('COMMIT_CHANGED')
    for path,digest in p['source_hashes'].items():
        if sha(work/path)!=digest: raise RuntimeError(f'SOURCE_CHANGED {path}')
    for package,version in p['environment']['packages'].items():
        if importlib.metadata.version(package)!=version: raise RuntimeError(f'DEPENDENCY_CHANGED {package}')
    mem=psutil.virtual_memory().available/1024**3
    if mem<2: raise RuntimeError(f'MEMORY_GATE {mem:.3f}')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    save(out/'start.json',dict(command=sys.argv,commit=p['commit'],manifest_sha256=sha(manifest),memory_gib=mem,response=args.response,cell=args.cell))
    config=json.loads(Path(p['config']).read_text(encoding='utf-8')); workload=pd.read_csv(p['workload'])
    c=make_cluster(config,workload,p['registry'],p['models'],args.cell,args.response,seed=True)
    a=Accumulator(c)
    try:
        with gzip.open(out/'rounds.jsonl.gz','xt',encoding='utf-8') as f:
            for i in range(p['max_rounds']):
                if c.all_complete(): break
                c.step(p['interval']); record=c.joint_round_logs[-1]
                # Persist BEFORE independent audit, preserving any bad round.
                f.write(json.dumps(record,default=json_default,separators=(',',':'))+'\n'); f.flush()
                a.consume(c,record)
                c.joint_round_logs.clear(); c.logs.clear(); c.rescale_events.clear(); c.migrate_events.clear()
                if (i+1)%100==0: print(json.dumps(dict(round=i+1,time=c.current_time,completed=sum(j.completion_time is not None for j in c.jobs))),flush=True)
        result=a.summary(c,p); save(out/'summary.json',result)
        save(out/'output_hashes.json',{f.name:sha(f) for f in out.iterdir() if f.is_file()})
        print(json.dumps({k:v for k,v in result.items() if k not in ('jobs','node_resource_hours','node_peaks')}),flush=True)
    except Exception as exc:
        save(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),rounds=a.rounds,time=c.current_time,
            policy_last_decision=c.policy.last_decision,allocations=c.allocations,resources=c.policy.current_resources,
            ready=c.policy.liveness.ready,services={k:{x:y for x,y in v.items() if x!='column'} for k,v in c.policy.liveness.services.items()},
            jobs={j.name:dict(progress=float(j.progress),completion=j.completion_time,placement=j.placement,delay=j.rescale_time) for j in c.jobs}))
        raise

if __name__=='__main__': main()
