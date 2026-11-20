"""One pre-registered full trace, independent process, durable gzip evidence."""
import argparse,collections,gzip,hashlib,importlib.metadata,json,math,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from joint_prototype.full160 import make_cluster
from run_stage2c_cell import Accumulator,independent_rho,sha,save
from run_cpu_bottleneck_smoke import json_default

class FullMetrics(Accumulator):
    """Own decision audit; delegate unchanged direct execution/time integration."""
    def __init__(self,c):
        super().__init__(c);self.fallbacks=collections.Counter();self.clipping=collections.Counter();self.selected_count=0
        self.last_locked={};self.tune=collections.Counter();self.scheduling_wall=[]

    def consume(self,c,r):
        d=r['decision_for_next_interval'];res=d.get('result',{});audit=res.get('audit',{})
        fresh=d.get('round')!=getattr(self,'last_decision_round',None)
        if fresh:
            if not audit.get('valid'):raise RuntimeError('INVALID_DECISION')
            self.status[res['status']]+=1;self.solve.append(res['solve_seconds'])
            self.events.update(e['event'] for e in d.get('liveness_events',[]))
            self.tune.update(e['action'] for e in audit.get('tune_trace',[]))
            if res['status'].startswith('FALLBACK_'):self.fallbacks[res['status']]+=1
            totals={f'{t}:0':dict(gpu=0,cpu=0.,ram=0.) for t in c.ngpus_per_node}
            for name,col in res.get('selected',{}).items():
                cfg=col['config'];typ=cfg['gpu_type'];cap=c.policy.node_capacities[typ];ng=c.ngpus_per_node[typ]
                pc=cap['cpu_cores']/ng;pm=cap['ram_gb']/ng
                expected=independent_rho(c.policy.oracle,name,cfg['config_id'],col['cpu_per_gpu'],col['ram_gb_per_gpu'])
                if not math.isclose(col['rho'],expected,rel_tol=1e-12):raise RuntimeError('DECISION_RHO')
                if c.policy.mode in ('b1','u_jp') and (col['cpu_per_gpu'],col['ram_gb_per_gpu'])!=(pc,pm):raise RuntimeError('PROPORTIONAL_CONTRACT')
                # TUNE demand/fair is not the JO proportional-progress-floor objective.
                if c.policy.mode in ('u_jp','u_jo'):
                    if expected+1e-9<independent_rho(c.policy.oracle,name,cfg['config_id'],pc,pm):raise RuntimeError('PROGRESS_FLOOR')
                    self.floor_checks+=1
                row=c.policy.oracle.reference(name,cfg['config_id']);self.selected_count+=1
                if cfg['cpu_demand_scale']!=row['phi_c'] or cfg['memory_demand_scale']!=row['phi_m']:raise RuntimeError('FIXED_PHI')
                for prefix,cpu,ram in [('numerator',col['cpu_per_gpu'],col['ram_gb_per_gpu']),('denominator',3.,62.5)]:
                    self.clipping[prefix+'_cpu']+=int(not 1<=cpu/row['phi_c']<=24)
                    self.clipping[prefix+'_ram']+=int(not 20.83<=ram/row['phi_m']<=250)
                for node,g in col['placement'].items():
                    v=totals[node];v['gpu']+=g;v['cpu']+=g*col['cpu_per_gpu'];v['ram']+=g*col['ram_gb_per_gpu']
            for node,v in totals.items():
                typ=node.rsplit(':',1)[0];cap=c.policy.node_capacities[typ];ng=c.ngpus_per_node[typ]
                if v['gpu']>ng or v['cpu']>cap['cpu_cores']+1e-7 or v['ram']>cap['ram_gb']+1e-7:raise RuntimeError('INDEPENDENT_NODE_CAPACITY')
                self.idle_saturated+=int(v['cpu']>=cap['cpu_cores']-1e-7 and v['gpu']<ng)
                for k in v:self.node_peaks[node][k]=max(self.node_peaks[node][k],v[k])
            if d.get('strict_full_intent'):
                intended=c.policy.canonical(d['original_sia_allocations'],c.allocations)
                if c.policy.canonical(c.allocations,c.allocations)!=intended:raise RuntimeError('OFFICIAL_INTENT_NOT_PRESERVED')
            # Prevent old Accumulator's L2-only progress-floor decision audit;
            # ALL decisions were independently checked above. Execution audit
            # below remains unchanged and uses directly recorded raw rates.
            self.last_decision_round=d.get('round')
        for e in r['execution']:
            if e['resource_allocation'] and e['checkpoint_delay_consumed']>0:
                if e['checkpoint_remaining_before_decision']>0 and e['completion_time'] is None:
                    old=e['resource_allocation'];new=c.policy.current_resources.get(e['job_id'])
                    for key in ('config_id','placement','cpu_per_gpu','ram_gb_per_gpu'):
                        if new is None or old[key]!=new[key]:raise RuntimeError('CHECKPOINT_GEOMETRY_CHANGED')
        super().consume(c,r)

    def summary(self,c,p):
        result=super().summary(c,p)
        result.update(fallback_count=sum(self.fallbacks.values()),fallback_status=dict(self.fallbacks),
            clipping_counts=dict(self.clipping),clipping_denominator_selected_columns=self.selected_count,
            tune_actions=dict(self.tune))
        return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True,type=Path);ap.add_argument('--cell',required=True);ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args();p=json.loads(a.manifest.read_text());cell=next(v for v in p['runs'] if v['id']==a.cell)
    for path,h in p['input_hashes'].items():
        if sha(path)!=h:raise RuntimeError(f'INPUT_CHANGED {path}')
    work=Path(__file__).resolve().parent
    if subprocess.check_output(['git','-C',str(work),'rev-parse','HEAD']).decode().strip()!=p['commit']:raise RuntimeError('COMMIT_CHANGED')
    for path,h in p['source_hashes'].items():
        if sha(work/path)!=h:raise RuntimeError(f'SOURCE_CHANGED {path}')
    for pkg,ver in p['packages'].items():
        if importlib.metadata.version(pkg)!=ver:raise RuntimeError('DEPENDENCY_CHANGED')
    mem=psutil.virtual_memory().available/1024**3
    if mem<2:raise RuntimeError(f'MEMORY_GATE {mem}')
    a.output.mkdir(parents=True,exist_ok=False)
    save(a.output/'start.json',dict(command=sys.argv,cell=cell,commit=p['commit'],manifest_hash=sha(a.manifest),memory_gib=mem))
    config=json.loads(Path(p['config']).read_text());trace=p['traces'][str(cell['trace'])]
    c=make_cluster(config,pd.read_csv(trace['file']),p['registry'],p['models'],cell['method'],cell['response'],cell['capacity'],True)
    m=FullMetrics(c)
    try:
        with gzip.open(a.output/'rounds.jsonl.gz','xt',encoding='utf-8') as f:
            for i in range(6000):
                if c.all_complete():break
                tick=time.perf_counter();c.step(60);record=c.joint_round_logs[-1];record['round_wall_seconds']=time.perf_counter()-tick
                record['job_end_state']={j.name:dict(restarts=j.num_restarts or 0,migrations=j.num_migrations or 0,
                    wasted_gpu_seconds=j.wasted_gpu_seconds,completion=j.completion_time) for j in c.jobs}
                f.write(json.dumps(record,default=json_default,separators=(',',':'))+'\n');f.flush()
                m.consume(c,record)
                c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
                if (i+1)%100==0:print('PROGRESS '+json.dumps(dict(round=i+1,time=c.current_time,completed=sum(j.completion_time is not None for j in c.jobs))),flush=True)
        result=m.summary(c,{'rmst_tau_seconds':trace['rmst_tau']});save(a.output/'summary.json',result)
        save(a.output/'output_hashes.json',{f.name:sha(f) for f in a.output.iterdir() if f.is_file()})
        print('RESULT '+json.dumps({k:v for k,v in result.items() if k not in ('jobs','node_resource_hours','node_peaks')}),flush=True)
    except Exception as exc:
        live=getattr(c.policy,'liveness',None)
        save(a.output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),rounds=m.rounds,time=c.current_time,
            last_failure=c.policy.last_failure,last_decision=c.policy.last_decision,allocations=c.allocations,
            ready=live.ready if live else None,services=list(live.services) if live else None,
            jobs={j.name:dict(progress=float(j.progress),completion=j.completion_time,delay=j.rescale_time,placement=j.placement) for j in c.jobs}))
        raise
if __name__=='__main__':main()
