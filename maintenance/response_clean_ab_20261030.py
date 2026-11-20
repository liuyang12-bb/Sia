"""Post hoc descriptive response accounting from immutable formal logs only."""
import collections,functools,gzip,itertools,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_clean_ab_20261030_run2';SRC=OUT/'frozen_source'
sys.path[:0]=[str(SRC),str(SRC/'sia-simulator')]
from joint_prototype.joint_profile import load_synergy_template
def main():
 p=json.loads((OUT/'preregistration.json').read_text());reg={r['key']:r for r in json.loads((SRC/'inputs/registry.json').read_text())}
 from applications import APPLICATIONS
 work={app:float(APPLICATIONS['aws'][app].get_progress(APPLICATIONS['aws'][app].max_epochs)) for app in ('cifar10','bert','deepspeech2')}
 data={}
 for w,n in p['cells']:
  if w=='A20':continue
  name=w+'_'+n;counts=collections.Counter();apps=collections.defaultdict(collections.Counter)
  with gzip.open(OUT/'cells'/name/'rounds.jsonl.gz','rt',encoding='utf8') as f:
   for line in f:
    rr=json.loads(line);by_node=collections.defaultdict(list)
    for e in rr['execution']:
     a=e['resource_allocation']
     if not a:continue
     app=e['job_id'].rsplit('-',1)[0];ref=reg[app+'/'+a['config_id']]
     inside=all([1.-1e-12<=a['cpu_per_gpu']/ref['phi_c']<=24.+1e-12,20.83-1e-12<=a['ram_gb_per_gpu']/ref['phi_m']<=250.+1e-12,
        1.-1e-12<=3/ref['phi_c']<=24.+1e-12,20.83-1e-12<=62.5/ref['phi_m']<=250.+1e-12])
     for z in e['training_segments']:
      weight=z['raw_rate_before_rho']*z['seconds']/work[app];delta=weight*(z['rho']-1.)
      counts['fixed_trajectory_normalized_progress_delta']+=delta
      counts['receiver_gain']+=max(0.,delta);counts['donor_loss']+=max(0.,-delta)
      counts['training_job_seconds']+=z['seconds'];apps[app]['progress_delta']+=delta
      apps[app]['accelerated_seconds']+=z['seconds']*int(z['rho']>1.+1e-10)
      apps[app]['slowed_seconds']+=z['seconds']*int(z['rho']<1.-1e-10)
      for node,g in a['placement'].items():
       if g==1:by_node[node].append(dict(app=app,start=z['start_time'],end=z['start_time']+z['seconds'],rho=z['rho'],inside=inside))
    for spans in by_node.values():
     for a,b in itertools.combinations(spans,2):
      if a['app']==b['app']:continue
      dt=max(0.,min(a['end'],b['end'])-max(a['start'],b['start']));counts['mixed_coproductive_node_seconds']+=dt
      if a['inside'] and b['inside']:counts['supported_mixed_coproductive_node_seconds']+=dt
      cf=a if a['app']=='cifar10' else b;peer=b if cf is a else a
      if cf['rho']>1.+1e-10 and peer['rho']>=1.-1e-10:
       counts['receiver_faster_donor_not_slower_node_seconds']+=dt
       if a['inside'] and b['inside']:counts['supported_receiver_faster_donor_not_slower_node_seconds']+=dt
  data[name]=dict(totals=dict(counts),applications={a:dict(v) for a,v in apps.items()})
 with (OUT/'response_realization.json').open('x',encoding='utf8') as f:json.dump(dict(results=data,
  warning='Fixed actual GPU/batch/time trajectory versus rho1; response accounting NOT feasible alternate schedule/JCT attribution. Work fractions normalized per app. Co-residence uses actual productive segment overlaps; zero-time boundary segments contribute0.'),f,indent=2)
 print('RESPONSE_ACCOUNTED',len(data))
if __name__=='__main__':main()
