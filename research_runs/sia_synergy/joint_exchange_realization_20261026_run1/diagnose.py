"""Six structural states, fixed selected jobs/GPU/placement, offline allocation."""
import collections,gzip,hashlib,itertools,json,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
PARENT=OUT.parent/'joint_opportunity_experiments_20261026_run1'
sys.path.insert(0,str(PARENT/'source'))
from joint_prototype.joint_profile import JobSpec,GpuConfig
from joint_prototype.r0_liveness import R0Oracle
from joint_prototype.exchange_netprogress import fair_utility
def save(p,v):
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def main():
 save(OUT/'selection_rules.json',dict(rules=['First decision with a shared node and different apps','First decision with a shared node and same app','First decision with a single CIFAR GPU job on a node'],traces=['D3_queued_training_proposal_medium_sia_p','D3_queued_training_proposal_cpu_symmetric_sia_p'],limit=6,scoring='same selected set/config/placement; sum normalized effective rates and existing rounded fairness; fixed RAM and full existing RAM grid; checkpoints exact',no_training=True))
 reg={v['key']:v for v in json.loads((PARENT/'inputs/registry.json').read_text())};states=[]
 for trace in ['D3_queued_training_proposal_medium_sia_p','D3_queued_training_proposal_cpu_symmetric_sia_p']:
  summary=json.loads((PARENT/'cells'/trace/'summary.json').read_text());nodes=summary['cell']['nodes'];found=set();last=None
  with gzip.open(PARENT/'cells'/trace/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
   for line in f:
    r=json.loads(line);d=r['decision_for_next_interval']
    if last==d['round']:continue
    last=d['round'];sel=d['result']['selected'];by=collections.defaultdict(list)
    for j,c in sel.items():
     for n in c['placement']:by[n].append(j)
    for kind in range(3):
     matches=[]
     for n,js in by.items():
      apps={j.rsplit('-',1)[0] for j in js}
      if (kind==0 and len(js)>=2 and len(apps)>1) or (kind==1 and len(js)>=2 and len(apps)==1) or (kind==2 and len(js)==1 and apps=={'cifar10'} and sel[js[0]]['config']['gpus']==1):matches.append(n)
     if kind in found or not matches:continue
     found.add(kind);node=sorted(matches)[0];jobs=by[node];cap=nodes[node.rsplit(':',1)[0]];specs={v['job_id']:v for v in d['probe']['specs']}
     oracle=R0Oracle(PARENT/'inputs/registry.json',PARENT/'inputs/models',{j:j.rsplit('-',1)[0] for j in sel},'medium')
     base=[];options={}
     for j in jobs:
      c=sel[j];g=GpuConfig(**c['config']);sp=JobSpec(j,reg[j.rsplit('-',1)[0]+'/'+g.config_id]['template'],(g,));v=specs[j]
      bp=oracle.query(sp,g,c['cpu_per_gpu'],c['ram_gb_per_gpu'],(cap['cpu']/cap['gpus'],cap['ram']/cap['gpus']))
      base.append(dict(job=j,cpu=bp.cpu_per_gpu,ram=bp.ram_gb_per_gpu,rho=bp.rho,rate=bp.effective_goodput,clipped=bp.source_coordinate_clipped or bp.baseline_coordinate_clipped,gpus=g.gpus))
      for grid in ['legacy','integer_cpu']:
       for ram_mode in ['fixed_ram','grid_ram']:
        points=[]
        cpus=[1,2,3,4,6,9,12,24] if grid=='legacy' else list(range(1,int(cap['cpu'])+1))
        rams=[c['ram_gb_per_gpu']] if ram_mode=='fixed_ram' else sorted(set([20.83,62.5,125,187.5,250,c['ram_gb_per_gpu']]))
        if v['checkpoint_locked']:cpus=[v['current_cpu_per_gpu']];rams=[v['current_ram_gb_per_gpu']]
        for cpu in cpus:
         for ram in rams:
          if cpu*g.gpus>cap['cpu'] or ram*g.gpus>cap['ram']:continue
          q=oracle.query(sp,g,cpu,ram,(cap['cpu']/cap['gpus'],cap['ram']/cap['gpus']))
          points.append(dict(job=j,cpu=cpu,ram=ram,rho=q.rho,rate=q.effective_goodput,clipped=q.source_coordinate_clipped or q.baseline_coordinate_clipped,gpus=g.gpus))
        options[j,grid,ram_mode]=points
     def score(ps):return dict(rate=sum(v['rate'] for v in ps),fair=sum(fair_utility(v['rate']) for v in ps))
     results={}
     for grid,rm in itertools.product(['legacy','integer_cpu'],['fixed_ram','grid_ram']):
      feasible=[ps for ps in itertools.product(*(options[j,grid,rm] for j in jobs)) if sum(v['cpu']*v['gpus'] for v in ps)<=cap['cpu']+1e-8 and sum(v['ram']*v['gpus'] for v in ps)<=cap['ram']+1e-8]
      for floor,clean in itertools.product([False,True],[False,True]):
       fp=[ps for ps in feasible if (not floor or all(v['rho']>=b['rho']-1e-9 for v,b in zip(ps,base))) and (not clean or all(not v['clipped'] for v in ps))]
       key=f'{grid}_{rm}_floor{floor}_clean{clean}'
       if fp:
        best=max(fp,key=lambda ps:(score(ps)['fair'],score(ps)['rate']))
        results[key]=dict(points=best,score=score(best),count=len(fp),rate_gain=score(best)['rate']/score(base)['rate']-1,fair_gain=score(best)['fair']-score(base)['fair'])
       else:results[key]=None
     states.append(dict(trace=trace,category=kind,round=d['round'],node=node,jobs=jobs,capacity=cap,baseline=base,baseline_score=score(base),results=results,full_state=d))
    if len(found)==3:break
 save(OUT/'same_state_diagnostic.json',states)
 for s in states:
  print(s['trace'],s['category'],s['round'],s['jobs'])
  for k in ['legacy_fixed_ram_floorFalse_cleanTrue','integer_cpu_fixed_ram_floorFalse_cleanTrue','integer_cpu_grid_ram_floorFalse_cleanTrue']:
   v=s['results'][k];print(k,None if v is None else (v['rate_gain'],v['fair_gain'],[(p['cpu'],p['ram']) for p in v['points']] ))
if __name__=='__main__':main()
