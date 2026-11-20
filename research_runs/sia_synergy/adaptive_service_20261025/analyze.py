"""Independent event/segment, node capacity and adaptive budget verification."""
import json,gzip,math,hashlib,sys,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent;OLD=OUT.parent/'small_mechanism_20261025';sys.path.insert(0,str(ROOT/'sia-joint-adaptive-service/sia-simulator'))
from joint_prototype.execution_audit import audit_execution
from joint_prototype.r0_liveness import R0Oracle
from run_stage2c_cell import independent_rho
def save(name,x):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2)
def close(a,b):return math.isclose(a,b,abs_tol=1e-6,rel_tol=1e-10)
def main():
 p=json.loads((OUT/'preregistration.json').read_text());allmetrics=[];allchecks=[];details={}
 for response in ['neutral','medium']:
  cell=response+'_joint_r';summary=json.loads((OUT/cell/'summary.json').read_text());maps={j:j.split('-')[0] for j in summary['jobs']};o=R0Oracle(p['registry'],p['models'],maps,response);costs=json.loads((OUT/cell/'start.json').read_text())['actual_delay_costs']
  states=collections.defaultdict(lambda:dict(wait=0.,checkpoint=0.,training=0.,gpu_hours=0.,cpu_core_hours=0.,ram_gb_hours=0.,completion=None,progress=0.));counts=collections.Counter();actions=collections.Counter();maxbudget=0.;maxage=0.;boost=0.;training=0.;rho_weight=0.;bert=collections.defaultdict(float);lastd=None;fixes=[];clipping=collections.Counter();nsel=0
  with gzip.open(OUT/cell/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
   for line in f:
    r=json.loads(line);d=r['decision_for_next_interval'];fresh=d['round']!=lastd;sel=d['result']['selected'];audit=d['result']['audit']
    for e in r['execution']:
     assert audit_execution(e)['valid'];counts['execution']+=1;v=states[e['job_id']];assert close(v['progress'],e['progress_before']);v['progress']=e['progress_after'];v['completion']=e['completion_time']
     for k,src in [('wait','waiting_seconds'),('checkpoint','checkpoint_delay_consumed'),('training','training_seconds')]:v[k]+=e[src]
     training+=e['training_seconds'];boost+=e['training_seconds'] if e['rho']>1.000001 else 0;rho_weight+=e['training_seconds']*e['rho'];a=e['resource_allocation']
     if a:
      expected=independent_rho(o,e['job_id'],a['config_id'],a['cpu_per_gpu'],a['ram_gb_per_gpu']);assert close(e['rho'],expected)
      if response=='neutral':assert e['rho']==1
      dt=e['allocated_seconds']/3600;v['gpu_hours']+=a['gpus']*dt;v['cpu_core_hours']+=a['gpus']*a['cpu_per_gpu']*dt;v['ram_gb_hours']+=a['gpus']*a['ram_gb_per_gpu']*dt
      if e['job_id']=='bert-28':bert[a['config_id']]+=e['training_seconds']
      if e['checkpoint_remaining_before_decision']>0 and e['completion_time'] is None:
       c=sel[e['job_id']];assert c['config']['config_id']==a['config_id']
       for k in ['placement','cpu_per_gpu','ram_gb_per_gpu']:assert c[k]==a[k]
       counts['true_checkpoint_lock']+=1
    if not fresh:continue
    lastd=d['round'];counts['decisions']+=1
    usage={t:[0.,0.,0.] for t in ['aws','dgx-ext','rtx']}
    for j,c in sel.items():
     t=c['config']['gpu_type'];g=c['config']['gpus'];usage[t][0]+=g;usage[t][1]+=g*c['cpu_per_gpu'];usage[t][2]+=g*c['ram_gb_per_gpu'];expected=independent_rho(o,j,c['config']['config_id'],c['cpu_per_gpu'],c['ram_gb_per_gpu']);assert close(expected,c['rho']);counts['selected_response']+=1;nsel+=1
     for prefix,point in [('numerator',c['source_coordinates']['requested']),('denominator',c['baseline_source_coordinates']['requested'])]:
      clipping[prefix+'_cpu']+=int(not 1<=point['cpu']<=24);clipping[prefix+'_ram']+=int(not 20.83<=point['ram_gb']<=250)
    for t,g in [('aws',4),('dgx-ext',8),('rtx',8)]:assert usage[t][0]<=g and usage[t][1]<=g*3+1e-7 and usage[t][2]<=g*62.5+1e-7;counts['node_capacity']+=1
    actual={}
    for event in r['rescale_events']+r['migrate_events']:
     if event[1]==r['interval_end']:actual[event[0]]=event[-1]
    for j,change in audit['service_budget'].items():
     before=change['before'];after=change['after'];charged=change['charged_this_round'];assert charged>=0 and after['delay_charged']<=250;maxbudget=max(maxbudget,after['delay_charged']);maxage=max(maxage,r['interval_end']-after['issued_time']);assert r['interval_end']-after['issued_time']<=610
     assert close(after['delay_charged'],(before['delay_charged'] if before else 0)+charged)
     assert close(charged,actual.get(j,0.)),(response,j,charged,actual)
     if before:
      assert after['issued_time']==before['issued_time'] and after['productive_base']==before['productive_base'];counts['service_clock_not_reset']+=1
     if change['change']=='GPU_GEOMETRY':
      c=sel[j];ev=audit['adaptive_candidate_filters'][j];H=ev['productive_remaining'];D=costs[j][c['config']['gpu_type']];assert (H-D)*c['effective_goodput']>H*ev['current_effective_goodput'];assert d['probe']['provenance'][j][c['config']['config_id']]['goodput_source'] in ['direct_profile','projected_profile'];assert states[j]['training']>before['last_geometry_productive'];fixes.append(dict(time=r['interval_end'],job=j,config=c['config']['config_id'],delay_charged=after['delay_charged'],rate=c['effective_goodput'],reference_rate=ev['current_effective_goodput']));counts['geometry_payback_and_profile']+=1
     if change['change']=='AUX_QUOTA':assert sel[j]['effective_goodput']+1e-9>=audit['adaptive_candidate_filters'][j]['current_effective_goodput'];assert charged==0;counts['aux_no_slowdown']+=1
    actions.update(e['event'] for e in d['liveness_events'])
  done=[]
  for j,v in states.items():
   row=summary['jobs'][j]
   for k in ['wait','checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours']:assert close(v[k],row[k]),(j,k,v[k],row[k])
   assert v['completion']==row['completion']
   if v['completion'] is not None:done.append(v['completion']-row['arrival'])
  if len(done)==3:assert close(sum(done)/3,summary['mean_jct'])
  else:assert summary['mean_jct'] is None
  allmetrics.append(summary);allchecks.append(dict(cell=cell,valid=True,checks=dict(counts),maximum_service_delay_budget=maxbudget,maximum_observed_service_age=maxage,actions=dict(actions),boosted_training_seconds=boost,weighted_rho=rho_weight/training,clipping_counts=dict(clipping),selected_count=nsel));details[cell]=dict(geometry_adaptations=fixes,bert_training_by_config=dict(bert))
 for response in ['neutral','medium']:
  for method in ['sia_p','sia_aux','joint_o']:allmetrics.append(json.loads((OLD/(response+'_'+method)/'summary.json').read_text()))
 save('comparison.json',allmetrics);save('validation.json',allchecks);save('adaptation_trace.json',details)
 print('RAW_VALIDATED',allchecks)
if __name__=='__main__':main()
