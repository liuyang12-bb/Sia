import collections,gzip,json,math
from pathlib import Path
O=Path(__file__).resolve().parent
def frames(method):
 with gzip.open(O/'cells'/method/'rounds.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:yield json.loads(line)
def main():
 output={};targets={'bert-107','bert-159','cifar10-5'}
 for method in ('OLD_FULL','NEW_FULL'):
  chains={j:[] for j in targets};states={};trajectory={j:[] for j in targets}
  for r in frames(method):
   d=r['decision_for_next_interval']
   if r['interval_end'] in (4800,5820):
    ev=d['result']['audit'].get('release_completion')
    states[str(r['interval_end'])]=dict(selected=d['result']['selected'],evaluation=ev,
      progress_before_decision={e['job_id']:e['progress_after'] for e in r['execution']},
      declared_work=d.get('declared_work'),events=d.get('liveness_events'),
      ledger=d['probe'].get('l2_after'))
   for e in r['execution']:
    j=e['job_id'];a=e['resource_allocation']
    if j not in targets or not a or not e['allocated_seconds']:continue
    cfg=a['config_id'];start=e['interval_start'];end=start+e['allocated_seconds']
    if chains[j] and chains[j][-1]['config']==cfg and chains[j][-1]['end']==start:
     z=chains[j][-1];z['end']=end
    else:
     z=dict(config=cfg,start=start,end=end,training=0.,checkpoint=0.,raw_integral=0.,effective_integral=0.,cpu_integral=0.,ram_integral=0.,rho_integral=0.)
     chains[j].append(z)
    z['checkpoint']+=e['checkpoint_delay_consumed'];z['training']+=e['training_seconds']
    for seg in e['training_segments']:
     z['raw_integral']+=seg['raw_rate_before_rho']*seg['seconds'];z['effective_integral']+=seg['effective_rate']*seg['seconds'];z['rho_integral']+=seg['rho']*seg['seconds']
     trajectory[j].append(dict(time=seg['start_time'],config=cfg,raw=seg['raw_rate_before_rho'],rho=seg['rho'],effective=seg['effective_rate'],epoch=seg['epoch_before'],seconds=seg['seconds']))
    z['cpu_integral']+=a['cpu_per_gpu']*e['allocated_seconds'];z['ram_integral']+=a['ram_gb_per_gpu']*e['allocated_seconds']
  output[method]=dict(states=states,chains=chains,actual_rate_trajectory=trajectory)
 with (O/'prediction_realization.json').open('x',encoding='utf8') as f:json.dump(output,f,indent=2)
 for m,x in output.items():
  print(m)
  for t,st in x['states'].items():
   print('STATE',t)
   for j,v in st['selected'].items():
    if j in targets:print(j,v['config']['config_id'],v['cpu_per_gpu'],v['ram_gb_per_gpu'],v['rho'])
   ev=st['evaluation']
   if ev and ev['status']=='OK':print('PREDICTED',ev['baseline']['completion_times'],ev['selected']['completion_times'])
  for j,cs in x['chains'].items():
   print('CHAIN',j)
   for z in cs:print(z['config'],z['start'],z['end'],'training',z['training'],'checkpoint',z['checkpoint'],'average_raw',z['raw_integral']/z['training'] if z['training'] else None)
if __name__=='__main__':main()
