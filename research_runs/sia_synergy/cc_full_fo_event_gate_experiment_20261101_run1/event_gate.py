import json,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent
def qualify(snapshot,rows,key,allowed):
 e=snapshot['event'];t=snapshot['time'];r=next((r for r in rows if r['time']==t),None)
 if not r or not set(r['fo'].get('geometry_event_reasons') or [])&set(allowed):return None
 if not e['structural'] or key not in e['changed'] or any(j==key for j,m in e['clipped_changed']):return None
 x,y=[snapshot['selected'][m][key]['config']['config_id'] for m in ('FO','FULL')]
 if x==y:return None
 span=[];productive=0;progress={'fo':0.,'full':0.}
 lookup={r['time']:r for r in rows}
 for r in rows:
  if r['time']<t:continue
  if not r['same_service_set'] or r['fo']['configs'].get(key)!=x or r['full']['configs'].get(key)!=y:break
  if r['time']+60 not in lookup:break
  span.append(r['time']);n=lookup[r['time']+60]
  for m in progress:progress[m]+=n[m]['progress'].get(key,0.)
  productive+=n['full']['progress'].get(key,0.)>0
 if len(span)<6 or productive<2:return None
 return dict(key=key,event_time=t,fo_config=x,full_config=y,start=t,end=span[-1]+60,intervals=len(span),productive_intervals=productive,actual_key_progress=progress,other_changed_point_clipping=e['clipped_changed'],event_objectives={k:e[k] for k in ('fo_objective','full_objective')})
def main():
 p=json.loads((O/'preregistration.json').read_text());g=p['corrected_gate'];P=Path(g['evidence_parent'])
 for n,h in p['evidence_hashes'].items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest().upper()==h
 assert json.loads((P/'gate_independent_audit.json').read_text())['valid']
 states=json.loads((P/'gate_evidence.json').read_text());rows=json.loads((P/'actual_prefix_comparison.json').read_text())['rows'];found=[]
 for owner,v in states.items():
  for snap in v['first_divergences']:
   for key in sorted(snap['event']['changed']):
    q=qualify(snap,rows,key,g['event_types'])
    if q:found.append(dict(evidence_owner=owner,**q))
 found.sort(key=lambda x:(x['event_time'],x['key'],x['evidence_owner']))
 result=dict(passed=bool(found),witnesses=found,confidence='fixed R0/E-legacy synthetic model, key comparison supported, other clipped donors disclosed; not independent replication or JCT evidence')
 with (O/'event_gate_result.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
