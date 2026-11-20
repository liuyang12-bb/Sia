"""Post-run admission duration and named loss drivers, no new counterfactual."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3]
p=json.loads((OUT/'preregistration.json').read_text())
def main():
    entries={}
    for cell in p['runs']:
        if cell['method']!='joint_s':continue
        folder=OUT/cell['id'];s=json.loads((folder/'summary.json').read_text());m=json.loads((folder/'mechanism.json').read_text());rows=m['rounds']
        full=[r for r in rows if r['active']==s['total'] and r['selected']==s['total']]
        entry=dict(initial_selected=rows[0]['selected'],initial_active=rows[0]['active'],
            first_full_initial_job_set_decision_time=full[0]['time'] if full else None,
            full_initial_job_set_decision_count=len(full),
            first_completion=min(v['completion'] for v in s['jobs'].values() if v['completion'] is not None),
            pre_first_completion=[dict(time=r['time'],selected=r['selected'],required_count=len(r['required']),optional_count=len(r['optional'])) for r in rows if r['time']<min(v['completion'] for v in s['jobs'].values() if v['completion'] is not None)],
            interpretation='decision timestamps and counts, not independent trials nor training seconds')
        for j in ['bert-54','bert-99']:
            if j in s['jobs']:entry[j]=dict(metrics=s['jobs'][j],training_by_config=m['training_by_config'].get(j,{}))
        entries[cell['id']]=entry
    with (OUT/'admission_mechanism.json').open('x') as f:json.dump(entries,f,indent=2)
    print('ADMISSION_MECHANISM',[(k,v['initial_selected'],v['first_full_initial_job_set_decision_time']) for k,v in entries.items()],flush=True)
if __name__=='__main__':main()
