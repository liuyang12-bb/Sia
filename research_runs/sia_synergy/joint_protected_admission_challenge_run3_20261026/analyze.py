import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
data=json.loads((OUT/'comparison_audit.json').read_text());assert data['valid']
summaries={r['cell']['method']:r['summary'] for r in data['results']}
comparisons={}
for base in ['sia_p','sia_aux','G00','G11']:
 a=summaries[base];b=summaries['G111']
 jobs=[dict(job=j,baseline_jct=v['jct'],new_jct=b['jobs'][j]['jct'],
            improvement_seconds=v['jct']-b['jobs'][j]['jct']) for j,v in a['jobs'].items()]
 comparisons[base]=dict(improvement_percent=100*(a['mean_jct']-b['mean_jct'])/a['mean_jct'],
   improved=sum(v['improvement_seconds']>0 for v in jobs),degraded=sum(v['improvement_seconds']<0 for v in jobs),jobs=jobs,
   time_difference_new_minus_baseline={key:b['totals'][key]-a['totals'][key] for key in ['wait','checkpoint','training','restarts']})
 print(base,comparisons[base]['improvement_percent'],comparisons[base]['improved'],comparisons[base]['degraded'])
gate=json.loads((OUT/'cells/G111/opportunity_gate.json').read_text());assert gate['passed']
with (OUT/'paired_comparison.json').open('x',encoding='utf-8') as f:
 json.dump(dict(status='ALL_OPPORTUNITIES_TRIGGERED_ALL_COMPLETE_JCT_REGRESSION',summaries=summaries,
 comparisons_g111_vs_baseline=comparisons,gate=gate['checks']),f,indent=2)
