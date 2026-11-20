"""Reference-data availability gate only; never mutate execution profiles."""
import dataclasses,json,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;P=json.loads((OUT/'preregistration.json').read_text());sys.path.insert(0,P['tree'])
from scenario import make,opportunity
from joint_prototype.joint_profile import GpuConfig
c=make('G111');initial,provenance=c.policy._job_specs(c.get_job_infos(),{})
specs=[]
for j in initial:
 app=c.policy.oracle.job_applications[j.job_id]
 rows=[c.policy.oracle.reference(j.job_id,f'{t}:1:{g}') for t,n in P['scenario']['gpus'].items() for g in [1,2] if g<=n]
 scale=min(r['T0'] for r in rows)
 configs=tuple(GpuConfig(r['config_id'],r['config_id'].split(':')[0],int(r['config_id'].split(':')[-1]),1,
    r['T0']/scale,r['phi_c'],r['phi_m']) for r in rows)
 specs.append(dataclasses.replace(j,configs=configs))
s=c.policy._scheduler(c.policy._nodes(c.get_node_infos()),specs);columns=s.build_joint_columns()
prop,flips,exchanges=opportunity(columns,P['scenario']['capacities'],P['scenario']['gpus'])
result=dict(valid=bool(flips and exchanges),no_training_steps=True,
 purpose='Offline R0 measured reference throughput availability, NOT current Sia Goodput or live ranking',
 correction='Cold state had no gradients and used equal placeholders; all execution initialization remains identical and untouched',
 initial_provenance=provenance,reference_specs=[dataclasses.asdict(j) for j in specs],
 rank_reversals=len(flips),exchanges=len(exchanges),
 rank_examples=[[dataclasses.asdict(v) for v in p] for p in flips[:10]],
 exchange_examples=[[dataclasses.asdict(v) for v in p] for p in exchanges[:10]])
with (OUT/'offline_reference_precheck.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print('REFERENCE_GATE',result['valid'],'rank_reversals',len(flips),'exchanges',len(exchanges))
if not result['valid']:raise SystemExit(2)
