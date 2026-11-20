import dataclasses,json,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
P=json.loads((OUT/'preregistration.json').read_text());sys.path.insert(0,P['tree'])
from scenario import make,opportunity
c=make('G111');jobs=c.get_job_infos();specs,provenance=c.policy._job_specs(jobs,{})
s=c.policy._scheduler(c.policy._nodes(c.get_node_infos()),specs)
columns=s.build_joint_columns()
prop,flips,exchanges=opportunity(columns,P['scenario']['capacities'],P['scenario']['gpus'])
evidence=dict(valid=bool(flips and exchanges),no_training_steps=True,profile_initialization=[1,8],
 candidate_count=len(columns),proportional_points=len(prop),rank_reversals=len(flips),feasible_exchanges=len(exchanges),
 rank_examples=[[dataclasses.asdict(v) for v in pair] for pair in flips[:10]],
 exchange_examples=[[dataclasses.asdict(v) for v in quad] for quad in exchanges[:10]],
 provenance=provenance,all_specs=[dataclasses.asdict(j) for j in specs])
with (OUT/'offline_precheck.json').open('x',encoding='utf-8') as f:json.dump(evidence,f,indent=2)
print('OFFLINE',evidence['valid'],'flips',len(flips),'exchanges',len(exchanges))
if not evidence['valid']:raise SystemExit(2)
