import hashlib,importlib.metadata,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
TREE=ROOT/'sia-joint-exchange-netprogress/sia-simulator'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
parent=json.loads((ROOT/'research_runs/sia_synergy/b3_version_bridge_20261026/preregistration.json').read_text(encoding='utf-8'))
import pandas as pd
work=pd.read_csv(parent['datasets']['original20']).iloc[:3].copy()
assert list(work.application)==['bert','cifar10','deepspeech2']
work.to_csv(OUT/'real_workload3.csv',index=False)
cases=[
 dict(id='cpu_complement',nodes=[['aws',2,6,250]],jobs=[['receiver',[['aws',1,1,'cpu_receiver']]],['donor',[['aws',1,1,'cpu_donor']]]]),
 dict(id='memory_complement',nodes=[['aws',2,6,250]],jobs=[['receiver',[['aws',1,1,'ram_receiver']]],['donor',[['aws',1,1,'ram_donor']]]]),
 dict(id='gpu_rank_reversal',nodes=[['aws',1,2,250],['dgx-ext',1,4,250]],jobs=[['receiver',[['aws',1,2,'cpu_fast'],['dgx-ext',1,1,'neutral']]],['donor',[['aws',1,1.6,'neutral'],['dgx-ext',1,1,'neutral']]]]),
 dict(id='elastic_cpu_limit',nodes=[['aws',3,6,750]],jobs=[['receiver',[['aws',1,1,'neutral'],['aws',2,1.8,'cpu_elastic']]],['donor',[['aws',1,1,'neutral']]]]),
 dict(id='expensive_switch',nodes=[['aws',1,3,125],['rtx',1,3,125]],jobs=[['receiver',[['aws',1,1,'neutral'],['rtx',1,1.1,'neutral']]]],current={'receiver':'aws:1:1'},delay=120),
 dict(id='neutral_control',nodes=[['aws',2,6,250]],jobs=[['receiver',[['aws',1,1,'neutral']]],['donor',[['aws',1,1,'neutral']]]]),
 dict(id='ample_cpu_control',nodes=[['aws',2,12,250]],jobs=[['receiver',[['aws',1,1,'cpu_receiver']]],['donor',[['aws',1,1,'cpu_donor']]]]),
]
curves=dict(cpu_receiver=[.4,.7,1,1.5,1.5],cpu_donor=[.7,.95,1,1,1],
 cpu_fast=[1/6,1/3,.5,2/3,1],cpu_elastic=[.25,.5,.75,1,1],
 ram_receiver=[.3,.5,1,1.5,1.5],ram_donor=[.7,.95,1,1,1],neutral=[1,1,1,1,1])
fixture=dict(evidence='DECLARED_ANALYTIC_SYNTHETIC_RESPONSE_NOT_SATURN_OR_SYNERGY_MEASUREMENTS',
 cpu_grid=[1,2,3,4,6],ram_grid=[20.83,62.5,125,187.5,250],curves=curves,cases=cases,
 methods=['fixed_prop','fixed_aux','joint_prop','G00','G10','G01','G11'],
 objective='shared p=-0.5 lambda_n=1.1; optional fixed H=300 cost replaces proxy; exact all-job equality',
 delay_rule='initial=50 seconds; current geometry=0; geometry change=case delay or 50',
 legacy_restart_factor=1.,interpretation='static states only; no inferred JCT')
with (OUT/'mechanism_cases.json').open('x',encoding='utf-8') as f:json.dump(fixture,f,indent=2)
source={str(p.relative_to(TREE)):sha(p) for p in TREE.rglob('*.py') if '__pycache__' not in p.parts}
with zipfile.ZipFile(OUT/'execution_source.zip','x',zipfile.ZIP_DEFLATED) as z:
 for p in source:z.write(TREE/p,p)
inputs={k:sha(Path(k)) for k in [parent['config'],parent['registry'],parent['datasets']['original20'],str(OUT/'real_workload3.csv'),str(OUT/'mechanism_cases.json')]}
inputs.update({str(p):sha(p) for p in Path(parent['models']).glob('*.py')})
plan=dict(tree=str(TREE),config=parent['config'],registry=parent['registry'],models=parent['models'],
 workload=str(OUT/'real_workload3.csv'),source_hashes=source,input_hashes=inputs,
 runner_hashes={p.name:sha(p) for p in OUT.glob('*.py')},
 packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','cvxpy','cylp','psutil']},
 python=sys.version,interval=60,max_rounds=300,horizon=300,capacity='half',seed_profiles=[1,8],
 runs=[dict(id=f'medium_{m}',method=m,response='medium') for m in ['sia_p','sia_aux','G00','G10','G01','G11']]+
 [dict(id=f'neutral_{m}',method=m,response='neutral') for m in ['G00','G11']],
 memory_gate_gib=2,disk_gate_gib=5,algorithm_admission='UNCHANGED_G_FIFO_Q300_D250',
 warning='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE')
with (OUT/'preregistration.json').open('x',encoding='utf-8') as f:json.dump(plan,f,indent=2)
print('FROZEN 7 analytic states x7 methods; 8 real Sia three-job cells; 300 rounds fixed')
