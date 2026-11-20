import hashlib,importlib.metadata,json,sys,zipfile
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
TREE=ROOT/'sia-joint-protected-admission/sia-simulator'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
parent=json.loads((ROOT/'research_runs/sia_synergy/b3_version_bridge_20261026/preregistration.json').read_text(encoding='utf-8'))
work=pd.read_csv(parent['datasets']['original20']).iloc[:6].copy()
assert list(work.application)==['bert','cifar10','deepspeech2']*2
work.to_csv(OUT/'workload.csv',index=False)
gpus={'aws':2,'dgx-ext':2,'rtx':1}
scenario=dict(base_config=parent['config'],registry=parent['registry'],models=parent['models'],gpus=gpus,
 capacities={t:dict(cpu_cores=n*1.5,ram_gb=n*62.5) for t,n in gpus.items()},
 response='FIXED_R0_E_LEGACY_MEDIUM_ALPHA_0.5',synthetic_capacity=True,
 selection_rule='first six records of previously frozen original20, no result-based selection',
 gates=dict(prefix_rounds=60,max_rounds=600,consecutive_queue_rounds=3,receiver_speed_gain_min=.05,donor_max_loss=.1,
    offline_rank_reversal_min=1,offline_feasible_exchange_min=1,
    online_rank_reversal_min=1,online_selected_rank_reversal_min=1,online_selected_exchange_min=1),
 gate_failure_rule='preserve prefix and stop all JCT comparison; do not tune or substitute workload',
 warning='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE')
with (OUT/'scenario.json').open('x',encoding='utf-8') as f:json.dump(scenario,f,indent=2)
sources={str(p.relative_to(TREE)):sha(p) for p in TREE.rglob('*.py') if '__pycache__' not in p.parts}
inputs={str(p):sha(p) for p in [Path(parent['config']),Path(parent['registry']),Path(parent['datasets']['original20']),OUT/'workload.csv',OUT/'scenario.json']}
inputs.update({str(p):sha(p) for p in Path(parent['models']).glob('*.py')})
inputs.update({str(p):sha(p) for p in (TREE/'traces').rglob('*.csv')})
with zipfile.ZipFile(OUT/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
 for p in sources:z.write(TREE/p,p)
plan=dict(tree=str(TREE),source_hashes=sources,input_hashes=inputs,
 tools_hashes={p.name:sha(p) for p in OUT.glob('*.py')},runs=['G111','sia_p','sia_aux','G00','G11'],
 interval=60,p=-.5,lambda_n=1.1,Q=300,D=250,H=300,seed_profiles=[1,8],
 packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','cvxpy','cylp','pandas','psutil']},python=sys.version,
 memory_gate_gib=2,disk_gate_gib=5,scenario=scenario,
 execution_rule='G111 prefix -> gates -> continue same process to fixed cap, then remaining four methods; compare JCT only after gates')
with (OUT/'preregistration.json').open('x',encoding='utf-8') as f:json.dump(plan,f,indent=2)
print('FROZEN one six-job five-GPU scenario; first G111 gate then five-method paired comparison')
