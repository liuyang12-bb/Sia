import hashlib,importlib.metadata,json,os,platform,shutil,sys,time,zipfile
from pathlib import Path
ROOT=Path(r'D:\科研启航\Sia');TREE=ROOT/'cc-joint-core80-8h-v3';OUT=ROOT/'research_runs/sia_synergy/cc_core80_8h_20261030_run3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf8')
budget=json.loads((OUT/'budget.json').read_text());print(budget)
env=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages={p:importlib.metadata.version(p) for p in ('numpy','scipy','pandas','cvxpy','cylp','psutil')},solver=dict(clean_time_limit=30.,mip_rel_gap=0.,cbc='existing Sia settings unchanged',highs_version='Bundled with scipy1.10.1; independent version not exposed by queried Python interface'),PYTHONHASHSEED=os.environ.get('PYTHONHASHSEED'))
save(OUT/'environment.json',env)
files={str(p.relative_to(TREE)).replace('\\','/'):sha(p) for p in TREE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
base=json.loads((OUT/'baseline_source_hashes.json').read_text())
if 'files' in base:base=base['files']
changed={k:(v,files.get(k)) for k,v in base.items() if files.get(k)!=v}
save(OUT/'source_delta.json',dict(changed=changed,added={k:v for k,v in files.items() if k not in base}))
if changed:raise RuntimeError('BASELINE_SOURCE_CHANGED '+str(changed))
cells=[]
for m in ('SIA_P','FULL','SIA_AUX','PROP','CPU','MEM','FO'):cells.append(dict(id='dual_medium_'+m,method=m,response='medium',scenario='dual',optional=False))
for m in ('A00','A01','A10','A11'):cells.append(dict(id='dual_neutral_'+m,method=m,response='neutral',scenario='dual',optional=False))
for scenario,methods in [('cpu_only',('PROP','CPU')),('ram_only',('PROP','MEM'))]:
 for m in methods:cells.append(dict(id=scenario+'_medium_'+m,method=m,response='medium',scenario=scenario,optional=True))
inputs={str(TREE/'inputs/core80_workload.csv'):sha(TREE/'inputs/core80_workload.csv'),str(TREE/'inputs/core80_config.json'):sha(TREE/'inputs/core80_config.json'),str(TREE/'inputs/registry.json'):sha(TREE/'inputs/registry.json')}
inputs.update({str(p):sha(p) for p in (TREE/'inputs/models').glob('*') if p.is_file()})
p=dict(version='CORE80_8H_V3_EQUIVALENT_ZERO_OPTIONAL_CERTIFICATE',output=str(OUT),source_root=str(TREE),source_hashes=files,input_hashes=inputs,cells=cells,
 hard_deadline_unix=budget['deadline_unix'],formal_stop_unix=budget['formal_stop_unix'],cell_wall_limit_seconds=1800,
 interval=60,max_rounds=6000,seed_profiles=[1,8],capacity='synthetic3CPU62.5GB_per_GPU;AWS4/DGX8/RTX8',response='R0+E-legacy;alpha0/.5 unchanged',
 cpu_grid='Historical source slope knots + one-peer complements unchanged; original8CPU point retained',
 SIA_P_neutral_reuse='Proportional3/62.5 equals fixed reference; all33 configs rho=1 plus real bounded exact regression. Original GPU-only decisions independent of alpha.',
 execution='Independent Python processes, serial,2GiB/5GiB before each, hard1800sec percell scheduling-computation budget (not shortened training work/round horizon).',
 timeout='COMPUTE_TIMEOUT distinct from6000round censor; do not report overall JCT for partials. Keep every job.',
 failure='Preserve failed evidence and continue other fixed cells; no tuning; optional only if11core ALL_COMPLETE and pair has2*1800+60sec budget.',
 primary_comparisons=['FULL vs SIA_P','FULL vs SIA_AUX','FULL vs PROP','FULL vs CPU','FULL vs MEM','FULL vs FO','A10-A00;A01-A00;A11-A10;interaction'],
 partial='No complete paired JCT claim for any missing cell; report exact completed coverage at8hour deadline.',environment=env,
 warning='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE')
save(OUT/'preregistration.json',p)
with zipfile.ZipFile(OUT/'frozen_source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
 for f in files:z.write(TREE/f,f)
save(OUT/'freeze_hashes.json',dict(source_snapshot_sha256=sha(OUT/'frozen_source_snapshot.zip'),preregistration_sha256=sha(OUT/'preregistration.json'),source_file_count=len(files)))
with (OUT/'CURRENT_TASK.md').open('a',encoding='utf8') as f:f.write('\n\nFrozen execution: run '+str(TREE/'supervise80.py')+' '+str(OUT/'preregistration.json')+'. Source hash authority: preregistration.source_hashes; full snapshot zip. No algorithm file from291-file baseline changed.\n')
print('FROZEN',len(files),len(cells),sha(OUT/'preregistration.json'))
