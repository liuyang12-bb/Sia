import difflib,hashlib,importlib.metadata,json,sys,zipfile
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-residual-capacity-v1';OLD=R/'cc-joint-uncertainty-v4'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(n,x):
 with (O/n).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,ensure_ascii=False)
log=(O/'full_suite.log').read_text(encoding='utf8');assert '\nOK\n' in log and 'FAILED (' not in log
for n in ('original_sia_regression.json','FO_regression.json','FULL_regression.json'):assert json.loads((O/n).read_text())['exact']
assert len(json.loads((O/'same_state.json').read_text()))==4
base=json.loads((O/'baseline_hashes.json').read_text());assert all(sha(OLD/n)==h for n,h in base.items())
h={p.relative_to(T).as_posix():sha(p) for p in T.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
changed=[n for n in base if h[n]!=base[n]];new=sorted(set(h)-set(base));assert not changed
patch=[]
for n in new:patch.extend(difflib.unified_diff([], (T/n).read_text(encoding='utf8').splitlines(True),fromfile='a/'+n,tofile='b/'+n))
(O/'source_changes.patch').write_text(''.join(patch),encoding='utf8')
with zipfile.ZipFile(O/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
 for n in sorted(h):z.write(T/n,n)
save('source_hashes.json',h);save('source_receipt.json',dict(changed=changed,new=new,files=len(h),snapshot_sha256=sha(O/'source_snapshot.zip'),git='Complete actual source copy, no fabricated commit'))
env=dict(python=sys.version,executable=sys.executable,packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','cvxpy','cylp','psutil']})
inputs={str(p):sha(p) for p in (T/'inputs').rglob('*') if p.is_file()};inputs[str(O/'inputs/workload.csv')]=sha(O/'inputs/workload.csv')
save('environment.json',env);save('input_hashes.json',inputs)
save('preregistration.json',dict(version='EXECUTION_FLOOR_RESIDUALS_V1',source=str(T),source_hashes=h,input_hashes=inputs,environment=env,
 topology={'aws':2,'dgx-ext':2},capacity={'aws':{'cpu':6,'ram':125},'dgx-ext':{'cpu':6,'ram':125}},
 cells=[dict(id=x,method=m,new_candidates=n) for x,m,n in [('OLD_FO','FO',False),('OLD_FULL','FULL',False),('NEW_FO','FO',True),('NEW_FULL','FULL',True)]],
 workload=str(O/'inputs/workload.csv'),count=12,response='R0_E_LEGACY_MEDIUM',alpha=.5,seed_profiles=[1,8],interval=60,max_rounds=6000,memory_gib=2,disk_gib=5,
 solver={'time_limit':30,'mip_rel_gap':0},rule='Union existing breakpoints with (capacity - k*existing_execution_floor)/own_g; all integer k=0..free GPUs; tightest node; one pass; no task identities; FO/FULL same provider',
 execution_floor={'cpu':1,'ram':20.83},uncertainty_enabled=False,objective_changed=False,response_changed=False,
 failure='Keep failed directory and stop pipeline; no rerun or parameter change for performance',success='12/12 only; no JCT imputation',science_label='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE'))
print('FROZEN',len(h),sha(O/'preregistration.json'))
