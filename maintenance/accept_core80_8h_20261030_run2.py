import contextlib,copy,hashlib,importlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];TREE=ROOT/'cc-joint-core80-8h-v2';OUT=ROOT/'research_runs/sia_synergy/cc_core80_8h_20261030_run2'
sys.path[:0]=[str(TREE),str(TREE/'sia-simulator'),str(TREE/'sia-simulator/tests')]
def save(n,x):(OUT/'tests'/n).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf8')
from test_clean_ablation import controller,state,cs
from test_adaptive_service import spec
js=[spec(f'j{i:03}',index=i) for i in range(80)];s=state(js);a=controller(js)
try:a.decide(s,cs(s,{}),0.,{});raise AssertionError('Legacy unexpectedly accepted80')
except RuntimeError as e:
 assert 'SCOPE' in str(e) or 'MAX' in str(e);save('legacy_scale_red.json',dict(expected_failure=True,error=str(e),active_jobs=80,baseline_guard=a.MAX_ACTIVE_JOBS))
probe=TREE/'sia-simulator/check_stage1_legacy_regression.py';states={}
for name,sim in [('official',ROOT/'official-artifacts/sia-simulator'),('core80',TREE/'sia-simulator')]:
 r=subprocess.run([sys.executable,'-X','utf8','-B',str(probe),'--probe',str(sim)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (OUT/'tests'/f'original_sia_{name}.log').write_bytes(r.stdout);assert r.returncode==0
 states[name]=json.loads(next(x.split('PROBE_JSON=',1)[1] for x in r.stdout.decode('utf8').splitlines() if x.startswith('PROBE_JSON=')))
assert states['official']==states['core80'];save('original_sia_regression.json',dict(exact_state_match=True,rounds=3,states=states))
print('LEGACY_RED_AND_OFFICIAL_EXACT_OK')
