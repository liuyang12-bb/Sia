import hashlib,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PARENT=ROOT/'research_runs/sia_synergy/cc_cb_validation_20261029_run1';OUT=ROOT/'research_runs/sia_synergy/cc_cb_validation_20261029_ample_retry1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
OUT.mkdir(exist_ok=False);shutil.copytree(PARENT/'source',OUT/'source');shutil.copytree(PARENT/'inputs',OUT/'inputs')
p=json.loads((PARENT/'preregistration.json').read_text());p['cells']=p['cells'][-6:]
text=(PARENT/'run.py').read_text().replace('cc_cb_validation_20261029_run1','cc_cb_validation_20261029_ample_retry1');a=text.index('CELLS=');b=text.index('\n',a);text=text[:a]+'CELLS='+repr(p['cells'])+text[b:]
text=text.replace("assert(cp,mm)==(cpu/2,ram/2)","assert cp in {cpu/2,round(cpu/2,9)} and mm in {ram/2,round(ram/2,9)}")
text=text.replace('assert mm==ram/2','assert mm in {ram/2,round(ram/2,9)}').replace('assert cp==cpu/2','assert cp in {cpu/2,round(cpu/2,9)}')
text=text.replace("+=cp!=cpu/2","+=cp not in {cpu/2,round(cpu/2,9)}").replace("+=mm!=ram/2","+=mm not in {ram/2,round(ram/2,9)}")
text=text.replace("'/36'","'/6'").replace('total=36','total=6').replace('ALL36_ENDED','ALL6_ENDED')
(OUT/'run.py').write_text(text,encoding='utf8');p.update(runner_hash=sha(OUT/'run.py'),frozen_unix=time.time(),retry_reason='Verification-only repair: existing source _proportional_point rounds node per-GPU quotas to9 decimals. Accept only original literal point or that exact canonical point; no relaxed resource budget, no source/science/config changes. All6 AMPLE methods rerun, failed first cell retained in parent.',parent_preregistration_hash=sha(PARENT/'preregistration.json'))
for name in ['preregistration.json','execution_preregistration_v2.json']:
 with (OUT/name).open('x',encoding='utf8') as f:json.dump(p,f,indent=2,ensure_ascii=False)
(OUT/'CURRENT_TASK.md').write_text('Verification-only proportional canonicalization repair; six AMPLE cells. Same source/work/capacity/response/solver. Independent processes serial2GiB/5GiB. Preserve parent failure. Audit together with accepted30 parent cells, then stop.\n',encoding='utf8')
print('RETRY6_FROZEN',OUT,sha(OUT/'preregistration.json'))
