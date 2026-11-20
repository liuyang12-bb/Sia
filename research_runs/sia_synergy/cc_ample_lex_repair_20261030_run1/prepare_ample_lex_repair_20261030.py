import hashlib,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'research_runs/sia_synergy/cc_cb_validation_20261029_run1';OUT=ROOT/'research_runs/sia_synergy/cc_ample_lex_repair_20261030_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
OUT.mkdir(exist_ok=False)
seals={}
for d in [OLD,ROOT/'research_runs/sia_synergy/cc_cb_validation_20261029_ample_retry1']:
 s=json.loads((d/'delivery_hashes.json').read_text());assert all(sha(d/n)==h for n,h in s.items());seals[str(d)]=sha(d/'delivery_hashes.json')
shutil.copytree(OLD/'source',OUT/'source');shutil.copytree(OLD/'inputs',OUT/'inputs')
baseline={p.relative_to(OUT/'source').as_posix():sha(p) for p in (OUT/'source').rglob('*') if p.is_file()}
with (OUT/'baseline_source_hashes.json').open('x',encoding='utf8') as f:json.dump(baseline,f,indent=2)
shutil.make_archive(str(OUT/'baseline_source_snapshot'),'zip',OUT/'source')
with (OUT/'protection.json').open('x',encoding='utf8') as f:json.dump(dict(old_seals=seals,baseline_files=len(baseline),frozen=time.time()),f,indent=2)
(OUT/'CURRENT_TASK.md').write_text('ACTIVE 2026-10-30: diagnose and minimally repair AMPLE lexicographic false infeasibility. Preserve exact main objective, constraints, CPU/RAM response, grid, capacities, workload, old outputs. Capture bounded real CPU/MEM four-round red states, independent model replay, add regressions, run241+new tests and original Sia bounded regression, then frozen six AMPLE independent serial processes2GiB RAM5GiB disk60s6000rounds. No other training experiment. Design genuinely distinct workloads only after results. New source snapshot, diff, hashes, report. Stop after deliver.\n',encoding='utf8')
print(OUT)
