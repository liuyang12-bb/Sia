"""Only two original rows for five-round correctness equivalence probes."""
import csv,json,shutil
from inventory import ROOT,OUT,sha,save

history=ROOT/'research_runs/sia_synergy/cc_joint_sustained_cpu_20job_20261027_run1'
dest=OUT/'probe_inputs/inputs';dest.mkdir(parents=True,exist_ok=False)
for name in ['config_symmetric.json','registry.json']:shutil.copy2(history/'inputs'/name,dest/name)
shutil.copytree(history/'inputs/models',dest/'models')
with (history/'inputs/workload.csv').open(encoding='utf-8') as f:
    r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)[:2]
assert [r['name'] for r in rows]==['cifar10-0','deepspeech2-8']
with (dest/'workload.csv').open('x',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
save(OUT/'probe_preregistration.json',dict(jobs=[r['name'] for r in rows],rounds=5,interval=60,
    methods=['SIA_P','SIA_ECT_AUX','JOINT_CC_P','JOINT_CC'],responses=['neutral','medium'],
    purpose='Compare exact allocation/progress/checkpoint/batch/CPU/RAM trajectories before and after dependency cleanup; not a performance experiment.',
    workload_sha256=sha(dest/'workload.csv')))
print('PROBES_PREPARED')
