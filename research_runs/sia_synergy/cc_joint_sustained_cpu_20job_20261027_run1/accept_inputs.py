"""Independent extraction and controlled arrival acceptance; no training."""
import csv,json
from experiment import OUT,ROOT,load,save,sha

p=load()
with (OUT/'inputs/workload.csv').open(encoding='utf-8') as f:jobs=list(csv.DictReader(f))
with (OUT/'inputs/original_workload-1.csv').open(encoding='utf-8') as f:original=list(csv.DictReader(f))
assert len(jobs)==20 and len(original)==160
cf=[r for r in original if r['application']=='cifar10'][:18]
ds=[r for r in original if r['application']=='deepspeech2'][:2]
expected=[cf[0],ds[0],cf[1],ds[1]]+cf[2:]
assert len({j['name'] for j in jobs})==20
for i,(j,src) in enumerate(zip(jobs,expected)):
    assert all(j[k]==v for k,v in src.items() if k!='time'),j
    assert float(j['original_arrival'])==float(src['time'])
    assert original[int(j['original_source_row_index'])]==src
    assert int(j['source_row_index'])==i
    assert float(j['time'])==(0 if i<4 else 300*(1+(i-4)//2))
assert sha(OUT/'inputs/original_workload-1.csv')==sha(ROOT/'official-artifacts/sia-simulator/workloads/saturn/workload-1.csv')
assert sha(OUT/'inputs/config_symmetric.json')==sha(OUT.parent/'cc_joint_12job_20261026_run1/inputs/config_symmetric.json')
save(OUT/'input_acceptance.json',dict(valid=True,checks=25+20*5,
    original_nonarrival_fields_exact=True,all_20_unique=True,stable_controlled_FIFO_exact=True,
    original_source_trace_unchanged=True,capacity_response_config_byte_identical_to_parent=True,
    workload_sha256=sha(OUT/'inputs/workload.csv'),preregistration_sha256=sha(OUT/'preregistration.json')))
print('INPUT_ACCEPTANCE_PASS')
