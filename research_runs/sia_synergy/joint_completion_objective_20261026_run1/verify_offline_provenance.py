"""Read-only input/source provenance verification; writes only new audit evidence."""
import csv
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
registry=json.loads((OUT/'inputs/registry.json').read_text())
keys={f'{app}/{gpu}:1:{g}' for app in ['cifar10','deepspeech2'] for gpu in ['aws','dgx-ext'] for g in [1,2]}
rows=[r for r in registry if r['key'] in keys]
checks=[]
for r in rows:
    for source in r['source_rows']:
        path=Path(source['file'])
        checks.append(dict(kind='R0_source',registry_key=r['key'],path=str(path),line=source['line'],expected=source['sha256'],actual=sha(path),valid=sha(path)==source['sha256']))
for name in ['alexnet_1.py','deepspeech_1.py']:
    official=ROOT/'research_sources/Synergy/simulator/models'/name
    frozen=OUT/'inputs/models'/name
    checks.append(dict(kind='official_matrix_copy',path=str(official),frozen=str(frozen),official_sha256=sha(official),frozen_sha256=sha(frozen),valid=sha(official)==sha(frozen)))
trace=ROOT/'official-artifacts/sia-simulator/workloads/saturn/workload-1.csv'
with trace.open(encoding='utf-8',newline='') as f: originals=list(csv.DictReader(f))
for name in ['D2_gpu_auxiliary_coupling.csv','D3_queued_training_proposal.csv']:
    with (OUT/'inputs'/name).open(encoding='utf-8',newline='') as f: selected=list(csv.DictReader(f))
    for row in selected:
        original=originals[int(row['source_row_index'])]
        checks.append(dict(kind='source_workload_record',file=name,job=row['name'],source_row_index=int(row['source_row_index']),valid=all(row[k]==original[k] for k in ['name','application','num_replicas','batch_size']) and float(row['time'])==0 and row['original_arrival']==original['time']))
payload=dict(training_executed=False,all_valid=all(c['valid'] for c in checks),source_trace_sha256=sha(trace),checks=checks,script_sha256=sha(__file__),offline_gate_sha256=sha(OUT/'offline_gate.json'))
with (OUT/'offline_provenance_verification.json').open('x',encoding='utf-8') as f:json.dump(payload,f,indent=2,ensure_ascii=False)
assert payload['all_valid'], 'OFFLINE_PROVENANCE_FAILED'
print('OFFLINE_PROVENANCE_PASS',len(checks))
