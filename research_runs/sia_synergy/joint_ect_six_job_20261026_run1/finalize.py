"""Verify unchanged source/inputs and package completed evidence, no simulation."""
import csv,json,time,zipfile
from pathlib import Path
from experiment import OUT,ROOT,PREVIOUS,LIVE,load,sha,save

def main():
    p=load();previous=json.loads((PREVIOUS/'delivery_manifest.json').read_text(encoding='utf-8'))
    assert all(sha(LIVE/rel)==expected for rel,expected in previous['source_hashes'].items())
    source=list(csv.DictReader((ROOT/'official-artifacts/sia-simulator/workloads/saturn/workload-1.csv').open(encoding='utf-8')))
    rows=list(csv.DictReader((OUT/'inputs/workload.csv').open(encoding='utf-8')))
    for row in rows:
        original=source[int(row['source_row_index'])]
        for key in ['name','application','num_replicas','batch_size']:assert original[key]==row[key],(key,row['name'])
        assert float(row['time'])==0 and float(row['original_arrival'])==float(original['time'])
    models={str(f.relative_to(OUT)):sha(f) for f in (OUT/'inputs/models').glob('*.py')}
    assert all(sha(ROOT/'research_sources/Synergy/simulator/models'/Path(rel).name)==h for rel,h in models.items())
    audit=json.loads((OUT/'independent_audit.json').read_text())
    assert audit['status']=='ALL_COMPLETE_VALID'
    archive=OUT/'complete_experiment_evidence.zip'
    files=[f for f in OUT.rglob('*') if f.is_file() and f.suffix!='.zip']
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as stream:
        for f in files:stream.write(f,f.relative_to(ROOT))
    with zipfile.ZipFile(archive) as stream:assert stream.testzip() is None
    executions=[json.loads(line) for line in (OUT/'execution_manifest.jsonl').read_text().splitlines()]
    save(OUT/'delivery_manifest.json',dict(status='COMPLETE_ALL_FOUR_VALID',source_unchanged=True,
      parent_actual_source_unchanged=True,official_workload_rows_verified=True,official_matrix_copies_verified=True,
      algorithm_changed=False,response_changed=False,parameter_search=False,
      summary_means={m:v['summary']['mean_jct'] for m,v in audit['results'].items()},
      archive_sha256=sha(archive),archive_file=archive.name,
      wall_cell_process_seconds=sum(v['wall_seconds'] for v in executions),
      wall_seconds_to_delivery=time.time()-p['created_unix'],
      notes='Prior 188 tests belong to identical source; this round constructor + raw full-run audit. Analysis schema correction retained separately.',
      files_sha256={str(f.relative_to(OUT)):sha(f) for f in OUT.rglob('*') if f.is_file()},
      future_full160_authorized=False))
    print(json.dumps(dict(status='COMPLETE',archive_sha256=sha(archive),cell_wall_seconds=sum(v['wall_seconds'] for v in executions))))

if __name__=='__main__':main()
