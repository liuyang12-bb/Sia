"""Read-only paired analysis of two independently audited policy cells."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()

gate = json.loads((HERE / 'cells/G112/opportunity_gate.json').read_text())
assert gate['passed']
data = {}
for name in ['G112', 'G111']:
    directory = HERE / 'cells' / name
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['completed'] == summary['total'] == 6
    events = collections.Counter()
    shortened = 0
    scored = 0
    with gzip.open(directory / 'rounds.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            d = r['decision_for_next_interval']
            events.update(e['event'] for e in d.get('liveness_events', [])
                          if e['time'] == r['interval_end'])
            for score in d['net_progress_scores']:
                scored += 1
                shortened += score.get('service_horizon', 300.) < 300.
    data[name] = dict(summary=summary, events=dict(events),
                      scored_columns=scored, shortened_score_columns=shortened,
                      raw_log_sha256=sha(directory / 'rounds.jsonl.gz'))

a = data['G112']['summary']
b = data['G111']['summary']
keys = ['wait', 'checkpoint', 'training', 'restarts', 'gpu_hours',
        'cpu_core_hours', 'ram_gb_hours']
totals = {method: {key: sum(v[key] for v in entry['summary']['jobs'].values())
                   for key in keys} for method, entry in data.items()}
per_job = {j: dict(G112=a['jobs'][j]['jct'], G111=b['jobs'][j]['jct'],
                    delta=a['jobs'][j]['jct'] - b['jobs'][j]['jct'])
           for j in a['jobs']}
result = dict(status='AUDITED_COMPLETE_PAIRED', gate_checks=gate['checks'],
              preregistration_sha256=sha(HERE / 'preregistration.json'),
              source_snapshot_sha256=sha(HERE / 'source_snapshot.zip'),
              input_sha256=sha(HERE / 'workload.csv'),
              cells={k: dict(mean_jct=v['summary']['mean_jct'],
                             median_jct=v['summary']['median_jct'],
                             p95_jct=v['summary']['p95_jct'],
                             p99_jct=v['summary']['p99_jct'],
                             makespan=v['summary']['makespan'],
                             rounds=v['summary']['rounds'],
                             events=v['events'], totals=totals[k],
                             scored_columns=v['scored_columns'],
                             shortened_score_columns=v['shortened_score_columns'],
                             raw_log_sha256=v['raw_log_sha256'])
                     for k, v in data.items()},
              per_job=per_job,
              mean_jct_delta=a['mean_jct'] - b['mean_jct'],
              relative_improvement=(b['mean_jct'] - a['mean_jct']) / b['mean_jct'],
              total_delta={k: totals['G112'][k] - totals['G111'][k] for k in keys})
with (HERE / 'paired_analysis.json').open('x', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(json.dumps(result, ensure_ascii=False, indent=2))
