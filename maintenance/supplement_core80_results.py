"""Classify frozen outcomes and expose tail losses; no simulation/solver."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_runs/sia_synergy/cc_core80_8h_20261030_run3'

def save(path,obj):
    with path.open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,ensure_ascii=False)

def main():
    a=OUT/'audit';metrics=json.loads((a/'raw_recomputed_metrics.json').read_text());canon=json.loads((OUT/'canonical_result_paths.json').read_text());p=json.loads((OUT/'preregistration.json').read_text())
    ledger=[]
    for path in (OUT/'execution_manifest.jsonl',OUT/'execution_recovery1/execution_manifest.jsonl'):
        ledger += [json.loads(x) for x in path.read_text().splitlines()]
    classified=[]
    for cell in p['cells']:
        cid=cell['id'];m=metrics.get(cid,{})
        d=Path(canon['paths'][cid]);status=m.get('status','NOT_RUN')
        if status=='COMPUTE_TIMEOUT_OR_INTERRUPTED' and (d/'COMPUTE_TIMEOUT.json').exists():status='COMPUTE_TIMEOUT'
        last=next((e for e in reversed(ledger) if e['event']=='CELL_END' and e['cell']==cid),{})
        classified.append(dict(cell=cid,status=status,completed=m.get('completed'),declared=80,mean_jct=m.get('mean_jct'),
           process_wall_seconds=last.get('wall_seconds'),observed_simulated_end=m.get('observed_simulated_end'),arrived=m.get('arrived_by_observed_end'),
           error=m.get('failure_error'),directory=str(d),optional=cell['optional']))
    save(a/'termination_analysis.json',dict(cells=classified,counts={s:sum(c['status']==s for c in classified) for s in sorted({c['status'] for c in classified})},
        optional_not_started_reason='Core numerical/lexicographic failures and incomplete paired controls; registered optional gate fails. Insufficient reserved budget for both optional full-work pairs; no shortened substitute.',
        scope='COMPUTE_TIMEOUT is confirmed by external watchdog receipt; partial completed count is persisted-prefix evidence, never a full-run JCT mean.'))
    with (a/'core80_coverage.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,list(classified[0]));w.writeheader();w.writerows(classified)
    jobs={}
    with (a/'per_job.csv').open(encoding='utf8') as f:
        for r in csv.DictReader(f):jobs.setdefault(r['cell'],{})[r['job']]=r
    pairs=json.loads((a/'paired_comparisons.json').read_text());tails=[]
    for q in pairs:
        if not q.get('complete_pair'):continue
        worst=sorted(q['per_job_saved_seconds'],key=q['per_job_saved_seconds'].get)[:8]
        def view(r):
            return {k:r[k] for k in ('original_name','application','jct','completion','first_wait','later_wait','checkpoint','training','gpu_hours','restarts','gpu_changes')}
        tails.append(dict(candidate=q['candidate'],baseline=q['baseline'],by_application=q['by_application'],
             worst_eight=[dict(job=j,saved_seconds=q['per_job_saved_seconds'][j],candidate=view(jobs[q['candidate']][j]),baseline=view(jobs[q['baseline']][j])) for j in worst],
             diagnostic_only='All80 retained in primary results; never delete these losses or largest beneficiaries.'))
    save(a/'tail_and_application_diagnostics.json',tails)
    print('CLASSIFIED_COUNTS',{s:sum(c['status']==s for c in classified) for s in sorted({c['status'] for c in classified})})
if __name__=='__main__':main()
