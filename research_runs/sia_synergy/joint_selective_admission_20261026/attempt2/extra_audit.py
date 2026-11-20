"""Cross-check end-state restart counters, aliases and immutable prerequisites."""
import collections,gzip,hashlib,json,math
from pathlib import Path
OUT=Path(__file__).resolve().parent;p=json.loads((OUT/'preregistration.json').read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest().upper()
def main():
    findings=[]
    for cell in p['runs']:
        folder=OUT/cell['id'];s=json.loads((folder/'summary.json').read_text());last=None;first_progress={};completion_events={};statuses=collections.Counter();fallback=collections.Counter()
        last_round=None;max_decline=0.;minimum_rate=None;numerical_retries=0
        with gzip.open(folder/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);last=r
                d=r['decision_for_next_interval']
                if d['round']!=last_round:
                    status=d['result']['status'];statuses[status]+=1
                    if status.startswith('FALLBACK_'):fallback[status]+=1
                    for retry in d['result']['audit'].get('numerical_solver_retries',[]):
                        def worst(x):
                            value=max([abs(v-round(v)) for v in x]+[-v for v in x]+[v-1 for v in x])
                            for con in retry['model']['constraints']:
                                matrix=con['A']
                                rows=matrix if isinstance(matrix[0],list) else [matrix]
                                def bound(b,i):return b if isinstance(b,(int,float)) else b[0 if len(b)==1 else i]
                                for i,row in enumerate(rows):
                                    total=sum(a*b for a,b in zip(row,x));value=max(value,bound(con['lb'],i)-total,total-bound(con['ub'],i))
                            return value
                        assert worst(retry['original_x'])>1e-7
                        assert worst(retry['retry_x'])<=1e-7 and worst([round(v) for v in retry['retry_x']])<=1e-7
                        numerical_retries+=1
                    last_round=d['round']
                for e in r['execution']:
                    if e['progress_after']>e['progress_before'] and e['job_id'] not in first_progress:first_progress[e['job_id']]=r['interval_end']
                    if e['completion_time'] is not None:
                        if e['job_id'] in completion_events:assert completion_events[e['job_id']]==e['completion_time']
                        completion_events[e['job_id']]=e['completion_time']
                    for seg in e['training_segments']:
                        if seg['seconds']>0:
                            assert seg['raw_rate_before_rho']>0
                            minimum_rate=seg['raw_rate_before_rho'] if minimum_rate is None else min(minimum_rate,seg['raw_rate_before_rho'])
                            assert math.isclose(seg['effective_rate'],seg['raw_rate_before_rho']*seg['rho'],rel_tol=1e-12)
        assert last is not None
        for j,row in s['jobs'].items():
            end=last['job_end_state'][j]
            for key,alias in [('restarts','restarts'),('migrations','migrations'),('wasted_gpu_seconds','wasted_gpu_seconds')]:
                assert math.isclose(end[key],row[alias],abs_tol=1e-7),(cell['id'],j,key)
            assert end['completion']==row['completion']==completion_events.get(j)
            if row['completed']:assert row['jct']==completion_events[j]-row['arrival']
        assert dict(statuses)==s['solver_status']
        assert sum(fallback.values())==s['fallback_count']
        findings.append(dict(cell=cell['id'],valid=True,completion_events=len(completion_events),raw_rate_positive_min=minimum_rate,
            independently_checked_final_job_counters=len(s['jobs']),fallback_status=dict(fallback),solver_status=dict(statuses),
            numerical_retries_raw_and_rounded_independently_validated=numerical_retries))
    for path,h in p['hashes'].items():assert sha(path)==h,path
    for path,h in p['baseline_hashes'].items():assert sha(path)==h,path
    with (OUT/'extra_validation.json').open('x') as f:json.dump(dict(valid=True,cells=findings,source_input_hashes_verified=len(p['hashes']),history_files_verified=len(p['baseline_hashes'])),f,indent=2)
    print('EXTRA_AUDIT_PASSED',len(findings),flush=True)
if __name__=='__main__':main()
