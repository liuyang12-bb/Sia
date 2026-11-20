"""Read-only semantic prefix bridge against the sealed previous executions."""
import gzip,itertools,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_core80_repair_20261031_run1';OLD=ROOT/'research_runs/sia_synergy/cc_core80_8h_20261030_run3'

def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k!='template_source'}
 if isinstance(x,list):return [clean(v) for v in x]
 return x
def frames(path,n):
 with gzip.open(path,'rt',encoding='utf8') as f:
  ans=[]
  for line in itertools.islice(f,n):
   r=json.loads(line);d=r['decision_for_next_interval'];p=d['result']
   ans.append(clean(dict(execution=r['execution'],selected=p['selected'],objective=p['scheduling_objective'],actual_goodput=p['actual_total_goodput'])))
 return ans
def main():
 canon=json.loads((OLD/'canonical_result_paths.json').read_text());results=[]
 cases=[('FULL',4,'real80_FULL_8_early_zero'),('PROP',12,'real80_PROP_12_post_only'),('A10',12,'real80_A10_20_post_only')]
 for method,n,directory in cases:
  cid=('dual_neutral_' if method.startswith('A') else 'dual_medium_')+method
  a=frames(Path(canon['paths'][cid])/'rounds.jsonl.gz',n);b=frames(OUT/'tests'/directory/'prefix.jsonl.gz',n)
  differences=[i+1 for i,(u,v) in enumerate(zip(a,b)) if u!=v]
  results.append(dict(method=method,rounds=n,exact_semantic_match=len(a)==len(b)==n and not differences,different_rounds=differences,
    ignored='Relocated template_source strings only. Template SHA remains in comparison; full source/data byte hashes are checked separately.'))
 with (OUT/'tests/historical_prefix_bridges.json').open('x',encoding='utf8') as f:json.dump(results,f,indent=2)
 print(json.dumps(results))
 assert all(x['exact_semantic_match'] for x in results)
if __name__=='__main__':main()
