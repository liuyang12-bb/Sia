"""Small read-only summaries from sealed captured MIPs, no solving."""
import gzip,json,hashlib,math
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];T=R/'cc-joint-core80-full-lex-repair-v7';O=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run4'
records=[]
for name in ('core80_full_rank_step19.json.gz','core80_full_startup_step37.json.gz','core80_full_startup_step42.json.gz'):
 p=T/'sia-simulator/tests/fixtures'/name;d=json.loads(gzip.decompress(p.read_bytes()));m=d['model'];c=np.asarray(m['c']);x=np.asarray(d['last_valid_x']) if d.get('last_valid_x') is not None else None
 records.append(dict(fixture=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest().upper(),variables=len(c),job_count=len({q['job_id'] for q in m['columns']}),constraint_groups=len(m['constraints']),cost_range=[float(c.min()),float(c.max())],gcd=math.gcd(*(int(v) for v in c if v)),valid_incumbent_objective=float(c@x) if x is not None else None,original_status=d.get('original_status'),retry_status=d.get('retry_status'),messages={k:d[k] for k in ('original_message','retry_message','independent_cbc') if k in d},primary_band={k:m['constraints'][1][k] for k in ('lb','ub')},lp_summary={k:v for k,v in (d.get('dual_proof') or {}).items() if k!='fix_zero'}))
 del d,m,c,x
with (O/'failure_chain.json').open('x',encoding='utf8') as f:json.dump(records,f,ensure_ascii=False,indent=2)
print(json.dumps([{k:v for k,v in q.items() if k not in ('lp_summary','messages')} for q in records],ensure_ascii=False,indent=2))
