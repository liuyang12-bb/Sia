"""Read only complete records from an active gzip prefix; never repair it."""
import gzip,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;p=json.loads((OUT/'preregistration.json').read_text())
for c in p['runs']:
    folder=OUT/c['id']
    if not folder.exists():continue
    if (folder/'summary.json').exists():
        s=json.loads((folder/'summary.json').read_text());print(c['id'],'FINISHED',s['completed'],s['rounds'],s['mean_jct'],flush=True);continue
    if (folder/'failure.json').exists():print(c['id'],'FAILED',json.loads((folder/'failure.json').read_text())['error'],flush=True);continue
    last=None;count=0
    try:
        with gzip.open(folder/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                try:last=json.loads(line);count+=1
                except json.JSONDecodeError:break
    except (EOFError,OSError):pass
    if last:print(c['id'],'ACTIVE_PREFIX',count,'time',last['interval_end'],'selected',len(last['decision_for_next_interval']['result']['selected']),flush=True)
