"""Recover the preregistered analysis version exactly, verified by its hash."""
import hashlib,json,zipfile
from pathlib import Path
OUT=Path(__file__).resolve().parent
p=json.loads((OUT/'attempt1/preregistration.json').read_text())
new="""                if fresh and e['checkpoint_remaining_before_decision']>0 and e['completion_time'] is None:
                    new=d['result']['selected'].get(e['job_id'])
                    assert new and new['candidate_kind']=='locked'
                    for key in ('cpu_per_gpu','ram_gb_per_gpu','placement'):assert new[key]==a[key]
                    assert new['config']['config_id']==a['config_id'];checks['checkpoint_lock']+=1
"""
old="""                if fresh and e['checkpoint_delay_consumed']>=active and active>0 and e['completion_time'] is None:
                    # Exact lock only if still in checkpoint at next boundary:
                    # expiry at boundary may legitimately reconfigure. Source
                    # candidate kind is checked separately by precommit audit.
                    new=d['result']['selected'].get(e['job_id'])
                    if new and new['candidate_kind']=='locked':
                        for key in ('cpu_per_gpu','ram_gb_per_gpu','placement'):assert new[key]==a[key]
                        assert new['config']['config_id']==a['config_id'];checks['checkpoint_lock']+=1
"""
text=(OUT/'audit_results.py').read_text();assert text.count(new)==1
raw=text.replace(new,old).encode('utf-8')
assert hashlib.sha256(raw).hexdigest().upper()==p['input_hashes'][str(OUT/'audit_results.py')]
with zipfile.ZipFile(OUT/'attempt1/external_tools_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
    for n in ('preflight.py','freeze_run.py','audit_results.py'):
        b=raw if n=='audit_results.py' else (OUT/n).read_bytes()
        assert hashlib.sha256(b).hexdigest().upper()==p['input_hashes'][str(OUT/n)]
        z.writestr(n,b)
print('Original external tool hashes reconstructed exactly')
