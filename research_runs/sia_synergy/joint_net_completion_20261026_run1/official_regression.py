"""Three official rounds, one real job, exact original Sia regression."""
from pathlib import Path
import json, subprocess, sys, time
from prepare import OUT, ROOT, TREE, save, sha

script=TREE/'check_stage1_legacy_regression.py'
results={};commands=[];started=time.time()
for name,sim in [('official',ROOT/'official-artifacts/sia-simulator'),('new',TREE)]:
    cmd=[sys.executable,'-X','utf8','-B',str(script),'--probe',str(sim)]
    commands.append(cmd)
    with (OUT/(name+'_sia_bounded.log')).open('x',encoding='utf-8') as f:
        proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    assert proc.returncode==0,name
    lines=(OUT/(name+'_sia_bounded.log')).read_text(encoding='utf-8').splitlines()
    results[name]=json.loads(next(l[len('PROBE_JSON='):] for l in lines if l.startswith('PROBE_JSON=')))
save(OUT/'official_regression.json',dict(exact_match=results['official']==results['new'],
    commands=commands,rounds=3,jobs=1,states=results,seconds=time.time()-started,
    logs={name:sha(OUT/(name+'_sia_bounded.log')) for name in results}))
assert results['official']==results['new']
print('OFFICIAL_SIA_EXACT_MATCH_3_ROUNDS')
