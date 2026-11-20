"""Self-contained frozen execution prerequisites plus final output hash index."""
import json,hashlib,zipfile,subprocess
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
p=json.loads((OUT/'preregistration_v2.json').read_text())
with zipfile.ZipFile(OUT/'frozen_prerequisites.zip','x',zipfile.ZIP_DEFLATED) as z:
    for name,h in p['hashes'].items():
        f=Path(name);assert hashlib.sha256(f.read_bytes()).hexdigest().upper()==h
        z.write(f,f.relative_to(ROOT))
status=subprocess.check_output(['git','-C',p['tree'],'status','--short','--branch']).decode()
diff=subprocess.check_output(['git','-C',p['tree'],'diff','--stat']).decode()
assert not diff,'TRACKED_SOURCE_CHANGED'
snapshot=dict(status=status,tracked_diff_stat=diff,commit=p['base_commit'],
    files={str(f.relative_to(OUT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.rglob('*') if f.is_file()},
    original_104_batch='PAUSED; never launched here',algorithm_changes='none; explicit tested scope adapter only')
with (OUT/'delivery_manifest.json').open('x',encoding='utf-8') as f:json.dump(snapshot,f,indent=2)
print('PACKED',len(p['hashes']),'frozen prerequisites;',len(snapshot['files']),'artifacts')
