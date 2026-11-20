"""Preserve pre-run Windows psutil Path error and seal a string-path adapter."""
import json,hashlib,zipfile
from pathlib import Path
OUT=Path(__file__).resolve().parent
p=json.loads((OUT/'preregistration.json').read_text());work=Path(p['tree'])/'sia-simulator'
assert not list(OUT.glob('n8_*')),'NO_RUNS_PERMITTED_BEFORE_AMENDMENT'
p['amendment']='Before first cell: psutil Windows disk_usage needs str, not Path. Only runner path conversion; strategy/parameters unchanged. Prior manifest/snapshot preserved.'
for f in work.rglob('*.py'):p['hashes'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest().upper()
with zipfile.ZipFile(OUT/'source_snapshot_v2.zip','x',zipfile.ZIP_DEFLATED) as z:
    for f in work.rglob('*.py'):z.write(f,f.relative_to(work.parent))
with (OUT/'preregistration_v2.json').open('x',encoding='utf-8') as f:json.dump(p,f,indent=2)
print('FROZEN_V2_BEFORE_FIRST_CELL')
