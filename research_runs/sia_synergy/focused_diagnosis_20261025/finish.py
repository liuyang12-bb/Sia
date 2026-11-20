"""Reuse six completed independent raw audits, recalculate static comparison only."""
import json
from pathlib import Path
import diagnose as d
initial=d.OUT
def cached(folder,cell,p):
 return json.loads((initial/(cell['id']+'_raw_audit.json')).read_text(encoding='utf-8'))
d.audit_results.cell_audit=cached
d.OUT=initial/'review2'
d.OUT.mkdir(exist_ok=False)
d.main()
