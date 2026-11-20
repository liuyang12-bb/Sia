"""Preserve failed launcher; correct script location only, no science changes."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1];p=ROOT/'research_runs/sia_synergy/cc_cifar_bert12_20261029_run1'
assert not (p/'cells').exists()
t=(p/'run_scenario.py').read_text()
t=t.replace('ROOT=Path(__file__).resolve().parents[1]','ROOT=Path(__file__).resolve().parents[3]').replace("OUT/'preregistration.json'","OUT/'execution_preregistration_v2.json'")
n=p/'run_scenario_v2.py';assert not n.exists();n.write_text(t,encoding='utf8')
r=json.loads((p/'preregistration.json').read_text());r['runner_hash']=hashlib.sha256(n.read_bytes()).hexdigest().upper()
r['operational_correction']='Only launcher workspace ROOT depth and corresponding hash. Initial failed before creating cluster/output cell. Scientific conditions/source identical.'
with (p/'execution_preregistration_v2.json').open('x',encoding='utf8') as f:json.dump(r,f,ensure_ascii=False,indent=2)
base=(ROOT/'maintenance/gate_cifar_bert12_20261029.py').read_text();head=base.split('def main():',1)[0];tail=base.split(' gates={}',1)[1]
out=head+"def main():\n assert json.loads((OUT/'tests_canonical.json').read_text())['exit_code']==0\n gates={}"+tail
out=out.replace('run_scenario.py','run_scenario_v2.py').replace('GATE_{method}.log','GATE_{method}_v2.log')
(ROOT/'maintenance/gate_cifar_bert12_v3_20261029.py').write_text(out,encoding='utf8')
