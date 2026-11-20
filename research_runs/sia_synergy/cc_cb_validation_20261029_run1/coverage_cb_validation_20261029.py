import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_cb_validation_20261029_run1'
p=json.loads((OUT/'preregistration.json').read_text())
text=(ROOT/'maintenance/gate_cifar_bert12_20261029.py').read_text().split(' start=time.time()')[0]
text=text.replace('cc_cifar_bert12_20261029_run1','cc_cb_validation_20261029_run1')
for cap,(cpu,ram) in p['capacities'].items():
 code=text.replace("2,6.,125.)",f"2,{cpu!r},{ram!r})").replace("cp=(3.,)",f"cp=({cpu/2!r},)").replace("|{3.}",f"|{{{cpu/2!r}}}")
 code=code.replace("mm=(62.5,)",f"mm=({ram/2!r},)").replace("|{62.5}",f"|{{{ram/2!r}}}")
 code=code.replace('>6+1e-9',f'>{cpu!r}+1e-9').replace('>125+1e-9',f'>{ram!r}+1e-9')
 code=code.replace('offline_boundary_coverage.json',f'offline_coverage_{cap}.json')
 ns={'__file__':__file__,'__name__':'offline_only'};exec(compile(code,__file__,'exec'),ns);ns['main']()
 print('FULL_DOMAIN_CENSUS',cap,flush=True)
