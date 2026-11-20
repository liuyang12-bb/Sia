"""Freeze an isolated scenario; source and execution physics stay byte-identical."""
import csv,hashlib,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1/attempt3'
OUT=ROOT/'research_runs/sia_synergy/cc_cifar_bert12_20261029_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def main():
 OUT.mkdir(exist_ok=False);(OUT/'inputs').mkdir()
 seal=json.loads((OLD/'delivery_hashes_final.json').read_text());assert all(sha(OLD/n)==h for n,h in seal.items())
 shutil.copytree(OLD/'source',OUT/'source')
 trace=ROOT/'official-artifacts/sia-simulator/workloads/saturn/workload-1.csv'
 rows=list(csv.DictReader(trace.open()));cf=[(i,r) for i,r in enumerate(rows) if r['application']=='cifar10'][:6];bt=[(i,r) for i,r in enumerate(rows) if r['application']=='bert'];assert len(bt)==6
 selected=[]
 for k in range(6):
  for i,r in [cf[k],bt[k]]:
   selected.append(dict(r,time=0 if k<2 else (k-1)*300,original_time=r['time'],original_source_row_index=i))
 with (OUT/'inputs/CB12_workload.csv').open('x',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=selected[0]);w.writeheader();w.writerows(selected)
 text=(ROOT/'maintenance/run_breakpoint30_precision_v3_20261029.py').read_text()
 text=text.replace("OUT=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1/attempt3'","OUT=ROOT/'research_runs/sia_synergy/cc_cifar_bert12_20261029_run1'")
 start=text.index('CELLS=');end=text.index('\n',start)
 methods=['SIA_P','SIA_AUX','CC_P','CC_CPU','CC_MEM','CC_FULL'];cells=[('CB12','RNEW','medium',m) for m in methods]
 text=text[:start]+'CELLS='+repr(cells)+text[end:]
 text=text.replace('from resource_study import CAPACITIES','from resource_study import CAPACITIES\n  CAPACITIES[\'RNEW\']=(6.,125.)')
 text=text.replace('for i in range(6000):','for i in range(60 if mix==\'GATE\' else 6000):')
 text=text.replace('setup();name=label','setup();name=label')
 text=text.replace("assert [mix,scenario,response,method] in p['cells']","assert [mix,scenario,response,method] in p['cells'] or [mix,scenario,response,method] in p['gate_cells']")
 text=text.replace('/30','/6').replace('total=30','total=6').replace('ALL30_ENDED','ALL6_ENDED')
 runner=OUT/'run_scenario.py';runner.write_text(text,encoding='utf8')
 shutil.copyfile(OUT/'inputs/CB12_workload.csv',OUT/'inputs/GATE_workload.csv')
 p=json.loads((OLD/'preregistration.json').read_text())
 p.update(cells=cells,gate_cells=[('GATE','RNEW','medium',m) for m in ['SIA_P','CC_CPU','CC_FULL']],
  capacities={'RNEW':[6.,125.]},runner_hash=sha(runner),input_hashes={p.name:sha(p) for p in (OUT/'inputs').iterdir()},
  frozen_unix=time.time(),scenario='First6 CF +all6 BERT real workload1 records, interleaved. First4 t0, pairs t300/600/900/1200. Full original work/batch/demand.',
  gate='60 rounds maximum per SIA_P/CC_CPU/CC_FULL; actual productive CF+BERT same-node overlap and an executed differential allocation with positive fixed-geometry receiver rho gain and no donor rho loss are required before formal6. No retuning on gate failure.',
  source_changed=False,model_changed=False,source_trace_hash=sha(trace),parent_seal_hash=sha(OLD/'delivery_hashes_final.json'),
  observation_limit=6000,resource_gate={'memory_gib':2,'disk_gib':5},
  boundary_contract='Existing E-legacy applies equally to all. Strict-supported AWS witness separately verified; BERT DGX reference and other clipped candidates retained and counted, never silently excluded.')
 with (OUT/'preregistration.json').open('x',encoding='utf8') as f:json.dump(p,f,ensure_ascii=False,indent=2)
 shutil.copyfile(ROOT/'research_runs/sia_synergy/cc_cpu_ram_increment_diagnosis_20261029_run1/scenario_opportunity.json',OUT/'strict_witness.json')
 shutil.make_archive(str(OUT/'source_snapshot'),'zip',OUT/'source')
 (OUT/'CURRENT_TASK.md').write_text('CF6+BERT6 approved scenario. Frozen code identical to prior attempt3. Capacity6CPU/125GB each2GPU AWS/DGX node. Medium alpha.5. Run offline full candidate coverage and241 tests, then three60-round gates, then six independent serial full runs only if actual opportunity gate passes. No104/160, no objective/model/parameter tuning. Preserve all failures. Output root is this directory.\n',encoding='utf8')
 print('FROZEN',OUT,sha(OUT/'preregistration.json'))
if __name__=='__main__':main()
