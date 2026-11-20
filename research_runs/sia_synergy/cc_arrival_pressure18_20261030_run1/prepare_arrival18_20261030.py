import csv,hashlib,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'research_runs/sia_synergy/cc_ample_lex_repair_20261030_run1';OUT=ROOT/'research_runs/sia_synergy/cc_arrival_pressure18_20261030_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
OUT.mkdir(exist_ok=False);(OUT/'inputs').mkdir()
seal=json.loads((OLD/'delivery_hashes_final.json').read_text());assert all(sha(OLD/n)==h for n,h in seal.items())
shutil.copytree(OLD/'source',OUT/'source');rows=list(csv.DictReader((OLD/'inputs/BASE_workload.csv').open()))
for name,times in [('BURST',[0]*12),('COMPACT',[int(r['time']) for r in rows]),('SPREAD',[i*300 for i in range(12)])]:
 with (OUT/'inputs'/f'{name}_workload.csv').open('x',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(dict(r,time=t) for r,t in zip(rows,times))
shutil.copyfile(OUT/'inputs/BURST_workload.csv',OUT/'inputs/GATE_workload.csv')
methods=['SIA_P','SIA_AUX','CC_P','CC_CPU','CC_MEM','CC_FULL'];cells=[(cohort,'RNEW','medium',m) for cohort in ['BURST','COMPACT','SPREAD'] for m in methods]
text=(OLD/'run.py').read_text().replace('cc_ample_lex_repair_20261030_run1','cc_arrival_pressure18_20261030_run1');a=text.index('CELLS=');b=text.index('\n',a);text=text[:a]+'CELLS='+repr(cells)+text[b:]
text=text.replace("'/6'","'/18'").replace('total=6','total=18').replace('ALL6_ENDED','ALL18_ENDED')
# Expand only after a successful independent audit of the first complete six.
text=text.replace("  if proc.returncode:return\n print('ALL18_ENDED'", "  if proc.returncode:return\n  if i==5:\n   audit=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'maintenance/audit_arrival18_20261030.py'),'burst'])\n   if audit.returncode:raise RuntimeError('BURST_VALIDATION_GATE_FAILED')\n print('ALL18_ENDED'")
(OUT/'run.py').write_text(text,encoding='utf8')
p=json.loads((OLD/'preregistration.json').read_text());p.update(cells=cells,gate_cells=[('GATE','RNEW','medium',m) for m in ['SIA_P','CC_CPU','CC_MEM','CC_FULL']],capacities={'RNEW':[6.,125.]},input_hashes={f.name:sha(f) for f in (OUT/'inputs').iterdir()},runner_hash=sha(OUT/'run.py'),frozen_unix=time.time(),parent_seal_hash=sha(OLD/'delivery_hashes_final.json'),scenario='Same accepted12 original jobs/work/batch/demand/order; only actual arrival differs. BURST all0; COMPACT original first4t0 then pairs300..1200; SPREAD i*300.',
 actual_gate='All4 methods at most60rounds, no extension. FULL must have>=60 productive CF+BERT shared-node seconds with CF faster than same-GPU proportional and BERT no slower; total all-running-set normalized fraction progress delta positive AND sum same-state remaining-ETA savings positive. All commits legal. CPU-only/MEM-only gate evidence reported without requiring wins. No JCT-based gating.',
 expansion_gate='BURST all6 complete12/12 and independent raw audit passes; then COMPACT/SPREAD. No JCT-win requirement.',max_rounds=6000,source_changed=False,model_changed=False)
for n in ['preregistration.json','execution_preregistration_v2.json']:
 with (OUT/n).open('x',encoding='utf8') as f:json.dump(p,f,indent=2,ensure_ascii=False)
for n in ['all_tests.log','all_tests.log.json','original_sia_regression.json']:shutil.copyfile(OLD/n,OUT/('prior_'+n))
shutil.make_archive(str(OUT/'source_snapshot'),'zip',OUT/'source')
(OUT/'CURRENT_TASK.md').write_text('ACTIVE authorized18 cells at RNEW6CPU125RAM, AWS/DGX2GPU each, Medium. Frozen repaired source identical to accepted lex repair. First4 bounded BURST gate methods60rounds, then6 BURST complete/audit, only then6COMPACT and6SPREAD. 60s6000rounds, serial independent processes2GiB RAM5GiB disk. No algorithm/grid/response/capacity tuning; preserve failures. No other batches. Final raw audits, comparative report and source/input/output seals; then STOP.\n',encoding='utf8')
print('FROZEN18',OUT,sha(OUT/'preregistration.json'),flush=True)
