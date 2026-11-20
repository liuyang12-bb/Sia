import csv,hashlib,json,shutil,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'research_runs/sia_synergy/cc_cifar_bert12_20261029_run1'
OUT=ROOT/'research_runs/sia_synergy/cc_cb_validation_20261029_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 OUT.mkdir(exist_ok=False);(OUT/'inputs').mkdir()
 seal=json.loads((OLD/'delivery_hashes.json').read_text());assert all(sha(OLD/n)==h for n,h in seal.items())
 shutil.copytree(OLD/'source',OUT/'source');shutil.copyfile(OLD/'strict_witness.json',OUT/'strict_witness.json')
 sys.path[:0]=[str(OUT/'source'),str(OUT/'source/sia-simulator')]
 import numpy as np
 from joint_prototype.joint_profile import load_synergy_template,CPU_COORDS,RAM_COORDS
 reg=json.loads((OUT/'source/inputs/registry.json').read_text());needs=[]
 for r in reg:
  if r['application'] not in ['cifar10','bert'] or r['config_id'] not in ['aws:1:1','aws:1:2','dgx-ext:1:1','dgx-ext:1:2']:continue
  a=load_synergy_template(OUT/'source/inputs/models'/r['template']).matrix
  ci=next(i for i in range(len(CPU_COORDS)) if np.all(a[i:,:]==a[-1,:]))
  mi=next(i for i in range(len(RAM_COORDS)) if np.all(a[:,i:]==a[:,-1,None]))
  needs.append(dict(key=r['key'],cpu=float(CPU_COORDS[ci]*r['phi_c']),ram=float(RAM_COORDS[mi]*r['phi_m'])))
 ample=[2*max(r['cpu'] for r in needs),2*max(r['ram'] for r in needs)]
 traces={};hashes={}
 for t in [1,5,10]:
  p=ROOT/f'official-artifacts/sia-simulator/workloads/saturn/workload-{t}.csv';hashes[str(t)]=sha(p)
  traces[t]=[dict(r,original_trace=t,original_source_row_index=i,original_name=r['name'],original_time=r['time']) for i,r in enumerate(csv.DictReader(p.open()))]
 def take(t,app,n):return [dict(r) for r in traces[t] if r['application']==app][:n]
 def write(name,cf,bt):
  assert len(cf)+len(bt)==12
  # Evenly distribute each class in stable source order, no result-dependent ordering.
  rows=[];ic=ib=0
  for k in range(12):
   choose_cf=ic<len(cf) and (ib==len(bt) or (ic+1)/len(cf)<=(ib+1)/len(bt))
   r=dict(cf[ic] if choose_cf else bt[ib]);ic+=choose_cf;ib+=not choose_cf
   r['name']=r['application']+'-'+str(int(r['original_trace'])*1000+int(r['original_source_row_index']))
   r['time']=0 if k<4 else ((k-4)//2+1)*300;rows.append(r)
  with (OUT/'inputs'/f'{name}_workload.csv').open('x',newline='',encoding='utf8') as f:
   w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
  return rows
 w5=take(5,'bert',6);w5+=take(10,'bert',6-len(w5))
 cohorts={'W1':write('W1',take(1,'cifar10',6),take(1,'bert',6)),
          'W5P':write('W5P',take(5,'cifar10',6),w5),
          'W10':write('W10',take(10,'cifar10',6),take(10,'bert',6)),
          'C9B3':write('C9B3',take(1,'cifar10',9),take(1,'bert',3)),
          'C3B9':write('C3B9',take(1,'cifar10',3),take(1,'bert',6)+take(5,'bert',3))}
 # Preserve exact accepted baseline byte-for-byte for the reuse and ample comparison.
 shutil.copyfile(OLD/'inputs/CB12_workload.csv',OUT/'inputs/BASE_workload.csv')
 methods=['SIA_P','SIA_AUX','CC_P','CC_CPU','CC_MEM','CC_FULL']
 # W1 is additionally run with namespaced IDs to expose deterministic tie sensitivity.
 cells=[(w,'RNEW','medium',m) for w in cohorts for m in methods]+[('BASE','AMPLE','medium',m) for m in methods]
 runner=(OLD/'run_scenario_v2.py').read_text()
 runner=runner.replace("cc_cifar_bert12_20261029_run1","cc_cb_validation_20261029_run1")
 start=runner.index('CELLS=');end=runner.index('\n',start);runner=runner[:start]+'CELLS='+repr(cells)+runner[end:]
 runner=runner.replace("CAPACITIES['RNEW']=(6.,125.)",f"CAPACITIES['RNEW']=(6.,125.)\n  CAPACITIES['AMPLE']={tuple(ample)!r}")
 runner=runner.replace("'/6'","'/36'").replace('total=6','total=36').replace('ALL6_ENDED','ALL36_ENDED')
 (OUT/'run.py').write_text(runner,encoding='utf8')
 p=json.loads((OLD/'execution_preregistration_v2.json').read_text());p.update(cells=cells,gate_cells=[],capacities={'RNEW':[6.,125.],'AMPLE':ample},input_hashes={x.name:sha(x) for x in (OUT/'inputs').iterdir()},runner_hash=sha(OUT/'run.py'),frozen_unix=time.time(),source_trace_hashes=hashes,cohort_rows=cohorts,ample_derivation=needs,
  protocol='36 new sequential runs; W1/W5P/W10 each6+6, C9B3/C3B9 composition; accepted BASE rerun under analytically saturated AMPLE. W5 lacks6 BERT: first4 native+first2 trace10. Namespaced IDs preserve application; W1 renamed vs accepted baseline is a disclosed tie-order diagnostic, not independent repeat. All12 full work, first4 t0 then pairs300/600/900/1200. 6000 rounds. No tuning.',
  fixed_mechanism='AWS2GPU6CPU125GB, CIFAR0 and BERT28 each1GPU, full workload batch and work; fixed witness P/CPU/MEM/FULL quotas; baseline compares productive rates plus checkpoint, fixed GPU/batch. Not a scheduling benchmark.',
  parent_seal_hash=sha(OLD/'delivery_hashes.json'),abundant_rule='Each node capacity=2 times largest per-GPU saturation coordinate across all8 app/config registry rows. Independent of JCT.')
 save(OUT/'execution_preregistration_v2.json',p);save(OUT/'preregistration.json',p)
 shutil.make_archive(str(OUT/'source_snapshot'),'zip',OUT/'source')
 (OUT/'CURRENT_TASK.md').write_text('Authorized boundary-supported fixed2-job4quota mechanism +36 frozen six-method replication/composition/ample cells. Existing algorithm/model byte-identical. Resource gate2GiB RAM5GiB disk;60s6000rounds. Preserve failures. No tuning, no104/160. Finish independent audit/report then STOP.\n',encoding='utf8')
 print('FROZEN',OUT,ample,sha(OUT/'preregistration.json'),flush=True)
if __name__=='__main__':main()
