import collections,csv,datetime,hashlib,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_runs/sia_synergy/cc_core80_8h_20261030_run1'
TREE=ROOT/'cc-joint-core80-8h'
BASE=ROOT/'research_runs/sia_synergy/cc_clean_ab_20261030_run3'
START=datetime.datetime.fromisoformat('2026-10-30T18:49:54+00:00').timestamp()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def hashes(p):return {f.relative_to(p).as_posix():sha(f) for f in p.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc'}
def save(p,x):
 with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,ensure_ascii=False)
def main():
 assert not OUT.exists() and not TREE.exists()
 OUT.mkdir();(OUT/'inputs').mkdir();(OUT/'tests').mkdir();(OUT/'cells').mkdir();(OUT/'preflight').mkdir()
 save(OUT/'budget.json',dict(start_unix=START,deadline_unix=START+8*3600,formal_stop_unix=START+7*3600,
     start_utc='2026-10-30T18:49:54Z',deadline_utc='2026-10-31T02:49:54Z',
     rule='8h includes preparation. All workers must stop before final audit/report; no simulation resumed from old104. Wall time interruption is COMPUTE_TIMEOUT, not simulated JCT censoring.'))
 (OUT/'CURRENT_TASK.md').write_text('''# Authorized current task:8-hour80-job core evidence package

Start2026-10-30 18:49:54UTC; hard deadline2026-10-31 02:49:54UTC. Final audit reserve1hour; formal stop01:49:54UTC. Do not resume the historical104/160 experiment. New root cc_core80_8h_20261030_run1; isolated source cc-joint-core80-8h. Base=clean_ab_run3/frozen_source,291 exact files. Preserve current algorithms/official sources/historical results. New opt-in scale/interface/logging adapters only, no objective/grid/response tuning.

80 real declarations:48CF+16DS+16BERT, fixed trace1..10 source-row selection, interleaveCF,CF,CF,DS,BERT. First40at0, next20at600, final20at1200.20GPUs:AWS4/DGX8/RTX8, one node each. Main perGPU3CPU/62.5GB; Neutral service4cells and Medium resource7cells (actualSiaP,actualSiaAux,CC-P,CPU,MEM,FULL,FO). May reuse SiaP Neutral only after exact equivalence proof. Optional CPUonly2 and RAMonly2 cells only as complete pairs and only if time permits. Independent processes serial;2GiB memory5GiB disk gates.

Do not shorten epochs/work, omit slow jobs, truncate queue/candidates, change p/lambda/response alpha/phi/clipping/grid, or insert L2 into Sia. Scale80 must be tested, not just remove guard. Purely equivalent unused-diagnostic elimination is permissible only with strict regression proof. Source/input/dependencies freeze before formal. Preflight must show actual pressure and feasible resource complementarity; static opportunity is not guaranteed JCT gain.

Resume: read budget.json, DESIGN.md, preregistration.json and execution_manifest.jsonl. Never duplicate running processes or overwrite partial cells. At deadline stop, retain interrupted logs, report only complete comparable results. Formal script and process session will be appended after setup.
''',encoding='utf8')
 seal=json.loads((BASE/'delivery_hashes_final_v2.json').read_text())
 for n,h in seal['files'].items():assert sha(BASE/n)==h,n
 shutil.copytree(BASE/'frozen_source',TREE,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
 basehash=hashes(BASE/'frozen_source');assert hashes(TREE)==basehash
 save(OUT/'baseline_source_hashes.json',basehash)
 pools=collections.defaultdict(list);sourcehash={}
 trace=ROOT/'official-artifacts/sia-simulator/workloads/saturn'
 for t in range(1,11):
  file=trace/f'workload-{t}.csv';sourcehash[str(file)]=sha(file)
  for index,row in enumerate(csv.DictReader(file.open(encoding='utf8'))):
   if row['application'] in ('cifar10','deepspeech2','bert'):
    pools[row['application']].append((t,index,row))
 selected={a:iter(pools[a][:n]) for a,n in [('cifar10',48),('deepspeech2',16),('bert',16)]}
 data=[]
 for cycle in range(16):
  for a in ('cifar10','cifar10','cifar10','deepspeech2','bert'):
   t,index,row=next(selected[a]);i=len(data)
   out=dict(row);out.update(name=f'{a}-{10000+i}',time=0 if i<40 else 600 if i<60 else 1200,
       original_name=row['name'],original_arrival=row['time'],source_trace=f'workload-{t}.csv',original_source_row_index=index,source_row_index=i)
   data.append(out)
 fields=list(data[0]);assert all(set(x)==set(fields) for x in data)
 with (OUT/'inputs/workload80.csv').open('x',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
 cfg=json.loads((TREE/'inputs/config_symmetric.json').read_text());gpus={'aws':4,'dgx-ext':8,'rtx':8}
 cfg['cluster'].update(nnodes={t:1 for t in gpus},ngpus_per_node=gpus,max_physical_nodes={t:1 for t in gpus})
 cfg['cluster']['capacity_scenarios']={'main':{t:dict(cpu_cores=3*g,ram_gb=62.5*g) for t,g in gpus.items()}}
 save(OUT/'inputs/config80.json',cfg)
 save(OUT/'input_origin.json',dict(rule='Trace1..10 stable source row, first48CF/16DS/16BERT, fixed3:1:1 interleave; NOT raw trace replay',source_trace_hashes=sourcehash,
    original_work_fields_unchanged=['application','num_replicas','batch_size'],new_identifiers='Deterministic globally unique app-ID; originalname/trace/row preserved',counts=dict(collections.Counter(x['application'] for x in data))))
 (OUT/'DESIGN.md').write_text('''#8h core80 protocol draft, fixed before any formal JCT

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE

Core11 unique runs: Medium dual main:SIA_P,FULL,SIA_AUX,PROP,CPU,MEM,FO; Neutral main:A00,A01,A10,A11. ActualSiaP can serve as Neutral external baseline ONLY if every proportional config has rho1 and real bounded Neutral/Medium trajectories are exact. Otherwise a distinct SIA_P_NEUTRAL becomes mandatory and is reported as12th core cell.
Optional pairs: CPU-limited(AuxRAM source-saturation supplied):PROP,CPU; RAM-limited(AuxCPU source-saturation supplied):PROP,MEM. Derived ample dimension uses fixed source matrices and R0 scales, not JCT. Start optional pair only if time reserve allows BOTH; no result-based selection. Existing all-ample negative control not repeated. No second-load or160-job claims.

A: same Sia fair score/cold start, jointly legal GPU/placement domain, lifecycle startup250 and exact checkpoints; onlyFIFO head/drain and300productive-second versus completion service differ. Qualification remains common across service expiry, zero-cost reissue preserves productive evidence. Not an officialSia variant; actualSia external has no L2.
B: same reciprocal physical ETA including actual startup, complete-universe score normalization BEFORE auxiliary factor filtering. FULL,PROP,CPU,MEM and FO all share FIFO/continuity. FO first optimizes proportional-domain GPU/service intent, freezes GPU type/count/nodecount/service set, then aux/placement; SIA_AUX is different: actualSia GPU+placement followed by existingECT aux allocation, noL2.

Seedprofiles(1,8),60s/6000rounds, full public epoch work and batch adaptation preserved. R0+Elegacy unchanged; Neutralalpha0/Medium0.5. All queries/clipping tracked.80job queue must be complete and capacity/lock/service ledgers tested. No stale column/rho fallback. Stop explicit on unprovable allocation or solver failure. No partial commits.

Budget: hard8h includes all preparation and final audit. Reserve final1h; no later formal process alive at that cutoff. Resource2GiB/5GiB. First SIA_P and FULL are real registered runs, not extra pilot data. Use only wall time/model size, never JCT, to forecast completion and optional-pair admission. No shortening work/observation windows or hiding computation failure as completed-onlyJCT.

Outcomes: actual completion JCT, perjob wait(first/later), checkpoint, productive, resource hours, restart/waste, actual co-residence/gain/donor loss, head drain, geometry changes, solver/column count and response clipping. Complete paired comparisons only. Simulated round-limit censor and wall-clockCOMPUTE_TIMEOUT distinct; incomplete group cannot support completeJCT claims. No percentages added across A/B. All jobs retained; leave-largest-beneficiary-out diagnostic only. Report conditional contribution and interaction, not assume all factors win.
''',encoding='utf8')
 print('PREPARED',OUT,len(basehash),dict(collections.Counter(x['application'] for x in data)),flush=True)
if __name__=='__main__':main()
