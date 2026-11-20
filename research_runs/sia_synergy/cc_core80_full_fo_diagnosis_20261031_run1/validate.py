"""Validate new read-only diagnostic arithmetic against closed run artifacts."""
import json,math,hashlib,sys
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-core80-full-lex-repair-v8'
S=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run5'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
rows=read(O/'per_job_comparison.json');methods=read(O/'method_mechanisms.json');states=read(O/'structural_states.json');late=read(O/'late_budget_states.json')
metrics=read(S/'audit/raw_recomputed_metrics.json');registry={x['key']:x for x in read(T/'inputs/registry.json')}
assert len(rows)==80 and len({r['job'] for r in rows})==80
for m in ('FULL','FO'):
 for k in ('jct','wait','checkpoint','training'):
  total=sum(r[k+'_'+m] for r in rows)
  expected=80*metrics['dual_medium_'+m]['mean_jct'] if k=='jct' else metrics['dual_medium_'+m]['totals'][k]
  assert math.isclose(total,expected,abs_tol=1e-6)
 assert methods[m]['optional_selected']==0
 assert math.isclose(sum(v['training'] for v in methods[m]['application_execution'].values()),metrics['dual_medium_'+m]['totals']['training'],abs_tol=1e-6)
 assert methods[m]['application_execution']['bert']['time_weighted_rho']==1.
assert sum(r['jct_saved'] for r in rows)==5742
assert methods['FO']['serial_rounds']==302 and methods['FO']['serial_gpu_violations']==0
assert len(states)+len(late)<=6
support=[]
def point(app,cid,cpu,ram):
 ref=registry[app+'/'+cid];a=cpu/ref['phi_c'];b=ram/ref['phi_m'];base=(3/ref['phi_c'],62.5/ref['phi_m'])
 return dict(requested_cpu=a,requested_ram=b,reference_cpu=base[0],reference_ram=base[1],
  numerator_inside=1<=a<=24 and 20.83<=b<=250,denominator_inside=1<=base[0]<=24 and 20.83<=base[1]<=250)
for x in late:
 c=x['current'];b=x['alternatives'][0];g=int(b['config'].split(':')[-1]);free=b['node_free_excluding_this_job']
 assert all(need<=cap+1e-8 for need,cap in zip((g,3*g,62.5*g),free))
 assert x['delay_ledger']['delay_charged']==250 and x['filters']['rejection_counts']['SERVICE_DELAY_BUDGET']>0
 current=point('deepspeech2',c['config']['config_id'],c['cpu_per_gpu'],c['ram_gb_per_gpu'])
 alt=point('deepspeech2',b['config'],3,62.5)
 support.append(dict(method=x['method'],job=x['job'],time=x['time'],current=current,alternative=alt,
  ratio=b['proxy_ratio_to_current'],direct_profile=b['profile']['goodput_source']=='direct_profile'))
 assert current['numerator_inside'] and current['denominator_inside'] and alt['numerator_inside'] and alt['denominator_inside']
frozen=read(S/'final_integrity_hashes.json');needed=read(O/'provenance.json')['inputs']
assert all(sha(S/n)==v==frozen[n] for n,v in needed.items())
p=read(S/'preregistration.json');assert all(sha(T/n)==v for n,v in p['source_hashes'].items())
result=dict(status='PASS',paired_jobs=80,exact_net_jct_task_seconds_saved=5742,
 exclusive_time_decomposition_verified=True,fo_gpu_intent_preserved=True,structural_states=len(states)+len(late),
 source_files_verified=len(p['source_hashes']),original_input_hashes_verified=len(needed),late_state_support=support,
 simulation_invoked=False,new_algorithm_tests_claimed=False)
with (O/'validation.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False))
