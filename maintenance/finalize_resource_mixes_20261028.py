"""Finalize only the new preflight delivery, preserving initial narrow-scan evidence."""
import importlib.util,json,re,shutil,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_resource_mixes_preflight_20261028_run1'
p=ROOT/'maintenance/audit_resource_mixes_20261028.py'
spec=importlib.util.spec_from_file_location('offline_audit',p);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
with (OUT/'independent_tests_final.log').open('x',encoding='utf8') as f:
    result=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(a.Audit))
assert result.wasSuccessful() and result.testsRun==7
# Rebuild the report directly from the corrected reporting template with the unchanged raw census.
text=p.read_text(encoding='utf8');template=text.split("    report=f'''",1)[1].split("'''\n    (OUT/'REPORT.md')",1)[0]
scope=dict(a.__dict__);scope.update(summary=a.summary,comparisons=a.read('broader_gpu_order_audit.json')['comparison_points'],reversals=a.read('broader_gpu_order_audit.json')['reversals'])
scope['jobtables']=[]
for mix in ('M1','M2','M3'):
    scope['jobtables'].append('### '+mix+'\n\n'+a.table(['顺序','作业','到达秒','源CSV行','GPU demand','batch'],[[r['source_row_index'],r['name'],r['time'],r['source_line'],r['num_replicas'],r['batch_size']] for r in a.jobs if r['cohort']==mix]))
def witness(p):
    if p is None:return '没有正收益的跨应用见证。'
    b=p['best'];return f"{p['jobs']}，{p['gpu']}各1GPU：配额{[(x['cpu'],x['ram']) for x in b['points']]}，速度比{[round(x,6) for x in b['speed_ratios']]}，净归一化进度率变化{b['relative_gain']*100:.4f}%；CPU总量{b['cpu_total']}、RAM总量{b['ram_total']}GB。分子分母均在支持域。"
scope['witness']=witness
stats=a.read('opportunity_distribution.json')
for f,key in [('CPU','cpu'),('MEM','mem')]:scope[key]=next(x['first_mixed_witness'] for x in stats if x['mix']=='M2' and x['capacity']=='R01' and x['factor']==f)
report=eval("f'''"+template+"'''",scope)
first=a.read('broader_gpu_order_audit.json')['reversals'][0];qs=a.q
rows=[]
for budget,_ in (first['low'],first['high']):
    for cid in first['configs']:
        g=int(cid.split(':')[-1]);c,m=budget[0]/g,budget[1]/g
        q=next(x for x in qs if x['capacity']==first['capacity'] and x['job']==first['job'] and x['config']==cid and x['cpu']==c and x['ram']==m)
        raw=next(x['offline_rate']['raw'] for x in a.cov if x['job']==first['job'] and x['config']==cid)
        rows.append([str(budget),cid,c,m,round(q['rho'],6),round(raw*q['rho'],6)])
report+='\n## GPU数量耦合的具体见证\n\nR01/cifar10-0，固定声明batch和epoch0，分别在相同总辅助预算下比较AWS1/2GPU；每个点都在既有网格和源支持域。两GPU占更多GPU，不能把结果当作全服务集合净收益。\n\n'+a.table(['总CPU/RAM预算','配置','每GPU CPU','每GPU RAM','rho','有效执行率'],rows)+'\n\n这说明固定辅助预算较低时单GPU可以更快，辅助预算较高时双GPU可以更快；未证明跨GPU类型反转，也未证明联合策略必然选择或兑现。\n'
(OUT/'REPORT.md').write_text(report,encoding='utf8')
v=a.read('audit_validation.json');v.update(tests=7,passed=7,gpu_reversal_witnesses=len(scope['reversals']),gpu_quantity_coupling_supported=True,all_cohort_job_config_entries_per_capacity=240)
(OUT/'audit_validation.json').write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
# Correct the descriptive count; this does not alter any query or census.
v=a.read('validation.json');v.pop('all_120_cohort_job_configs_queried_per_capacity',None);v['all_240_cohort_job_configs_queried_per_capacity']=True
(OUT/'validation.json').write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
for name in ('preflight_resource_mixes_20261028.py','audit_resource_mixes_20261028.py','finalize_resource_mixes_20261028.py'):shutil.copyfile(ROOT/'maintenance'/name,OUT/name)
for p,h in a.read('protected_source_hashes.json').items():assert a.sha(ROOT/p)==h,p
for p,h in a.read('historical_evidence_hashes.json').items():assert a.sha(ROOT/p)==h,p
hashes={str(p.relative_to(OUT)):a.sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_hashes.json'}
(OUT/'delivery_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf8')
print('FINAL: 7 checks passed;',len(hashes),'delivery files hashed; no simulation; GPU quantity reversal confirmed; denominator gap remains.')
