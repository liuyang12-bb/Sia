"""Independent arithmetic audit and delivery for the offline-only census."""
import ast, collections, csv, hashlib, itertools, json, math, shutil, sys, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_runs/sia_synergy/cc_resource_mixes_preflight_20261028_run1'
TREE=ROOT/'cc-joint-resource-study'
def read(n):return json.loads((OUT/n).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(n,x):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def table(headers,rows):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
q=read('all_queries.json');cov=read('coverage.json');pairs=read('pair_census.json');summary=read('summary.json');jobs=read('job_manifest.json')['jobs']
sources={}; matrices={}
for app,template in [('cifar10','alexnet_1.py'),('deepspeech2','deepspeech_1.py')]:
    p=TREE/'inputs/models'/template; sources[app]=dict(path=str(p),sha256=sha(p))
    tree=ast.parse(p.read_text(encoding='utf8'))
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='tput' for t in node.targets):
            v=node.value; matrices[app]=np.array(ast.literal_eval(v.args[0] if isinstance(v,ast.Call) else v),dtype=float)
appmap={r['name']:r['application'] for r in jobs}
def interp(app,c,m):
    # Independent sequential 1D interpolation; does not call the production bilinear helper.
    row=[np.interp(m,[20.83,62.5,125,187.5,250],v) for v in matrices[app]]
    return float(np.interp(c,[1,2,3,4,5,6,9,12,24],row))
class Audit(unittest.TestCase):
    def test_frozen_source(self):
        for p,h in read('protected_source_hashes.json').items():self.assertEqual(sha(ROOT/p),h,p)
    def test_unique_source_and_arrivals(self):
        orig=list(csv.DictReader((ROOT/'official-artifacts/sia-simulator/workloads/saturn/workload-1.csv').open()))
        for mix in ('M1','M2','M3'):
            rs=[r for r in jobs if r['cohort']==mix];self.assertEqual(len(rs),20);self.assertEqual(len({r['name'] for r in rs}),20)
            for i,r in enumerate(rs):
                for k in ('name','application','batch_size','num_replicas'):self.assertEqual(r[k],orig[r['original_source_row_index']][k])
                self.assertEqual(r['time'],0 if i<4 else 300*(1+(i-4)//2))
    def test_independent_all_response_values(self):
        for x in q:
            a=appmap[x['job']]; ratio=interp(a,x['source_cpu'],x['source_ram'])/interp(a,x['baseline_cpu'],x['baseline_ram'])
            self.assertTrue(math.isclose(x['rho'],math.sqrt(ratio),rel_tol=1e-12))
    def test_full_configuration_domain(self):
        for mix in ('M1','M2','M3'):
            names={r['name'] for r in jobs if r['cohort']==mix}
            for cap in ('R01','R11'):self.assertEqual(sum(v['job'] in names and v['capacity']==cap for v in cov),80)
    def test_witness_capacity_support_and_donor(self):
        for p in pairs:
            if not p['positive_supported_pairs']:continue
            b=p['best'];cpu=12 if p['capacity']=='R01' else 6
            self.assertLessEqual(b['cpu_total'],cpu+1e-9);self.assertLessEqual(b['ram_total'],187.5+1e-9)
            self.assertTrue(all(not(v['numerator_clipped'] or v['denominator_clipped']) for v in b['points']))
            self.assertTrue(math.isclose(sum(b['individual_fraction_rate_delta']),b['net_fraction_rate'],abs_tol=1e-15))
            self.assertGreater(b['net_fraction_rate'],0)
    def test_proportional_coverage_gap_not_hidden(self):
        for s in summary:self.assertLess(s['proportional_supported'],s['job_configs'])
    def test_gpu_reversal_independent_arithmetic(self):
        witnesses=read('broader_gpu_order_audit.json')['reversals'] if (OUT/'broader_gpu_order_audit.json').exists() else [dict(capacity='R01',job='cifar10-0',configs=['aws:1:1','aws:1:2'],contract='equal_total_cpu_ram',low=[[12.,187.5],None],high=[[6.,125.],None])]
        for witness in witnesses:
            diffs=[]
            for total,old_delta in (witness['low'],witness['high']):
                rr=[]
                for cid in witness['configs']:
                    g=int(cid.split(':')[-1]);c,m=(total[0]/g,total[1]/g) if witness['contract']=='equal_total_cpu_ram' else total
                    v=next(x for x in q if x['capacity']==witness['capacity'] and x['job']==witness['job'] and x['config']==cid and x['cpu']==c and x['ram']==m)
                    self.assertFalse(v['numerator_clipped'] or v['denominator_clipped'])
                    rate=next(x['offline_rate']['raw'] for x in cov if x['job']==witness['job'] and x['config']==cid)
                    rr.append(rate*math.sqrt(interp(appmap[v['job']],v['source_cpu'],v['source_ram'])/interp(appmap[v['job']],v['baseline_cpu'],v['baseline_ram'])))
                diffs.append(rr[0]-rr[1])
                if old_delta is not None:self.assertTrue(math.isclose(diffs[-1],old_delta,rel_tol=1e-10))
            self.assertLess(diffs[0]*diffs[1],0)

if __name__=='__main__':
    with (OUT/'independent_tests.log').open('x',encoding='utf8') as f:
        result=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Audit))
    if not result.wasSuccessful():raise RuntimeError('Independent audit failed')
    # Broader, finite GPU-order check including both-coordinate changes and equal-total-resource comparisons.
    by=collections.defaultdict(dict); rates={(v['job'],v['config']):v['offline_rate']['raw'] for v in cov}
    for x in q:
        if not(x['numerator_clipped'] or x['denominator_clipped']):by[x['capacity'],x['job'],x['config']][x['cpu'],x['ram']]=x
    reversals=[]; comparisons=0
    for cap in ('R01','R11'):
        for name in sorted(appmap):
            configs=sorted(k[2] for k in by if k[:2]==(cap,name))
            for ca,cb in itertools.combinations(configs,2):
                for contract in ('equal_per_gpu_quota','equal_total_cpu_ram'):
                    a,b=by[cap,name,ca],by[cap,name,cb]
                    if contract=='equal_total_cpu_ram':
                        ga,gb=int(ca.split(':')[-1]),int(cb.split(':')[-1]);a={(c*ga,m*ga):x for (c,m),x in a.items()};b={(c*gb,m*gb):x for (c,m),x in b.items()}
                    common=sorted(a.keys()&b.keys()); values=[(z,rates[name,ca]*a[z]['rho']-rates[name,cb]*b[z]['rho']) for z in common]
                    comparisons+=len(values)
                    lo=min(values,key=lambda v:v[1]) if values else None;hi=max(values,key=lambda v:v[1]) if values else None
                    if lo and lo[1]<-1e-9 and hi[1]>1e-9:reversals.append(dict(capacity=cap,job=name,configs=[ca,cb],contract=contract,low=lo,high=hi))
    save('broader_gpu_order_audit.json',dict(comparison_points=comparisons,reversals=reversals,interpretation='Fixed epoch0 declared batch only; no reversal here is not proof against adaptive-batch or other-epoch coupling.'))
    stats=[]
    for s in summary:
        ps=[p for p in pairs if p['mix']==s['mix'] and p['capacity']==s['capacity']]
        for f in ('CPU','MEM','FULL'):
            positive=[p for p in ps if p['factor']==f and p['positive_supported_pairs']]
            gains=[p['best']['relative_gain'] for p in positive]
            mixed=[p for p in positive if len(set(p['applications']))==2]
            stats.append(dict(mix=s['mix'],capacity=s['capacity'],factor=f,positive_states=len(positive),
                mixed_application_positive_states=len(mixed),median_gain=float(np.median(gains)) if gains else None,
                min_gain=min(gains) if gains else None,max_gain=max(gains) if gains else None,
                first_mixed_witness=mixed[0] if mixed else None))
    save('opportunity_distribution.json',stats)
    counter=collections.Counter()
    for v in cov:
        p=v['proportional']
        if p['numerator_clipped'] or p['denominator_clipped']:counter[v['application'],v['config']]+=1
    boundaries=[]
    for cap in ('R01','R11'):
        for app in ('cifar10','deepspeech2'):
            vs=[x for x in q if x['capacity']==cap and appmap[x['job']]==app]
            boundaries.append(dict(capacity=cap,application=app,queries=len(vs),
                numerator_cpu=sum(not 1<=x['source_cpu']<=24 for x in vs),numerator_ram=sum(not 20.83<=x['source_ram']<=250 for x in vs),
                denominator_cpu=sum(not 1<=x['baseline_cpu']<=24 for x in vs),denominator_ram=sum(not 20.83<=x['baseline_ram']<=250 for x in vs)))
    save('boundary_counts.json',boundaries)
    # Read-only historical support: not a counterfactual for these new cohorts.
    old=ROOT/'research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1'
    history={str(p.relative_to(ROOT)):sha(p) for p in (old/'completed_analysis_20261028_run1').iterdir() if p.is_file()}
    save('historical_evidence_hashes.json',history)
    with (OUT/'task_configuration_evidence.csv').open('x',encoding='utf8',newline='') as f:
        fields=['capacity','job','application','config','template','phi_c','phi_m','proportional','best','feasible_points','supported_points']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for v in cov:w.writerow({k:json.dumps(v[k],ensure_ascii=False) if isinstance(v[k],dict) else v[k] for k in fields})
    jobtables=[]
    for mix in ('M1','M2','M3'):
        jobtables.append('### '+mix+'\n\n'+table(['顺序','作业','到达秒','源CSV行','GPU demand','batch'],[[r['source_row_index'],r['name'],r['time'],r['source_line'],r['num_replicas'],r['batch_size']] for r in jobs if r['cohort']==mix]))
    mem=next(x['first_mixed_witness'] for x in stats if x['mix']=='M2' and x['capacity']=='R01' and x['factor']=='MEM')
    cpu=next(x['first_mixed_witness'] for x in stats if x['mix']=='M2' and x['capacity']=='R01' and x['factor']=='CPU')
    def witness(p):
        if p is None:return '没有正收益的跨应用见证。'
        b=p['best'];return f"{p['jobs']}，{p['gpu']}各1GPU：配额{[(x['cpu'],x['ram']) for x in b['points']]}，速度比{[round(x,6) for x in b['speed_ratios']]}，净归一化进度率变化{b['relative_gain']*100:.4f}%；CPU总量{b['cpu_total']}、RAM总量{b['ram_total']}GB。分子分母均在支持域。"
    report=f'''# 三种负载的CPU/RAM离线机会预检

结论：三种负载均有现有网格内、节点容量合法、计入donor减速后仍为正的CPU及RAM互补机会；扩展为相同总CPU/RAM预算比较后，也找到GPU数量排序反转。**当前不建议直接启动原定42格**：完整候选域的参考分母支持不闭合。辅助资源交换和GPU耦合的局部机制见证通过，但完整域严格支持门槛尚未通过。未运行训练、调度器或42格。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE

## 1. 冻结范围与作业选择

当前算法为cc-joint-resource-study中的原CC-TAIL；其实际文件完整hash见protected_source_hashes.json。源码目录没有.git，以实际文件快照版本为依据；未创建分支、提交或改算法。源workload-1.csv有160作业，CIFAR10=134、DeepSpeech2=10。按原始行顺序分别取前N条，无组内复制；三组是有意嵌套的受控混合比例，不能当作三个独立自然trace。

M1=18:2、M2=14:6、M3=10:10。按预注册floor规则交错；首4作业0秒到达，其后每300秒2个直到2400。保留name/application/num_replicas/batch及原始arrival和行号；总工作量由Artifact固定epoch声明保留。新M1到达排列与历史20作业不同，不宣称逐轮复现旧M1。

两容量：R01每节点12CPU/187.5GB；R11每节点6CPU/187.5GB。AWS和DGX-ext各节点2GPU，均为合成规格。Medium alpha=.5、R0、固定参考(3,62.5)、插值和E-legacy均沿用。CPU实际执行网格包括既有8点，另包含比例点；RAM包含既有比例93.75点。没有扩网格。

## 2. 数据和计算口径

遍历全部作业×AWS/DGX-ext×1/2GPU×两容量的所有可落地网格点。JobSpec纯数据结构，未创建Job或Cluster、未调用step/optimize_policy。固定声明batch，读取Artifact epoch0梯度/step/sync，按原累积和gain公式计算离线执行率；保留atomic/实际batch、accum、校准和完整工作量，见coverage.json。不把这些离线真实profile值灌入在线策略，也不把它们当作未知在线模型预测。

rho=[F(3c/(3phi_c),62.5m/(62.5phi_m))/F(3/phi_c,62.5/phi_m)]^.5；即实际分子为(c/phi_c,m/phi_m)，因固定基准为3CPU/62.5GB。容量比例点是R01(6,93.75)、R11(3,93.75)，不是固定归一化点，rho不要求等于1。源坐标范围CPU[1,24]、RAM[20.83,250]；正机会只接受分子分母均不裁剪的点。

净收益=Σraw_i*(rho_new-rho_prop)/W_i；没有相加不同应用原始Goodput。另算300秒内按总工作量归一化且封顶1的进度增量。它不是ΣJCT优化证明，不能由它断言训练完成更早。相同状态固定GPU/batch，假设零辅助资源调整成本，与既有执行模型一致；真实硬件开销未测。

CPU-only固定RAM比例，RAM-only固定CPU比例，FULL同时枚举。所有190个不同job pair×两GPU类型共380个静态共驻状态逐一检查；只统计可容纳的两单GPU任务。它们是可能共驻的离线组合，不是已观察到的实际并发。

## 3. 机会数量与覆盖

{table(['负载','容量','CPU正收益状态/380','RAM正收益状态/380','FULL正收益状态/380','比例点受支持配置/80'],[[s['mix'],s['capacity'],s['positive_jobpair_gpu_states']['CPU'],s['positive_jobpair_gpu_states']['MEM'],s['positive_jobpair_gpu_states']['FULL'],str(s['proportional_supported'])+'/80'] for s in summary])}

计数表示作业对×GPU类型，不是独立实验次数。扩大DeepSpeech比重后RAM正收益状态36→84→100，但这主要是两类应用配对数增加（18×2、14×6、10×10），不能把计数增加本身解释为更大JCT收益。CPU状态包括同类不同batch的速度差异，非全为跨应用互补；详见opportunity_distribution.json的mixed_application_positive_states。

CPU首个跨应用见证（按源顺序，不按最大收益选）：{witness(cpu)}

RAM首个跨应用见证：{witness(mem)}

RAM接收CF并非无损拿走DS资源，DS确实减速。原始矩阵只有源模板；CIFAR→AlexNet、DS→DeepSpeech、phi、迁移、归一化及乘Sia速率均是合成假设。支持域内不等于已被目标硬件验证。

## 4. 不能隐藏的两个阻塞

**参考分母越界：所有CIFAR的dgx-ext:1:2配置。** CPU参考坐标3/phi_c=0.9408050181，小于源矩阵最低1；RAM参考约35.0001GB在域内。这与容量无关，所以R01/R11都发生。M1/M2/M3分别18/14/10个配置受影响，其他比例点支持。该配置的分子可以在域内，也不能把其rho叫作完全有数据支持；禁止只按分子判合格。各资源、分子/分母全域计数见boundary_counts.json。

不删除该合法GPU配置，不改phi/参考点，不把它作为严格无裁剪主证据。沿用E-legacy可以运行探索性实验，但需要明确允许裁剪；这与前述“完整域支持闭合”的严格门槛不同，本轮未自行放宽。

**GPU配置耦合存在。** 初步相同每GPU配额的一维检查未发现反转；扩展检查相同每GPU配额及相同总CPU/RAM预算、两维配额全部支持点，共{comparisons}个比较点；发现{len(reversals)}个配置对反转见证。应以broader_gpu_order_audit.json为最终结论，早期gpu_ranking_reversals.json仅覆盖较窄的一维定义。见证均为同任务AWS单/双GPU的相同总辅助预算比较，不能称为已证明跨GPU类型排序反转。只检查epoch0及声明batch，不声称完整训练路径的JCT改善。

## 5. 持续竞争与JCT边界

三负载到达密度相同，资源互补组合更常见；未仿真，不能声称实际持续排队、同一对必然共驻或收益持续到完成。离线300秒正收益只是固定状态条件证据。历史相同模板/网格确实触发过RAM125/62.5的训练差异及CF提前42秒，来源为cc_joint_30cell_diagnosis_20261028_run1/REPORT.md；不是这些新混合负载的运行结果。新的场景完整训练前仍需检查持续机会和重配置成本是否可兑现。

现有倒数ETA目标曾选择短任务节省约20秒、长任务预测多约3433秒的交换；新数据不能自动修复目标偏差。即便机会数量更多，FULL也可能不如CPU-only，必须保留这种结果。

## 6. 验收与建议

7项独立审计通过：源文件未改、作业与到达提取一致、所有rho用独立逐轴插值重算、完整80配置覆盖、净收益含donor且容量/支持合法、分母缺口未隐藏、GPU排序反转独立复算。初始预检使用实际oracle；复核未调用oracle插值函数。training_runs=0、scheduler_solves=0、未运行42格。

建议先明确现有参考分母边界在下一轮主结论中的角色：完整E-legacy探索性结果可以保留裁剪并逐项披露，但不能称为全域源数据支持；严格主验证需要研究者审定边界数据方案。**值得推进CPU/RAM感知及GPU数量耦合验证，但此刻不直接启动完整42格；先解决完整域分母支持缺口，不自动换参考点、删合法配置或改phi。** 没有自动改模型、删作业或启动实验。

## 7. 完整作业清单

{chr(10).join(jobtables)}

## 8. 复现与产物

工作区D:/科研启航/Sia；Python .venv-sia-baseline/Scripts/python.exe，-X utf8 -B。离线脚本maintenance/preflight_resource_mixes_20261028.py、maintenance/audit_resource_mixes_20261028.py；输出目录必须全新，脚本拒绝覆盖。preregistration.json在查询前写入。全部原始查询all_queries.json、任务配置coverage.json/CSV、pair_census.json、GPU排序检查、manifest、独立测试、受保护文件hash均在本目录。delivery_hashes.json为最终交付校验。
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf8')
    for p in [ROOT/'maintenance/preflight_resource_mixes_20261028.py',ROOT/'maintenance/audit_resource_mixes_20261028.py']:
        shutil.copyfile(p,OUT/p.name)
    save('audit_validation.json',dict(status='OFFLINE_OPPORTUNITIES_VALID_FULL_BATCH_GATE_BLOCKED',tests=7,passed=7,training_runs=0,scheduler_solves=0,
        supported_cpu_memory_opportunities_all_cohorts=True,full_proportional_domain_supported=False,gpu_reversal_witnesses=len(reversals),algorithm_modified=False))
    (OUT/'CURRENT_TASK.md').write_text('COMPLETE_OFFLINE_ONLY. REPORT.md contains supported CPU/RAM opportunities and full-domain denominator gap. 42-cell batch NOT started or approved. Historical algorithms and results protected. Stop for review.\n',encoding='utf8')
    save('delivery_hashes.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()})
    print('7 independent checks passed. Full batch blocked; report:',OUT/'REPORT.md')
