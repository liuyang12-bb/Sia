import csv,hashlib,json
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-competition-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
p=json.loads((O/'preregistration.json').read_text(encoding='utf8'));s=json.loads((O/'raw_audit.json').read_text(encoding='utf8'));pairs=json.loads((O/'comparison_summary.json').read_text(encoding='utf8'));jobs=json.loads((O/'per_job.json').read_text(encoding='utf8'))
assert all(sha(T/k)==v for k,v in p['source_hashes'].items())
assert all(sha(Path(k))==v for k,v in p['input_hashes'].items())
assert all(sha(O/k)==v for k,v in p['runner_hashes'].items())
for name in ('cc_competition_scenario_design_20261101_run1','cc_fo_full_mismatch_20261101_run1','cc_residual_capacity_20261101_run1'):
 d=O.parent/name;h=json.loads((d/'artifact_hashes.json').read_text(encoding='utf8'));assert all(sha(d/k)==v for k,v in h.items())
assert all(v['completed']==12 and not v['errors'] for v in s.values())
with (O/'per_job_utf8.csv').open('x',encoding='utf-8-sig',newline='') as f:
 fields=['cell','job','arrival','completion','jct','wait','training','checkpoint','restarts','gpu_hours','cpu_core_hours','ram_gb_hours'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for cell,rows in jobs.items():
  for j,v in rows.items():w.writerow(dict(cell=cell,job=j,**{k:v[k] for k in fields[2:]}))
lines=['\n## 最终解读\n','8/8格均12/12，正式流水线墙钟464.621秒（7.74分钟，不含门槛、测试和分析）。两项参数化适配/三轮精确回归测试通过（60.121秒）。旧343源文件不变；旧393项通过证据保留，本轮未再次全套运行。',
'\n|配对|FO均值|FULL均值|改善%|改善/退化/持平|\n|---|---:|---:|---:|---|']
for n,v in pairs.items():lines.append(f"|{n}|{s[n]['mean']:.3f}|{s[v['full']]['mean']:.3f}|{v['mean_improvement_pct']:.3f}|{v['better']}/{v['worse']}/{v['equal']}|")
lines += ['\n主场景净节省总JCT1817秒：等待减少720秒、训练减少1117秒、checkpoint增加20秒。7个作业改善、2个退化、3个持平；最大受益cifar10-3节省709秒，占净收益39.02%，bert-54退化704秒。不能把所有改善都归给240秒单任务CPU收益。',
'\n主场景Median4115→3825，P95 6912.9→6681.3，P99 7424.18→7030.66，makespan8452→8018秒。CPU小时15.7731→16.2967、RAM GB小时316.9945→328.4624、GPU小时8.0564→8.2217；重启16→15，wasted GPU秒2530→2860。JCT改善并不代表所有资源开销都降低。',
'\n到达对照的净收益5546秒主要来自bert-159提前6695秒，其余作业合计退化1149秒；去掉它的诊断性均值反而退化104.45秒。Neutral对照全部收益5980秒来自bert-159，其余11作业持平；不能将这些10%左右的差异归因于CPU/RAM响应。对称供给组优势8.07%，高于新主场景3.70%。资源错配使局部联合收缩可见，但没有放大全体平均优势。',
'\n主场景82.54%（FO）/81.17%（FULL）的所选查询至少一处分子/分母坐标触发E-legacy裁剪。该比例按所选查询计数，不是时间比例。AWS收缩比较点在支持范围内，整体仍依赖合成边界假设。',
'\n结论：新场景确认真实串行选择会错过一次合法GPU收缩，FULL选中并在训练推进中兑现；平均JCT正收益3.70%，幅度中等。机会只持续一轮，FO随后纠正，后续服务和batch/epoch自适应抵消部分局部收益。不能据此宣称CPU/内存联合优化带来巨大或普遍增量。',
'\n一次CSV导出命令使用了不支持的编码别名utf8-sig，在打开文件前退出；改为标准utf-8-sig导出，未触碰实验数据或重跑实验。',
'\n本轮完成并停止，不调参、不自动扩大实验。']
with (O/'REPORT.md').open('a',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
save('validation.json',dict(status='COMPLETE',cells=8,all_completed=True,capacity_lock_rho_execution_errors=0,source343_unchanged=True,inputs_frozen=True,runners_frozen=True,historical_seals_unchanged=True,algorithm_changed=False,gate_passed=True,independent_raw_event_recompute=True,tests=2))
(O/'CURRENT_TASK.md').write_text('完成并停止。完整12前缀门槛通过，240秒CIFAR2 AWS2→AWS1。8格均12/12，独立raw审计0。primary FO4096.9167 FULL3945.5改善3.6959%；其余10.2938/10.2684/8.0680%。主机会仅持续一周期，目标作业最终省21秒；Neutral收益集中bert159。报告REPORT.md。源码/输入/历史hash不变；正式墙钟464.621秒。不得自动扩大/调参/重启。',encoding='utf8')
save('artifact_hashes.json',{str(q.relative_to(O)):sha(q) for q in sorted(O.rglob('*')) if q.is_file() and q.name!='artifact_hashes.json'})
print('Final integrity and artifact sealing PASS')
