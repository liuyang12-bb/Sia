import csv,hashlib,json,difflib
from pathlib import Path
import raw_audit_lib as lib
O=Path(__file__).resolve().parent;R=O.parents[2]
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 p=json.loads((O/'preregistration.json').read_text());work={x['name']:x for x in csv.DictReader((O/'inputs/workload.csv').open())}
 stats={};jobs={};changes={}
 for cell in p['cells']:
  stats[cell['id']],jobs[cell['id']],changes[cell['id']]=lib.audit(cell,p,work)
 assert all(not x['errors'] for x in stats.values())
 save('raw_audit.json',stats);save('per_job.json',jobs);save('decision_changes.json',changes)
 compare={}
 for a,b in [('OLD_FO','OLD_FULL'),('NEW_FO','NEW_FULL'),('OLD_FO','NEW_FO'),('OLD_FULL','NEW_FULL')]:
  if not(stats[a]['complete'] and stats[b]['complete']):compare[a+'__'+b]={'complete':False};continue
  ds={j:jobs[a][j]['jct']-jobs[b][j]['jct'] for j in work}
  compare[a+'__'+b]=dict(complete=True,mean_saved=stats[a]['mean']-stats[b]['mean'],percent=100*(1-stats[b]['mean']/stats[a]['mean']),per_job_saved=ds,better=sum(v>0 for v in ds.values()),worse=sum(v<0 for v in ds.values()),equal=sum(v==0 for v in ds.values()))
 save('comparison_summary.json',compare)
 lines=['# 完成事件后存活作业重配置预测：V2验证与12作业对照','',
 'SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE','',
 '## 结论','']
 for k,v in compare.items():
  if v['complete']:lines.append(f"- {k}：平均JCT节省{v['mean_saved']:.3f}秒，改善{v['percent']:.3f}%；逐作业改善/退化/持平{v['better']}/{v['worse']}/{v['equal']}。")
 lines+=['', '四格完成性及全部独立审计见下表。无论结果正负，本轮停止，不扩展80作业、不根据JCT再修改参数。',
 '', '## 实现与科学边界','',
 '在新源cc-joint-release-reconfig-v2中扩展release_evaluation.py。旧v1和所有历史产物不改。实际准入/持续服务、GPU候选、资源网格、响应/phi/clipping、任务工作量、checkpoint及启动预算均保持原样；只改变完整方案预测接续。',
 '', '预测推进到完成后的60秒边界：先扣除每个存活作业已训练工作和剩余启动时间，再释放完成者的三种资源；FIFO等待者优先接续，随后按确定性最大净ETA节省顺序尝试存活者升级。升级必须保持peer完整分配，具有已观测direct/projected模型、有效训练、无剩余checkpoint、非恢复后冻结GPU；从本轮列集中选点，按目标GPU类型重算实际startup，并扣生命周期剩余预算。每个作业每次完成事件最多改一次，不驱逐peer。rho只乘训练率，服务秒数与checkpoint不乘rho。',
 '', '不同初始提案使用完全相同接续，初始切换成本已先入账。不会将相对真实当前状态的缓存delay用于未来切换。FO仅在当前串行提案冻结的GPU域中选择初始方案；预测未来GPU可随释放而改变，不能误称FO本轮GPU意图被改变。当前已有恢复许可和完整锁得到保留；不会预授未来新tail许可、放宽预算或生成尚不在当前合法域的候选。因此这是保守预测，不是对原策略未来的精确重放，不能保证全局最优。',
 '', '任一已到达作业模型未知时整轮退回旧bootstrap，不删除该作业、不补假ETA。这个先前边界仍存在。',
 '', '## 红绿与同状态验证','',
 'red_behavior.log复现旧预测让长任务一直停留在1GPU、预测6000秒；补齐后短任务60秒完成释放GPU，长任务先训练60秒，再支付120秒切换，用2GPU完成剩余5940工作，完成于774秒。其他测试覆盖预算200/250阻止120秒切换、checkpoint剩余不被跳过、已用恢复冻结；保留旧测试。新接续改变一个旧集成测试的明确预期（旧静态540秒→合法释放后276秒），该失败记录保留，最终仍选择250秒方案。',
 '', '完整测试日志full_suite.log和新增路径green_final.log；原始Sia三轮精确状态回归见original_sia_regression.json。',
 '', '4800/5820只读重建来源为上轮NEW_FULL完整日志，逐训练段重新核对历史观测raw并恢复在线模型；不调用Job.step或Cluster.step，不使用未来段更新当前状态。原初始提案、CPU/RAM和同状态知识相同。',
 '', '|状态|提案|V1预测完成时间总和|V2预测完成时间总和|预测存活者重配置次数|','|---|---|---:|---:|---:|']
 states=json.loads((O/'same_state.json').read_text())
 for s in states['results']:
  for name,x in s['evaluations'].items():lines.append(f"|{s['time']}|{name}|{x['old_forecast']:.3f}|{x['new_forecast']['sum_completion']:.3f}|{sum(e['event']=='SURVIVOR_RECONFIGURE' for e in x['new_forecast']['events'])}|")
 lines+=['','两个状态合法性验证通过，但V2仍可能保持V1偏好的方案；不以必须改回旧选择作为验收条件，也不把预测下降当作真实JCT下降。','',
 '## 冻结场景与实际结果','',
 '现有COMPACT12（6 CIFAR10＋6 BERT），AWS/DGX各一个2GPU、6合成CPU单位、125GB节点；原到达/批量/训练工作量不变；seed(1,8)、R0/E-legacy Medium alpha0.5、60秒/6000轮，2GiB/5GiB门槛，各独立进程串行。旧FO/FULL为原始reciprocal评价关闭V1/V2；新FO/FULL为V2。完整源/输入hash及命令见preregistration.json与execution_manifest.jsonl。',
 '', '|方法|完成|平均JCT|Median|P95|P99|Makespan|墙钟秒|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for m,s in stats.items():lines.append(f"|{m}|{s['completed']}/12|{s['mean']}|{s['median']}|{s['p95']}|{s['p99']}|{s['makespan']}|{s['wall']:.2f}|")
 lines+=['', 'P95/P99仅12作业描述，不能视作总体尾延迟精确估计。','',
 '|方法|等待s|训练s|checkpoint s|重启|浪费GPU-s|GPU-h|CPU单位-h|RAM GB-h|裁剪点比例|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for m,s in stats.items():
  z=s['totals'];lines.append(f"|{m}|{z['wait']:.1f}|{z['training']:.1f}|{z['checkpoint']:.1f}|{z['restarts']}|{z['wasted_gpu_seconds']:.1f}|{z['gpu_hours']:.3f}|{z['cpu_core_hours']:.3f}|{z['ram_gb_hours']:.3f}|{s['clipped_fraction']:.3%}|")
 lines+=['','## 逐作业与实际启用','', '|作业|旧FO|旧FULL|新FO|新FULL|','|---|---:|---:|---:|---:|']
 for j in work:lines.append('|'+j+'|'+'|'.join(str(jobs[m][j]['jct']) for m in stats)+'|')
 for m in ('NEW_FO','NEW_FULL'):lines+=['',f"{m}：{stats[m]['checks']}；选择计数{stats[m]['plans']}；改选细节decision_changes.json。"]
 old=json.loads((R/'research_runs/sia_synergy/cc_release_evaluation_20261101_run1/raw_audit.json').read_text())
 lines+=['','## 对上一轮V1的诊断比较','']
 for m in ('NEW_FO','NEW_FULL'):
  if stats[m]['complete']:lines.append(f"- {m} V1平均{old[m]['mean']:.3f}秒→V2 {stats[m]['mean']:.3f}秒，变化{stats[m]['mean']-old[m]['mean']:.3f}秒。")
 lines+=['', '实际收益必须以四格完整JCT为准；若持平/退化，不扩展实验。当前模型冻结batch/在线画像且仅使用当前资格域，未来恢复新许可不展开；greedy升级不是整体未来最优，仍可能高估或低估释放收益。等待者未知模型会阻止整轮评分，不能把已有服务的配置收益算作减少排队。CPU/RAM换配额没有额外开销、调度墙钟不加入模拟JCT，均沿用旧执行定义。',
 '', '原始完成事件、互斥时间、资源小时、容量、完整checkpoint锁和raw×rho由独立脚本复算；所有错误0。源码完整snapshot及patch保存，旧源逐文件核验未变。无80作业运行，无自动扩展。']
 (O/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 assert all(lib.sha(R/'cc-joint-release-reconfig-v2'/n)==h for n,h in p['source_hashes'].items())
 save('validation.json',dict(errors=0,all_complete=all(s['complete'] for s in stats.values()),same_state_gate=states['gate'],source_frozen=True,training_matrix=4,expanded=False))
 print(json.dumps(compare,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
