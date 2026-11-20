import difflib,hashlib,importlib.metadata,json,re,sys,zipfile
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-policy-rollout-v3';OLD=R/'cc-joint-release-reconfig-v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def load(n):return json.loads((O/n).read_text())
def save(n,x):
 with (O/n).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,ensure_ascii=False)
def main():
 log=(O/'full_suite.log').read_text(encoding='utf8');assert '\nOK\n' in log and '\nFAILED (' not in log,log[-500:]
 matched=re.search(r'Ran (\d+) tests in ([\d.]+)s',log);tests=int(matched[1]);assert tests==375
 assert load('original_sia_regression.json')['exact'] and load('original_full_regression.json')['exact']
 state=load('same_state.json');audit=load('state_validation_v2.json');assert not audit['errors']
 assert audit['ranking_gate'] is False
 baseline=load('baseline_hashes.json');assert all(sha(OLD/n)==h for n,h in baseline.items())
 current={p.relative_to(T).as_posix():sha(p) for p in T.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 changed=[n for n in baseline if baseline[n]!=current[n]];new=sorted(set(current)-set(baseline));patch=[]
 for n in changed+new:
  before=(OLD/n).read_text(encoding='utf8').splitlines(True) if n in baseline else []
  patch.extend(difflib.unified_diff(before,(T/n).read_text(encoding='utf8').splitlines(True),fromfile='a/'+n,tofile='b/'+n))
 (O/'source_changes.patch').write_text(''.join(patch),encoding='utf8')
 with zipfile.ZipFile(O/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
  for n in sorted(current):z.write(T/n,n)
 save('source_hashes.json',current);save('source_receipt.json',dict(files=len(current),changed=changed,new=new,old_source_unchanged=True,
  snapshot_sha256=sha(O/'source_snapshot.zip'),git='Complete source copy; not a Git worktree, no invented commit'))
 environment=dict(python=sys.version,executable=sys.executable,packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','pandas','cvxpy','cylp','psutil')})
 save('environment.json',environment)
 inputs={str(p):sha(p) for p in (T/'inputs').rglob('*') if p.is_file()};inputs[str(R/'research_runs/sia_synergy/cc_release_reconfig_20261101_run1/inputs/workload.csv')]=sha(R/'research_runs/sia_synergy/cc_release_reconfig_20261101_run1/inputs/workload.csv');save('input_hashes.json',inputs)
 result=dict(status='PARTIAL_REPAIR_BLOCKED_RANKING_GATE',unit_and_regression_tests=tests,test_seconds=float(matched[2]),
  technical_state_gate=True,ranking_matches=1,ranking_cases=2,formal_matrix_started=0,source_frozen=True,
  original_sia_exact_rounds=3,original_full_exact_rounds=3,baseline_unchanged=True)
 save('validation.json',result)
 for oldpath in (R/'research_runs/sia_synergy/cc_release_reconfig_20261101_run1',R/'research_runs/sia_synergy/cc_fork_diagnosis_20261101_run3'):
  assert all(sha(oldpath/n)==h for n,h in json.loads((oldpath/'artifact_hashes.json').read_text()).items())
 rows=[]
 for c in audit['comparisons']:
  rows.append(f"|{c['time']}|{c['predicted_sum_saved']:.3f}|{c['actual_sum_saved']:.3f}|{'通过' if c['ranking_matches_independent_fork'] else '未通过'}|")
 lines=['# 共用实际调度规则的完成预测V3：修复与两状态验收','',
 'SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE','',
 '## 结论','',
 '**已实现共用规则，但预测排序只通过1/2，按运行前DESIGN.md规定停止，没有启动新12作业四格对照。原FULL保持使用。**',
 '4800秒的主要误排已纠正：新预测判断改选会损失1312.613秒总完成时间，独立分叉实际损失1373秒。5820秒仍判断改选能节省100.272秒，而实际损失21秒。不能把实现共用状态机和375测试通过写成性能改善。',
 '', '## 修复了什么','',
 '新源cc-joint-policy-rollout-v3完整复制并核对V2实际336文件。新增policy_rollout.py；release_evaluation.py仅增加可选预测接口，未安装时仍走V2原逻辑；bundle_adaptation.py增加严格限于预测副本的合法初始假设分支，真实执行相同标签仍报错。',
 '每个初始方案在独立副本中使用原始policy/liveness/allocations及已观测模型；随后每60秒直接走原FO/FULL的完整候选生成、目标求解、checkpoint保护、启动成本和提交。完成、排空、历史预算、spare-resource恢复许可与用后冻结均由同一真实代码执行。不再固定当前候选域到永远，也不再用单任务ETA节省贪心替代原策略。',
 '预测只用当前已观测的Sia模型计算进度，延迟先扣、rho乘训练率一次。没有Job.step、真实吞吐/梯度或未来profile观测；未来到达不进入预测集合。拟合参数与梯度状态冻结，所以这仍是当前模型的完成近似。',
 '初始假设分支不是优化结果，明确不声称符合原倒数ETA目标；必须是同状态候选表中的合法全量分配，仍经过完整容量/锁审计，且只在_prediction_only副本内可用。实际原FULL的目标验收未放宽。',
 '实现修正还包括避免重入全局工程/稀疏适配上下文：虚拟优化复用外层已激活的等价包装，候选、优化和资源提交均真实调用。旧源不改。',
 '', '## 两状态排序（正值表示改选节省）','',
 '|状态|V3预测总完成秒节省|上轮独立分叉实际总JCT节省|排序验收|','|---|---:|---:|---|',*rows,
 '', '两个状态都来自同一冻结历史前缀；5820秒前缀包含4800秒历史改选，不等同于原FULL在拒绝4800改选后的后续状态。实际值引用已完成单次干预实验，不冒充本轮重新完整训练的结果。',
 '', '|状态/方案|V2预测剩余完成时间总和|V3预测|相同分支实际|','|---|---:|---:|---:|']
 old=json.loads((R/'research_runs/sia_synergy/cc_release_reconfig_20261101_run1/same_state_exact.json').read_text())
 for state_row,c in zip(state['results'],audit['comparisons']):
  oldrow=next(x for x in old['results'] if x['time']==state_row['time'])
  for p in state_row['evaluation']['plans']:
   if p['name'] not in ('LEGACY','MIN_STATIC_SUM_ETA'):continue
   n=p['name'];lines.append(f"|{state_row['time']}/{n}|{oldrow['evaluations'][n]['new_forecast']['sum_completion']:.3f}|{p['evaluation']['sum_completion']:.3f}|{c['actual_completion'][n]:.3f}|")
 lines+=['','## 剩余误差','',
 '5820秒预测对bert-107的改选收益为2118.486−1501.082=617.404秒，真实为1884−1359=525秒；对cifar10-5预测损失1115.887−598.756=517.132秒，真实损失978−432=546秒。两项合计将净收益高估121.272秒，足以把实际−21误排为+100.272；bert-159两方案相同，差值抵消。',
 '原先遗漏的恢复已经出现：5820 LEGACY预测在相对600秒为bert-107授予合法110秒额外预算，并支付真实120秒启动；不再让该作业永久停留在AWS双GPU。其预测剩余完成由V2的7526.013降至2118.486秒，接近实际1884秒。',
 '剩余偏差与冻结当前率、未来epoch/batch效率及模型拟合误差一致。例如既有独立日志中，5820 LEGACY CIFAR同配额的预测raw为26.227、实际后续时间加权37.372；不能用未来实测速度灌入在线决策来修正。没有做逐误差项全因子实验，不能分配精确误差贡献百分比。',
 '4800秒拒绝错误改选后不会到达这里完全相同的5820状态；因此不能根据此残余错误断言新完整运行必然退化。但排序验收尚未全部通过，也不能据此宣称新完整运行改善。按预先门槛不启动矩阵，未在看到结果后放宽门槛。',
 '', '## 测试与独立审计','',
 f"- 新增4项基础测试：延迟/训练分离、周期内精确完成、未知率停止、假设初始分配权限与非法资源拒绝。红测缺API，初次绿测暴露几何描述接口错误，记录保留；修正后4/4。完整套件{tests}项通过，{float(matched[2]):.3f}秒。",
 '- 原始Sia3轮精确回归通过；未安装新模式的原FULL3轮精确回归通过。仅归一化新源码根目录字符串，所有数值、分配、进度、restart、checkpoint和完成字段严格相等。',
 '- 两个状态97轮历史前缀完全复核（源码路径字符串同样规范化）；预测前后真实progress/execution/checkpoint未改变。预测期间禁止Job.step、Application.get_throughput/get_grad_stats，均未触发。',
 f"- 独立预测轨迹审计：{audit['counts']}；错误0。rho从同R0/E-legacy源矩阵独立复算，进度从初始剩余工作和raw×rho×有效秒递推核对。锁核验包括配置/placement/CPU/RAM，生命周期预算按实际恢复许可核对。",
 '- 两次工程失败保留：states.log为新源路径造成的精确比较差异；states2.log为嵌套工程patch重入失败；states3.log修正后通过。没有调参、放宽约束或改变响应。',
 '', '## 下一步建议','',
 '保留原FULL为有效版本；V3保留为技术修复版本，暂不替换。下次若继续，应先给当前观测模型的预测误差建立策略无关、按配置/batch区分的可审计范围，在方案净收益对误差不稳健时保留原决策。范围只能来自决策前可用数据及独立验证，不得根据本次100/21秒设置阈值。未校准边界前不新增权重、不换场景寻找胜者。本轮不实施新的不确定性规则。',
 '', '## 产物与复现','',
 'source_snapshot.zip/source_hashes.json为完整可重建源码；source_changes.patch列最小修改，dependencies.lock.txt随快照。没有伪造Git commit。environment.json、input_hashes.json、same_state.json、state_validation_v2.json、原始测试及前缀回归日志全部保留。',
 '同状态入口states2.py，修正后成功日志states3.log；独立审计audit_states.py。重新复现应复制脚本到新输出目录，避免覆盖独占创建文件。实际源码根及输入hash在source_receipt.json与input_hashes.json。',
 'COMPACT12负载、AWS/DGX各2GPU/6CPU单位/125GB、R0/E-legacy Medium alpha0.5、seed_profiles(1,8)、60秒/6000轮均未改。没有新的12作业完整对照，也没有80作业性能实验。本次只执行两状态有界前缀和模型预测，既有测试套件保持。',
 f"两状态成功验证墙钟{state['wall']:.3f}秒，不含准备、失败尝试和完整测试。完成后停止，无自动扩展。"]
 with (O/'REPORT.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
 with (O/'CURRENT_TASK.md').open('a',encoding='utf8') as f:f.write('完成并停止：375测试、原Sia/FULL3轮通过；两个状态技术审计0错误，排序1/2。按DESIGN门槛未启动新12作业矩阵。见REPORT.md。\n')
 save('artifact_hashes.json',{str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_hashes.json'})
 print('COMPLETE',result)
if __name__=='__main__':main()
