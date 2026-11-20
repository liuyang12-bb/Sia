import collections,csv,gzip,hashlib,json,re
from pathlib import Path
import raw_audit_lib as lib
O=Path(__file__).resolve().parent;R=O.parents[2]
def save(n,x):
 with (O/n).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,ensure_ascii=False)
def main():
 p=json.loads((O/'preregistration.json').read_text());work={x['name']:x for x in csv.DictReader((O/'inputs/workload.csv').open())};stats={};jobs={};changes={};unc={};frames={}
 for cell in p['cells']:
  name=cell['id'];stats[name],jobs[name],changes[name]=lib.audit(cell,p,work);assert not stats[name]['errors']
  with gzip.open(O/'cells'/name/'rounds.jsonl.gz','rt',encoding='utf8') as f:rs=[json.loads(x) for x in f]
  frames[name]=rs;counter=collections.Counter();screens=[];last_decision=None
  for r in rs:
   decision=r['decision_for_next_interval']
   if decision['round']==last_decision:continue
   last_decision=decision['round']
   ev=r['decision_for_next_interval']['result']['audit'].get('release_completion',{});u=ev.get('uncertainty')
   if u:
    counter[u['reason']]+=1
    if u.get('point_selected_plan') not in (None,'LEGACY'):screens.append(dict(time=r['interval_end'],selected_plan=ev['selected_plan'],screen=u))
  unc[name]=dict(counts=dict(counter),potential_changes=screens)
  record=O/'cells'/name/'calibration_records.json.gz'
  if record.exists():
   with gzip.open(record,'rt',encoding='utf8') as f:rows=json.load(f)
   assert all(z['origin']<z['observed_end']<=z['observed_at'] for z in rows)
   unc[name]['calibration_rows']=len(rows);unc[name]['strata']=len({z['key'] for z in rows})
  stats[name]['max_round_wall']=max(r['round_wall_seconds'] for r in rs)
 save('raw_audit.json',stats);save('per_job.json',jobs);save('decision_changes.json',changes);save('uncertainty_summary.json',unc)
 comparisons={};exact={}
 for a,b in [('OLD_FO','OLD_FULL'),('NEW_FO','NEW_FULL'),('OLD_FO','NEW_FO'),('OLD_FULL','NEW_FULL')]:
  if not (stats[a]['complete'] and stats[b]['complete']):comparisons[a+'__'+b]=dict(complete=False);continue
  delta={j:jobs[a][j]['jct']-jobs[b][j]['jct'] for j in work}
  comparisons[a+'__'+b]=dict(complete=True,mean_saved=stats[a]['mean']-stats[b]['mean'],percent=100*(1-stats[b]['mean']/stats[a]['mean']),per_job_saved=delta,better=sum(v>0 for v in delta.values()),worse=sum(v<0 for v in delta.values()),equal=sum(v==0 for v in delta.values()))
 for a,b in [('OLD_FO','NEW_FO'),('OLD_FULL','NEW_FULL')]:
  exact[a+'__'+b]=len(frames[a])==len(frames[b]) and all(x['execution']==y['execution'] and x['job_end_state']==y['job_end_state'] and x['decision_for_next_interval']['result']['selected']==y['decision_for_next_interval']['result']['selected'] for x,y in zip(frames[a],frames[b]))
 save('comparison_summary.json',comparisons);save('trajectory_comparison.json',exact)
 gate=json.loads((O/'state_validation_final.json').read_text());cv=json.loads((O/'calibration_validation.json').read_text());states=json.loads((O/'same_state_final.json').read_text())
 tests=re.search(r'Ran (\d+) tests in ([\d.]+)s',(O/'full_suite_final.log').read_text())
 lines=['# 决策前误差校准与可靠性筛查：12作业对照','',
 'SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE','',
 '## 结论','']
 for pair in ('OLD_FULL__NEW_FULL','OLD_FO__NEW_FO'):
  c=comparisons[pair];lines.append(f"- {pair}：平均JCT节省{c.get('mean_saved')}秒、改善{c.get('percent')}%；逐作业改善/退化/持平{c.get('better')}/{c.get('worse')}/{c.get('equal')}。")
 lines+=['',f'新旧FO/FULL实际执行轨迹完全一致性：{exact}。不能把原FULL相对FO的既有收益归因于本轮误差筛查。若本轮持平，表示没有证明新增JCT收益，而非宣称没有资源优化机会。',
 '', '## 算法与冻结规则','',
 '新增uncertainty.py，继承V3共用真实FO/FULL接续。每个真实间隔开始前记录模型raw、应用、GPU配置、atomic batch、accumulation、epoch；下次决策前以首个有效训练段的直接raw形成r=raw_observed/raw_predicted。只用当前运行已发生的观测，不从未来JCT校准。',
 '按(application, GPU config, atomic, accum, epoch)分层，同一应用不同作业共享需条件可交换假设；CPU/RAM配额不进入raw层，因为统一执行模型中其作用已单独乘rho。每层至少两个不同已观测间隔才有支持；I=[min(1,历史r),max(1,历史r)]。无支持不补1；使用所有前缀数据，不选有利窗口、分位数或收益阈值。',
 '点预测提出改选后，若原/候选预测轨迹需要的任一层缺支持则保留原方案；全部覆盖才对所有误差端点组合进行配对接续，原方案和候选使用同一因子。任何净收益非正、场景新层、预测删失或失败即拒绝。所有端点正收益才允许改选。沿用原1e-7秒数值容差。',
 '此范围不是置信区间；端点检查不保证非线性接续的所有内部误差世界，历史范围也不保证未来epoch误差不越界。没有据两个分叉的事后胜负选择阈值。只改变新增评价器的改选验收；原FULL本身不加校准准入。预测误差因子不进入真实Job.step或CPU/RAM响应。',
 '', '## 同状态验收','', '|状态|决策|依据|决策前观测条数|','|---|---|---|---:|']
 for s in states['results']:
  ev=s['evaluation'];lines.append(f"|{s['time']}|{ev['selected_plan']}|{ev['uncertainty']['reason']}|{len(s['calibration_records'])}|")
 lines+=['','4800秒点预测已认为原方案更好；5820秒原点预测仍认为改选节省100.272秒，但误差支持不全而保留。这里的通过是正确保守处理与来源审计通过，并没有把5820秒的点预测误差校正成真值。',
 '5820缺失层：BERT/aws双GPU/atomic12/accum0/epoch0；BERT/dgx双GPU/atomic47/accum0/epoch0；CIFAR/aws双GPU/atomic264/accum0/epoch39；CIFAR/dgx单GPU/atomic709/accum1/epoch39。两个已覆盖BERT/dgx单GPU层的经验比率范围约[0.9510,1.0891]（epoch0）和[0.9510,1.0]（epoch1）。缺失与已有GPU Goodput模型是否存在不是同一件事：种子画像不等于同配置/批量/epoch的前瞻执行误差已校准。',
 '', '## 完整训练对照','',
 '相同COMPACT12（6 CIFAR10＋6 BERT），原声明工作量、batch、到达序列不变；AWS/DGX各2GPU、6合成CPU单位、125GB，R0/E-legacy Medium alpha0.5，seed_profiles(1,8)，60秒6000轮；每格独立进程串行，启动前2GiB/5GiB门槛。',
 '', '|方法|完成|平均JCT|Median|P95|P99|Makespan|轮数|墙钟s|最大单轮墙钟s|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name,s in stats.items():lines.append(f"|{name}|{s['completed']}/12|{s['mean']}|{s['median']}|{s['p95']}|{s['p99']}|{s['makespan']}|{s['rounds']}|{s['wall']:.3f}|{s['max_round_wall']:.3f}|")
 lines+=['','P95/P99仅12作业描述，不能视作总体尾延迟高精度估计。','',
 '|方法|等待s|训练s|checkpoint s|restart|浪费GPU-s|GPU-h|CPU单位-h|RAM GB-h|裁剪点比例|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name,s in stats.items():
  t=s['totals'];lines.append(f"|{name}|{t['wait']}|{t['training']}|{t['checkpoint']}|{t['restarts']}|{t['wasted_gpu_seconds']}|{t['gpu_hours']:.5f}|{t['cpu_core_hours']:.5f}|{t['ram_gb_hours']:.5f}|{s['clipped_fraction']:.3%}|")
 lines+=['','## 逐作业JCT','', '|作业|旧FO|旧FULL|新FO|新FULL|','|---|---:|---:|---:|---:|']
 for j in work:lines.append('|'+j+'|'+'|'.join(str(jobs[n][j]['jct']) for n in stats)+'|')
 lines+=['','## 筛查是否真正介入','']
 for name in ('NEW_FO','NEW_FULL'):lines.append(f"- {name}：{unc[name]}")
 lines+=['', '上述原因来自每轮真实日志。缺失支持不是候选不可行或没有CPU互补；当前无法用已观测误差为新增改选背书。如果没有端点接受事件，必须明确：本轮只验证了保守回退，不能宣称端点机制已在真实负载上带来收益。构造的全覆盖正收益测试证明机制并非逻辑上永远拒绝，不能代替真实负载的性能证据。',
 '', '## 正确性、复现与限制','',
 f"完整最终测试{tests[1]}项通过（{tests[2]}秒），最终14项针对测试通过；原Sia与原FULL各3轮精确回归通过。两个状态97轮前缀精确复核，预测禁用真实训练/吞吐/梯度调用。决策前校准来源独立审计错误{len(cv['errors'])}；四格原始执行审计错误{sum(len(s['errors']) for s in stats.values())}。",
 '原始raw、rho、进度、容量、完整锁、完成事件、互斥时间和资源小时独立复算，未用effective/rho反推raw。observed_at与observed_end双重截止，延迟到达的标签不能提前使用。checkpoint不产生训练误差样本。求解/预测失败只保留已审计原方案，不部分提交。',
 '完整源码source_snapshot.zip与source_hashes.json、最小diff、依赖environment.json、冻结输入preregistration.json、命令execution_manifest.jsonl和calibration_records.json.gz可复核；旧源V3及历史产物未修改。源为完整复制，没有伪造Git commit。运行前DESIGN.md规则保持不变。',
 'raw_audit.json、per_job.json、uncertainty_summary.json、trajectory_comparison.json、calibration_validation.json与全部日志保留。初次基础测试因计时fixture缺少新元数据失败已保留，修正fixture未伪造真实模型；旧测试原树不改。中间回归保留，最终全套以full_suite_final.log为准。',
 '限制：经验覆盖可能过于稀疏，跨作业分层共享及冻结未来参数是近似；原FULL的性能模型本身也有误差，本轮只防止新增完成评价在证据不足时改选。R0跨硬件映射及裁剪假设未被验证为真实硬件性能。原FULL对FO的既有优势也不等于本轮新增CPU/内存收益。优化墙钟仍未计入模拟JCT，沿用历史口径；新增预测可能明显增加实际计算代价，表中给出实测最大单轮墙钟，不能把模拟持平理解为没有开销。',
 '下一步先补充策略无关的预测误差覆盖与独立留出验证，或预注册可解释的分层共享规则；不能看到这四个缺失层后直接放宽直到通过。本轮不实现探索、不调参、不扩大规模。',
 f"四格实测墙钟{json.loads((O/'pipeline_complete.json').read_text())['wall_seconds']:.3f}秒，不含实现、测试与同状态预检。完成后停止，不80作业。"]
 # Avoid embedding all scenario traces in prose; machine-readable records retain them.
 for name in ('NEW_FO','NEW_FULL'):
  bulky=f"- {name}：{unc[name]}"
  concise=f"- {name}：原因计数{unc[name]['counts']}；校准记录{unc[name].get('calibration_rows',0)}条、{unc[name].get('strata',0)}层；逐次潜在改选及缺失层见uncertainty_summary.json。"
  lines=[concise if x==bulky else x for x in lines]
 with (O/'REPORT.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
 assert all(lib.sha(R/'cc-joint-uncertainty-v4'/n)==h for n,h in p['source_hashes'].items())
 save('validation.json',dict(all_complete=all(s['complete'] for s in stats.values()),raw_errors=sum(len(s['errors']) for s in stats.values()),source_frozen=True,state_gate=gate['ranking_gate'],calibration_errors=len(cv['errors']),tests=int(tests[1]),formal_runs=4))
 with (O/'CURRENT_TASK.md').open('a',encoding='utf8') as f:f.write('完成并停止。四格与审计见REPORT.md；未调参、未80作业。\n')
 save('artifact_hashes.json',{str(f.relative_to(O)):lib.sha(f) for f in O.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.name!='artifact_hashes.json'})
 print(json.dumps(comparisons,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
