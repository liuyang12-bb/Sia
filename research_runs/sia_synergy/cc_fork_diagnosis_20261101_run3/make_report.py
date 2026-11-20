import hashlib,json,zipfile
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];P=R/'research_runs/sia_synergy/cc_release_reconfig_20261101_run1'
def load(n):return json.loads((O/n).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def main():
 stats=load('raw_audit.json');pairs=load('fork_comparisons.json');err=load('prediction_errors.json');v=load('validation.json');p=load('preregistration.json')
 lines=['# 4800/5820秒单决策分叉：预测误差诊断','',
 'SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE','',
 '## 结论','',
 '保留原FULL。两次被新评价器看好的改选，在统一原FULL后续策略下均未改善平均JCT。主要问题已有直接证据：预测接续不是实际原FULL的接续；当前候选与恢复预算被固定到未来，遗漏后续合法恢复；冻结速率还包含批量、梯度效率和在线模型误差。不能用本轮直接确定这些因素各自占多少误差，也不能把有限状态归纳为普遍结论。',
 '', '## 预注册与控制','',
 '4条独立进程、12作业完整训练轨迹。4800秒对照从共同历史前缀分叉；5820秒对照共同前缀包含4800秒的历史改选，故两组均为严格同状态比较。每组只在目标轮选择LEGACY或MIN_STATIC_SUM_ETA；随后全部关闭新增评价，执行原FULL。不是把两个不同历史状态直接相比。',
 'COMPACT12、6 CIFAR10＋6 BERT，AWS/DGX各2GPU/6合成CPU单位/125GB；R0/E-legacy Medium alpha0.5，seed_profiles(1,8)，60秒/6000轮。旧工作量、到达、容量、响应、网格、求解设置不变。每进程2GiB内存/5GiB磁盘门槛。源cc-joint-release-reconfig-v2只读，原FULL不修改；只新增诊断hook，使用原原子试算、候选与提交校验。',
 '新目录run3，run1/run2脚本故障证据保留：run1在60秒因tuple/list表示差异断言停止；run2在300秒因实例绑定hook绕开trial对象失败。最终使用类级hook，JSON同编码后逐字段严格比较，不删字段、不放宽数值。失败都在目标干预前，未作为性能结果。',
 '', '## 完整结果','',
 '|分叉状态|一次决策|完成|平均JCT|Median|P95|P99|Makespan|','|---|---|---:|---:|---:|---:|---:|---:|']
 for n,s in stats.items():lines.append(f"|{n.split('_')[0]}|{n.split('_',1)[1]}|{s['completed']}/12|{s['mean']:.3f}|{s['median']:.3f}|{s['p95']:.3f}|{s['p99']:.3f}|{s['makespan']:.3f}|")
 lines+=['','P95/P99只描述12作业，不能视作总体尾延迟精确估计。','',
 '|状态|预测改选节省总完成秒|实际改选节省总JCT秒|实际平均JCT节省|改善/退化/持平|','|---|---:|---:|---:|---|']
 for x in pairs:lines.append(f"|{x['time']}|{x['predicted_sum_saved']:.3f}|{x['actual_sum_saved']:.3f}|{x['actual_mean_saved']:.3f}|{x['better']}/{x['worse']}/{x['equal']}|")
 lines+=['','正值=节省，负值=恶化。实际差值由原始完成事件复算；同一作业arrival相同，配对完成时间差等于JCT差。预测只包括当时尚未完成者；其余作业在两分支完全相同。','',
 '## 逐作业因果对比','', '|作业|4800改选节省JCT|5820改选节省JCT|','|---|---:|---:|']
 for j in pairs[0]['per_job_saved']:lines.append(f"|{j}|{pairs[0]['per_job_saved'][j]:.3f}|{pairs[1]['per_job_saved'][j]:.3f}|")
 lines+=['','## 预测误差与执行证据','',
 '|状态/方案|作业|预测剩余秒|实际剩余秒|预测−实际|首训练raw预测/实际|','|---|---|---:|---:|---:|---|']
 for n,d in err.items():
  for j,z in d['per_job'].items():
   f=z['first_training'];rates=f"{f['predicted_raw']:.5f}/{f['segment']['raw_rate_before_rho']:.5f}" if f else '无'
   lines.append(f"|{n}|{j}|{z['predicted_remaining']:.3f}|{z['actual_remaining']:.3f}|{z['error']:.3f}|{rates}|")
 lines+=['',
 '1. **未来任务选择不同。** release_evaluation.py:55–71只在预测完成事件后，按单个存活者自己的ETA节省贪心分配资源；它没有调用实际原FULL的联合目标和资格状态机。4800 LEGACY预测在相对720秒时先将bert-159迁往DGX，再将bert-107扩为AWS双GPU；实际在5400秒（相对600秒）将DGX给bert-107、AWS双GPU给bert-159。两者即便拥有相同资源，也不是同一个后续决策问题。',
 '2. **未来恢复许可未展开。** release_evaluation.py:26固定当前候选表；:64–66受当前frozen/known/limit限制；:196仅构建当时budget。真实bounded_correction.py:11–60和tail_recovery.py:20–58在之后资源空出、训练和锁条件满足时可以授予一次有真实startup的合法恢复。4800 LEGACY实际bert-107于5820秒获110秒额外预算，bert-159于6480秒再获合法许可；累计360秒不等于超配或越过真实规则。V2保守停留在初始250秒预算，特别高估了原决策下bert-107的完成时间。未来许可应由预测状态合法触发，而非一开始预授，也不应永久假设不会发生。',
 '3. **固定训练率不等于实际全程速率。** completion_time.py:31–62用当前拟合模型、当前梯度状态和批量估计raw；job.py:465、496、550按实际epoch/吞吐更新训练和批量。首段raw差异与整条剩余ETA误差不是一回事；例如4800 LEGACY bert-107首段约0.944/0.904，4%左右差异解释不了8386秒对1668秒的完成预测差，后续GPU选择和恢复遗漏是明确的大项。CIFAR后续效率变化仍有误差，不能宣称全部偏差仅来自恢复许可。',
 '', '本轮不是分别将每个误差项修正后的全因子实验，因此不提供虚构的误差贡献百分比。预测错误可以严格写成：各方案残差e=预测Σ剩余完成−实际Σ剩余完成；错误排序幅度=e(LEGACY)−e(改选)。机器可读文件给出逐作业残差，可复算。',
 '', '## 资源、时间与恢复','', '|分支|等待s|训练s|checkpoint s|restart|GPU-h|CPU单位-h|RAM GB-h|浪费GPU-s|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for n,s in stats.items():
  z=s['totals'];lines.append(f"|{n}|{z['wait']:.0f}|{z['training']:.0f}|{z['checkpoint']:.0f}|{z['restarts']}|{z['gpu_hours']:.4f}|{z['cpu_core_hours']:.4f}|{z['ram_gb_hours']:.4f}|{z['wasted_gpu_seconds']:.0f}|")
 lines+=['',
 '实际恢复事件、候选外未来转换、每个作业完整轨迹见prediction_errors.json与trajectories.json。forecast与实际都支付checkpoint；问题不是通过免除真实成本得到优势。',
 '', '## 最小修复方案（本轮不实施）','',
 '优先修复预测与实际接续的一致性，不调整公平目标、候选点、响应或启动预算：',
 '1. 将现有候选资格、checkpoint锁、生命周期账本、恢复许可条件整理为可复用的无副作用状态转换。预测每个调度边界用虚拟当前状态重建合法域；真实状态和预测状态调用同一规则。只用当时已观测模型，不读取未来实际速度、到达或完成事件。',
 '2. 用原FULL自身的选择目标生成未来接续，替换“最大单任务ETA节省”的贪心分配。完成释放、可支付恢复及其冻结都由虚拟状态触发。分支有初始成本差异，但后续决策规则必须一致。不要用固定当前列和固定budget代表整个未来。',
 '3. 预测状态机尚未闭合、模型未知或对可行后续顺序结论不稳定时保留原FULL。速率不确定范围需要由独立预测残差预注册；本轮不依据两个状态的胜负设阈值。不要把未知替换成假ETA。',
 '4. 验收先覆盖本次两个状态的未来许可、相同资源下不同survivor选择、batch/epoch变化、checkpoint/预算原子性；再测试独立构造的相反排序案例。修复后的预测必须先解释误差，才允许新的12作业对照。原FULL继续作为外部对照，而不能硬编码这两个状态的已知胜者。',
 '', '不能承诺上述修复必然提高JCT；它首先让评价问题与实际执行问题相符。避免再叠加复杂预测而没有同状态证据。',
 '', '## 验收、复现与限制','',
 f"原FULL4800 LEGACY逐轮执行、最终状态及分配精确回归：{v['old_full_4800_legacy_exact']}。两个分叉同状态、同候选表精确一致；前缀校验轮数{v['prefix_frames']}。四格12/12，原始执行容量/锁/rho/进度/互斥时间/资源小时审计错误{v['raw_audit_errors']}，生命周期预算独立校验次数{sum(z['lifetime_budget_checks'] for z in err.values())}。",
 '本轮调度源码0修改；既有371测试的日志保留于V2历史目录，未冒称本轮重新执行完整套件。本轮新增验证是4条完整小型轨迹、同状态前缀断言、单点干预断言和独立原始事件审计。历史报告、源和结果均保留。源不具有新的Git commit；复现依赖真实336文件hash及V2完整snapshot，不能用裸HEAD替代。',
 f"四条完整运行墙钟{load('pipeline_complete.json')['wall']:.3f}秒（不含脚本准备和两次早期失败）。",
 '输入/源hash为preregistration.json；启动前脚本hash见execution_receipt.json；最终脚本与产物hash见artifact_hashes.json。运行命令：.venv-sia-baseline/Scripts/python.exe -X utf8 -B <新目录>/pipeline.py，然后analyze.py、make_report.py。复现须复制到新输出目录，不能覆盖本目录。',
 '本轮仅解释12作业两个状态。响应仍是合成跨硬件迁移，较多裁剪点的外部有效性问题仍在；这不改变本次同输入下预测与真实决策不一致的发现。没有80作业运行、无参数搜索、无算法修复实施，完成后停止。']
 with (O/'REPORT.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
 receipt=dict(source=str(R/'cc-joint-release-reconfig-v2'),source_hashes=p['source_hashes'],snapshot=str(P/'source_snapshot.zip'),snapshot_sha256=sha(P/'source_snapshot.zip'),dependency_lock_sha256=sha(R/'cc-joint-release-reconfig-v2/dependencies.lock.txt'))
 with (O/'source_snapshot_reference.json').open('x') as f:json.dump(receipt,f,indent=2)
 with zipfile.ZipFile(O/'diagnostic_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
  for n in ('run_fork.py','pipeline.py','analyze.py','make_report.py','raw_audit_lib.py','preregistration.json','inputs/workload.csv','source_snapshot_reference.json'):z.write(O/n,n)
 (O/'CURRENT_TASK.md').open('a',encoding='utf8').write('完成并停止：4/4各12/12；同状态前缀与原FULL精确回归通过；独立审计0错误。两个单次改选均恶化平均JCT，见REPORT.md。未修改算法，无80作业。\n')
 hashes={str(f.relative_to(O)):sha(f) for f in O.rglob('*') if f.is_file() and '__pycache__' not in str(f) and f.name!='artifact_hashes.json'}
 with (O/'artifact_hashes.json').open('x') as f:json.dump(hashes,f,indent=2)
 print('REPORT',O/'REPORT.md','sealed',len(hashes))
if __name__=='__main__':main()
