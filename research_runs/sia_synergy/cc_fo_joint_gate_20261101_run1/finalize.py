import csv,difflib,hashlib,importlib.metadata,json,shutil,subprocess,sys,time,zipfile
from pathlib import Path
R=Path(r'D:/科研启航/Sia');T=R/'cc-joint-fo-audit-repair-v2';O=Path(__file__).resolve().parent;OLD=R/'cc-joint-bounded-correction-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 text=(O/'full_suite.log').read_text(encoding='utf8');assert 'Ran 353 tests' in text and '\nOK\n' in text and '\nFAILED (' not in text
 probes={}
 for tree in ('official-artifacts',T.name):
  log=O/('sia_probe_'+tree+'.log')
  with log.open('xb') as f:r=subprocess.run([sys.executable,'-X','utf8','-B',str(T/'sia-simulator/check_stage1_legacy_regression.py'),'--probe',str(R/tree/'sia-simulator')],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  assert r.returncode==0
  probes[tree]=json.loads(next(s.removeprefix('PROBE_JSON=') for s in log.read_text(encoding='utf8').splitlines() if s.startswith('PROBE_JSON=')))
 assert probes['official-artifacts']==probes[T.name]
 save('original_sia_regression.json',dict(exact=True,rounds=3,states=probes))
 base=json.loads((O/'baseline_hashes.json').read_text());h={p.relative_to(T).as_posix():sha(p) for p in T.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 assert all(sha(OLD/n)==v for n,v in base.items())
 changed=[n for n,v in base.items() if h.get(n)!=v];new=sorted(set(h)-set(base))
 assert set(changed)=={'sia-simulator/joint_prototype/clean_ablation.py','sia-simulator/joint_prototype/bundle_adaptation.py'}
 assert new==['sia-simulator/tests/test_serial_acceptance_domain.py']
 save('source_hashes.json',h)
 patch=[]
 for n in changed+new:
  before=(OLD/n).read_text(encoding='utf8').splitlines(True) if n in base else []
  patch.extend(difflib.unified_diff(before,(T/n).read_text(encoding='utf8').splitlines(True),fromfile='a/'+n,tofile='b/'+n))
 (O/'source_changes.patch').write_text(''.join(patch),encoding='utf8')
 with zipfile.ZipFile(O/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
  for n in sorted(h):z.write(T/n,n)
 save('source_receipt.json',dict(files=len(h),changed=changed,new=new,old331_unchanged=True,snapshot_sha256=sha(O/'source_snapshot.zip'),git='Full source copy; no invented commit identity'))
 save('environment.json',dict(python=sys.version,executable=sys.executable,packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','cvxpy','cylp','psutil']}))
 obs=json.loads((O/'reconstruction_attempt3/model_reconstruction_validation.json').read_text());failed=json.loads((O/'reconstruction_attempt3/failed_decision_reconstruction.json').read_text());fixed=json.loads((O/'reconstruction_attempt3/repaired_decision.json').read_text());opp=json.loads((O/'opportunity_results.json').read_text());late=json.loads((O/'late_state_opportunity.json').read_text())
 assert failed['reproduced'] and not obs['raw_rate_mismatches'] and not obs['candidate_model_mismatches']
 same=all(v['config']==fixed['result']['selected'][j]['config']['config_id'] and v['cpu']==fixed['result']['selected'][j]['cpu_per_gpu'] and v['ram']==fixed['result']['selected'][j]['ram_gb_per_gpu'] for j,v in failed['selected'].items())
 assert same and fixed['result']['audit']['valid']
 assert not opp['gate_pass'],'Unexpected positive gate: stop for separate pre-registration, never launch from this finalizer'
 inputs=[R/'research_runs/sia_synergy/cc_clean_ab_20261030_run3/inputs/COMPACT_workload.csv',R/'research_runs/sia_synergy/cc_bounded_correction_20261031_run1/cells/dual_medium_FO/failure.json',R/'research_runs/sia_synergy/cc_bounded_correction_20261031_run1/cells/dual_medium_FO/rounds.jsonl.gz']+list((T/'inputs').rglob('*'))
 save('input_hashes.json',{str(p):sha(p) for p in inputs if p.is_file()})
 save('validation.json',dict(status='PASS_CORRECTNESS_GATE_FAIL_OPPORTUNITY',tests=353,old_tests=348,new_tests=5,observed_segment_rate_checks=obs['observed_segments'],previous_candidate_model_mismatches=0,original_guard_reproduced=True,repair_selected_allocation_unchanged=same,atomic=True,original_sia_exact_rounds=3,opportunity_gate=False,formal_runs_started=0))
 lines=['# FO验收修复与小配对机会预检','',
  '**结论：FO验收基准错误已定位并修复；正确性验证通过。预先选定12作业COMPACT负载的同状态机会门槛未通过，本轮不启动FO/FULL训练配对，不重跑80作业。**','',
  'SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE','',
  '## 1. 失败轮重建','',
  '从80作业失败现场抽取10个存活作业的历史观测，不重放整个训练。逐条重建1719段梯度/吞吐模型更新，原始执行率与失败前合法GPU候选Goodput全部吻合。随后仅对10个存活作业推进一个60秒诊断片段，并重求该单轮调度；一作业在此片段完成。没有80作业重跑或12作业完整训练。',
  '', '第一次重建漏了epoch_duration统计，第二次漏了_last_committed_columns当前配额证据，均显式失败、保留日志；第三次补齐日志可恢复状态后成功。不是通过修改Goodput、额外seed、关闭锁或跳过失败轮获得通过。',
  '',f"旧检查原样复现：旧几何基线目标{failed['comparison']['baseline_objective']:.12f}；FO结果{failed['comparison']['chosen_objective']:.12f}，原错误GLOBAL_BUNDLE_OBJECTIVE_REGRESSION。FO自身比例提案目标0.758384579073，辅助阶段提高到0.766192288847。旧基线涉及四个不同GPU配置，落在本轮FO冻结GPU域之外，因此不能用作该阶段的全局最优下界。",
  '', '修复后的同状态FO选择（GPU、CPU、RAM）与旧检查前结果逐项完全相同，原子资源审计有效。修复只改变验收，不使FO强行选择FULL的结果。',
  '', '## 2. 最小修复与保护','',
  '- clean_ablation.py记录旧完整分配是否属于本轮实际过滤后的候选域。',
  '- bundle_adaptation.py仅在物理合法、当前候选资格合法、且符合FO冻结GPU域时，将旧分配作为可比较基线。',
  '- FO每次辅助优化必须不低于自己的比例提案；真下降仍抛SERIAL_AUX_OBJECTIVE_REGRESSION。FULL对可比较可行基线的GLOBAL_BUNDLE_OBJECTIVE_REGRESSION继续保留。',
  '- 发生真退化时保存基线、选中分配和串行阶段诊断。没有扩大容差或修改目标、候选网格、phi、响应和容量。checkpoint、容量、当前候选与原子提交检查继续执行。',
  '', '## 3. 测试与可重建版本','',
  '红测用真实候选、oracle、容量审计和MIP，复现FO不应被全局基线约束的反例。旧检查抛错；第二项红测还暴露新增比较域字段缺失，不能当成第二个独立算法缺陷。新5项测试覆盖FO域不一致、FULL保留更好基线、串行真下降拒绝、FULL真下降拒绝和完整失败诊断。',
  '', '完整353项通过（原348＋新5），原始Sia3轮逐状态精确回归。完整源码332项、旧331项逐项核验未变，完整snapshot及patch可重建；仅修改2个验收相关文件并新增1个测试。所有历史现场保留。',
  '', '## 4. 12作业预检的冻结条件','',
  '使用既有COMPACT12作业（6 CIFAR10＋6 BERT），原到达、批量与工作量不改；AWS/DGX各1个2GPU节点，各6合成CPU单位/125GB RAM；固定R0/E-legacy、Medium alpha0.5及现有网格。未按新JCT挑选负载。opportunity_preregistration.json在扫描结果前冻结三个首次命中的结构条件。',
  '', '比较固定相同当前服务集合、信息、完整锁和历史资格。FO固定已有真实比例阶段的GPU意图，再以零gap重新优化CPU/RAM；其最优目标复现原轮FO目标。FULL在相同合法域联合选GPU和配额。两者均计入各作业真实目标启动成本，不忽略让出方。仅离线MILP，无训练推进。',
  '', '|首次结构状态|模拟秒|候选列|FO目标|FULL目标|FO/FULL剩余ETA总和|通过|','|---|---:|---:|---:|---:|---:|---|']
 for x in opp['results']:lines.append(f"|{x['rule']}|{x['time']}|{x['columns']}|{x['fo']['objective']:.9f}|{x['joint']['objective']:.9f}|{x['fo']['sum_eta']:.3f}/{x['joint']['sum_eta']:.3f}|否|")
 lines+=['', '完成事件＋混合已画像无锁＋有排队的第三条件没有命中，明确记为缺失。前两状态目标最优值相同；所返分配ETA也相同，未证明严格增量。不排除其他同分解、未检查状态或闭环配置路径产生收益；这不能否定历史COMPACT闭环JCT差异。',
  '', '边界：420秒两策略各2个选中点、1200秒各3个选中点涉及分子或分母裁剪；没有改动边界响应。不能将其称为目标硬件实测证据。',
  '', '## 5. 失败轮的局部联合机会为何仍不够','',
  f"在重建的13500秒同状态上，固定相同9个服务作业、当前模型、历史恢复许可和候选资格，对50070列离线求解。FO目标{late['fo']['objective']:.9f}，FULL目标{late['joint']['objective']:.9f}，提高{100*late['objective_gain']/late['fo']['objective']:.3f}%。但恒定当前速率剩余ETA总和由{late['fo']['sum_eta']:.3f}变为{late['joint']['sum_eta']:.3f}秒，增加{-late['net_eta_saved']:.3f}秒。两方法各有1个选中点触及边界。",
  '', '|作业|FO ETA|FULL ETA|节省秒（负为退化）|','|---|---:|---:|---:|']
 for j,v in sorted(late['eta_saved_by_job'].items(),key=lambda x:x[1]):lines.append(f"|{j}|{late['fo']['jobs'][j]['eta']:.3f}|{late['joint']['jobs'][j]['eta']:.3f}|{v:.3f}|")
 lines+=['', 'deepspeech2-10078从RTX2低辅助配额切至DGX1高辅助配额，预计ETA缩短约4197秒；但deepspeech2-10063被减配后预计增加约6957秒，另有其他让出方损失。联合机制确实能改变GPU与辅助资源，但不能只展示接收方加速。',
  '', '当前目标是倒数ETA之和，数学上不等于ETA之和，更不等于动态闭环总JCT。短ETA的进一步缩短可能在当前目标中抵消非常长任务的延后。这里给出了明确的局部取舍证据，不证明FULL最终JCT一定更差：任务完成释放资源后的再次调度、epoch和模型变化都未在恒定速率ETA中展开。',
  '', '## 6. 门槛与下一步','',
  '本轮没有得到预注册所要求的“FO已优化辅助资源后，联合配置带来正目标增量且所有让出方计入后剩余完成时间净改善”的见证。故不启动12作业配对，不改数据/参数继续寻找有利结果，也不重跑80作业。',
  '', '最小下一步是单独审定完成时间收益准则：比较当前剩余ETA与完成事件后资源释放的有限阶段估计，明确可接受的donor延迟与防饥饿保护；或先预注册一个由GPU/辅助配额排序反转定义、能够给出全服务集合净完成时间收益的场景。当前任务不实施目标改动，不承诺FULL必须胜过FO。',
  '', '所有机器可读结果：failed_decision_reconstruction.json、repaired_decision.json、model_reconstruction_validation.json、opportunity_results.json、late_state_opportunity.json、validation.json、source_receipt.json与input_hashes.json。原始失败和历史报告保持不变。']
 (O/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 with (O/'CURRENT_TASK.md').open('a',encoding='utf8') as f:f.write('\n结束：FO验收修复完成，353测试/原始Sia3轮通过。12作业离线门槛未过，训练配对0格，80作业未重跑。详见REPORT.md。无后台实验。\n')
 save('artifact_hashes.json',{p.relative_to(O).as_posix():sha(p) for p in O.rglob('*') if p.is_file() and p.name!='finalize.log'})
 print('COMPLETE_CORRECTNESS_PASS_OPPORTUNITY_FAIL_NO_FORMAL',flush=True)
if __name__=='__main__':main()
