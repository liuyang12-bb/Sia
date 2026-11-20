"""Finalize software evidence ONLY. Does not launch any experiment."""
import csv,difflib,gzip,hashlib,importlib.metadata,itertools,json,re,subprocess,sys,time,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];T=R/'cc-joint-core80-repair-v3';O=R/'research_runs/sia_synergy/cc_core80_repair_20261031_run3';OLD=R/'research_runs/sia_synergy/cc_core80_8h_20261030_run3'
sys.path[:0]=[str(T),str(T/'sia-simulator')]
from joint_prototype.execution_audit import audit_execution
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def clean(v):
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items() if k!='template_source'}
 if isinstance(v,list):return [clean(x) for x in v]
 return v
def frames(p,n=None):
 with gzip.open(p,'rt',encoding='utf8') as f:return [json.loads(l) for l in (f if n is None else itertools.islice(f,n))]
def semantic(rs):
 return [clean(dict(execution=r['execution'],selected=r['decision_for_next_interval']['result']['selected'],objective=r['decision_for_next_interval']['result']['scheduling_objective'],actual_goodput=r['decision_for_next_interval']['result']['actual_total_goodput'])) for r in rs]
def main():
 assert (O/'FORMAL_BLOCKED_AUTO_REVIEW.json').exists()
 log=(O/'tests/full_suite.log').read_text(encoding='utf8');match=re.search(r'Ran (\d+) tests in ([\d.]+)s',log);assert match and int(match[1])==302 and '\nOK\n' in log
 original=read(O/'tests/original_sia_exact_regression.json');assert original['exact_match']
 cases=[('FULL',12,'cbc_safety'),('A10',20,'primal_dual'),('PROP',12,'primal_dual')];data={};gates=[];violations=[];locks=0;executions=0
 for name,n,suffix in cases:
  p=O/'tests'/f'real80_{name}_{n}_{suffix}';status=read(p/'result.json');assert status['status']=='PASS' and status['boundaries']==n;rs=frames(p/'prefix.jsonl.gz');assert len(rs)==n;data[name]=rs;gates.append(status)
  for r in rs:
   d=r['decision_for_next_interval'];res=d['result'];specs={j['job_id']:j for j in d['probe']['specs']};used={t+':0':[0.,0.,0.] for t in ('aws','dgx-ext','rtx')};caps={'aws:0':(4,12,250),'dgx-ext:0':(8,24,500),'rtx:0':(8,24,500)}
   for j,s in specs.items():
    if s['checkpoint_locked'] and s['current_config_id']:
     locks+=1;c=res['selected'].get(j)
     if c is None or c['config']['config_id']!=s['current_config_id'] or c['placement']!=dict(s['current_placement']) or (c['cpu_per_gpu'],c['ram_gb_per_gpu'])!=(s['current_cpu_per_gpu'],s['current_ram_gb_per_gpu']):violations.append('LOCK')
   for c in res['selected'].values():
    for node,g in c['placement'].items():
     for k,v in enumerate((g,g*c['cpu_per_gpu'],g*c['ram_gb_per_gpu'])):used[node][k]+=v
   if any(v>b+1e-7 for node,vs in used.items() for v,b in zip(vs,caps[node])):violations.append('CAPACITY')
   for e in r['execution']:
    executions+=1;a=audit_execution(e)
    if not a['valid']:violations.extend(a['violations'])
   assert res['audit']['valid']
 assert not violations,violations
 canon=read(OLD/'canonical_result_paths.json');bridges=[]
 for name,n in [('FULL',4),('PROP',12),('A10',12)]:
  cid=('dual_neutral_' if name.startswith('A') else 'dual_medium_')+name
  old=frames(Path(canon['paths'][cid])/'rounds.jsonl.gz',n);new=data[name][:n];equal=semantic(old)==semantic(new)
  bridges.append(dict(method=name,rounds=n,exact_execution_score_allocation=equal,ignored='Relocated template_source strings only; template SHA remains.'))
 assert all(b['exact_execution_score_allocation'] for b in bridges),bridges
 prior=read(OLD/'preregistration.json');baseline=read(O/'baseline_source_hashes.json');files=sorted(p for p in T.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts);hashes={p.relative_to(T).as_posix():sha(p) for p in files};inputs={}
 for name,h in prior['input_hashes'].items():
  p=T/Path(name).relative_to(Path(prior['source_root']));assert sha(p)==h;inputs[str(p)]=h
 for name,h in prior['source_hashes'].items():
  if name.startswith('inputs/') or name.startswith('sia-simulator/traces/'):assert hashes[name]==h
 changed=[n for n,h in hashes.items() if baseline.get(n)!=h];removed=sorted(set(baseline)-set(hashes));assert not removed
 save(O/'source_hashes.json',hashes);save(O/'input_hashes.json',inputs);save(O/'source_delta.json',dict(changed_or_added=changed,removed=removed,unchanged_count=sum(hashes.get(n)==h for n,h in baseline.items())))
 with zipfile.ZipFile(O/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
  for p in files:z.write(p,p.relative_to(T).as_posix())
 with (O/'source_changes.patch').open('x',encoding='utf8') as f:
  for n in changed:
   a=R/'cc-joint-core80-8h-v3'/n;b=T/n
   try:f.writelines(difflib.unified_diff(a.read_text(encoding='utf8').splitlines(True) if a.exists() else [],b.read_text(encoding='utf8').splitlines(True),fromfile='baseline/'+n,tofile='repaired/'+n))
   except UnicodeDecodeError:f.write('\nBINARY '+n+' '+hashes[n]+'\n')
 env=dict(python=sys.version,executable=sys.executable,packages={k:importlib.metadata.version(k) for k in prior['environment']['packages']},integer_backends='SciPy bundled HiGHS; existing CyLP/CBC via CVXPY, no installation',solver_options=dict(highs=dict(time_limit=30.,mip_rel_gap=0.),existing_retry=dict(time_limit=30.,mip_rel_gap=0.,presolve=False),cbc_safety=dict(maximumSeconds=30.,allowableFractionGap=0.)))
 save(O/'environment.json',env);(O/'dependencies.lock.txt').write_bytes((T/'dependencies.lock.txt').read_bytes())
 # Historical regression tests read external fixtures and official files.
 # Preserve them too, rather than claiming the source zip alone suffices.
 external={}
 for folder in (R/'research_sources/Synergy/simulator/models',R/'research_runs/sia_synergy/sia_r0_full160_comparison/inputs',R/'research_runs/sia_synergy/joint_opportunity_experiments_20261026_run1/inputs'):
  for q in folder.rglob('*'):
   if q.is_file() and '__pycache__' not in q.parts and q.suffix in ('.py','.json','.csv','.txt','.yaml','.yml'):external[q.relative_to(R).as_posix()]=q
 for rel in ('research_runs/sia_synergy/small_mechanism_20261025/preregistration.json','research_runs/sia_synergy/small_mechanism_20261025/workload.csv','research_runs/sia_synergy/stage2b_neutral/g0_verified/registry.json','research_runs/sia_synergy/b3_multiworkload_medium/followup_w10_late_6000/frozen_config_6000.json'):
  q=R/rel;assert q.exists(),q;external[rel]=q
 for q in (R/'official-artifacts/sia-simulator').rglob('*'):
  if q.is_file() and '__pycache__' not in q.parts and q.suffix in ('.py','.csv','.json','.txt','.yaml','.yml'):external[q.relative_to(R).as_posix()]=q
 with zipfile.ZipFile(O/'external_test_dependencies.zip','x',zipfile.ZIP_DEFLATED) as z:
  for rel,q in sorted(external.items()):z.write(q,rel)
 save(O/'external_test_dependency_hashes.json',{rel:sha(q) for rel,q in external.items()})
 protected={}
 for name in ('official-artifacts','research_sources/Synergy'):
  h=subprocess.run(['git','rev-parse','HEAD'],cwd=R/name,capture_output=True,text=True);d=subprocess.run(['git','diff','--quiet'],cwd=R/name);status=subprocess.run(['git','status','--short'],cwd=R/name,capture_output=True,text=True)
  assert h.returncode==0 and d.returncode==0;protected[name]=dict(head=h.stdout.strip(),tracked_clean=True,status=status.stdout)
 save(O/'protected_repositories.json',protected)
 acceptance=dict(existing_tests=276,new_tests=26,total=302,passed=302,seconds=float(match[2]),bounded_gates=gates,original_sia_exact=True,bridges=bridges,execution_audits=executions,checkpoint_lock_checks=locks,violations=violations)
 save(O/'software_acceptance.json',acceptance)
 planned={k:prior[k] for k in ('interval','max_rounds','seed_profiles','capacity','response','cpu_grid')};planned.update(environment=env,status='PENDING_EXPLICIT_REAUTHORIZATION_AUTO_REVIEW_BLOCK',cells=[x for x in prior['cells'] if not x['optional']],source_snapshot_sha256=sha(O/'source_snapshot.zip'),source_hashes=hashes,input_hashes=inputs,no_launch_performed=True,no_parameter_tuning=True,freeze_note='Protocol only; no executable output/deadline registration. Must resolve two automatic review rejections before launching. No104/160/optional4.')
 save(O/'planned_preregistration.json',planned)
 save(O/'freeze_receipt.json',dict(time=time.time(),source_files=len(files),snapshot_sha256=sha(O/'source_snapshot.zip'),patch_sha256=sha(O/'source_changes.patch'),scientific_inputs_verified=len(inputs),status='SOFTWARE_VALIDATED_FORMAL_BLOCKED'))
 with (O/'REBUILD.md').open('x',encoding='utf8') as f:f.write(f'''# Rebuild without losing historical work\n\nSource archive: source_snapshot.zip ({len(files)}files), verify every entry using source_hashes.json. This isolated source copy is not a Git checkout; hashes/full bytes are the version authority. Extract only into a new directory below the workspace; never overwrite an existing tree.\n\nLegacy test fixtures/official baseline: external_test_dependencies.zip, {len(external)}files at workspace-relative paths; verify existing files rather than overwrite. The original protected repositories remain untouched. The legacy small_mechanism preregistration contains absolute D:/科研启航/Sia paths: tests reproduce directly under that workspace layout. When relocating, repoint ONLY copied path strings and record original/relocated hashes; do not change numerical configuration, data or original artifacts.\n\nRuntime: Python3.10.10 and dependencies.lock.txt/environment.json; use independent venv, never global upgrades. All302 software tests: from NEW_TREE/sia-simulator run the matching venv python -X utf8 -B -m unittest discover -s tests -p test_*.py -v. This command is software verification; no formal launcher is invoked.\n\nFormal11-cell restart remains blocked. planned_preregistration.json is a reviewable protocol, deliberately not an executable launch manifest. Do not use old wait_accept/accept_and_launch scripts to bypass FORMAL_BLOCKED_AUTO_REVIEW.json.\n''')
 # The final report is engineering-only; no new overall JCT fabricated.
 report=f'''# 80作业工程修复验收：正式补跑待授权\n\nSYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE\n\n## 当前状态\n\n本轮工程修复及软件验收完成；原11格80作业完整对照尚未补齐。自动审批两次拒绝“验收后自动启动11格补跑”，理由是认定较晚的实验暂停尚未解除，已提交最新80作业授权供复核仍被拒绝。未绕过审批；当前没有正式批次进程。不能报告本修复版80/80完成、总体JCT均值或CPU/内存收益。\n\n## 原因与最小范围\n\n旧FULL模型68394列首次HiGHS status0，整数误差6.048878e-7；旧1e-7门槛先于已有1e-6规范化检查，导致有效二进制解被拒绝。先规范化，再验证原始容量/主目标/锁行修复此问题。窄带±1e-9不放宽；普通/窄带验收仍1e-7/1e-10。\n\n随后第10轮121133列模型两次HiGHS30秒都超时。连续松弛下界169.2247，当前整数成本295；LP舍入容量越界7.937，禁止提交。它不是资源不可行，也不能把295的已有可行解直接冒充最优。相同原始模型在仅做可证明支配变量消去后，用已安装CBC在5.8246秒证明最优启动成本仍为295，原始行违规0。\n\n实现仅包括：完整原网格最大rho缓存与已有因子前置过滤；同几何支配/零效用可选变量的精确消去；保守Decimal80位对偶下界、整数成本格点证书和阶段内严格劣化变量固定；一个LP支撑集整数子问题仅提供更好可行上界，不能冒称全局最优；两次原生status1后，显式启用已有CBC独立零gap求解，仍需原始行验证。所有解均原子审计后提交，超时/非法向量不算最优。原Sia决策逻辑未改。\n\n数学上，对Ax≤b、0≤x≤1、y≤0，L=bᵀy+Σmin(0,c−Aᵀy)为保守下界。若L+max(r_i,0)>当前可行成本V，可证明x_i不能在任何全局最优解中取1；仅本词典序阶段消去，后续阶段重建。整数成本公因子g满足L>V−g才可直接证明当前整数解最优。证明用验证容差外包松弛仅使下界更保守，执行/提交域不放宽。\n\n## 软件验收\n\n- 原276项测试保留，新增26项；**302/302通过，用时{float(match[2]):.3f}秒**。原始日志tests/full_suite.log。\n- 真实FULL前12次完整调度通过，跨过旧第5次及新增第10次失败；A10前20次、PROP前12次通过。这是固定窗口工程回归，不是80作业完成实验。\n- 三段旧成功前缀：FULL4、A10/PROP各12轮，分配、分数、实际执行逐项精确匹配，仅迁移后的template_source路径不同且模板SHA仍相同。\n- 原始Sia3轮分配/进度/restart/完成事件逐项一致。\n- 独立审计{executions}条执行记录、{locks}次checkpoint锁；容量/锁/执行违规0。\n- 红测、失败变体、完整失败模型、未采用的selector扩展静态求解均保留。错误fixture/旧测试预期修正单独记录，不冒称生产缺陷。\n\n## 保护与可重建版本\n\n执行源：`{T}`。实际源码快照{len(files)}文件，SHA-256 `{sha(O/'source_snapshot.zip')}`。旧302文件中299个未改，0删除；38个科学输入内容hash完全一致。源码patch、依赖锁、Python/NumPy/SciPy/CVXPY/CyLP版本、完整输入/源码SHA及原始日志均在本目录。官方Sia/Synergy tracked状态干净，既有未跟踪文件不删除。\n\n保持原80声明（48CIFAR10/16DeepSpeech2/16BERT）、原batch/工作量、0/600/1200到达、3CPU/62.5GB每GPU、R0/E-legacy、alpha0/.5、seed_profiles(1,8)、60秒/6000轮不变。p、lambda、bonus、冷启动、准入/持续服务、候选网格与响应模型均未改变。CBC是已经安装的独立整数后端重试，maximumSeconds30/allowableFractionGap0；原HiGHS30秒/zero-gap及原有无presolve重试不变。数值后端重试须与策略fallback分开统计。\n\n## 仍待完成\n\n11格完整80作业对照仍未完成。run1/attempt2的FULL失败和A10中断、run2/run3全部技术失败证据原样保留；当前修复版没有完整JCT性能结果。planned_preregistration.json只是待批准协议，不含可自动启动批次的运行目录或截止时间。正式11格需研究者明确解除本轮暂停并通过自动审批后，使用新目录及同一冻结源码完成；不恢复104格、不运行160作业、不加入4个可选格、不调参。\n\n工程源/测试已就绪不代表性能优势已得到验证。完成正式对照后，仍须从原始完成事件重算JCT、互斥时间、资源小时及裁剪比例，再报告CPU/内存优化与减少排队各自贡献。\n'''
 for p in (O/'REPORT.md',R/'research_notes/synergy/CC_CORE80_REPAIR_REPORT_20261031.md'):
  with p.open('x',encoding='utf8') as f:f.write(report)
 with (O/'CURRENT_TASK.md').open('a',encoding='utf8') as f:f.write('\nSOFTWARE VALIDATED/FROZEN. See REPORT.md,software_acceptance.json,source_snapshot.zip. Formal11 remains blocked by auto-review; no launch allowed until explicit researcher reconfirmation and new review. No running tests/processes after final seal. Do not resume any old launcher.\n')
 hashes={str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file() and p.name not in ('packet_hashes.json','seal_receipt.json')};save(O/'packet_hashes.json',hashes);save(O/'seal_receipt.json',dict(files=len(hashes),packet_sha256=sha(O/'packet_hashes.json'),sealed_unix=time.time(),formal_status='BLOCKED',violations=0))
 print('SOFTWARE_PACKET_READY',len(hashes),'files; formal remains blocked')
if __name__=='__main__':main()
