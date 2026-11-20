"""Seal independent audit, history protection and complete report; no simulation."""
import csv,gzip,hashlib,json
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-recovery-equivalent-v1';OLD=O.parent/'cc_persistent_opportunity_experiment_20261101_run1'
def read(n):return json.loads((O/n).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
p=read('preregistration.json');a=read('raw_audit.json');jobs=read('per_job.json');pairs=read('comparison_summary.json');recs=read('recovery_events.json');gate=read('primary_receiver_gate_audit.json');repair=read('repair_effect.json')
for rel,h in p['source_hashes'].items():assert sha(T/rel)==h,rel
for rel,h in p['input_hashes'].items():assert sha(Path(rel))==h,rel
for rel,h in p['script_hashes'].items():assert sha(O/rel)==h,rel
base=read('baseline_receipt.json')
for rel,h in base['hashes'].items():assert sha(Path(base['parent'])/rel)==h,rel
oldhash=json.loads((OLD/'artifact_hashes.json').read_text())
for rel,h in oldhash.items():assert sha(OLD/rel)==h,rel
assert all(v['complete'] and not v['errors'] for v in a.values())
assert pairs['balanced_neutral_FO']['equal']==6
assert not recs['balanced_medium_FO'] and not recs['balanced_medium_FULL']
late={}
for m in ('FO','FULL'):
 late[m]=[]
 for l in gzip.open(O/'cells'/('balanced_medium_'+m)/'rounds.jsonl.gz','rt',encoding='utf8'):
  r=json.loads(l)
  if r['interval_end'] in (240,1800,4920,5940):
   late[m].append(dict(time=r['interval_end'],selected={j:dict(config=c['config']['config_id'],cpu=c['cpu_per_gpu'],ram=c['ram_gb_per_gpu'],rho=c['rho']) for j,c in r['decision_for_next_interval']['result']['selected'].items()},
    jobs=r['job_end_state']))
save('mechanism_states.json',late)
v=dict(status='COMPLETE_REPAIR_AND_SIX_CELLS_AUDITED',tests=read('test_summary.json'),
 source_files=len(p['source_hashes']),source_snapshot_sha256=p['source_snapshot_sha256'],preregistration_sha256=sha(O/'preregistration.json'),
 source_unchanged=True,input_unchanged=True,frozen_scripts_unchanged=True,prior_source_unchanged=True,old_sealed_artifacts_verified=len(oldhash),
 cells_completed=6,raw_errors=0,neutral_per_job_equal=True,new_medium_tail_recoveries=0,primary_gate=gate,
 engineering_change_only=['solo recovery quota closure on same GPU geometry','consume and log actual selected permit column'],
 limits=['synthetic transfer and E-legacy clipping','one small controlled trace','GPU/CPU/RAM joint effect, no independent CPU-only/RAM-only JCT ablation','later trajectories differ; common-time states are not exact counterfactuals'])
save('validation.json',v)
for cell in p['cells']:
 for file,h in json.loads((O/'cells'/cell['id']/'output_hashes.json').read_text()).items():assert sha(O/'cells'/cell['id']/file)==h
F=a['balanced_medium_FO'];J=a['balanced_medium_FULL'];P=pairs['balanced_medium_FO']
imp={key:100*(1-J[key]/F[key]) for key in ('mean','median','p95','p99','makespan')}
save('metric_improvements.json',imp)
lines=['# 恢复资格等效修复与持续资源竞争场景',
'','SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE','',
'## 结论','',
'恢复漏洞已修复。旧12作业场景修复后FO平均3318.50秒、FULL3273.75秒，FULL优势由14.28%降至1.35%。FO的bert-159从12286降至6278秒；其他11作业JCT未变。不能继续用旧14.28%作为干净的联合优化增量。',
f'新6作业场景Medium下FO平均{F["mean"]:.2f}秒、FULL{J["mean"]:.2f}秒，改善{imp["mean"]:.3f}%；Neutral两者均4262.833秒，六个作业JCT完全相同。新场景的两种方法均没有触发尾部/多任务恢复例外，因此新收益不是恢复能力不等效造成的。',
'净JCT节省1134秒＝训练少1034秒＋checkpoint少100秒＋等待少0秒。4作业改善、1退化、1持平；这次不是靠减少排队得到平均收益，也不能说所有任务更快。',
'','## 工程修复和边界','',
'父源码为cc-joint-persistent-v1的完整343文件实际执行快照；新树cc-joint-recovery-equivalent-v1，共344文件。两者不是Git worktree，不能拿裸HEAD代替执行版本。本轮未提交或推送。可重建版本是source_snapshot.zip＋preregistration.json的逐文件SHA-256；本轮仅改两个生产文件并新增一个测试文件。',
'- `sia-simulator/joint_prototype/clean_ablation.py::CleanAdmission.shared_eligible`：保持原最优目标GPU geometry＝config＋placement；将该geometry内当前已生成、已画像、独立合法性审计通过、正有限ETA/raw、实际启动成本一致、严格优于当前ETA的辅助配额候选共同获准。没有新增网格，没有放行其他geometry，没有加bonus或放宽预算。',
'- `sia-simulator/joint_prototype/tail_recovery.py::TailRecoveryAdmission._decide`：许可由单column_id扩为同geometry的合法column_ids，按实际选中列核销一次许可并记录实际eta_after/成本。旧单列多任务纠偏保持兼容。真实checkpoint、一次恢复、恢复后GPU冻结和全量原子审计保留。',
'- `sia-simulator/tests/test_recovery_equivalence.py`：真实oracle/candidate/MIP的7项红绿测试。旧版4失败，新版7通过；覆盖FO/FULL等效资格、比例/非比例实际核销、锁、不能偿还成本、未知模型/延迟错误、列顺序、求解失败回滚和不能二次恢复。',
'所谓等效是资格规则和可达合法辅助形状相同，不是强迫FO/FULL选择相同GPU。如果比例点本身不能抵偿真实切换成本，仍拒绝；不能以公平为由放宽物理约束。',
'400项完整测试全部通过（343.699秒）；另2项参数化适配/精确回归通过。原始Sia、FO/FULL各三轮精确回归通过。旧全空反例等历史测试保留。本轮没有修改目标函数、候选网格、响应、phi、clipping、solver参数或训练总工作量。',
'','## 为何调整场景','',
'旧DGX极快：两个peer在1320/1380附近完成，接收者随后迁到DGX，早期AWS进度优势被后续高吞吐摊薄。旧cifar10-2的114秒JCT收益实际是100秒checkpoint＋14秒训练，而非114秒全部来自资源加速。1200秒新任务到达也会迫使FO/FULL都降到AWS1。仅延迟到达不足以普遍解决这两个结构问题。',
'本次是预先固定的一种新场景，不是参数搜索：官方workload-1前6个CIFAR记录，源行0–5，batch/demand/应用工作量保留；按源行顺序分两组，在0和1800秒到达。1800秒规则为300秒冷启动预算＋1500秒竞争预算，未看新JCT后修改。两组不是独立重复实验。',
'AWS节点2GPU/9合成CPU/62.5GB；RTX节点2GPU/3合成CPU/187.5GB。将旧DGX换为RTX，合计仍4GPU/12CPU/250GB。R0单GPU吞吐参考AWS747.86、RTX736.21，相对接近；双GPU1304.92/948.96，扩容收益不同。它们只用于场景设计与固定phi，不灌入在线Goodput；真实训练仍走Sia模型。',
'这是合成节点规格，不是AWS/RTX真实主机。全部方法使用同一R0/E-legacy、alpha=.5/.0、seed_profiles(1,8)、60秒周期、6000轮上限及2GiB内存/5GiB磁盘门槛。FO仍是本项目完成导向的串行GPU→辅助资源策略，不冒称原始Sia。',
'','|作业|到达s|原demand|原batch|','|---|---:|---:|---:|']
with (O/'inputs/grouped6.csv').open(newline='') as f:
 for r in csv.DictReader(f):lines.append(f'|{r["name"]}|{r["time"]}|{r["num_replicas"]}|{r["batch_size"]}|')
lines+=['','## 数据机会、持续性与收益传导','',
'四个合法GPU配置的离线覆盖闭合，422个物理合法辅助候选，缺失0。固定AWS2时节点已经满载，每GPU最多4.5CPU/31.25GB，rho最高约0.618327；改变GPU数量后，AWS1可以得到9CPU/62.5GB，rho约1.611957。FO在固定GPU后无法凭辅助优化跨越这个容量上限。',
'分轴离线查询进一步表明这里是CPU/RAM耦合而非纯CPU场景：AWS1的(4.5,31.25)rho=.620047；仅加CPU到(9,31.25)仅.622926；仅加RAM到(4.5,62.5)为1.209282；两者共同(9,62.5)为1.611957。这四点分子、分母均在矩阵支持范围内。不能把联合JCT差值全归CPU单轴。',
'240秒真实同状态优化确认FO自然选择AWS2、FULL选择AWS1；同服务集合，另外两个RTX配置相同，FULL共同目标更高且sumETA更低。第一次机会既不是强制placement，也不是手写Goodput。',
'主要接收者cifar10-2的配置差异持续240–1800，共26周期/1560秒。该窗口FULL推进13107.31（其声明总工作量33.55%），FO推进8494.83，差4612.48；cifar10-1窗口推进少390.68，cifar10-0相同，净增4221.80。不能忽略peer代价。',
'注意窗口增量与累计状态不同：包括180–240冷启动阶段，cifar10-1在1800的累计进度反而领先52.66。窗口内-390.68还包含此前状态分叉后的梯度/epoch演化，不应直接称为纯CPU让出损失。primary_receiver_gate_audit.json将短暂CF1 geometry变化与持久CF2收缩分开，声明总工作由当前remaining_work＋progress复算，没有读取未来完成时间。',
'60轮预检只证实一段持续窗口，未证明第二批必然再次触发。完整训练后来出现新的实际资源组织差异：4920秒两种方法都服务CF3/4/5，FO让CF3占RTX2，CF4/5分享AWS；FULL让CF3/4各用RTX1，将AWS1的CPU9/RAM62.5集中给CF5。5940秒两者都只服务CF4/5，CF4都是RTX2，而CF5在FO为AWS2低配额、FULL为AWS1高配额。直到6720秒又持续出现耦合配置差异。',
'这些后期记录来自已经分叉的真实轨迹，不是完全相同训练状态的精确反事实。它们解释资源释放与重配置的传导，不能证明某一个局部决定独自造成全部JCT变化。没有新前瞻模型、没有读取未来到达。',
'','## 六格完整指标','',
'|格|完成|平均JCT|Median|P95|P99|Makespan|','|---|---:|---:|---:|---:|---:|---:|']
for n,s in a.items():lines.append(f'|{n}|{s["completed"]}|{s["mean"]:.2f}|{s["median"]:.2f}|{s["p95"]:.2f}|{s["p99"]:.2f}|{s["makespan"]:.2f}|')
lines+=['','新6作业Medium：'+', '.join(f'{k}改善{v:.2f}%' for k,v in imp.items())+'。P95/P99仅为6个作业的经验分位，不能当总体尾延迟精确估计。','',
'|新Medium作业|FO JCT|FULL JCT|节省s|等待节省s|训练节省s|checkpoint节省s|','|---|---:|---:|---:|---:|---:|---:|']
for j,f in jobs['balanced_medium_FO'].items():
 u=jobs['balanced_medium_FULL'][j]
 lines.append('|'+j+'|'+f'{f["jct"]:.0f}|{u["jct"]:.0f}|'+ '|'.join(f'{f[k]-u[k]:.0f}' for k in ('jct','wait','training','checkpoint'))+'|')
lines+=['','CF2的361秒改善含311秒训练和50秒checkpoint；CF3退化256秒，包含训练多306秒、checkpoint少50秒。CF4/5则分别改善318/707秒。整体并非所有作业更快，CF3可以被看作后期资源配置取舍中的受损任务。',
'CF5贡献707/1134＝62.35%净JCT节省；去掉该任务后其余5作业诊断性均值仍减少85.4秒。该诊断不替代全6作业主结果。新场景没有BERT，FO/FULL恢复例外次数均0，不能将CF5收益误归于修复后的尾恢复。',
'','|格|等待s|训练s|checkpoint s|restart|GPU h|CPU h|RAM GB h|wasted GPU s|裁剪查询比例|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for n,s in a.items():
 t=s['totals'];lines.append(f'|{n}|{t["wait"]:.0f}|{t["training"]:.0f}|{t["checkpoint"]:.0f}|{t["restarts"]}|{t["gpu_hours"]:.4f}|{t["cpu_core_hours"]:.4f}|{t["ram_gb_hours"]:.4f}|{t["wasted_gpu_seconds"]}|{100*s["clipped_fraction"]:.2f}%|')
lines+=['','## 恢复修复与历史证据','',
'旧FO与修复FO前121轮完整分配/作业状态相同，首次分叉7320秒；只有bert-159的JCT变化（减少6008秒）。FULL的全部作业JCT保持相同，但7260及下一轮辅助配额存在同速候选的不同选择，不能说全部逐轮资源字节一致。实际许可按选中CPU1/RAM20.83列核销，FO的比例第一阶段已能合法选择同GPU geometry。',
'新的旧场景FO/FULL净差537秒：等待360、训练77、checkpoint100。相比先前6545秒净差，大部分差距确实来自基线恢复不等效，而不是原策略突然失效。旧文件没有覆盖，repair_effect.json保留逐作业及逐轮对比。',
'','## 审计与限制','',
'独立audit从原始完成事件、训练片段与资源分配复算均值、JCT互斥时间、资源小时、rho、锁和容量；6格全部完成，错误0，未插补。额外核验locked作业不能消失、FO两阶段GPU意图固定、恢复许可实际列核销及预算。',
'新Medium已选择查询的裁剪比例FO55.42%、FULL62.33%；Neutral98.35%（alpha0时rho仍严格1）。比例定义是一次已选oracle查询的分子或分母任一坐标裁剪，不是训练时间比例。主AWS比较点受支持，不代表RTX peer或全部迁移模型都经过真实硬件验证。',
'Neutral两策略逐作业和聚合持平，Medium差189秒/作业，因此条件交互差为189秒。该结果支持联合决策在这一受控响应下有额外价值；无法据此拆成CPU与RAM各自的因果JCT贡献。Medium下FULL仍比自己的Neutral基线慢62.5秒/作业，不能称瓶颈已经消除。',
'这只是一种6作业合成场景；两批之间状态相关，不能按6个独立重复计算统计显著性。FO/FULL有相同准入/服务与恢复规则，GPU与配额选择不同。求解墙钟不计模拟JCT，沿用旧实验口径。',
f'正式六格墙钟{read("pipeline_complete.json")["wall"]:.3f}秒；门槛、回归与离线审计另计。没有80/160作业正式运行，没有改参或失败后重选数据。',
'','## 可重建交付及后续判断','',
f'- 源码快照SHA-256：`{p["source_snapshot_sha256"]}`',
f'- 预注册SHA-256：`{sha(O/"preregistration.json")}`',
'- baseline_receipt.json、repair.diff、source_snapshot.zip：实际父源码哈希、最小差异、修复源码与依赖锁。',
'- design_manifest.json、inputs/、offline_coverage.json：固定选择规则、六作业源行、容量及422列响应数据。',
'- red.log、green.log、full_tests.log、test_summary.json、regressions.log、*_regression.json：红绿和回归。',
'- gate_*.jsonl.gz、gate_result.json、primary_receiver_gate_audit.json：同状态预测、真实推进及持续窗口。',
'- cells/、execution_manifest.jsonl、pipeline_complete.json：六格原始轮次、命令、资源门槛、完成与输出哈希。',
'- raw_audit.json、per_job_utf8.csv、comparison_summary.json、repair_effect.json、recovery_events.json、key_job_transitions.json、mechanism_states.json：独立复算、逐任务退化、恢复核销与资源传导。',
'- validation.json、artifact_hashes.json：新旧完整性核验和封存。',
'结论：新场景更直接触及固定GPU之后无法解决的每GPU辅助供给不足，获得4.19%平均JCT下降，训练时间贡献占净收益91.18%，且P95/P99/makespan同时改善；证据比存在恢复漏洞的14.28%更干净，但仍不足以宣称大幅、普遍、真实集群提升。下一步应以该已修复版本做独立的预注册同机制复验及轴向消融，而不是继续从本次JCT选容量。本轮已完成并停止，没有自动扩大。']
(O/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
(O/'CURRENT_TASK.md').write_text('完成并停止。恢复资格GPU geometry闭合＋实际列许可核销修复；400测试及Sia/FO/FULL三轮精确回归通过。旧12作业修复后FO3318.50/FULL3273.75，优势1.3485%。新6作业AWS+RTX两批场景MediumFO4514.333/FULL4325.333，改善4.1867%，Neutral两者4262.833。六格全部完成原始审计0，未改参。新Medium净训练节省1034秒、checkpoint100秒、等待0；CF3退化256秒，CF5贡献62.35%。新场景无恢复例外。历史source343及全部旧封存文件核验未变。详见REPORT.md/validation.json。不要自动扩大或恢复历史任务。\n',encoding='utf8')
save('artifact_hashes.json',{str(x.relative_to(O)):sha(x) for x in sorted(O.rglob('*')) if x.is_file() and x.name!='artifact_hashes.json'})
print('SEALED COMPLETE',imp)
