"""Seal a design/offline evidence package; never runs training or changes code."""
import hashlib,json,csv
from pathlib import Path
O=Path(__file__).resolve().parent; R=O.parents[2]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
task=json.loads((O/'CURRENT_TASK.json').read_text()); T=Path(task['algorithm_source'])
source_check={p:sha(T/p)==h for p,h in task['code_source_hashes'].items()}
assert all(source_check.values())
src=R/'official-artifacts/sia-simulator/workloads/saturn/workload-1.csv'
assert sha(src)==task['official_workload_sha256']
rows=list(csv.DictReader(src.open(encoding='utf8')))
for n in (6,8):
 items=list(csv.DictReader((O/f'inputs/proposed_{n}job.csv').open(encoding='utf8')))
 expected=[(i,r) for i,r in enumerate(rows) if r['application']=='cifar10'][:n]
 assert len(items)==n
 for x,(i,y) in zip(items,expected):
  assert int(x['original_source_row_index'])==i and float(x['original_time'])==float(y['time']) and float(x['time'])==0
  assert all(x[k]==v for k,v in y.items() if k!='time')
e=json.loads((O/'conditional_envelope.json').read_text()); c=json.loads((O/'coverage.json').read_text())
m=e['cases']['medium']; neutral=e['cases']['neutral']
design={
 'status':'DESIGN_ONLY_NOT_AUTHORIZATION_OR_COMPLETED_SIMULATION',
 'label':'SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE',
 'central_hypothesis':'辅助配额优化后GPU配置排序翻转。FULL联合选择的价值应超过FO在既定GPU配置上充分优化辅助配额后的最好结果，而非仅超过比例分配。',
 'mathematical_scope':{
  'rate':'v_i(g,q)=raw_current_training_rate_i(g)*rho_i(g,q)',
  'eta':'d_i(g,placement,current_state)+remaining_work_i/v_i(g,q)',
  'shared_objective':'U=sum_i 1/eta_i (current implementation; same compulsory service set, legality and locks)',
  'fo':'g_FO=argmax U(g,proportional(g)); then maximize U over current legal auxiliary quotas/placement at that fixed config_id',
  'full':'maximize U over legal GPU configurations and auxiliary quotas/placement together',
  'distinct_opportunity':'argmax of proportional score differs from argmax of optimized auxiliary envelope',
  'limits':'候选包含只证明同状态、同服务集合、同约束且精确求解时FULL目标不劣；不证明全局JCT最优或所有作业更快。FO是项目完成导向串行策略，不是原始Sia。'},
 'scenario':{
  'name':'GPU配置自由度下的持续辅助资源竞争',
  'main_workload':'官方workload-1源行0–5的6个CIFAR；同时到达0；全量batch/demand/声明工作保留；不强制放置或改GPU候选',
  'nodes':[{'type':'aws','gpus':4,'synthetic_cpu':18,'ram_gb':125},{'type':'rtx','gpus':4,'synthetic_cpu':12,'ram_gb':375}],
  'reason_for_gpu_count':'6作业/8GPU允许至少两个额外GPU的分配选择；不再在4个必须服务作业占4GPU时将所有作业强制为单GPU。合法类型内配置仍完整保留1/2/4GPU。',
  'reason_for_arrival':'取消第二批晚到达在固定时刻把数量选择压回全单GPU的结构；不按新JCT选到达时间。',
  'reason_for_cpu':'AWS按上一场景同比扩大；RTX每GPU3CPU是现有整数网格中覆盖CIFAR RTX1 phi_c=2.2016867的最小点，避免比例点低CPU越界；不是按JCT选容量。',
  'expected_but_not_forced':'可能出现4个RTX单GPUpeer以及2个AWS任务：FO先选AWS2+2，FULL能选AWS1+1，把副本占用的辅助资源转给有效训练。实际选中任务/类型必须由原控制器自然产生。',
  'unchanged':['现已修复的FO/FULL恢复能力','目标和准入/持续服务','候选网格与残余容量规则','Medium alpha=.5及R0/E-legacy映射','checkpoint及真实切换成本','60s周期和6000轮统一观察上限','画像初始化seed_profiles(1,8)'],
  'control':'同8GPU资源，取官方前8个CIFAR（源行0–6、10）同时到达；仅当8任务确实共同服务且均至少1GPU时，GPU数量自由度消失。GPU类型仍可选择、任务完成后自由度会恢复，因此不是要求整条轨迹FO=FULL。'},
 'proposed_matrix_not_started':[
  {'scenario':'6job_8gpu','response':'medium','policies':['FO','FULL']},
  {'scenario':'6job_8gpu','response':'neutral','policies':['FO','FULL']},
  {'scenario':'8job_8gpu_degree_control','response':'medium','policies':['FO','FULL']}],
 'offline_evidence':{
  'state_rule':e['observed_state']['selection_rule'],'state_time':e['observed_state']['time'],'jobs':e['jobs'],
  'conditional_only':e['scope'],
  'fo_aux_search_feasible_pairs':m['best_fixed_gpu_fo']['feasible_pairs'],
  'full_search_feasible_pairs':m['best_joint_full']['feasible_pairs'],
  'fo_gpu':[x['config'] for x in m['best_fixed_gpu_fo']['selected']],
  'full_gpu':[x['config'] for x in m['best_joint_full']['selected']],
  'objective_gain_pct':100*m['relative_u_improvement'],
  'supported_witness_objective_gain_pct':100*(m['best_joint_supported_witness']['objective']/m['best_fixed_gpu_fo']['objective']-1),
  'hold_current_quota_forever_eta_sum_change_pct':100*(-m['sum_eta_improvement']),
  'neutral_objective_gain_pct':100*neutral['relative_u_improvement'],
  'critical_interpretation':'FULL使较近完成者更快但另一任务在当前低配额下变慢。静态sumETA增大不能忽略；它又假定首个完成后配额永远不变，不能反向证明完整JCT更差。未来必须记录合法资源释放和donor实际追赶。不得把任何静态数字写成JCT收益。',
  'known_confounders':['移植旧1860s训练状态到新容量，不是六任务新初态','两个旧当前配置都AWS1，FO2GPU各含50s切换而FULL1GPU为0；不代表相同初始启动条件','其余4个RTX peer仅是固定可行见证，未证明控制器自然选中','FO最优和FULL最优donor均涉及clipping；另有完全受支持FULL见证，但仍不能声明全轨迹无裁剪','既有恢复许可/持续服务状态未参与此资源子问题；需要真实当前资格检查']},
 'coverage':{
  'gpu_configs':c['domain'],'missing':c['missing'],
  'per_job_candidate_count':sum(v['columns'] for v in c['coverage'].values()),
  'numerator_clipped_count':sum(v['numerator_clipped'] for v in c['coverage'].values()),
  'denominator_clipped_count':sum(v['denominator_clipped'] for v in c['coverage'].values()),
  'all_proportional_numerator_denominator_supported':all(not p['source_coordinate_clipped'] and not p['baseline_coordinate_clipped'] for v in c['coverage'].values() for p in v['proportional']),
  'warning':'候选枚举含raw=1占位，仅用于物理/响应覆盖；静态评分用历史raw。未观测AWS4/RTX4的新在线速率未用R0伪造。6任务候选/求解规模尚未验收。'},
 'next_gate_design_only':{
  'preflight':['复核344源码、模型和输入哈希','400项既有测试及新增容量适配回归，不能修改目标/网格','一次候选规模/内存预检；2GiB内存/5GiB磁盘门槛；不能为省时静默删候选'],
  'real_decision_gate':'固定60轮机会窗口（最多3600s），FO/FULL前缀各自在相同真实状态克隆做只决策比较；使用同一服务集合、真实已观测性能、checkpoint和许可预算；不强制GPU类型/数量；不根据新JCT延长或换样本。',
  'opportunity_pass':['固定FO GPU的辅助优化已充分完成，而FULL自然选择不同合法GPU配置','至少6个连续调度周期出现同方向配置机会；记录窗口覆盖的实际工作量，不能只记录次数','关键receiver/donor比较点分子分母均有支持；无法满足则标模型依赖而非可信机制通过','检查所有活动作业的真实净进度、启动损失及资源占用，不仅receiver；记录必要的合法后续重配置/释放资格','真实controller的恢复资格相同、锁不破坏、整体原子提交无违规'],
  'performance_evaluation':'通过机会和正确性门槛后才建议完整6格；FO/FULL完整全部作业JCT是最终判据。均值、逐作业、tail、makespan、服务时间分解及资源小时并列报告；只会计receiver收益不算通过。若机会未出现或完整JCT不改善，保存结果停止，不从多个容量/种子选最好。',
  'separation':'机会门槛检验机制能否实际施展，不能以未来JCT收益作为选择场景的门槛；完整结果允许持平或退化。'},
 'alternative_not_ready':'CPU丰富/RAM丰富节点配合不同任务的GPU类型重配也能形成FO固定类型后无法修复的机会，但现有固定模板没有天然正交的纯CPU/纯RAM两类；BERT在支持域平坦且GPU亲和性强。未取得同预算最优FO仍落后的证据前，不把CIFAR+BERT命名混合当作必然互补。',
 'what_was_not_done':['未启动训练或正式运行','未修改算法/官方源/恢复预算/响应','未新增调度参数','未覆盖历史文件','未承诺新场景JCT提升比例']}
save('proposed_design.json',design)
save('validation.json',{'actual_source_files':len(source_check),'source_unchanged':all(source_check.values()),'official_trace_hash_valid':True,'first6_first8_extraction_valid':True,'new_training_runs':0,'simulation_started':False,'static_enumeration_completed':True,'scope':'resource subproblem and model-domain coverage only','initial_diagnostic_error':'normal rounds logs strip continuity_scores; preserved initial failure, then read already-existing rich gate logs without repairing historical logs'})
save('artifact_hashes.json',{str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})
print('Design and offline evidence saved; no simulation run; source files unchanged:',len(source_check))
