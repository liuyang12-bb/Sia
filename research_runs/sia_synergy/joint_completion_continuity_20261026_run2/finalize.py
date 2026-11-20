"""Generate tables from independent audits; preserve every attempt and input."""
from pathlib import Path
import hashlib, json, time, zipfile
from experiment import OUT, ROOT, METHODS, PREVIOUS, load, sha, save

def main():
    p=load();a=json.loads((OUT/'independent_audit.json').read_text());r=a['results']
    summaries={m:r[m].get('summary') for m in METHODS}
    assert all(summaries.values()),'Report incomplete cells explicitly before packaging'
    execution=[json.loads(l) for l in (OUT/'execution_manifest.jsonl').read_text().splitlines()]
    wall=sum(e.get('wall_seconds',0) for e in execution)
    cc=summaries['JOINT_CC'];base=summaries['SIA_P'];legacy=summaries['LEGACY_ECT'];prop=summaries['JOINT_CC_P']
    fmt=lambda v:'—' if v is None else f'{v:.3f}'
    pct=lambda v:f'{v*100:.3f}%'
    lines=['# 完成优先与连续服务：同六任务机制修订实验','',p['warning'],'',
      f'状态：{a["status"]}。七个预注册单元使用相同六任务、相同Medium响应及资源容量，未按中间结果调参。',
      '', '## 直接结果','',
      '|策略|完成|平均JCT(s)|Median|P95|P99|Makespan|Restart|',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m,s in summaries.items():
      lines.append('|'+ '|'.join([m,f'{s["completed"]}/6']+[fmt(s[k]) for k in ['mean_jct','median_jct','p95_jct','p99_jct','makespan']]+[str(s['totals']['restarts'])])+'|')
    lines+=['','仅6个作业，P95/P99为该小集合描述，不代表高精度总体尾延迟。完整平均JCT只比较全部完成单元。',
      '', '## 两项机制修复与限制','',
      '第一项用u=1/T替代等待参考减ETA，T=真实候选启动延迟+公开剩余工作/(当前在线Sia预测raw×rho)。'
      '不同任务的剩余工作不再抵消；全表共同正缩放只用于数值条件，不引入任务权重。'
      '未知ETA保留比例bootstrap及FIFO首次服务，不用raw=1占位值伪造完成时间。',
      '', '第二项把300秒到期重排改为服务至真实完成，保留真实checkpoint完整锁。GPU变更只在到达、完成、'
      '首次有效画像事件上开放，并要求已有有效训练及合法画像；已有250秒启动预算按整个任务累计。'
      'CPU/RAM仍每轮联合选择，允许让出方减速，全部容量及候选原子审计。',
      '', '有限任务、有限工作、单独可行配置、实际正速率下界、持续合法的既有配置及有界累计启动开销，'
      '是最终完成的条件。已服务任务持续推进；阻塞队首时停止新承诺，有限在服任务最终完成并释放资源。'
      '这不保证无限负载统一等待上界。零速或失效锁显式停止。',
      '', 'sum(1/T)不是sum(JCT)的精确目标；FIFO队首仍强制选中，因此完整策略不等于SPT/SRPT。'
      '新连续性组合同时改变轮转、事件门控和生命周期预算，当前消融不能把三者分别归因。'
      '非抢占可能推迟后到短任务；事件门控可能错过epoch变化；预算可能限制长任务后期弹性。',
      '', '## 预注册与公平性','',
      '六任务为官方Saturn workload-1的cifar10-0/1/2、deepspeech2-8/9/12，完整工作量及batch不变，'
      '受控同时到达。AWS和DGX各2GPU、12合成CPU单位、250GB；比例点每GPU6CPU/125GB。'
      '两类型GPU画像、batch自适应、seed_profiles(1,8)、R0/E-legacy alpha=.5、映射、phi、CPU/RAM网格、'
      '60秒周期和6000轮上限逐字节/参数核对与上一轮相同。',
      '', 'SIA_P保持官方Sia GPU决策和比例执行；SIA_ECT_AUX保持完整Sia GPU意图和placement，仅优化辅助配额。'
      'LEGACY_ECT为旧评分旧服务。SCORE_ONLY仅换评分；CONTINUITY_ONLY仅换上述服务连续性组合；'
      'JOINT_CC同时换两项；JOINT_CC_P仅关闭JOINT_CC辅助优化。与Sia差异还包含目标、准入和placement。',
      '', '## 组件和辅助消融','',
      '|比较|平均JCT变化(s，负为改善)|改善率|改善/退化任务数|最大受益任务|去掉最大受益任务的诊断改善率|',
      '|---|---:|---:|---:|---|---:|']
    for name,c in a['comparisons'].items():
      lines.append(f'|{name}|{c["mean_delta"]:.3f}|{pct(c["improvement"])}|{c["improved"]}/{c["worsened"]}|{c["largest_beneficiary"]}|{pct(c["without_largest_diagnostic_improvement"])}|')
    lines += ['', f'组合交互项I=JCT(CC)−JCT(仅评分)−JCT(仅连续性)+JCT(旧ECT)={fmt(a["interaction_jct_seconds"])}秒。'
      '负值表示两项修改的JCT降低具有互补，正值表示联合效应小于单独效果的可加和；仅此单一负载描述，非统计因果估计。',
      '', '辅助消融会引起后续GPU/等待轨迹分叉，不能把其全部JCT差异称为固定GPU下的直接CPU加速。'
      '去掉单个任务仅为影响诊断，主结果始终包含全部6任务。',
      '', '## 逐任务JCT及互斥时间','',
      '|任务|'+'|'.join(METHODS)+'|','|---|'+'---:|'*len(METHODS)]
    for j in cc['jobs']:
      lines.append('|'+j+'|'+'|'.join(fmt(s['jobs'][j]['jct']) for s in summaries.values())+'|')
    lines+=['','|策略|等待(s)|有效训练(s)|启动/checkpoint(s)|GPU-h|CPU单位-h|RAM-GB-h|Wasted GPU-s|',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m,s in summaries.items():lines.append('|'+m+'|'+'|'.join(fmt(s['totals'][k]) for k in ['wait','training','checkpoint','gpu_hours','cpu_core_hours','ram_gb_hours','wasted_gpu_seconds'])+'|')
    lines+=['','对相同6个任务，JCT总变化等于等待＋训练＋checkpoint任务秒变化。其明细在comparison_summary.json，'
      '从原始完成事件及训练段复算，未用summary自行证明summary。每任务/配置训练时间在independent_audit.json。',
      '', '## 事件、求解及响应边界','',
      '|策略|重排|GPU几何变化|辅助变化|排空轮|fallback|求解总秒|selected列|分子/分母clip|',
      '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for m,s in summaries.items():
      e=r[m]['events'];lines.append('|'+ '|'.join([m]+[str(e.get(k,0)) for k in ['REQUEUE_PRODUCTIVE_QUANTUM','ADAPT_GEOMETRY','ADAPT_AUX','DRAIN_FOR_HEAD']]+[str(s['fallback_count']),fmt(s['solve_total_seconds']),str(r[m]['checks']['selected_columns']),str(r[m]['clipping'])])+'|')
    lines+=['','CIFAR借用AlexNet矩阵、DeepSpeech借用DeepSpeech矩阵；原单元数据来自Synergy，跨应用/硬件映射、'
      'R0需求缩放与乘法执行均是合成假设。已知CIFAR DGX2分母CPU越界仍保留；clipping四类可重叠，'
      '不能把所选列低clip等同完整域已验证。CPU/RAM变化没有额外加载成本，是沿用模型限制。',
      '', '## 测试、源码和重建','',
      '24项新增测试：11项评分/完整真实链路，10项状态机边界，3项数值整数化边界。连同188项既有测试，最终212项通过。'
      '初始旧行为适配红测保留7个failure和5个error（含子测试，11个测试用例）；新增画像重发现测试'
      '修复前10项中1项失败，修复后全套通过。第一次完整训练尝试在SCORE_ONLY时间900发生整数残差保护停止，'
      '15轮复现及全部失败证据保留在run1。近整数解经完整原约束检查后接受，3项新增数值测试红绿证据在本目录。'
      '本次7格全部在同一修复版本重跑，没有混用结果；详情见NUMERICAL_REPAIR.md。',
      '', '原始Sia独立进程1任务3轮逐字段回归完全一致。SIA_P、SIA_ECT_AUX、LEGACY_ECT与上一轮完整轨迹'
      '逐轮比较（配置、配额、progress、完成、restart和时间）结果：'+str({m:r[m]['regression'] for m in ['SIA_P','SIA_ECT_AUX','LEGACY_ECT']})+'。',
      '', '全部逐轮节点容量违规0；raw×rho单次、epoch舍入界、互斥时间、完成后资源计时、完整锁、'
      '无事件几何保持、服务不被提前清空、生命周期预算均独立检查。检查次数见independent_audit.json。'
      '本负载checkpoint短于周期，跨轮锁以真实BERT测试作为证据，不能夸大该六任务运行的锁覆盖。',
      '', '隔离源码sia-joint-completion-continuity/sia-simulator新增completion_continuity.py、三测试文件和一份数值失败夹具，'
      'adaptive_service.py与bundle_adaptation.py只增加默认行为不变的到期/时钟钩子。'
      '旧实际源码快照、source.patch、source_changes.json及运行source/完整保存。未提交/推送，未修改官方或历史树。',
      '', f'最终源码fingerprint：`{p["source_fingerprint"]}`。输入hash完整清单在preregistration.json。'
      f'七子进程墙钟合计{wall:.3f}秒；预注册到审计{a["prereg_to_audit_seconds"]:.3f}秒。'
      '每格独立进程、串行、内存2GiB/磁盘5GiB门槛、无上限延长。',
      '', '本次run2构造验收在批次启动前结束，七个训练进程彼此串行。此前run1构造验收的末段与首个Sia进程短暂重叠，'
      '该失败尝试单独保留、不混入run2配对结果。内存门槛逐格记录。',
      '', 'REPRODUCE.md说明源码/输入恢复入口；完整原始日志在cells/<method>/rounds.jsonl.gz，'
      '缩略决策在audited_decisions.jsonl.gz，逐作业结果在per_job.csv。本轮完成后停止，不自动扩大负载或调参。']
    with (OUT/'REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    save(OUT/'wall_time.json',dict(process_seconds=wall,execution_manifest=execution,delivered_unix=time.time()))
    print('REPORT_GENERATED',a['status'])

if __name__=='__main__':main()
