"""Render results without selecting favorable runs or omitting any jobs."""
import json
from experiment import OUT, METHODS, load, save


def main():
    p=load(); a=json.loads((OUT/'independent_audit.json').read_text())
    if a['status'] != 'ALL_COMPLETE_VALID':
        lines=['# 净完成时间决策修正：未完整配对', '', p['warning'], '',
            '固定6000轮观察窗或安全停止导致本轮没有六策略完整配对结果。不得插补JCT或只用已完成任务均值宣布收益。',
            '', '独立审计详情见independent_audit.json；原始日志和失败文件完整保留。']
        for method, result in a['results'].items():
            s=result.get('summary',{})
            lines.append(f'- {method}: {result.get("status",s.get("termination"))}; completed={s.get("completed")}; RMST={s.get("rmst")}')
        with (OUT/'REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
        print('REPORT_INCOMPLETE_STOP')
        return
    r=a['results']; s={m:r[m]['summary'] for m in METHODS}
    gate=json.loads((OUT/'offline_gate.json').read_text())
    reference=json.loads((OUT/'reference_audit.json').read_text())
    loss=json.loads((OUT/'loss_evidence.json').read_text())
    assert reference['valid'] and loss['valid']
    main_result=s['JOINT_NC'];prop=s['JOINT_NC_P']
    executions=[json.loads(v) for v in (OUT/'execution_manifest.jsonl').read_text().splitlines()]
    wall=sum(v.get('wall_seconds',0.) for v in executions)
    lines=['# 同服务集合净完成时间决策修正实验','',p['warning'],'',
        '## 直接结论','',f'审计状态：{a["status"]}。固定同一六任务、Medium、资源容量和全部物理执行参数；无结果驱动调参。',
        '', '**工程实现通过，性能改进未通过；不建议用本轮NC替换上一版CC。** '
        f'新NC平均JCT为{main_result["mean_jct"]:.2f}秒，相比旧CC增加'
        f'{100*(main_result["mean_jct"]/s["JOINT_CC"]["mean_jct"]-1):.2f}%；'
        f'相对自身比例消融NC-P也增加{100*(main_result["mean_jct"]/prop["mean_jct"]-1):.2f}%。'
        f'虽然仍比Sia减少{100*(1-main_result["mean_jct"]/s["SIA_P"]["mean_jct"]):.2f}%，不能把这一点包装成本次改动成功。'
        '六组全部6/6完成，无删失；旧四组逐轮轨迹完全复现。',
        '', '|策略|完成|平均JCT(s)|Median|P95|P99|Makespan|重启|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m,v in s.items():
        lines.append('|'+m+'|'+str(v['completed'])+'/6|'+'|'.join(f'{v[k]:.3f}' for k in ['mean_jct','median_jct','p95_jct','p99_jct','makespan'])+'|'+str(v['totals']['restarts'])+'|')
    lines += ['', 'SIA_P是官方Sia GPU决策＋比例辅助执行适配；SIA_ECT_AUX是固定完整Sia意图后优化辅助配额。'
        'JOINT_CC/P为上一版原样复跑；JOINT_NC/P为本轮净代价修正及比例消融。P仅限制辅助配额。',
        '', '## 修改的机制和未修改的限制','',
        '第一阶段按旧reciprocal ETA及全部原tie规则得到参考服务集合S；第二阶段固定S，联合选择GPU类型/数量、放置、CPU/RAM，'
        '最小化Σ[d+R/(raw×rho)]。所有任务的预计秒数等权，不再把长任务延后一秒按1/T²折价。'
        '未知ETA的参考bootstrap列完整固定，省略的是候选间相同未知常量，不是伪造0秒；已知作业禁止未知替换。',
        '', '这不是全局JCT最优或下一完成事件规划。服务集合仍来自前一阶段，非抢占、FIFO队首、GPU事件门控和250秒累计启动预算均保留。'
        '后续完成释放资源、epoch速率改变和预测误差仍可能让局部sum ETA与真实JCT不一致。全局训练仿真之前，已明确预注册该限制。',
        '', '源码只新增net_completion.py、test_net_completion.py，以及adaptive_service.py的默认不变primary钩子、'
        'bundle_adaptation.py的baseline可比较性钩子。旧算法目标、工作量、数据、rho、GPU网格、配额网格和容量未更改。',
        '', '## 先失败后通过及同状态机会核验','',
        '旧控制器红测15项中3失败：核心案例真实选(9,3) CPU、预计总时间1280秒，而比例(6,6)为1100秒。'
        '新实现拒绝该交换；当交换总时间为1090秒时仍接受donor稍慢，未恢复逐任务不得减速的限制。'
        '新增17项＋既有212项，最终229项测试通过。真实unseeded BERT完整12轮调度链、跨轮checkpoint、比例契约、原子故障均覆盖。',
        '', '原始Sia独立进程1任务3轮完全匹配；前四个完整六任务对照的配置/配额/进度/完成/restart/时间轨迹逐轮匹配上一版。',
        '', f'固定历史CC round4（首个非比例且ETA完整状态）离线核验，无训练推进：旧倒数目标的预计剩余时间和{gate["old_reciprocal"]["sum_eta"]:.3f}秒，'
        f'比例方案{gate["proportional"]["sum_eta"]:.3f}秒，新sum ETA方案{gate["new_sum_eta"]["sum_eta"]:.3f}秒。'
        f'对固定GPU几何的{gate["enumerated_node_bundles"]}个节点级组合独立穷举，核对两个MIP最优值。'
        '这是局部预测代价修正证据，不是实际JCT必然降低的证明。',
        '', '## 平均与逐任务比较','',
        '|比较|平均JCT变化(s，负为改善)|改善率|改善/退化任务数|最大受益任务|去掉该任务后的诊断改善率|',
        '|---|---:|---:|---:|---|---:|']
    for name,v in a['comparisons'].items():
        lines.append(f'|{name}|{v["mean_delta"]:.3f}|{100*v["improvement"]:.3f}%|{v["improved"]}/{v["worsened"]}|{v["largest_beneficiary"]}|{100*v["without_largest_diagnostic_improvement"]:.3f}%|')
    lines += ['', '主结果包含所有作业。去除单一高影响作业仅用于诊断，不代替主结果；本单一六任务集合的P95/P99不代表总体尾延迟。'
        '辅助开关可能改变后续GPU和等待轨迹，不能把全部差异称为固定GPU的直接CPU加速。',
        '', '|任务|'+'|'.join(METHODS)+'|','|---|'+'---:|'*len(METHODS)]
    for job in main_result['jobs']:
        lines.append('|'+job+'|'+'|'.join(f'{v["jobs"][job]["jct"]:.3f}' for v in s.values())+'|')
    lines += ['', '## 退化原因：能确定什么，不能确定什么','',
        '1. **不只CPU交换有问题。** 关闭辅助优化后，NC-P平均JCT仍从CC-P的4476增至4970.833秒，恶化11.06%。'
        '因此此次更换目标也改变了GPU配置轨迹，不可把全部损失推给CPU画像。',
        '', '2. **首个偏离参考方案的决定是一次GPU互换，不是CPU配额变化。** '
        'NC round3（t=240秒）将CF0从DGX1换到AWS1，同时将CF2从AWS1换到DGX1，两者CPU/RAM仍为6/125。'
        '预计CF2减少2720.967秒，CF0增加2677.098秒，净减少43.868秒（参考总和的0.659%）；两次50秒启动已经计入。'
        '见reference_audit.json的first_structural_difference。这个小净收益依赖当前速度外推，不能证明真实完成事件更早。',
        '', '3. **反复在允许事件换GPU，随后耗尽原有预算。** CF0在round0/1/3/15/17经历'
        'AWS→DGX→AWS→DGX→AWS，累计启动时间50/100/150/200/250秒。round18因SERVICE_DELAY_BUDGET排除70列。'
        '实际在AWS1训练3148秒、DGX1训练140秒；旧CC为10秒和589秒。其JCT从759增至3598秒，增加2839秒。'
        'loss_evidence.json保存预算和拒绝证据，mechanism_trace.json保存全作业配额/配置轨迹。'
        '证据支持“局部目标、当前速率估计、事件门控和累计预算交互造成慢GPU驻留”，'
        '但没有隔离反事实，不能将全部2839秒因果归于round3一次决策。',
        '', '4. **候选ETA不是动态完成时间。** completion_time.py用当前模型、当前batch的速率外推全部剩余工作；'
        'job.py会逐epoch更新速率/batch，作业完成又会释放GPU/CPU。新目标补齐了donor代价，却仍漏掉'
        '“谁先完成、释放资源后其他作业如何加速”的价值，也可能把临时让出资源当成持续至完成的损失。'
        '当前未服务作业的后续等待也不进入第二阶段目标。',
        '', '一个仅作数学说明的反例：两个作业剩余工作1、2；两台设备速度2、1，启动成本0，作业各用一台。'
        '静态sum ETA偏好快设备给长作业，代价2；给短作业代价2.5。但允许短作业完成后转移释放的快设备，'
        '后者完成时间为0.5和1.25，真实总完成时间1.75；前者为1和1，总计2。'
        '因此即使预测率完全准确，静态sum ETA也不等于有资源释放的sum JCT最优。该反例未用于修改数据或参数。',
        '', '5. **训练损失超过等待/重启节省。** NC对NC-P：等待少300任务秒、checkpoint少75秒，但有效训练墙钟多2436秒，'
        '总JCT多2061秒；主要损失CF0 +2143秒、DS12 +461秒。与Sia相比仅2个作业改善、4个退化，'
        'CF2节省6346秒，超过净节省1933秒；诊断性排除CF2后平均JCT反而比Sia高16.63%。'
        '主结果仍保留六个作业，不事后删掉任何作业。',
        '', '本轮NC所选列的分子/分母均无clipping，旧CC也无；故此次退化不能直接归咎所选响应点的裁剪。'
        '其余策略仍有CIFAR DGX2分母裁剪，完整候选域缺口未修复。四个联合策略实际所选RAM均125GB/GPU，'
        '这轮只证明CPU配额/GPU决策变化，不能声称已证明内存差异化优化收益。']
    lines += ['', '## 时间与资源分解','',
        '|策略|等待任务秒|有效训练任务秒|checkpoint任务秒|GPU-h|CPU单位-h|RAM-GB-h|浪费GPU-s|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m,v in s.items():
        lines.append('|'+m+'|'+'|'.join(f'{v["totals"][k]:.3f}' for k in
            ['wait','training','checkpoint','gpu_hours','cpu_core_hours','ram_gb_hours','wasted_gpu_seconds'])+'|')
    for name,v in a['comparisons'].items():
        lines.append(f'\n{name}：等待/checkpoint/有效训练的总秒变化为{v["time_deltas"]}，三者之和等于6倍平均JCT变化。')
    lines += ['', '## 正确性与版本证据','',
        '|策略|容量违规|净代价检查轮|selected列|clipping计数|fallback|记录的solve秒|',
        '|---|---:|---:|---:|---|---:|---:|']
    for m,v in r.items():
        lines.append(f'|{m}|{v["capacity_violations"]}|{v["checks"].get("net_cost_audits",0)}|{v["checks"]["selected_columns"]}|{v["clipping"]}|{s[m]["fallback_count"]}|{s[m]["solve_total_seconds"]:.3f}|')
    lines += ['', '从原始完成事件、执行片段独立复算JCT、互斥等待/训练/checkpoint、结束后停止资源计时；独立检查rho公式、'
        'raw×rho单次及epoch舍入边界、逐节点容量、真实锁、无事件GPU几何保持和启动预算。'
        '六任务checkpoint短于周期，不能以此替代真实BERT跨轮锁测试。求解字段不等于包含所有模型/日志工作的端到端墙钟。',
        '', 'NC每轮保存reference_jobs、reference_columns、reference_eta_sum、chosen_eta_sum和unknown_frozen，'
        '审计固定集合且新预计代价不超过参考代价；不使用此不等式证明真实JCT改善。',
        '', 'reference_audit.py为独立补充审计，未修改冻结运行脚本/审计器：按原始候选重新计算参考与所选ETA、'
        '两侧Oracle rho、参考容量、成员及未知列。NC-P的151轮、NC的153轮均通过；分别7轮和58轮严格降低当前参考sum ETA。'
        '这直接展示了“局部预测不劣”与“完整JCT退化”可以同时成立，跨轮重叠ETA不得相加当作总收益。',
        '', '所有策略使用相同R0/E-legacy响应。源Synergy矩阵与跨硬件映射/phi/插值/裁剪必须区分；'
        '已知候选域覆盖缺口仍在，不补phi、不改clipping。本轮CPU/RAM换配额仍沿用无额外加载开销假设。',
        '', f'源码fingerprint：`{p["source_fingerprint"]}`；全部输入、源码、脚本和依赖hash在preregistration.json、source_changes.json及delivery_manifest.json。'
        f'六个独立训练子进程总墙钟{wall:.3f}秒；全部串行，逐格内存≥2GiB、磁盘≥5GiB。',
        '', '原始源码快照parent_actual_source.zip、source.patch、source/、inputs/、全部红绿日志、offline_gate.json、'
        '逐轮cells/<method>/rounds.jsonl.gz、per_job.csv及independent_audit.json可复核。旧目录和官方源码未覆盖，未推送。',
        '', '## 是否继续与下一步边界','',
        '不把NC升级为默认算法，不扩大性能实验；旧CC/CC-P和全部历史产物保持原样。'
        '当前主要问题是动态决策目标，而不只是donor计价。若后续另行授权，优先研究包含下一完成事件、'
        '资源释放、等待者代价和必要恢复成本的短视前瞻；先用可独立精确计算的小例子证明目标方向，'
        '保持资源/锁/活性正确性，不能靠放宽250秒预算、调权重或换任务来弥补本轮结果。'
        '这属于下一次机制设计，不在本轮实施。',
        '', '本轮只实施一个预注册目标修正；完成后停止，不自动扩大负载或继续修改到获胜。']
    with (OUT/'REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    save(OUT/'wall_time.json',dict(process_seconds=wall,execution_manifest=executions))
    save(OUT/'verdict.json',dict(engineering='PASS_229_TESTS_AND_RAW_AUDIT',
        performance='REJECT_AS_IMPROVEMENT_OVER_PARENT',replace_previous_default=False,
        joint_nc_mean_jct=main_result['mean_jct'],
        comparisons=a['comparisons'],data_scope=p['warning']))
    print('REPORT_READY',a['status'])


if __name__=='__main__':main()
