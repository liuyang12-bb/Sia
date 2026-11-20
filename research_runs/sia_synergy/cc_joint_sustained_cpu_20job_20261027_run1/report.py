"""Generate a new report from audited results only, including bad outcomes."""
import csv,json,time
from experiment import OUT,ROOT,PARENT,RUNS,METHODS,load,save,sha,WARNING

def f(v):return '—' if v is None else f'{v:.2f}'
def main():
    p=load();a=json.loads((OUT/'independent_audit.json').read_text())
    cov=json.loads((OUT/'coverage_preflight.json').read_text());ref=cov['reference_gate']
    mech=json.loads((OUT/'mechanism_analysis.json').read_text())
    exch=json.loads((OUT/'exchange_composition.json').read_text())
    obs=json.loads((OUT/'outcome_analysis.json').read_text())
    opportunities=json.loads((OUT/'observed_opportunities.json').read_text())
    alloc_opp=json.loads((OUT/'allocation_opportunities.json').read_text())
    prepared=json.loads((OUT/'prepared.json').read_text())
    manifests=[]
    for path in sorted(OUT.glob('*execution_manifest.jsonl')):
        manifests += [dict(manifest=path.name,**json.loads(l)) for l in path.read_text().splitlines()]
    launches=[e for e in manifests if 'exit_code' in e]
    assert len({(e['response'],e['method']) for e in launches})==len(launches)
    assert all(e['exit_code']==0 for e in launches)
    records=[]
    for response,method in RUNS:
      d=OUT/'cells'/response/method
      if not (d/'summary.json').exists():continue
      start=json.loads((d/'start.json').read_text())
      assert start['preregistration_sha256']==sha(OUT/'preregistration.json')
      assert start['resources']['memory_gib']>=2 and start['resources']['disk_gib']>=5
      records.append(dict(response=response,method=method,resources=start['resources'],
        hashes={n.name:sha(n) for n in d.iterdir() if n.is_file()}))
    save(OUT/'run_manifest.json',dict(status=a['status'],planned=RUNS,unique_launched_cells=len(launches),
      execution_records=manifests,cells=records,source_fingerprint=p['source_fingerprint'],algorithm_changes=False,
      preregistration_sha256=sha(OUT/'preregistration.json')))
    lines=['# CC-Joint：20任务持续CPU互补场景实验','',WARNING,'',
      f'状态：{a["status"]}。本轮算法源码404文件逐字节不变；只改变预注册的负载组成与到达时序。',
      '', '## 结论与主要指标','',
      '|响应|策略|完成|平均JCT(s)|Median|P95|P99|Makespan|Restart|',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for response,method in RUNS:
      s=a['results'][response+'/'+method].get('summary')
      if not s:lines.append(f'|{response}|{method}|NOT COMPLETE|—|—|—|—|—|—|');continue
      lines.append(f'|{response}|{method}|{s["completed"]}/20|'+'|'.join(f(s[k]) for k in ['mean_jct','median_jct','p95_jct','p99_jct','makespan'])+f'|{s["totals"]["restarts"]}|')
    lines += ['', '|响应|CC-Joint相对对照|均值改善率|平均差(CC−对照)|改善/退化/不变|最大贡献作业|排除该作业诊断改善率|',
      '|---|---|---:|---:|---:|---|---:|']
    for response,comp in a['comparisons'].items():
      for method,v in comp.items():
        lines.append(f'|{response}|{method}|{100*v["improvement"]:.2f}%|{v["mean_delta"]:.2f}|{v["improved"]}/{v["worsened"]}/{v["unchanged"]}|{v["largest_beneficiary"]}|{100*v["without_largest_diagnostic_improvement"]:.2f}%|')
    medium=a['comparisons'].get('medium',{})
    if medium:
      aux=medium['JOINT_CC_P']['improvement']
      lines += ['',f'Medium完整版相对SIA-P的平均JCT改善{100*medium["SIA_P"]["improvement"]:.2f}%，'
        f'相对串行辅助方案改善{100*medium["SIA_ECT_AUX"]["improvement"]:.2f}%。'
        f'关键配额消融CC相对CC-P改善{100*aux:.2f}%（负数表示退化）。'
        +('因此本场景支持差异化辅助配额的增量JCT收益。' if aux>0 else '因此本场景仍未证实差异化辅助配额的增量JCT收益。'),
        '这不是跨负载普适结论，更不是目标集群实测。正改善率表示更快；保留全部20个作业。']
      v=medium['JOINT_CC_P'];cf=v['groups']['cifar_sensitive'];ds=v['groups']['deepspeech_weak']
      lines += ['', f'配额消融的任务级归因：18个CIFAR合计JCT变化{18*cf["mean_delta"]:.0f}秒，'
        f'2个DeepSpeech合计变化{2*ds["mean_delta"]:.0f}秒，净变化{20*v["mean_delta"]:.0f}秒。'
        f'CIFAR组均值改善{100*cf["improvement"]:.2f}%，DeepSpeech组变化改善率{100*ds["improvement"]:.2f}%（负值为退化）。'
        f'这属于开启配额自由度的闭环总效应，包含后续GPU/放置/排队反馈，不等于固定GPU时CPU加速率。'
        f'最大受益{v["largest_beneficiary"]}占净节省{100*v["largest_saving"]/(-20*v["mean_delta"]):.2f}%；'
        f'诊断性去除后仍改善{100*v["without_largest_diagnostic_improvement"]:.2f}%，主结果仍保留全部任务。',
        '必须保留长任务损失：deepspeech2-8等作业可能明显延后，后文给全部逐作业数据、P95/P99及makespan。'
        '本研究主目标平均JCT可改善，但不能描述为所有作业都更快或所有资源指标都更好。']
    lines += ['', '## 预注册输入与公平性','',
      '从官方workload-1依原始行序选前18条CIFAR10、前2条DeepSpeech2。'
      't0固定交错CF0、DS8、CF1、DS9；剩余CIFAR每300秒两条，到2400秒。'
      '原batch、GPU demand和完整应用训练工作量不改。合成到达时间与原始时间、行号均另列。'
      'source_row_index为新场景的稳定队列顺序，original_source_row_index保留原始出处。',
      '', '|Job|受控Arrival|原Arrival|原GPU demand|Batch|原行(0-based)|', '|---|---:|---:|---:|---:|---:|']
    for r in prepared['jobs']:
      lines.append(f'|{r["name"]}|{r["time"]}|{r["original_arrival"]}|{r["num_replicas"]}|{r["batch_size"]}|{r["original_source_row_index"]}|')
    lines += ['', 'AWS、DGX-ext各一节点，每节点2GPU、12合成CPU单位、250GB RAM。'
      '比例点6CPU/125GB每GPU；固定参考3CPU/62.5GB，R0/E-legacy、alpha0/.5、原网格及既有CPU8、seed_profiles(1,8)、batch自适应不变。'
      '60秒周期，6000轮共同观察窗，八格独立进程串行；逐格内存2GiB、磁盘5GiB门槛。',
      'SIA-P实际调用官方Sia GPU逻辑，不加CC准入；SIA-ECT-AUX固定完整Sia任务集合/GPU意图/放置优化辅助配额。'
      'CC-P/CC共享完成优先评分、GPU候选、FIFO持续服务、checkpoint保护和250秒终身启动预算，仅差辅助配额自由度。'
      'CC-P仍使用响应预测GPU配置速度，不是完全CPU盲基线。CC相对Sia还同时改变目标、准入与放置，不能把总收益都归因CPU。',
      '', '## 机会预检：在JCT实验之前完成','',
      f'固定一GPU、原声明batch、逐epoch官方公式的参考耗时，取较快GPU类型，CIFAR占{100*ref["shares"]["fastest_one_gpu"]["cifar_sensitive_share"]:.2f}%。'
      f'若固定AWS为{100*ref["shares"]["aws"]["cifar_sensitive_share"]:.2f}%，固定DGX为{100*ref["shares"]["dgx-ext"]["cifar_sensitive_share"]:.2f}%。'
      '参考占比依GPU和batch定义而变，不声称在所有策略实际路径下都占同样比例。'
      '该离线计算未向在线任务注入未来性能/完成信息，真实训练仍按原有自适应batch执行。',
      '', '|GPU|CF收益(s)|DS损失(s)|平均净节省(s)|CF/DS交换CPU|支持范围|', '|---|---:|---:|---:|---|---|']
    for w in ref['witnesses']:
      lines.append(f'|{w["gpu_type"]}|{w["receiver_saving"]:.2f}|{w["donor_delay"]:.2f}|{w["mean_saving"]:.2f}|8/4 vs6/6|'+('分子分母均支持' if w['fully_supported'] else '含越界')+'|')
    lines += ['', '上述解析任务对在CF完成后将CPU还给DS，计入50/25秒真实启动；是全任务平均速率近似，不是执行器精确完成仿真或全局最优证明。'
      'AWS的GPU耦合见证比较DS2→CF2串行与CF1+DS1→DS2，计入25秒缩放，主见证分子/分母均在支持域。'
      'DGX2分母已知裁剪，因此DGX GPU耦合数字不作为无裁剪主证据。'
      '预检证明机会可落地而非保证调度器获胜；实际触发与完整JCT要分别查验。',
      f'80个job×GPU配置均覆盖；全可行点边界计数{cov["query_clipping_counts"]}。'
      '全域沿用旧E-legacy裁剪，未偷偷删配置；主见证不依赖边界裁剪。映射、phi、参考、插值均属合成跨硬件假设，源矩阵并非Saturn测量。',
      '', '## 实际竞争、交换与时间分解','',
      f'观察到的两作业同节点构成：{exch["composition"]}；资源交换构成：{exch["exchanges_by_receiver_donor"]}。',
      f'首次跨应用交换：{json.dumps(exch["first_cross_application_exchange"],ensure_ascii=False)}。',
      f'实际完整训练区间内的支持域机会检查：{json.dumps(opportunities["counts"],ensure_ascii=False)}。'
      '这里固定GPU/batch和直接记录的raw执行率，比较6/6与8/4，双方按各自声明总工作量归一。'
      '正净工作分数包含donor损失，但依旧不等于完整运行的反事实JCT收益。',
      f'不局限跨应用共驻的全节点同状态CPU枚举：{json.dumps({m:v["counts"] for m,v in alloc_opp["results"].items()},ensure_ascii=False)}。'
      '枚举全部原有CPU点，固定GPU/batch/placement与完整共驻任务集合，RAM固定125；容量12CPU。'
      '未省略任何donor，主见证分子分母均支持；边界点和checkpoint/部分区间另记排除数。'
      '该离线诊断不修改配置或启动仿真，也不是按JCT选参数。',
      '', '|单元|CIFAR训练任务秒占比|初始等待|后续等待|Checkpoint|有效训练|GPU-h|CPU单位-h|RAM-GB-h|',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key,r in a['results'].items():
      if 'summary' not in r:continue
      s=r['summary'];o=obs[key];w=o['waiting_phases'];t=s['totals']
      lines.append(f'|{key}|{100*o["cifar_productive_wall_fraction"]:.2f}%|{sum(v.get("initial_wait",0) for v in w.values()):.2f}|{sum(v.get("post_allocation_wait",0) for v in w.values()):.2f}|'+
        '|'.join(f(t[k]) for k in ['checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours'])+'|')
    if medium:
      for baseline,v in medium.items():lines.append(f'\nCC相对{baseline}的总时间差（任务秒，负值为减少）：{v["time_delta"]}。')
      target=a['results']['medium/JOINT_CC']['summary']
      for baseline in METHODS[:-1]:
        b=a['results']['medium/'+baseline]['summary']
        changes={k:100*(target[k]/b[k]-1) for k in ['median_jct','p95_jct','p99_jct','makespan']}
        lines.append(f'\nMedium CC相对{baseline}的Median/P95/P99/Makespan变化率（正数为变差）：{changes}。'
          '平均值若改善而尾部/整体结束变差，不能隐去冲突。')
    mc=mech['results'].get('medium/JOINT_CC',{})
    if mc:
      counts={k:sum(v.get(k,0) for v in mc['jobs'].values()) for k in ['above_proportional_training_seconds','below_proportional_training_seconds','equal_proportional_training_seconds']}
      lines += ['',f'完整CC实际训练墙钟中，相对同GPU比例点更快/更慢/相同的任务秒：{counts}。'
        f'资源交换的决策级统计：{mc["counts"]}。'
        '若这些机会被选中却未降低JCT，应区分候选/评分、放置、准入、重配置和任务间损失，而不是继续只增加任务数。']
    lines += ['', '等待、checkpoint、有效训练互斥且总和逐作业等于JCT。'
      '执行训练片段保存raw_before_rho、rho、effective rate与epoch边界残差。'
      '同状态比例时间等价变化只作局部诊断，不能当作真实反事实JCT节省。',
      '', '|单元|排队轮|最大等待作业|CPU饱和GPU空闲节点轮|Fallback|容量违规|所选列裁剪|',
      '|---|---:|---:|---:|---:|---:|---|']
    for key,r in a['results'].items():
      if 'summary' not in r:continue
      q=r['queue'];lines.append(f'|{key}|{q.get("rounds_with_waiters",0)}|{q.get("max_unserved_active_jobs",0)}|{q.get("cpu_saturated_gpu_idle_node_rounds",0)}|{r["fallback_count"]}|{r["capacity_violations"]}|{r["clipping"]}|')
    lines += ['', '|单元|求解状态计数|优化总墙钟(s)|优化P99(s)|', '|---|---|---:|---:|']
    for key,r in a['results'].items():
      if 'summary' not in r:continue
      s=r['summary'];lines.append(f'|{key}|{r["solver_status"]}|{s["solve_total_seconds"]:.3f}|{s["solve_p99_seconds"]:.3f}|')
    lines += ['', '优化器墙钟没有额外推进仿真时间，四策略统计一致，不能据此声称真实计算零成本。'
      '辅助配额变化的额外数据加载开销未建模，沿用原执行假设。',
      '', '## 逐作业JCT与退化','']
    for response in ['neutral','medium']:
      lines += [f'### {response}','', '|Job|'+'|'.join(METHODS)+'|', '|---|'+'---:|'*4]
      for j in [r['name'] for r in prepared['jobs']]:lines.append('|'+j+'|'+'|'.join(f(a['results'][response+'/'+m].get('summary',{}).get('jobs',{}).get(j,{}).get('jct')) for m in METHODS)+'|')
    lines += ['', '## 测试、版本与局限','',
      f'212旧测试通过；原始Sia独立3轮分配/进度/重启/完成事件回归一致；独立输入验收通过。'
      f'Neutral内部轨迹对照：{mech["neutral_controls"]}。完整容量、锁、有效服务、预算、目标/近整数解、响应单次倍率审计见independent_audit.json。',
      '构造验收未完成时曾提前调用批次，入口在缺少验收文件处拒绝，未启动任何训练；证据prebatch_ordering_stop.md。'
      '所有正式运行仅启动一次，安全门槛或其他停止均保留在manifest。没有按结果调参或挑选重跑。',
      f'源码fingerprint：{p["source_fingerprint"]}。版本是实际404文件快照，不冒称裸HEAD能重建。'
      '输入/脚本/包版本见preregistration.json，依赖见dependencies.lock.txt；可重建归档及逐文件hash见delivery_manifest.json。',
      f'八格训练进程总墙钟{sum(e["wall_seconds"] for e in launches):.2f}秒。'
      '单一有意构造的场景、20作业不是20次独立重复。P95/P99只作描述，不能作为精确总体尾延迟。'
      '辅助配额变化没有额外加载成本、跨硬件响应有合成假设、ETA是当前速率近似，均限制推广。'
      '与先前12作业不同负载和到达，不能把两轮均值直接相减作因果效果。本轮不自动扩展。']
    with (OUT/'REPORT.md').open('x',encoding='utf-8') as f0:f0.write('\n'.join(lines)+'\n')
    save(OUT/'conclusions.json',dict(status=a['status'],all_cells_complete=a['status']=='EIGHT_COMPLETE_VALID',
      old_cc_source_unchanged=True,medium_comparisons=medium,neutral_comparisons=a['comparisons'].get('neutral',{}),
      training_wall_seconds=sum(e['wall_seconds'] for e in launches),next_experiment_started=False))
    print('REPORT_READY',a['status'])

if __name__=='__main__':main()
