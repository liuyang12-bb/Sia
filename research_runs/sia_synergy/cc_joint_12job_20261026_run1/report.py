"""Render every frozen cell, including censored or failed cells."""
import json,time
from experiment import OUT, PARENT, METHODS, RUNS, load, save, sha, WARNING

def fmt(v):return '—' if v is None else f'{v:.2f}'
def main():
    p=load();a=json.loads((OUT/'independent_audit.json').read_text())
    cov=json.loads((OUT/'coverage_preflight.json').read_text());prepared=json.loads((OUT/'prepared.json').read_text())
    manifest=[json.loads(l) for l in (OUT/'execution_manifest.jsonl').read_text().splitlines()]
    for extra in ['resume_execution_manifest.jsonl','last_execution_manifest.jsonl']:
        if (OUT/extra).exists():manifest += [json.loads(l) for l in (OUT/extra).read_text().splitlines()]
    mechanism=json.loads((OUT/'mechanism_analysis.json').read_text())
    exchange=json.loads((OUT/'exchange_composition.json').read_text())
    actual=[e for e in manifest if 'exit_code' in e]
    assert len({(e['response'],e['method']) for e in actual})==len(actual)
    if a['status']=='EIGHT_COMPLETE_VALID':
        assert [(e['response'],e['method']) for e in actual]==RUNS and all(e['exit_code']==0 for e in actual)
    launches=[]
    for response,method in RUNS:
        d=OUT/'cells'/response/method
        if not (d/'summary.json').exists():continue
        start=json.loads((d/'start.json').read_text())
        assert start['resources']['memory_gib']>=2 and start['resources']['disk_gib']>=5
        assert start['preregistration_sha256']==sha(OUT/'preregistration.json')
        launches.append(dict(response=response,method=method,resources=start['resources'],
            completed=a['results'][response+'/'+method]['summary']['completed'],
            output_hashes={f.relative_to(OUT).as_posix():sha(f) for f in d.iterdir() if f.is_file()}))
    save(OUT/'run_manifest.json',dict(status=a['status'],planned=RUNS,execution_records=manifest,
        unique_launched_cells=len(actual),launches=launches,source_fingerprint=p['source_fingerprint'],
        preregistration_sha256=sha(OUT/'preregistration.json'),algorithm_changes=False,
        recovery='Two serial batches stopped at memory gates; only never-started suffix cells resumed after hash verification. Each cell ran once.'))
    lines=['# CC-Joint：12任务负载扩展实验','',WARNING,'',
      '本轮保留此前六任务平均JCT改善15.28%的JOINT_CC，正式称CC-Joint；比例消融为CC-Joint-P。'
      '新NC方案没有混入本轮。源码404文件与父版本逐字节一致，没有算法修改或结果驱动调参。',
      '',f'原始事件审计：{a["status"]}。完成率与删失先于均值解释；完整均值仅用于全部完成配对。',
      '', '**结论：保留旧版并扩大任务后，整体平均JCT优势仍成立，而且在本场景中更明显；辅助配额优化的额外JCT收益仍未成立。** '
      'Neutral较Sia下降31.02%；Medium较Sia下降42.85%、较串行辅助优化下降35.55%。'
      '但Medium完整版7073.50秒，比例消融7050.67秒，开启辅助优化反而恶化0.32%。'
      '因此本轮支持CC-Joint整套机制在该12任务场景有效，不支持把42.85%全部解释为CPU差异化分配的收益。',
      '', '## 主要结果','', '|响应|策略|完成|平均JCT(s)|Median|P95|P99|Makespan|Restart|',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for response,method in RUNS:
      r=a['results'][response+'/'+method];s=r.get('summary')
      if s:
        lines.append('|'+response+'|'+method+f'|{s["completed"]}/12|'+'|'.join(fmt(s[k]) for k in ['mean_jct','median_jct','p95_jct','p99_jct','makespan'])+f'|{s["totals"]["restarts"]}|')
      else:lines.append(f'|{response}|{method}|FAILED/NOT RUN|—|—|—|—|—|—|')
    lines += ['', 'SIA_P调用官方Sia GPU决策、原有放置与重配置，统一响应下按比例执行；'
      'SIA_ECT_AUX固定完整Sia意图/任务集合/放置后优化CPU/RAM，沿用已冻结ECT辅助评分。'
      'JOINT_CC_P与JOINT_CC共享完成感知目标、候选、FIFO/持续服务和原250秒启动预算，只切换配额自由度。'
      '相对Sia的差异还包含准入与放置，不能把全部收益归因CPU感知。',
      '', '比例消融CC-Joint-P仍按统一响应模型评价GPU候选；它关闭的是CPU/RAM配额自由度，'
      '并未关闭“辅助资源供给影响GPU配置速度”的建模。CC与Sia还存在公平目标/完成评分差异，'
      '当前四策略不能分别识别准入、目标和放置的全部贡献。',
      '', '|响应|CC-Joint对照|平均JCT改善率|改善/退化/不变作业|最大受益作业|排除该作业后的诊断改善率|',
      '|---|---|---:|---:|---|---:|']
    for response,comp in a['comparisons'].items():
      for method,v in comp.items():
        lines.append(f'|{response}|{method}|{100*v["improvement"]:.2f}%|{v["improved"]}/{v["worsened"]}/{v["unchanged"]}|{v["largest_beneficiary"]}|{100*v["without_largest_diagnostic_improvement"]:.2f}%|')
    lines += ['', '正改善率表示CC-Joint更快；负数为退化。单一12作业cohort的P95/P99是描述值，不能视作高精度总体尾延迟。'
      '主分析保留所有作业，排除最大受益作业仅作诊断。该受控混合不是12次独立重复。',
      '', '## 输入与扩大规则','',
      '保留旧CF0/1/2和DS8/9/12；按原始行序，给每个应用补入后续三条，得到CF3/4/5和DS16/21/30。'
      '全部来自官方workload-1.csv，训练工作量不缩短、原GPU demand和batch不改。'
      '统一同时到达t=0属于明确合成的持续排队压力场景，原到达时间另存。',
      '', '|Job|原Arrival|原GPU demand|Batch|Source row(0-based)|',
      '|---|---:|---:|---:|---:|']
    for r in prepared['jobs']:lines.append(f'|{r["name"]}|{r["time"]}|{r["num_replicas"]}|{r["batch_size"]}|{r["source_row_index"]}|')
    lines += ['', '硬件保持AWS、DGX-ext各一个2 GPU节点，每节点12合成CPU单位、250GB RAM，总4GPU。'
      '比例点每GPU6CPU/125GB；响应固定参考3CPU/62.5GB，R0/E-legacy和既有应用模板映射不改。'
      'Neutral alpha=0，Medium alpha=0.5；原网格含CPU8，初始化seed_profiles(1,8)，保留batch自适应。'
      '60秒周期、6000轮上限，八个独立进程固定顺序运行；逐格内存2GiB/磁盘5GiB门槛。',
      '', '仅增加任务数量而不增加硬件，能够增加排队与混合机会；新增真实作业也带来不同batch/GPU demand，'
      '因此这不是严格只改变N、其他任务完全相同的因果实验。不能预设更多任务一定改善CC。',
      '', '## 离线机会与数据边界','',
      f'离线覆盖48个作业×配置项（12×2类型×1/2GPU），均有R0记录与容量可行比例点。'
      f'全可行资源点的边界统计：{cov["query_clipping_counts"]}。完整查询和源模板/phi/坐标见coverage_preflight.json。',
      '', '|GPU|一GPU CIFAR/DS CPU|RAM|CIFAR速度比|DS速度比|相对比例之和|支持范围|',
      '|---|---|---|---:|---:|---:|---|']
    for w in cov['pair_witnesses']:
      v=w['best'];lines.append(f'|{w["gpu_type"]}|{v["cpu"]}|{v["ram"]}|{v["ratio"][0]:.5f}|{v["ratio"][1]:.5f}|{v["score"]:.5f}|分子/分母均无裁剪|')
    lines += ['', '按各自比例速度归一化后存在预算内互补；跨应用归一化速度之和不是绝对Goodput或JCT证据。'
      'Synergy源矩阵是原始数据，应用映射、跨GPU迁移、phi和固定参考均为合成假设。'
      '已知CIFAR DGX2分母裁剪仍保留并统计，没有补phi或改变边界规则。',
      '', '## 逐作业JCT','']
    for response in ['neutral','medium']:
      lines += [f'### {response}', '', '|Job|'+'|'.join(METHODS)+'|','|---|'+'---:|'*4]
      for j in [r['name'] for r in prepared['jobs']]:
        lines.append('|'+j+'|'+'|'.join(fmt(a['results'][response+'/'+m].get('summary',{}).get('jobs',{}).get(j,{}).get('jct')) for m in METHODS)+'|')
    lines += ['', '## 排队、执行与资源','',
      '|响应|策略|等待任务秒|训练任务秒|Checkpoint任务秒|GPU-h|CPU单位-h|RAM-GB-h|浪费GPU-s|',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,r in a['results'].items():
      if 'summary' not in r:continue
      response,method=key.split('/');t=r['summary']['totals']
      lines.append('|'+response+'|'+method+'|'+'|'.join(fmt(t[k]) for k in ['wait','training','checkpoint','gpu_hours','cpu_core_hours','ram_gb_hours','wasted_gpu_seconds'])+'|')
    lines += ['', '|单元|有排队轮|最大未服务活动作业|CPU饱和/GPU闲置节点轮|Fallback|容量违规|所选列裁剪|',
      '|---|---:|---:|---:|---:|---:|---|']
    for key,r in a['results'].items():
      if 'summary' not in r:continue
      q=r['queue'];lines.append(f'|{key}|{q.get("rounds_with_waiters",0)}|{q.get("max_unserved_active_jobs",0)}|{q.get("cpu_saturated_gpu_idle_node_rounds",0)}|{r["fallback_count"]}|{r["capacity_violations"]}|{r["clipping"]}|')
    lines += ['', '等待任务秒由真实执行区间累计；未服务作业边界次数和节点轮事件不是等待秒数。'
      '独立逐节点时长积分与利用率、求解状态/开销、事件与启动预算检查见independent_audit.json。',
      '', '## 收益来自哪里，CPU机会是否兑现','',
      'Medium中CC相对Sia：等待总量减少79020任务秒、checkpoint减少775秒，但有效训练墙钟增加16156任务秒，'
      '净减少63639任务秒，即平均JCT下降5303.25秒。主要观测机制是更早获得服务、不同GPU/并发配置和较少重启，'
      '并非所有作业训练阶段都更快。Sia重启47次，CC为15次。',
      '', 'Medium相对Sia有6作业改善、6作业退化；最大受益CF0节省17604秒，约占净节省27.66%。'
      '诊断性排除CF0后仍改善35.37%，收益不再完全由单一任务主导。'
      '但DS30相对Sia增加10617秒，不能宣称所有任务变快。总体P95改善14.15%、makespan改善13.68%，'
      '仍需保留这个明显受损作业。',
      '', 'Medium开启辅助优化相对CC-P：等待增加180任务秒、checkpoint减少50秒、训练增加144秒，'
      '总JCT增加274秒（平均22.83秒，0.32%）。DS21节省1733秒，而DS30增加1764秒，两者几乎抵消。'
      'CPU单位小时从109.02降至99.97（减少8.31%），这是资源使用减少，不能代替主要目标JCT改善。'
      '六任务时开启辅助优化恶化6.69%，十二任务时为0.32%；两个受控集合不同，该变化不能作统计显著性结论。',
      '', '串行SIA-AUX相对Sia平均JCT改善11.32%，说明本数据确实能产生辅助配额价值；'
      '然而CC-P已经联合考虑GPU候选和比例点下的响应，所以额外释放配额自由度不必再产生同等收益。',
      '', f'Neutral两组内部GPU/进度/完成/restart轨迹分别完全一致：{mechanism["neutral_controls"]}。'
      'Medium全CC有32个决策轮出现同节点receiver加速与donor减速，累计35个节点轮交换；'
      f'按应用分解为{exchange["exchanges_by_receiver_donor"]}。'
      f'两作业同节点构成：{exchange["composition"]}。'
      'CIFAR与DeepSpeech2混合共驻仅8个节点轮，其中7个发生资源交换，表明已有机会被策略利用，'
      '但持续互补的共驻时间有限，更多交换发生在两个CPU敏感CIFAR之间。节点轮与决策轮口径不同，不能互相当作秒数。',
      '', '第一个跨应用交换在t=1440秒：AWS1上的CF5分9CPU，DS8分3CPU，RAM均125GB，'
      '相对各自比例点速度变化+18.69%和−17.46%。所选Medium全CC列分子/分母均无裁剪。'
      '源矩阵支持下确有加速和减速，训练片段也真实施加rho；单次交换并不能证明整个JCT更好。',
      '', '输入顺序使六个CIFAR先于六个DeepSpeech2，旧FIFO/持续服务规则原样保留。'
      '它既可能帮助短作业尽早完成，也限制敏感/弱敏感任务同时共驻。'
      '未隔离准入与放置的反事实，不能据此直接断言改变排序一定改善JCT，也不能把这组受控到达外推到所有真实trace。'
      'mechanism_analysis.json记录全部任务的同状态比例时间等价变化，属于固定已观测GPU/batch路径的局部描述，'
      '不能把这些重叠量相加当成完整运行的因果JCT收益。',
      '', '## 与六任务结果的关系','',
      '历史Medium六任务：Sia5636.5秒、CC-Joint4775.5秒（15.28%改善）、CC-Joint-P4476秒。'
      '六与十二任务均值不能当成相同任务集合的配对结果。下表单独比较在十二任务竞争中的原六任务和新增六任务。',
      '', '|响应|对照|组别|平均JCT差(CC−对照)|改善率|','|---|---|---|---:|---:|']
    for response,comp in a['comparisons'].items():
      for m,v in comp.items():
        for group,g in v['groups'].items():lines.append(f'|{response}|{m}|{group}|{g["mean_delta"]:.2f}|{100*g["improvement"]:.2f}%|')
    lines += ['', '增加任务后排队更长不等于CPU优化收益更大；必须同时看CC-Joint与自身比例消融。'
      'Medium/Neutral相对效果变化是同一合成模型下的机制诊断，不能泛化为真实GPU集群测量。',
      '', '## 正确性与可复现性','',
      '旧版源码404文件原样复制到新树sia-joint-cc-12job/sia-simulator及source/，父版本hash全部匹配。'
      '首次测试因产物目录深度使历史fixture根路径错误，完整失败日志保留；恢复隔离工作树布局后重跑212项。'
      '最终212/212通过，不改测试断言或算法。tests_relocated.json记录结果，official_regression.json为原始Sia独立3轮逐状态一致。',
      '', '原批次在第7组启动前因1.8599GiB可用内存安全停止；恢复后第7组完成，第8组启动前再次因1.7800GiB停止。'
      '另一次第8组启动预检在1.8207GiB退出，未创建运行标记或启动训练；随后内存恢复，第8组在2.143GiB通过门槛启动。'
      '两次恢复均检查源/输入/完成前缀hash，只运行从未启动的后缀。八个实际训练格各运行一次，'
      '无因JCT选取重复结果；所有启动内存≥2GiB、磁盘≥5GiB。'
      '原manifest及安全停止证据没有覆盖，run_manifest.json整合三个启动清单。',
      '', '从原始完成事件与训练片段复算JCT、P95/P99、互斥时间、资源小时；验证rho公式、单次乘法、Neutral=1、'
      'checkpoint锁、连续服务/事件门控、启动预算、近整数规范化的原模型约束和比例契约。'
      '已有BERT短链覆盖真实跨轮checkpoint；本批应用短startup不能替代该单元验证。',
      '',f'源码fingerprint：{p["source_fingerprint"]}。源码、输入和执行脚本hash见preregistration.json，'
      '依赖版本见dependencies.lock.txt，完整归档与readback见delivery_manifest.json。'
      f'八格子进程总墙钟{sum(e.get("wall_seconds",0) for e in manifest):.2f}秒。'
      '旧版源、官方源和全部历史结果未修改；本轮不采用新NC目标，不推送。',
      '', '同一12任务队列只支持该受控场景判断。若收益主要由少数任务或准入差异贡献，应按数据报告；'
      '不声称已证明普适CPU/RAM优化收益。本轮结束后停止，不自动扩展或调参。']
    with (OUT/'REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    save(OUT/'wall_time.json',dict(training_process_seconds=sum(e.get('wall_seconds',0) for e in manifest),execution_manifest=manifest))
    save(OUT/'conclusions.json',dict(status=a['status'],all_cells_complete=a['status']=='EIGHT_COMPLETE_VALID',
        old_cc_source_unchanged=True,neutral_improvement=0.31018541100976105,
        medium_improvement=0.42848486072676595,medium_vs_serial_improvement=0.3555337904015671,
        medium_aux_increment_improvement=-0.0032384644478062885,
        interpretation='Overall benefit reproduced in controlled12-job load; differentiated auxiliary quota JCT increment remains unproven.',
        next_experiment_started=False,freeze_to_report_seconds=time.time()-p['created_unix']))
    print('REPORT_READY',a['status'])

if __name__=='__main__':main()
