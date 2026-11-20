"""Summarize pre-frozen comparisons; never change scheduling parameters."""
import collections,hashlib,json,math,statistics
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3]
p=json.loads((OUT/'preregistration.json').read_text())
def save(name,v):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
def pct(a,b):return 100*(a-b)/a if a else None
def comparison(base,new,label):
    result=dict(label=label,baseline=base['method'],candidate=new['method'],complete_pair=base['completed']==base['total'] and new['completed']==new['total'])
    assert set(base['jobs'])==set(new['jobs'])
    if not result['complete_pair']:return result
    differences={j:new['jobs'][j]['jct']-base['jobs'][j]['jct'] for j in base['jobs']}
    net=sum(differences.values());best=min(differences,key=differences.get);worst=max(differences,key=differences.get)
    result.update(mean_delta=new['mean_jct']-base['mean_jct'],mean_improvement_percent=pct(base['mean_jct'],new['mean_jct']),
        improved=sum(v<0 for v in differences.values()),worse=sum(v>0 for v in differences.values()),same=sum(v==0 for v in differences.values()),
        net_jct_delta=net,per_job_delta=differences,best_job=best,best_job_delta=differences[best],worst_job=worst,worst_job_delta=differences[worst],
        best_share_of_net=(differences[best]/net if net else None),
        diagnostic_without_best=dict(baseline_mean=(sum(v['jct'] for v in base['jobs'].values())-base['jobs'][best]['jct'])/(len(differences)-1),
            candidate_mean=(sum(v['jct'] for v in new['jobs'].values())-new['jobs'][best]['jct'])/(len(differences)-1)),
        totals_delta={k:sum(v[k] for v in new['jobs'].values())-sum(v[k] for v in base['jobs'].values()) for k in ['wait','checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours','restarts','wasted_gpu_seconds']})
    t=result['totals_delta'];assert math.isclose(net,t['wait']+t['checkpoint']+t['training'],abs_tol=1e-5)
    return result
def main():
    original=json.loads((ROOT/'research_runs/sia_synergy/joint_budgeted_adaptation_20261025/all_comparisons.json').read_text())
    lookup={(s['size'],s['response'],s['method']):s for s in original}
    fresh=[];validations=[];pairs=[];mechanisms={}
    for cell in p['runs']:
        folder=OUT/cell['id'];s=json.loads((folder/'summary.json').read_text());v=json.loads((folder/'validation.json').read_text())
        assert v['valid'];assert s['method']==cell['method'] and s['response']==cell['response']
        fresh.append(s);validations.append(v);mechanisms[cell['id']]=json.loads((folder/'mechanism.json').read_text())
        if cell['dataset'].startswith('original'):
            for method in ['sia_p','sia_aux','joint_r','joint_g']:
                pairs.append(comparison(lookup[(s['size'],s['response'],method)],s,f"{cell['dataset']}/{s['response']}/S_vs_{method}"))
    for response in ['neutral','medium']:
        data={s['method']:s for s in fresh if s['dataset']=='holdout20' and s['response']==response}
        for method in ['sia_p','sia_aux']:pairs.append(comparison(data[method],data['joint_s'],f'holdout20/{response}/S_vs_{method}'))
    for f,h in p['baseline_hashes'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest().upper()==h,f
    save('comparison_summary.json',dict(new_results=fresh,original_reference_results=original,paired=pairs))
    save('validation_summary.json',dict(valid=True,complete_cells=sum(v['all_complete'] for v in validations),planned_cells=len(p['runs']),cells=validations))
    report=['# JOINT-S：选择性准入验证结果','',
        'SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。','',
        '## 结论与边界','',
        '本轮仅改变新服务准入：未兑现承诺和FIFO队首必选，其他作业由相同原公平目标决定是否加入。旧G整组交换、真实checkpoint、300秒有效服务、250秒启动预算、R0/E-legacy响应和全部工作负载/容量/参数不变。SIA-AUX是固定官方Sia GPU意图后的本项目辅助资源MIP，不是官方Synergy-TUNE。',
        '',f"新矩阵{sum(v['all_complete'] for v in validations)}/{len(validations)}格全部完成；154项既有/新增测试的冻结源码与本轮一致。训练前内存阻塞证据保留在父目录，当前attempt2未覆盖它。104正式批次未恢复。",'',
        '**效果验收结论：不建议把新S直接作为旧G的升级替代。** 原8作业两格与G相同，原20作业Neutral/Medium分别比G慢0.09%/1.09%。独立20作业Medium中S比Sia快6.17%、比AUX快9.60%，但未运行同负载G，不能把这部分整体方法收益归因于新增准入。两组20作业Medium均优于Sia，只在一组优于AUX，说明优势依赖负载；这些不是目标硬件实测证据。关键机制解释见FINAL_REVIEW.md。','',
        '## 1. 原负载结果','',
        '|规模/响应|SIA-P|SIA-AUX|旧R|旧G|新S|S对P改善|S对AUX改善|S对G改善|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in fresh:
        if not s['dataset'].startswith('original'):continue
        n=s['size'];r=s['response'];b={m:lookup[n,r,m] for m in ['sia_p','sia_aux','joint_r','joint_g']}
        if s['completed']!=s['total'] or any(x['completed']!=x['total'] for x in b.values()):
            report.append(f'|{n}/{r}|含删失，禁止完整均值比较|—|—|—|—|—|—|—|');continue
        report.append(f"|{n}/{r}|{b['sia_p']['mean_jct']:.2f}|{b['sia_aux']['mean_jct']:.2f}|{b['joint_r']['mean_jct']:.2f}|{b['joint_g']['mean_jct']:.2f}|{s['mean_jct']:.2f}|{pct(b['sia_p']['mean_jct'],s['mean_jct']):+.2f}%|{pct(b['sia_aux']['mean_jct'],s['mean_jct']):+.2f}%|{pct(b['joint_g']['mean_jct'],s['mean_jct']):+.2f}%|")
    report+=['','## 2. 预先冻结的独立负载','',
        '在任何新结果出现前按相同轮转规则提取workload-5的20条真实记录，控制到达均为0；保留原batch、GPU demand、工作量与source row。它独立于当前设计所诊断的workload-1，不声称历史项目从未使用过该trace。不能将这一组确定性负载当作多次独立重复。','',
        '|响应|SIA-P|SIA-AUX|新S|S对P改善|S对AUX改善|','|---|---:|---:|---:|---:|---:|']
    for r in ['neutral','medium']:
        d={s['method']:s for s in fresh if s['dataset']=='holdout20' and s['response']==r}
        if any(s['completed']!=s['total'] for s in d.values()):report.append(f'|{r}|含删失，不比较完整均值|—|—|—|—|');continue
        report.append(f"|{r}|{d['sia_p']['mean_jct']:.2f}|{d['sia_aux']['mean_jct']:.2f}|{d['joint_s']['mean_jct']:.2f}|{pct(d['sia_p']['mean_jct'],d['joint_s']['mean_jct']):+.2f}%|{pct(d['sia_aux']['mean_jct'],d['joint_s']['mean_jct']):+.2f}%|")
    report+=['','### 响应开关的完整运行差分','',
        '|负载|方法|Neutral均值−Medium均值(s)|','|---|---|---:|']
    for dataset in ['original8','original20','holdout20']:
        for method in ['sia_p','sia_aux','joint_s']:
            if dataset=='holdout20':
                data={s['response']:s for s in fresh if s['dataset']==dataset and s['method']==method}
            elif method=='joint_s':
                data={s['response']:s for s in fresh if s['dataset']==dataset}
            else:
                n=8 if dataset=='original8' else 20
                data={r:lookup[n,r,method] for r in ['neutral','medium']}
            value=data['neutral']['mean_jct']-data['medium']['mean_jct'] if all(s['completed']==s['total'] for s in data.values()) else None
            report.append(f'|{dataset}|{method}|'+('删失' if value is None else f'{value:+.2f}')+'|')
    report+=['','正值表示Medium在该完整策略路径上更快，负值表示更慢；包含作业完成顺序、GPU决策、在线画像和重启的路径变化，不等于固定状态的纯CPU边际。即使每个合法辅助点不低于该GPU配置的比例基准，也不保证整条在线调度的JCT单调改善。','',
        '## 3. 尾部、资源与开销','',
        '|数据/响应/方法|完成|Median|P95|P99|Makespan|GPU-h|CPU-h|RAM-GB-h|等待s|checkpoint s|训练s|restart|wasted GPU-s|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in fresh:
        t={k:sum(v[k] for v in s['jobs'].values()) for k in ['wait','checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours','restarts','wasted_gpu_seconds']}
        metrics=[s[k] for k in ['median_jct','p95_jct','p99_jct','makespan']]
        fmt=lambda x: '删失' if x is None else f'{x:.2f}'
        report.append(f"|{s['dataset']}/{s['response']}/{s['method']}|{s['completed']}/{s['total']}|"+'|'.join(fmt(x) for x in metrics)+f"|{t['gpu_hours']:.3f}|{t['cpu_core_hours']:.3f}|{t['ram_gb_hours']:.3f}|{t['wait']:.0f}|{t['checkpoint']:.0f}|{t['training']:.0f}|{t['restarts']}|{t['wasted_gpu_seconds']:.0f}|")
    report+=['','P95/P99只基于8或20作业，不能解释为总体尾分位的高精度估计。等待、checkpoint与有效训练从原始执行片段互斥积分；资源小时按实际完成截断的allocated_seconds计算，rho不加速checkpoint/有效服务秒数。','',
        '## 4. 逐作业与净差分','']
    for pair in pairs:
        if not pair['complete_pair']:continue
        report += [f"### {pair['label']}",'',f"均值变化{pair['mean_delta']:+.2f}s，改善率{pair['mean_improvement_percent']:+.2f}%；{pair['improved']}改善/{pair['worse']}退化/{pair['same']}相同。总JCT变化{pair['net_jct_delta']:+.0f}s。最有利差值{pair['best_job']}={pair['best_job_delta']:+.0f}s，最不利差值{pair['worst_job']}={pair['worst_job_delta']:+.0f}s。",
            f"总差分=等待{pair['totals_delta']['wait']:+.0f}+checkpoint{pair['totals_delta']['checkpoint']:+.0f}+有效训练{pair['totals_delta']['training']:+.0f}s。",
            f"仅诊断去掉最大正贡献作业后，对照/新策略均值分别{pair['diagnostic_without_best']['baseline_mean']:.2f}/{pair['diagnostic_without_best']['candidate_mean']:.2f}s；主结果保留全部作业。",'',
            '|Job|新S−对照JCT(s)|','|---|---:|']
        for j,delta in pair['per_job_delta'].items():report.append(f'|{j}|{delta:+.0f}|')
        report.append('')
    report+=['## 5. 准入是否真正改变、保障与审计','',
        '|格|首轮选中|最大同时选中|optional未选总数|head保障轮|drain轮|最大累计启动|数值重求|weighted rho|分子CPU/RAM clipping|分母CPU/RAM clipping|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for v in validations:
        rows=mechanisms[v['cell']]['rounds'];cnt=v['counts'];cl=v['clipping']
        report.append(f"|{v['cell']}|{rows[0]['selected']}|{max(r['selected'] for r in rows)}|{cnt.get('omitted_optional_columns',0)}|{cnt.get('head_required_rounds',0)}|{v['actions'].get('DRAIN_FOR_HEAD',0)}|{v['maxbudget']:.0f}|{cnt.get('numerical_retries',0)}|{v['weighted_rho']:.4f}|{cl.get('numerator_cpu',0)}/{cl.get('numerator_ram',0)}|{cl.get('denominator_cpu',0)}/{cl.get('denominator_ram',0)}|")
    report+=['','clipping单位为每轮选中列，不是作业比例；它是已冻结合成模型的重要边界，不因无容量违规而变为真实画像。raw训练率直接记录、独立矩阵重算rho、checkpoint完整几何、进度单调、原始完成事件JCT、三资源容量、承诺/队首与可选集合、服务时钟不重置和实际启动预算均复核；全部见validation_summary和逐格validation。',
        '','## 6. 解释与下一阶段限制','',
        '待结合下方结论审查文件 FINAL_REVIEW.md 阅读。只能把新S与旧G的变化归因于本轮准入规则（含optional精确同分的确定性处理），不能把S相对Sia所有差异都归因于CPU优化：两者仍同时存在准入、GPU选择和放置差异。Neutral→Medium差分也包含完整路径变化，不是固定GPU的纯CPU边际。',
        '','保留FIFO队首与服务承诺避免全空和饥饿；但随着下一轮队首必选，仍可能重新增加同时运行数。若出现退化，如实记录，不在本轮按JCT再加入预留、调整Q/D、修改公平权重或响应指数。辅助资源改变无额外加载成本仍是合成执行假设，一节点每类型无法证明复杂跨节点放置收益。',
        '','## 7. 复现入口','',
        'preregistration.json与父冻结manifest逐字节一致；resume_manifest记录961源码/输入及112历史文件核验、启动前资源与无重复进程。每格独立Python进程串行，60秒/600轮/seed_profiles(1,8)，内存2GiB、磁盘5GiB。完整命令/退出码/墙钟和原始输出hash见execution_manifest.jsonl。源码在父execution_source.zip，前置材料在父frozen_prerequisites.zip，依赖锁与工程diff均已保留。旧安全停止报告不覆盖。',
        '','正式104格继续暂停，没有新增160作业、九cohort、Low/High或参数搜索。']
    with (OUT/'JOINT_SELECTIVE_ADMISSION_RESULTS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(report)+'\n')
    print('ANALYZED',len(fresh),'new cells',len(pairs),'paired comparisons',flush=True)
if __name__=='__main__':main()
