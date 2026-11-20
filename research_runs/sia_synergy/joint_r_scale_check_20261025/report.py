"""Generate the fixed comparison report only from audited cells."""
import json,hashlib,subprocess
from pathlib import Path
OUT=Path(__file__).resolve().parent
def read(f):return json.loads((OUT/f).read_text())
def main():
    p=read('preregistration_v2.json');lines=['# JOINT-R：8→20作业规模与竞争度验证','',
        'SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。','',
        '## 范围与预注册','',
        '只执行本轮8/20作业、SIA-P/SIA-AUX/JOINT-R、Neutral/Medium，不恢复104格。600轮×60秒为统一预定上限；未完成不插补、不延长。8的六格全部完成并通过原始事件审计后才准入20。',
        '','新树 `'+p['tree']+'`，分支exp/joint-r-scale-check；基线HEAD '+p['base_commit']+'，新增文件未提交，实际执行版本以preregistration_v2.json及source_snapshot_v2.zip为准。历史策略和AdaptiveAdmission逐字节保留，只新增显式上限20的子类和运行入口；未调整目标、Q300、D250、phi、响应指数、网格、容量、种子。',
        '','输入从官方workload-1按BERT→CIFAR→DeepSpeech轮转，各类依原行序，耗尽跳过、不复制；8为20的前缀。8为BERT3/CIFAR3/DeepSpeech2，20为6/7/7。到达全部设0用于受控竞争，原arrival及源行保留，工作量/参数未改。这不是自然到达trace，也不是同工作量下纯粹改变N的因果实验。',
        '','容量AWS4GPU/12CPU/250GB、DGX8/24/500、RTX8/24/500；CPU为synthetic单位。seed_profiles(1,8)，固定R0/E-legacy，alpha0/.5，独立进程串行。',
        '','SIA-P调用官方Sia GPU决策，无L2；SIA-AUX冻结同轮官方GPU意图后独立优化辅助资源；JOINT-R联合选择并保留FIFO服务及有界配置适应。三者准入/放置并非完全相同，JOINT-R收益不能仅归因CPU感知。',
        '','## 测试、完整性与原始审计','',
        '125项测试全部通过（原122＋新增3）：8/20全队列重复服务、负效用强制服务、排空、容量、故障原子回滚及21任务拒绝；原始Sia与真实未seed集成测试仍在完整套件中。新规模测试使用真实MILP状态机，不以人工toy速度代替实际实验输入。所有仿真使用真实Sia Job/profile。',
        '','首次准备有路径/数据类别数量预检错误，发生于任何仿真之前；Windows psutil Path适配修正后发布v2冻结，v1原manifest/快照保留。详见preparation_notes.md。没有按JCT修正策略。',
        '','分析器从原始完成事件复算JCT，从直接记录的rho之前速率验证进度积分，复算互斥等待/启动/训练、周期内完成资源小时、节点GPU/CPU/RAM、完整checkpoint锁、服务时钟/费用/回本。所有均与summary核对；clipping作为模型局限保留。']
    allresults=[];available=[]
    for n in [8,20]:
        if not (OUT/f'validation_{n}.json').exists():continue
        gate=read(f'validation_{n}.json');rs=read(f'comparison_{n}.json');available.append(n);allresults+=rs
        lines+=['',f'## {n}作业结果','',f"审计valid={gate['valid']}，六格全部完成={gate['all_complete']}。",'',
            '|响应|方法|完成|平均JCT|Median|P95|P99|Makespan|','|---|---|---:|---:|---:|---:|---:|---:|']
        for s in rs:
            q=s['independent_jct_metrics'];make=max((r['completion'] for r in s['jobs'].values() if r['completion'] is not None),default=0) if s['completed']==n else None
            def fmt(v):return '删失' if v is None else f'{v:.2f}'
            lines.append(f"|{s['response']}|{s['method']}|{s['completed']}/{n}|{fmt(q['mean'])}|{fmt(q['median'])}|{fmt(q['p95'])}|{fmt(q['p99'])}|{fmt(make)}|")
        lines+=['','P95/P99只基于该8或20作业样本，不是高精度总体尾部估计。','',
                '|响应|对照|JOINT-R平均JCT改善率（正为好）|逐作业改善/退化/相同|净收益最大贡献作业|去掉该作业诊断改善率|',
                '|---|---|---:|---|---|---:|']
        for response in p['responses']:
            rr=next(s for s in rs if s['response']==response and s['method']=='joint_r')
            for method in ['sia_p','sia_aux']:
                b=next(s for s in rs if s['response']==response and s['method']==method)
                if rr['completed']!=n or b['completed']!=n:
                    lines.append(f'|{response}|{method}|禁止完整均值比较（删失）|—|—|—|');continue
                dif={j:b['jobs'][j]['completion']-rr['jobs'][j]['completion'] for j in b['jobs']}
                top=max(dif,key=dif.get);diagb=sum(v['completion'] for j,v in b['jobs'].items() if j!=top)/(n-1);diagr=sum(v['completion'] for j,v in rr['jobs'].items() if j!=top)/(n-1)
                net=sum(dif.values());share=dif[top]/net*100 if net else 0
                lines.append(f"|{response}|{method}|{100*(1-rr['mean_jct']/b['mean_jct']):.2f}%|{sum(v>0 for v in dif.values())}/{sum(v<0 for v in dif.values())}/{sum(v==0 for v in dif.values())}|{top}: {dif[top]:.0f}s（净差额占比{share:.1f}%）|{100*(1-diagr/diagb):.2f}%|")
        lines+=['','移除作业仅是诊断，主结果保留全部作业；净差额接近0或为负时贡献占比可能大于100%或负值，不作概率解释。','',
                '|响应|方法|GPU-h|CPU-h|RAM-GB-h|等待s|checkpoint s|训练s|restart|wasted GPU-s|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for s in rs:
            sums={k:sum(v[k] for v in s['jobs'].values()) for k in ['gpu_hours','cpu_core_hours','ram_gb_hours','wait','checkpoint','training']}
            restart=sum(v.get('restarts',0) for v in s['jobs'].values());waste=sum(v.get('wasted_gpu_seconds',0) for v in s['jobs'].values())
            lines.append(f"|{s['response']}|{s['method']}|{sums['gpu_hours']:.3f}|{sums['cpu_core_hours']:.3f}|{sums['ram_gb_hours']:.3f}|{sums['wait']:.0f}|{sums['checkpoint']:.0f}|{sums['training']:.0f}|{restart}|{waste:.0f}|")
        lines+=['','### 是否真正存在竞争与响应机会','',
            '|单元|最高活跃/等待者|等待job-round|CPU满GPU闲node-round|可行加速job-round|rho>1训练秒|训练加权rho|重配置/排空事件|',
            '|---|---|---:|---:|---:|---:|---:|---|']
        for c in gate['cells']:
            counts=c['counts'];acts=c['actions'];op=c['opportunity_counts']
            lines.append(f"|{c['cell']}|{c['max_active']}/{c['max_waiters']}|{counts['waiting_job_rounds']}|{counts['cpu_saturated_gpu_idle_node_rounds']}|{op.get('job_round_with_individually_feasible_acceleration',0)}|{c['boosted_training_seconds']:.0f}|{c['training_weighted_rho']:.4f}|GPU适应{acts.get('ADAPT_GEOMETRY',0)} / 排空{acts.get('DRAIN_FOR_HEAD',0)}|")
        lines+=['','### 响应边界与求解状态','',
                '|单元|selected列数|分子CPU/RAM越界|分母CPU/RAM越界|状态计数|','|---|---:|---|---|---|']
        for c in gate['cells']:
            cl=c['clipping_counts']
            lines.append(f"|{c['cell']}|{c['selected_count']}|{cl.get('numerator_cpu',0)}/{cl.get('numerator_ram',0)}|{cl.get('denominator_cpu',0)}/{cl.get('denominator_ram',0)}|{c['statuses']}|")
        lines+=['','机会计数为当轮单任务存在节点内可行的优于比例点的候选，不意味着所有任务可同时获得它；排队job-round与节点事件不同单位，不能相加。',
            '','### Medium逐作业完成/JCT（全部到达0）','',
            '|Job|SIA-P|SIA-AUX|JOINT-R|R相对P差值（负为好）|R等待|R checkpoint|','|---|---:|---:|---:|---:|---:|---:|']
        by={s['method']:s for s in rs if s['response']=='medium'}
        for row in p['records'][str(n)]:
            j=row['name'];a=by['sia_p']['jobs'][j];b=by['sia_aux']['jobs'][j];r=by['joint_r']['jobs'][j]
            if all(v['completion'] is not None for v in [a,b,r]):lines.append(f"|{j}|{a['completion']}|{b['completion']}|{r['completion']}|{r['completion']-a['completion']:+.0f}|{r['wait']:.0f}|{r['checkpoint']:.0f}|")
            else:lines.append(f'|{j}|{a["completion"]}|{b["completion"]}|{r["completion"]}|删失不插补|{r["wait"]}|{r["checkpoint"]}|')
    executive=['## 直接回答：作业数量是否限制了优势','',
        '作业少可能限制互补机会，但不能仅靠增大N保证联合方案获胜。本次只使用一个嵌套、受控同时到达的负载家族；新增作业改变总工作量及细分构成，不是独立trace复验。',
        '','|规模|Medium JOINT-R对SIA-P改善|对SIA-AUX改善|','|---|---:|---:|',
        '|此前3作业|2.36%|-1.39%|']
    for n in available:
        rows=[s for s in allresults if s['size']==n and s['response']=='medium'];b={s['method']:s for s in rows}
        if all(s['completed']==n for s in rows):
            gp=100*(1-b['joint_r']['mean_jct']/b['sia_p']['mean_jct']);ga=100*(1-b['joint_r']['mean_jct']/b['sia_aux']['mean_jct'])
            executive.append(f'|{n}作业|{gp:.2f}%|{ga:.2f}%|')
        else:executive.append(f'|{n}作业|删失不作完整均值比较|删失不作完整均值比较|')
    executive+=['','8作业中JOINT-R相比SIA-AUX的净提前123秒，小于deepspeech2-9单作业提前357秒；移除该作业后其余7作业净慢234秒。故不能将1.04%均值改善解释为多数作业普遍改善。主分析仍包含全部作业。',
        '','三类应用、固定容量和同一响应下，SIA-AUX的Medium对照已证明存在可兑现的辅助资源收益；JOINT-R是否额外获益应以其相对AUX及逐作业差异判断，而不是仅对SIA-P获益。',
        '','抽样轮转仅决定纳入集合；同到达时FIFO仍按保存的原始source_row_index排序，不按人为轮转顺序改变优先级。Neutral较小配额资源小时下降只体现该人工平坦响应下的合法省配额，不是现实硬件节约证据。',
        '','---','']
    executive+=['### Neutral→Medium：辅助响应的实际增量','',
        '|规模|SIA-P均值降低秒|SIA-AUX均值降低秒|JOINT-R均值降低秒|R比AUX多降低秒|',
        '|---|---:|---:|---:|---:|']
    for n in available:
        b={(s['response'],s['method']):s for s in allresults if s['size']==n}
        if not all(s['completed']==n for s in b.values()):continue
        delta={m:b[('neutral',m)]['mean_jct']-b[('medium',m)]['mean_jct'] for m in p['methods']}
        executive.append(f"|{n}|{delta['sia_p']:.3f}|{delta['sia_aux']:.3f}|{delta['joint_r']:.3f}|{delta['joint_r']-delta['sia_aux']:.3f}|")
    executive+=['','该差分比较完整策略路径；开启响应后GPU/放置/服务路径也可能变化，不能解释为同状态下纯CPU的因果边际。Neutral的联合收益本身不来自CPU加速。','']
    if 20 in available:
        executive+=['### 20作业的关键反证与耗时定位','',
            'Medium JOINT-R平均3277.15秒，比SIA-P改善5.75%，但比SIA-AUX增加5.28%；相对AUX为7作业改善、13作业退化。因而不能把此前优势不足仅归因于作业太少，也没有随规模增加持续扩大联合增量。',
            '','相对AUX的总JCT净增加3289秒，可精确分解为：等待减少3120秒（4320→1200），checkpoint减少610秒（3920→3310），有效训练耗时增加7019秒（54014→61033）。restart也从58降到48。当前损失集中在实际训练耗时，而不是额外排队或checkpoint；GPU/服务选择的作用值得隔离，不应再简单加大N。',
            '','JOINT-R有37213个物理训练秒rho>1、训练秒加权rho约1.17，AUX为8009秒和约1.06；更高rho仍未带来更低JCT。因为GPU原始速率、资源份额、训练任务构成与完成顺序同时变化，不能把rho当作绝对效率指标。',
            '','20作业SIA-P/AUX真实出现等待（最高4个等待者），JOINT-R没有分配外等待轮，只有共同初始60秒；未出现DRAIN事件。故这组已验证资源竞争及准入差异，但没有验证排空在实际20作业轨迹中的效果（其证据仍来自状态机测试）。',
            '','相对AUX，bert-99慢1053秒、bert-54慢907秒，两者占净退化1960/3289=59.6%；即使诊断性移除这两项，剩余18作业仍净慢1329秒，不能用单一异常长尾解释。deepspeech2-31本轮反而比AUX慢196秒。主结果不删除任何作业。',
            '','下一步最少工作应是用这些已有同状态日志审查低GPU速率/服务内候选过滤导致的损失，再预注册固定服务集合与准入差异的隔离对照；本轮不实施、不调整Q或预算、不重复参数寻优。至少需要另一独立trace及自然到达对照，才可讨论泛化。','']
    lines[4:4]=executive
    events=[json.loads(l) for l in (OUT/'execution_manifest.jsonl').read_text().splitlines()]
    lines+=['','## 局限、复现与停止','',f"本次实际策略进程墙钟合计{sum(e['wall_seconds'] for e in events):.2f}秒；逐格日志/命令/hash在execution_manifest.jsonl。测试墙钟见test_result.json；编排及分析另计。",'',
        '不能将较多作业自动解释成更好：队列/排空和checkpoint可能抵消并行资源互补；公平目标与平均JCT仍不同。BERT不敏感、其他模板跨硬件转移，R0/E-legacy clipping未被验证为目标硬件真实性能。固定合成负载只有一组、不是独立重复、未检验真实到达或长期一般化。',
        '','原104格仍暂停，未运行任何160任务、九cohort、Low/High或参数搜索。旧目录/官方源码未改。本轮完成即停止；新测试或结果不构成自动扩大实验的授权。']
    with (OUT/'SCALE_COMPETITION_REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    with (OUT/'all_comparisons.json').open('x',encoding='utf-8') as f:json.dump(allresults,f,indent=2)
    for path,h in p['hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()==h,path
    manifest=dict(status='COMPLETE' if 20 in available else 'PARTIAL_STOP_20_NOT_VALIDATED',sizes=available,base_commit=p['base_commit'],working_tree='tracked source unchanged; explicit untracked adapters in snapshot',
        results={str(f.relative_to(OUT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.rglob('*') if f.is_file()},
        wall_seconds=sum(e['wall_seconds'] for e in events),frozen_hashes_rechecked=len(p['hashes']))
    with (OUT/'final_manifest.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2)
    print('SEALED',len(events),'cells',manifest['frozen_hashes_rechecked'],'frozen hashes')
if __name__=='__main__':main()
