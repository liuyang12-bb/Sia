"""Seal one principled revision, not an outcome-guided optimization loop."""
import json,hashlib,zipfile,subprocess
from pathlib import Path
OUT=Path(__file__).resolve().parent;A=OUT/'attempt3';ROOT=OUT.parents[2]
def read(f):return json.loads(Path(f).read_text())
def main():
    p=read(A/'preregistration.json');OLD=Path(p['baseline_directory']);results=[]
    for n in [8,20]:
        if (A/f'comparison_{n}.json').exists():
            results+=read(A/f'comparison_{n}.json')
        results+=read(OLD/f'comparison_{n}.json')
    rows=['# Sia＋Synergy：整组资源交换改进与验证','',
        'SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。','',
        '## 直接结论','',
        '新策略名JOINT-G。它不是调大bonus或改变响应：把持有服务的任务之间的GPU/CPU/RAM交换交给同一个原公平MIP，而不再要求每个任务单独不降速、在服务剩余时间内单独回本。真实锁、FIFO有效服务、实际启动预算仍保留。',
        '','SIA-P为官方Sia GPU决策＋共同辅助资源执行适配；SIA-AUX为Sia GPU固定后的本项目辅助MIP，不是官方Synergy-TUNE。JOINT-G借鉴Sia公平/异构候选和Synergy响应思想，不声称直接移植官方Synergy-OPT或TUNE；本轮没有重跑SY/TUNE独立对照。',
        '','|规模/响应|SIA-P均值|SIA-AUX均值|旧JOINT-R均值|新JOINT-G均值|G对P改善|G对AUX改善|G对R改善|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    lookup={(s['size'],s['response'],s['method']):s for s in results}
    for n in [8,20]:
        for r in ['neutral','medium']:
            if (n,r,'joint_g') not in lookup:continue
            b={m:lookup[n,r,m] for m in ['sia_p','sia_aux','joint_r','joint_g']}
            if any(v['completed']!=n for v in b.values()):rows.append(f'|{n}/{r}|删失，禁止完整均值比较|—|—|—|—|—|—|');continue
            g=b['joint_g']['mean_jct'];rows.append(f"|{n}/{r}|{b['sia_p']['mean_jct']:.3f}|{b['sia_aux']['mean_jct']:.3f}|{b['joint_r']['mean_jct']:.3f}|{g:.3f}|{100*(1-g/b['sia_p']['mean_jct']):.2f}%|{100*(1-g/b['sia_aux']['mean_jct']):.2f}%|{100*(1-g/b['joint_r']['mean_jct']):.2f}%|")
    if (20,'medium','joint_g') in lookup:
        s=lookup[20,'medium','joint_g'];b=lookup[20,'medium','sia_aux']
        if s['completed']==20:
            verdict='本受控20作业场景中，新策略优于串联。' if s['mean_jct']<b['mean_jct'] else '本受控20作业场景中，新策略仍未优于串联，不能宣布JCT目标已解决。'
            rows+=['',verdict+' 这只是一组冻结的合成响应、同时到达负载，不能当作跨workload或真实集群的普遍结论。']
            rows+=['','相对AUX，新G总JCT仍增加889秒，分解为等待减少3120秒、checkpoint减少780秒、有效训练耗时增加4789秒；restart为44而非AUX的58。旧R的额外训练为7019秒，新G减少了其中2230秒，但仍未抵消全部损失。不能把更少等待/更少重启直接当作更好JCT。',
                '','bert-54/99的RTX1训练从旧R各1200秒降为各1080秒，但仍比AUX慢810/720秒；二者合计1530秒，超过总净退化889秒（其他作业改善抵消部分损失）。这说明局部过滤不是唯一问题：当前公平目标、强制服务集合与GPU机会分配仍可能限制完成效率；具体贡献需要隔离，不能只凭日志断言删除L2就会更好。',
                '','后续最小设计方向是保留真实锁和未兑现服务承诺，FIFO队首仍必须获得有界服务；在承诺到期时，让其余未承诺作业成为可选集合，以同一公平效用选择，而非先贪心塞满所有可并发任务。队首不适配时继续排空，有限先行服务的有界退出用于防饥饿；冷启动队首不因负效用被清空。此为下一阶段建议，**本轮未实施或验证**，也不保证一定优于Sia。']
    rows+=['','### 响应开关的完整策略路径差分','',
        '|规模|AUX Neutral→Medium均值降低|旧R降低|新G降低|G比旧R多降低|','|---|---:|---:|---:|---:|']
    for n in [8,20]:
        if (n,'medium','joint_g') not in lookup:continue
        reduction={m:lookup[n,'neutral',m]['mean_jct']-lookup[n,'medium',m]['mean_jct'] for m in ['sia_aux','joint_r','joint_g']}
        rows.append(f"|{n}|{reduction['sia_aux']:.3f}|{reduction['joint_r']:.3f}|{reduction['joint_g']:.3f}|{reduction['joint_g']-reduction['joint_r']:.3f}|")
    rows+=['','该差分同时包含响应引起的GPU/配额/服务路径改变，不是固定同状态纯CPU边际。20作业新G的响应收益比旧R大110.6秒均值，但仍小于AUX，不能说CPU/RAM机会已经充分利用。']
    rows+=['','## 原因：不是简单的“作业太少”或“CPU画像没有机会”','',
        '旧20作业Medium相对AUX净慢3289秒，等待和checkpoint合计少3730秒，却多7019秒训练。bert-54/99在RTX1各训练1200秒；旧策略不止约束迁移次数，还逐任务禁止资源让渡。真正的联合交换有时必须让一个任务略慢，换取其他任务更大的收益；局部不降速不等于整组更优。',
        '','最多6个事先定义状态查询，5个匹配、对应4个不同全局状态；一个指定Neutral状态未发生，记录NO_MATCH，未补造替代状态。同状态恢复目标与原日志匹配；固定任务集合、GPU画像、响应、容量、锁及原效用，只比较候选过滤。',
        '','|规模/响应/时间|原目标|仅放开单任务加速回本门槛|整组交换目标|','|---|---:|---:|---:|']
    seen=set()
    for d in read(OUT/'same_state_bundle_diagnosis.json'):
        if d.get('status')=='NO_MATCH':continue
        key=d['cell'],d['time']
        if key in seen:continue
        seen.add(key);rows.append(f"|{d['cell']}/{d['time']}|{d['old']['scheduling_objective']:.2f}|{d['relaxed']['scheduling_objective']:.2f}|{d['global_bundle']['scheduling_objective']:.2f}|")
    rows+=['','例如240秒，DeepSpeech2-16从DGX1让到AWS1、CIFAR10-1从AWS1转DGX1，连同CPU重配，主效用8.17→8.48。前者个人速率下降，故旧筛选拒绝此类交换。原效用已有restart factor；另一层短服务回本硬条件并非公平最优的必要条件。该证据只是同状态预测，不是JCT反事实，也不是给某个模型硬编码GPU。',
        '','## 数学与实现','',
        '保持原列效用 `u=lambda_n-min(round((raw_sia_goodput*rho*restart_factor)^p,2),clip)`（当前p<0）。GPU配置/CPU/RAM/placement一起选择；每个已准入或持有服务任务Σx=1，逐节点三资源容量、合法候选和比例进度下限不变。真实checkpoint只允许完整旧分配。',
        '','GPU改变仍要求上次改变后已有正训练、目标direct/projected画像可用、每服务累计真实启动开销≤250。CPU/RAM-only改变沿用执行模型的零额外加载开销假设，不能解释成真实系统免费迁移。只取消局部AUX_REDUCES_CURRENT_RATE与NO_REMAINING_SERVICE_PAYBACK否决；选择是否让渡由整组效用决定。',
        '','主效用精确同分时二级最小真实新增启动秒数，再原词典序几何排序；无epsilon/新权重。服务身份、issued_time、productive_base不重置，Q300/D250/60秒不变。当前配置保留；若保持当前服务几何＋新任务分配可行，每轮独立校验所选主效用不低于该基准。基准本身不合法时明确记录，不声称单调保证。',
        '','启动预算与正执行率/有限可行资源/求解可决假设继续给出服务有界性，FIFO排空不删；最终完成仍需有限工作量及正速率下界。没有用瞬时效用最优推导JCT必然更优。',
        '','新增bundle_adaptation.py、run_bundle_cell.py及测试；新树adaptive_service.py增加默认no-op二级tie hook，并将原milp调用等价提取为默认参数不变的hook，只有新G覆写数值验证。execution_audit.py另有独立正确性修复。官方源和旧工作树未修改，历史策略定义保持。source_snapshot及差异保留；基线HEAD仍2d5f9db，实际执行版本由快照/hash标识，不能只引用HEAD。',
        '','## 红绿测试与失败保留','',
        '原125项保持。新整组交换测试在旧类上2失败/6通过（GPU协同交换和允许CPU候选让渡）；新模式133项通过，再加空状态及真实未seed BERT链路共135项通过。首次8任务Medium在720秒遭旧checkpoint审计假阳性；原失败JSON/完整轮次/日志/源码快照保留，不当作性能样本。',
        '','原始Sia会把不足半秒的最后epoch时间round为0。作业在checkpoint结束瞬间结算进度并完成，观测存活时间被截断到checkpoint结束，但请求周期仍有正训练窗口。未改动官方Job.step已真实复现同样的进度、epoch、completion和execution_time；旧审计混淆请求区间和完成截断区间。只修复审计，并增强“segment不得在checkpoint内部/越过请求端点”的检查，半秒速率残差限制保留。无训练/JCT/重配置参数变更。新增边界红测1失败/3通过，最终完整139项通过。详见AUDIT_CORRECTION.md。',
        '','attempt2的20任务Medium又在1860秒触发必选集合保护，求解器所谓最优向量通过物理容量审计但不满足必选任务等式，未提交。补齐原始/取整向量的全部线性约束验证，仅对status=0却不满足约束的返回值，以同一fingerprint数学模型关闭presolve重求一次；目标/候选/锁/容量/时间限/最优gap不变，第二次仍不合法或不可决就停止。正常有效解路径不改求解设置。两项故障红测1失败/1错误，最终完整141项通过。此为数值结果正确性处理，不是按JCT调参；不可把原返回物理安全误称为完整服务合法。',
        '','最终捕获了同一1860秒轮次5个词典序子问题的非法“最优”返回：第一组约束第3行deepspeech2-9要求1，原x及取整x均为0，违反量1.0。5次均重求同一模型得到满足全部约束的向量，原始/取整向量、矩阵/cost/边界、模型SHA及message可复算，见numerical_evidence.json。它们不是5个等待轮或已提交违规；没有丢掉deepspeech2-9。可定位当前SciPy/内置HiGHS返回路径，不据此猜测内部具体模块。',
        '','比较仅使用attempt3单一冻结版本：完整8配对及完整20配对重新运行。原始失败、已完成旧尝试和快照不覆盖，不挑最好结果；候选规则及主/次目标方法AST与attempt2完全一致。原12格只读hash复用，并由增强审计重新核对。新旧算法相对比较包括新数值保护及二级tie，不能把JCT所有变化单独归因于某一个过滤条件。',
        '','## 冻结输入和运行','',
        '同上轮8/20嵌套CSV，应用/原工作量/batch/num_replicas均未改，到达仍全部0。8为BERT3/CIFAR3/DeepSpeech2，20为6/7/7，FIFO同到达仍按源行顺序。AWS4GPU/12CPU/250GB、DGX8/24/500、RTX8/24/500；CPU为合成单位。R0/E-legacy、模板映射、phi、网格、alpha0/.5和seed_profiles(1,8)全部沿用。600轮固定上限，不根据结果延长。独立进程串行，2GiB/5GiB门槛，104自动化保持PAUSED。',
        '','## 尾部、资源与开销','',
        '|规模/响应/方法|Median|P95|P99|Makespan|GPU-h|CPU单位-h|RAM-GB-h|等待s|checkpoint s|训练s|restart|wasted GPU-s|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in results:
        if s['method']=='sia_p' and s['response']=='neutral':continue
        j=list(s['jobs'].values());q=s.get('independent_jct_metrics',{})
        sums={k:sum(v[k] for v in j) for k in ['gpu_hours','cpu_core_hours','ram_gb_hours','wait','checkpoint','training','restarts','wasted_gpu_seconds']}
        rows.append(f"|{s['size']}/{s['response']}/{s['method']}|{s['median_jct']:.2f}|{s['p95_jct']:.2f}|{s['p99_jct']:.2f}|{s['makespan']:.0f}|{sums['gpu_hours']:.3f}|{sums['cpu_core_hours']:.3f}|{sums['ram_gb_hours']:.3f}|{sums['wait']:.0f}|{sums['checkpoint']:.0f}|{sums['training']:.0f}|{sums['restarts']}|{sums['wasted_gpu_seconds']:.0f}|")
    rows+=['','P95/P99只用8或20个作业，不能解释为高精度总体尾估计。Neutral资源小时更低可能仅因平坦模型下允许低配额，不是目标硬件节约证据。',
        '','## Medium逐作业与长尾归因']
    for n in [8,20]:
        if (n,'medium','joint_g') not in lookup:continue
        b={m:lookup[n,'medium',m] for m in ['sia_p','sia_aux','joint_r','joint_g']}
        rows+=['',f'### {n}作业','', '|Job|SIA-P|AUX|旧R|新G|G-AUX|G等待|G checkpoint|','|---|---:|---:|---:|---:|---:|---:|---:|']
        dif={}
        for v in p['records'][str(n)]:
            j=v['name'];val={m:b[m]['jobs'][j]['jct'] for m in b};g=b['joint_g']['jobs'][j]
            if any(x is None for x in val.values()):rows.append(f'|{j}|删失不插补|—|—|—|—|—|—|');continue
            delta=val['joint_g']-val['sia_aux'];dif[j]=-delta
            rows.append(f"|{j}|{val['sia_p']}|{val['sia_aux']}|{val['joint_r']}|{val['joint_g']}|{delta:+.0f}|{g['wait']:.0f}|{g['checkpoint']:.0f}|")
        if len(dif)==n:
            top=max(dif,key=dif.get);net=sum(dif.values());ab=b['sia_aux']['mean_jct'];gg=b['joint_g']['mean_jct']
            rb=(n*ab-b['sia_aux']['jobs'][top]['jct'])/(n-1);rg=(n*gg-b['joint_g']['jobs'][top]['jct'])/(n-1)
            rows+=['',f"相对AUX：{sum(v>0 for v in dif.values())}改善/{sum(v<0 for v in dif.values())}退化/{sum(v==0 for v in dif.values())}相同。总JCT净提前{net:.0f}秒，最大正贡献{top}={dif[top]:.0f}秒；诊断性去掉它后均值改善率{100*(1-rg/rb):.2f}%。主结果不删任何作业。"]
            sums={m:{k:sum(v[k] for v in b[m]['jobs'].values()) for k in ['wait','checkpoint','training']} for m in ['sia_aux','joint_g']}
            rows+=['',f"总JCT变化G-AUX={-net:.0f}秒 = 等待变化{sums['joint_g']['wait']-sums['sia_aux']['wait']:+.0f} + checkpoint变化{sums['joint_g']['checkpoint']-sums['sia_aux']['checkpoint']:+.0f} + 有效训练变化{sums['joint_g']['training']-sums['sia_aux']['training']:+.0f}。"]
    rows+=['','## 新模式原始审计与模型边界','',
        '|单元|完成|执行片段|重新允许并实际选择的列|整组基准检查|最大累计启动|量化epoch事件|数值重求解|加速训练秒|训练加权rho|分子CPU/RAM截断|分母CPU/RAM截断|selected数|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|']
    for n in [8,20]:
        if not (A/f'validation_{n}.json').exists():continue
        for c in read(A/f'validation_{n}.json')['cells']:
            k=c['counts'];cl=c['clipping'];rows.append(f"|{c['cell']}|{c['all_complete']}|{k['execution']}|{k.get('selected_previously_vetoed_column',0)}|{k.get('nondecreasing_global_objective',0)}|{c['max_delay']:.0f}|{k.get('zero_duration_epoch_settlements',0)}|{k.get('independently_verified_numerical_retries',0)}|{c['boosted_training_seconds']:.0f}|{c['weighted_rho']:.4f}|{cl.get('numerator_cpu',0)}/{cl.get('numerator_ram',0)}|{cl.get('denominator_cpu',0)}/{cl.get('denominator_ram',0)}|{c['selected_count']}|")
    rows+=['','逐节点容量/完整checkpoint/直接raw×rho/互斥时间/完成资源截断/预算费用与实际事件/主目标及二级启动成本均从原始日志复算；非法或不可行则停止，不能部分提交。clipping计数单位为selected列，不能把它当事件或作业比例。完整求解状态和fallback、CPU满GPU闲、时钟与重启见validation及summary；不以摘要代替审计。',
        '','## 局限与下一步','',
        '本轮只实现一个机制版本，没有在看JCT后继续改Q、预算、bonus、目标权重或映射。公平效用不是JCT目标，固定服务任务集合仍可能过宽，按物理有效训练秒服务仍不同于按工作进度公平；本次不同时修改准入，以免无法解释。新结果若有效，也不能把GPU、CPU/RAM、准入与二级tie的贡献各自分离。',
        '','单一合成响应/同时到达家族，不是独立trace重复。R0跨应用/跨硬件迁移、phi、E-legacy clipping的不确定性保留；辅助配额加载开销为0只是当前假设。下一步应先在另一个预注册独立trace/真实相对到达上复验，并隔离服务集合/候选交换因素；此轮不自动实施、不恢复104格。',
        '','报告/机器汇总在本目录；有效新日志在attempt3。复现应从基线HEAD＋source_snapshot恢复，读取preregistration，重新核对全部hash，在全新目录运行；不要从裸HEAD假定未提交新代码已经存在。源码/输入的完整副本、最终hash和命令随交付封存。']
    v2events=[readline for readline in (A/'execution_manifest.jsonl').read_text().splitlines()]
    wall=sum(json.loads(e)['wall_seconds'] for e in v2events)
    rows+=['',f'新版本四格验证策略进程墙钟合计{wall:.2f}秒；失败首次尝试、单元测试、离线求解、分析另计，全部日志保留。']
    with (OUT/'JOINT_GLOBAL_EXCHANGE_REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(rows)+'\n')
    with (OUT/'all_comparisons.json').open('x',encoding='utf-8') as f:json.dump(results,f,indent=2)
    print('REPORT_WRITTEN',len(results),'strategy-response-size results')
if __name__=='__main__':main()
