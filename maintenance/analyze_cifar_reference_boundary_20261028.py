"""Read-only boundary diagnosis; no oracle/model edits or training calls."""
import ast, collections, hashlib, json, math, statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
TREE=ROOT/'cc-joint-resource-study'
PREF=ROOT/'research_runs/sia_synergy/cc_resource_mixes_preflight_20261028_run1'
OUT=ROOT/'research_runs/sia_synergy/cc_cifar_reference_boundary_20261028_run1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(n,x):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def table(h,rs):return '\n'.join(['|'+'|'.join(h)+'|','|'+'|'.join(['---']*len(h))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rs])
def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'REPORT.md').exists(), 'refuse completed directory'
    paths=[TREE/'inputs/registry.json',TREE/'inputs/models/alexnet_1.py',TREE/'sia-simulator/joint_prototype/joint_profile.py',PREF/'all_queries.json',PREF/'delivery_hashes.json']
    old=ROOT/'research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1/completed_analysis_20261028_run1/resource_and_clipping_summary.json';paths.append(old)
    protected={str(p.relative_to(ROOT)):sha(p) for p in paths}
    reg=read(paths[0]);r=next(r for r in reg if r['key']=='cifar10/dgx-ext:1:2')
    cf=[x for x in reg if x['application']=='cifar10'];assert math.isclose(min(x['per_gpu_T0'] for x in cf),r['denominator'])
    assert math.isclose(r['T0'],64/r['step_time']);assert math.isclose(r['phi_c'],r['per_gpu_T0']/r['denominator']);assert math.isclose(r['phi_m'],math.sqrt(r['phi_c']))
    source=TREE/'sia-simulator/traces/cifar10/placements-dgx-ext.csv'
    assert sha(source)==r['source_rows'][0]['sha256']
    text=paths[1].read_text(encoding='utf8');tree=ast.parse(text)
    mat=next(np.asarray(ast.literal_eval(n.value.args[0]),float) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='tput' for t in n.targets))
    def F(c,m):return float(np.interp(c,[1,2,3,4,5,6,9,12,24],[np.interp(m,[20.83,62.5,125,187.5,250],row) for row in mat]))
    u=3/r['phi_c'];v=62.5/r['phi_m'];d=F(1,v);slope=F(2,v)-d
    # Counterfactual arithmetic only; neither rule is installed in execution.
    linear=d+(u-1)*slope;origin=d*u
    comparisons=[]
    for c,m in [(3,62.5),(3,93.75),(6,93.75),(6,125)]:
        n=F(c/r['phi_c'],m/r['phi_m']); rho=math.sqrt(n/d)
        comparisons.append(dict(cpu=c,ram=m,source_cpu=c/r['phi_c'],source_ram=m/r['phi_m'],legacy_numerator=n,legacy_rho=rho,
            linear_denominator_only_ratio=math.sqrt(d/linear),note='Denominator-only diagnostic invalid as complete replacement where numerator also out of domain; fixed reference must remain rho=1.'))
    qs=read(PREF/'all_queries.json');ranges={k:[min(q[k] for q in qs),max(q[k] for q in qs)] for k in ('source_cpu','source_ram','baseline_cpu','baseline_ram')}
    impacts=[]
    for cell,x in read(old).items():
        impacts.append(dict(cell=cell,denominator_column_count=x['response_checks']['denominator_cpu_clip'],
            denominator_training_fraction=x['clipping_fraction_productive_job_seconds']['denominator_cpu'],
            numerator_training_fraction=x['clipping_fraction_productive_job_seconds']['numerator_cpu']))
    save('calculation.json',dict(registry_row=r,source_profile_hash_verified=True,minimum_reference_verified=True,
        denominator_coordinate=[u,v],cpu_relative_shortfall=1-u,legacy_denominator=d,
        hypothetical_local_linear_denominator=linear,hypothetical_origin_linear_denominator=origin,
        local_linear_rho_ratio_for_supported_numerator=math.sqrt(d/linear),origin_linear_rho_ratio_for_supported_numerator=math.sqrt(d/origin),
        examples=comparisons,full_grid_coordinate_ranges=ranges))
    save('historical_exposure.json',impacts)
    report=f'''# CIFAR10 DGX双GPU参考分母边界诊断

结论：这是冻结的R0跨硬件映射将固定参考点推到源矩阵之外，随后E-legacy按既有规则做常值边界延拓；不是浮点误差、GPU配置非法或重复乘rho。不能把坐标强行改成1后宣称“恢复到实测支持”：当前代码已经这样计算。

推荐最小方案是**不做数值修补**，把E-legacy的常值边界假设显式预注册并完整审计，将42格定位为固定合成执行模型下的机制验证。若要求所有合法候选均处于原矩阵支持范围，则现有数据不足，需要先补数据/审定迁移边界模型；仅修这个分母也不能达标。本轮只分析，未改模型、未训练或启动42格。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE

## 1. 完整来源与复算

registry.json:648–688给出cifar10/dgx-ext:1:2。Sia placements-dgx-ext.csv第113行、microbatch32、step={r['step_time']:.15f}s，2GPU无累积参考T0=64/step={r['T0']:.9f}样本/秒，每GPU={r['per_gpu_T0']:.9f}。11个原始合法GPU配置的per_gpu_T0最小值={r['denominator']:.9f}；本脚本核对原CSV哈希与最小值。该最小值来自rtx:1:8，而当前小集群没有RTX、也没有8GPU节点；固定R0沿用原完整域，不随当前拓扑重算。

R0规则phi_c=clip(per_gpu_T0/min_reference,0.5,4)={r['phi_c']:.12f}，phi_m=sqrt(phi_c)={r['phi_m']:.12f}。这些是策略无关的合成需求缩放，不是实测CPU需求。快GPU配置被映射成较高CPU需求，固定(3CPU,62.5GB)参考映射为(3/phi_c,62.5/phi_m)=({u:.12f},{v:.12f})。

CPU源网格最低1，因此越界{(1-u)*100:.4f}%；不是四舍五入误差。RAM在[20.83,250]内。源CPU第一行AlexNet矩阵为[.1,.3,.3,.3,.3]；在RAM={v:.6f}处插值得到当前分母F(1,v)={d:.12f}。CPU在该RAM处的1→2斜率={slope:.12f}。

证据：joint_profile.py:20–21源网格、181–182 np.clip；263–269分子分母映射；273–276比率及alpha幂。R0Oracle不改变基类插值，registry要求固定哈希，未用策略历史轨迹重新拟合phi。

## 2. 现有常值边界意味着什么

记C为按矩形支持域裁剪，F为域内插值。现执行rho=[F(C(c/phi_c,m/phi_m))/F(C(3/phi_c,62.5/phi_m))]^.5。对低于源最低CPU的点，E-legacy假设响应等于最低CPU行。原始矩阵没有测量0.9408，因此这部分是边界假设，不是实测。

该定义在实际固定参考(3,62.5)上，分子分母同点同裁剪，仍严格rho=1；R01比例点(6,93.75)和R11比例点(3,93.75)不等于固定参考，rho可不等于1。不能把容量比例点改成新分母来“修正”，那会改变本轮科学定义。

{table(['每GPU CPU','RAM','源CPU','源RAM','当前rho'],[[x['cpu'],x['ram'],round(x['source_cpu'],6),round(x['source_ram'],6),round(x['legacy_rho'],6)] for x in comparisons])}

## 3. 分母数值影响：能说到哪里

仅做两个未接入执行的离线反事实，均不是建议自动采用的新模型：

* 沿CPU=1→2局部斜率线性延伸到0.9408，分母={linear:.12f}。若某候选分子本来在支持域，rho将统一乘{math.sqrt(d/linear):.8f}，约增加{(math.sqrt(d/linear)-1)*100:.4f}%。
* 假设0CPU响应为0并连接到1CPU，分母={origin:.12f}，对应乘{math.sqrt(d/origin):.8f}，约增加{(math.sqrt(d/origin)-1)*100:.4f}%。

这两个数不是可信区间或上界；原矩阵不能决定0.9408处的唯一真值。若假设响应随CPU单调且固定支持域内分子，则真实低CPU分母不高于最低行，当前常值分母会低估该配置的rho，而非通过压低分母夸大加速。但任务选择/重配置的闭环影响没有固定方向，不能声称一定有利或不利于某策略。

只改分母、不同时处理越界分子会破坏参考点rho=1。固定GPU配置的两个支持域内配额rho之比，分母会抵消；因此局部配额速度比不受该分母常数影响。不同GPU配置之间的绝对速度排序和跨任务资源取舍仍可能变化。

## 4. 对已有结论的影响

历史Neutral alpha0恒rho1，分母问题对执行率无影响，因此已完成的Neutral排队消融不被它数值改变。Medium历史数值仍是冻结E-legacy模型下真实完成事件结果，不能自动作废，也不能升级为目标硬件证据。

{table(['已完成容量','SIA-P分母裁剪训练秒占比','SIA-AUX占比','CC四方法分母占比'],[[cap, str(round(next(x['denominator_training_fraction'] for x in impacts if x['cell']==cap+'_medium_SIA_P')*100,4))+'%',str(round(next(x['denominator_training_fraction'] for x in impacts if x['cell']==cap+'_medium_SIA_AUX')*100,4))+'%',str([round(x['denominator_training_fraction']*100,4) for x in impacts if x['cell'].startswith(cap+'_medium_CC_')])] for cap in ('R00','R10','R01','R11')])}

上述占比复用已有原始日志独立审计结果，不把候选越界计数当执行暴露。CC若未选到该配置，直接执行暴露为0仍不代表候选评分未受影响；未运行无裁剪反事实，不能量化历史JCT会改变多少。CPU5.28%–10.82%、RAM3.36%–5.09%的历史配额消融需继续标注模型依赖。

上一轮CPU与RAM网格内见证、AWS单/双GPU排序反转的分子分母均在支持域，不依赖本处裁剪，仍成立；它们只是单状态预测机会，不是新负载JCT证据。

## 5. 为什么只解决分母还不够

完整候选网格还存在分子CPU/RAM越界：本次全部查询源CPU范围{ranges['source_cpu']}，RAM范围{ranges['source_ram']}，均超出源矩阵低端；本域高端未越界。具体数量在上一轮boundary_counts.json。分母缺口集中在CIFAR DGX双GPU；分子缺口并不限于它。

不能通过删除合法配置、仅筛选获胜策略的执行点、补phi=1或悄悄限制资源网格来宣称全域闭合。即使补测0.9408分母，要坚持所有候选均域内仍需覆盖其他低资源点。

## 6. 处理方案比较与明确建议

|方案|变化和依据|评价|
|---|---|---|
|保留E-legacy，显式登记边界延拓|不改任何数值；保留所有合法候选，对各方法分别统计分子/分母裁剪|推荐的最小方案，适合已明确允许合成数据的机制实验；不能称全域实测支持|
|补低资源画像|测量相应源坐标，或有依据地建立并冻结完整边界模型|严格数据支持目标所需；当前没有数据，不可自行补造|
|线性外推/过原点延伸|增加未测假设，必须分子分母一致处理并另立模型版本|可作未来敏感性方案，不能自动代替已冻结响应；本轮仅算示例|
|把phi_c截到3、改参考吞吐或改参考点|连同域内曲线、GPU排序与既有结果一起改变|不推荐，不是单纯工程修复|
|删掉DGX双GPU或越界候选|改变Sia合法配置/联合候选，可能对方法不公平|不推荐|
|仅把分母坐标设为1|现有E-legacy已经这么算|没有解决新问题，只是重述当前实现|

**能否开展42格：**若接受“冻结E-legacy合成边界假设下的机制验证”，不需要数值修模型；三组负载均已有CPU/RAM互补及GPU数量耦合见证，完成预注册披露、六方法同函数/网格/容量核验、逐格执行裁剪统计后可以开展。若要求“整个合法候选域所有点都有原始矩阵支持”，目前不能开展该口径下的42格，需要先补完整边界数据；两者不能混称。

建议采用前一种研究口径，完整保留不利结果并将边界影响作为局限。它不保证结果稳健到其他外推模型；边界敏感性未验证前，不能宣称效果与迁移假设无关。此建议并未授权或启动实验，也没有修改原42格参数。

## 7. 验证与复现

只读矩阵、registry、原始源CSV及既有审计摘要；核对CSV哈希、T0公式、最小参考值、phi平方根、原有文件hash。所有新计算在calculation.json，历史逐格暴露在historical_exposure.json。已有算法、配置、预检和历史报告均未更改。独立脚本保存在本目录及maintenance/；没有Job/Cluster对象、求解器调用或训练步骤。
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf8')
    for p,h in protected.items():assert sha(ROOT/p)==h,p
    save('validation.json',dict(scope='OFFLINE_DIAGNOSIS_ONLY',source_profile_hash_verified=True,minimum_reference_and_phi_verified=True,protected_files_unchanged=True,training_runs=0,response_model_changed=False))
    save('input_hashes.json',protected)
    import shutil
    shutil.copyfile(Path(__file__),OUT/Path(__file__).name)
    save('delivery_hashes.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file()})
    print(json.dumps(dict(out=str(OUT),coordinate=[u,v],denominator=d,linear_rho_delta=math.sqrt(d/linear)-1,origin_rho_delta=math.sqrt(d/origin)-1),indent=2))
if __name__=='__main__':main()
