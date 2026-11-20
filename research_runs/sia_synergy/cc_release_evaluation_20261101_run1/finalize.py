import hashlib,json,math
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-release-evaluation-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def main():
 p=json.loads((O/'preregistration.json').read_text());a=json.loads((O/'raw_audit.json').read_text());v=json.loads((O/'validation.json').read_text())
 assert v['errors']==0 and v['all_completed'] and v['old_source_unchanged']
 assert all(sha(T/n)==h for n,h in p['source_hashes'].items())
 assert all(sha(n)==h for n,h in p['input_hashes'].items())
 report=(O/'REPORT.md').read_text(encoding='utf8');assert '## 最终判断' not in report
 verdict='''
## 最终判断

**评价方法的最小版本已实现、四格均12/12完成，但没有获得新一轮平均JCT改善，因此不扩大实验、不替换旧FULL。** 新FO与旧FO全部逐作业指标相同；新FULL均值3537.083秒，相对旧FULL3420.917秒恶化3.396%，相对新FO3724.667秒仍改善5.036%，低于旧FULL对旧FO的8.155%。这5.036%不是本次新增评价的收益。

新FULL相对旧FULL：Median不变；P95增加2.202%，P99增加3.715%；Makespan7233→7179秒，缩短0.747%。总等待完全相同16920秒，训练增加1364秒、checkpoint增加30秒，恰好对应总JCT增加1394秒（均值增加116.167秒）。不能用makespan小幅改善掩盖平均与P99退化。

四格串行仿真实测224.52秒（约3.74分钟）；不含代码开发、同状态分析及测试。旧353项回归均通过，新增14项测试通过：完整批次363项（含10项新增）344.309秒，最终新增14项单独复验通过，总计367个不同测试。原始Sia3轮精确一致；原始完成事件/互斥时间/资源小时独立复算错误0，容量、完整锁、rho单次施加违规0。源码335文件快照SHA256见source_receipt.json，旧332文件核验未变。

'''
 marker='## 机制与正确性'
 report=report.replace(marker,verdict+marker,1)
 analysis='''
## 为什么预测变好，实际平均JCT反而变差

**不是这次再次漏掉donor：新评价显式计入了全部donor损失。关键不足是最小接续预测只让资源释放后启动等待者，却让已运行的作业保持原配置直到预测结束，没有预测它们在资源释放后扩容或迁移。** 这既是本轮原先明确的保守假设，也是结果暴露的主要失效边界。完成后资源释放只做到了“释放→等待者接续”，尚未完整做到“释放→存活作业重新联合配置”；不能宣称用户希望的整个机制已经解决。

本轮新FO116轮中84轮因未知模型退回旧bootstrap、32轮完成新评价；新FULL119轮中79轮退回、40轮完成评价。**有等待者时成功完成新评价的轮数均为0**，预测等待者收益均为0。等待队列里的任务尚无已观测梯度/执行率，所以不伪造ETA；等它们都开始训练，队列已经空了。这使实际启用部分退化为静态ETA和比较，资源释放的排队收益仅有手算单元测试，尚无完整训练兑现证据。

新FULL只在4800、5820秒两轮改选，其他9个作业JCT完全不变。两次改选使用的当前raw和rho没有重复计乘，但对旧方案未来一直停留在慢GPU的假设过于悲观：

|4800秒第一处分叉|原FULL当前方案的预测剩余时间|新方案预测剩余时间|原FULL实际剩余时间|新FULL实际剩余时间|
|---|---:|---:|---:|---:|
|bert-159|15586.521|1836.912|2433|1677|
|cifar10-5|715.687|3866.076|559|1998|

评分把bert-159提前完成收益估为13749.610秒，计入cifar10-5推迟3150.389秒后，认为整组净收益10599.221秒。实际bert-159只提前756秒，而cifar10-5晚1439秒；bert-107还晚711秒，整组净损失1394秒。

为什么旧bert-159实际上没那么慢：旧FULL并没有一直让它使用AWS1。它在5400秒切到AWS2，在6480秒切到DGX2，实际于7233秒完成。原CIFAR于5359秒完成，释放DGX资源；旧bert-107也随后在5400秒迁入DGX1、5820秒扩到DGX2。新评分的冻结存活配置接续没有展开这些后续选择。新方案提前把DGX1给bert-159，让CIFAR从DGX1搬回AWS1；CIFAR后来又换AWS2、DGX1，延后释放，并影响另一BERT。5820秒第二次静态ETA改选再次让CIFAR让出快GPU。该链路由prediction_realization.json中的GPU轨迹、真实训练片段和完成事件支持，不是仅由平均值猜测。

|受影响作业|旧FULL JCT|新FULL JCT|变化（新－旧）|等待变化|训练变化|checkpoint变化|
|---|---:|---:|---:|---:|---:|---:|
|bert-159|6033|5277|-756|0|-636|-120|
|cifar10-5|4159|5598|+1439|0|+1289|+150|
|bert-107|5568|6279|+711|0|+711|0|

这不是证明当前速率模型完全错误，更不能说加入CPU/内存感知无效：接收方新方案预测1836.9秒与实际1677秒接近，主要高估来自对对照路径未来资源利用的过度静态化。但epoch/模型更新和进一步决策仍存在影响，本轮没有把它们全部因果隔离。

## 是否扩大与最小后续方向

**不扩大。旧FULL仍是当前已验证较好的版本，新评价保存为独立研究版本。** 下一项若获授权，应只补“完成事件之后，对存活任务也进行一次合法的重新配置预测”，统一计入实际startup、checkpoint锁、剩余生命周期预算及donor净代价；先在本轮4800/5820秒两个已冻结状态，检查能否重现旧方案“短作业完成释放快GPU，其他任务再扩容”的选择。不能给新方案预测更强的接续、却让对照一直留在慢GPU。

另一项独立边界是未知等待者：应先论证可审计的不确定性处理，不能新增seed、用真实未来速度或假ETA强行启用本评分。没有解决这两点前，继续增加作业数或扩大训练矩阵不会证明本机制有效。本轮没有自动实施这一后续设计，也没有调权、改网格、改响应或换负载追求获胜。
'''
 report+=analysis
 (O/'REPORT.md').write_text(report,encoding='utf8')
 with (O/'CURRENT_TASK.md').open('a',encoding='utf8') as f:f.write('\n完成并停止：4/4均12/12，独立审计0错误；新FO持平，新FULL较旧FULL平均JCT恶化3.396%。不扩大。缺口为未来释放后存活作业重配置及冷启动未知，详见REPORT.md。无后台实验。\n')
 save('final_validation.json',dict(status='IMPLEMENTATION_VALID_PERFORMANCE_GATE_FAILED',tests_unique=367,historical_tests=353,new_tests=14,complete_cells=4,all_jobs_completed=True,raw_audit_errors=0,source_unchanged_after_freeze=True,old_source_protected=True,expanded=False,formal_wall_seconds=json.loads((O/'pipeline_complete.json').read_text())['wall_seconds']))
 save('artifact_hashes.json',{x.relative_to(O).as_posix():sha(x) for x in O.rglob('*') if x.is_file() and '__pycache__' not in x.parts and x.name!='finalize.log'})
 print('FINAL_PERFORMANCE_GATE_FAILED_NO_EXPANSION',flush=True)
if __name__=='__main__':main()
