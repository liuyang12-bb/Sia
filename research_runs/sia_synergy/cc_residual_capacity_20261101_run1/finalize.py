import json,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent
stats=json.loads((O/'raw_audit.json').read_text());comp=json.loads((O/'comparison_summary.json').read_text())
metrics={}
for a,b in [('OLD_FO','NEW_FO'),('OLD_FULL','NEW_FULL'),('NEW_FO','NEW_FULL')]:
 metrics[a+'__'+b]={k:100*(1-stats[b][k]/stats[a][k]) for k in ('mean','median','p95','p99','makespan')}
interaction=comp['OLD_FULL__NEW_FULL']['mean_saved']-comp['OLD_FO__NEW_FO']['mean_saved']
with (O/'tail_and_interaction.json').open('x',encoding='utf8') as f:json.dump(dict(improvement_percent=metrics,extra_full_mean_saved_seconds=interaction),f,indent=2)
with (O/'REPORT.md').open('a',encoding='utf8') as f:
 f.write('\n## 最终判断与直接兑现证据\n\n')
 f.write('四格均12/12，完整393项测试通过347.793秒，包含旧385项、4项新增测试及4项重复复用的旧断点测试；不能把393−385都当独立新测试。四格流水线总墙钟224.341秒（不含测试与开发）。\n\n')
 f.write('统一补点有效，但幅度有限：FO平均JCT改善0.687%，FULL改善1.033%。FULL对FO优势从8.155%变为8.475%；FULL相比FO额外获得的补点均值收益仅9.75秒/作业，不能把两者共享的改善全归于GPU联合选择。\n\n')
 f.write('新FULL相对旧FULL，9个作业改善、3个持平、无退化；P95/P99/makespan均改善，完整百分比见tail_and_interaction.json。新FO有8改善/3退化/1持平，median反而恶化，收益不能描述为所有指标均改善。新FULL相对新FO的P99/makespan仍较差，历史长尾权衡没有彻底解决。\n\n')
 f.write('transmission.json核对首个实际决策分叉：FO/FULL均在240秒，相同真实决策前进度/状态，CIFAR0的GPU、RAM保持，CPU由4.399394升为5；下一训练段直接记录的raw完全相同，而effective_rate比值精确等于rho比值1.07343977。说明这次CPU补点确实作用于实际训练，不只是目标评分。之后全轨迹差异不能继续简单按7.34%线性推算JCT。\n\n')
 f.write('FULL总JCT节省424秒中，训练时间减少364秒、等待减少60秒、checkpoint不变；重启仍16次。最大单作业贡献65秒，占15.33%，收益没有集中于单个长尾作业。本轮不单独识别RAM补点的贡献，不宣称内存优化获得独立新收益。\n\n')
 f.write('建议保留新候选版作为小规模已验证分支，尚不因1.03%的单负载增量扩大实验或叠加预测。原FULL与全部历史保持；当前结论限于冻结合成模型。\n')
with (O/'CURRENT_TASK.md').open('a',encoding='utf8') as f:f.write('\n完成并停止：393测试通过；同状态及Sia/FO/FULL回归通过；四格12/12，raw审计0；新FO−0.687%、新FULL−1.033%平均JCT。源码342文件冻结，旧340文件及116旧产物未变。完整结果REPORT.md。未80作业、未调参。\n')
# Check Markdown tables in final report.
cols=None
for i,l in enumerate((O/'REPORT.md').read_text(encoding='utf8').splitlines(),1):
 if l.startswith('|'):
  c=l.count('|')-1
  if cols is None:cols=c
  assert c==cols,(i,c,cols)
 else:cols=None
print(json.dumps(dict(tails=metrics,extra_full_saved=interaction,complete=True),indent=2))
