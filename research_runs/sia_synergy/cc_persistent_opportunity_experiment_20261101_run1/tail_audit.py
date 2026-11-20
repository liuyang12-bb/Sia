"""Read-only baseline semantics audit; no simulation/optimization."""
import gzip,json,sys
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-persistent-v1'
sys.path[:0]=[str(T),str(T/'sia-simulator')]
from joint_prototype.r0_liveness import R0Oracle
from joint_prototype.joint_profile import JobSpec,GpuConfig
rows=[]
for m,t in [('FO',7320),('FULL',7260)]:
 for l in gzip.open(O/'cells'/('long_medium_'+m)/'rounds.jsonl.gz','rt',encoding='utf8'):
  r=json.loads(l)
  if r['interval_end']==t:
   a=r['decision_for_next_interval']['result']['audit'];f=a['shared_eligibility']['bert-159']
   rows.append(dict(method=m,time=t,required=a['required_jobs'],eligible_dgx2=[g for g in f['eligible_geometries'] if 'dgx-ext:1:2' in g],
    tail_grants=a['tail_grants'],tail_recoveries=a['tail_recoveries'],selected=r['decision_for_next_interval']['result']['selected']['bert-159']))
   break
q=R0Oracle(T/'inputs/registry.json',T/'inputs/models',{'bert-159':'bert'},'medium');ref=q.reference('bert-159','dgx-ext:1:2')
c=GpuConfig('dgx-ext:1:2','dgx-ext',2,1,1.,ref['phi_c'],ref['phi_m']);j=JobSpec('bert-159','bert_1.py',(c,))
ratios={str(ram):q.query(j,c,1.5,ram,(1.5,93.75)).rho for ram in (20.83,93.75)}
with (O/'tail_semantics_audit.json').open('x',encoding='utf8') as f:json.dump(dict(states=rows,rho=ratios,
 finding='sole tail exception is non-proportional RAM20.83; FO proportional GPU proposal discards it while equivalent RAM93.75 is absent; FULL uses it',
 scope='cross-policy baseline semantics risk; no physical violation; no counterfactual corrected JCT computed'),f,ensure_ascii=False,indent=2)
with (O/'REPORT.md').open('a',encoding='utf8') as f:
 f.write('''\n## 重要：长尾恢复与FO串行候选过滤存在语义交互\n
FO在7320秒只剩bert-159，已用240秒延迟预算；共享资格过滤额外放行的DGX2只有CPU1.5/RAM20.83这一列。FULL在7260秒也获同一形状的单列例外，成功迁到DGX2（实际120秒切换，并增加110秒恢复预算），最终提前完成。

但FO第一阶段只接受比例配额，DGX比例点为CPU1.5/RAM93.75。低RAM例外被过滤，比例形状又没有被例外放行，FO保持AWS2。离线同Oracle查询确认两个RAM点rho均为1，在相同GPU/batch下速度相同且单任务时均满足容量。

代码证据：clean_ablation.py:82的比例候选过滤，与同文件shared_eligible约340–355行按最小ETA/column_id仅加入一个tail例外的组合。虽然函数与预算规则共享，最终可达恢复域并非对两个策略等效。它可能是设计语义交互或对照构造缺口，不能仅凭“同一恢复代码”声称公平恢复能力完全一致。

所以14.28%是当前冻结实现的实际配对观测，但其净收益92.74%来自bert-159，不能作为纯辅助资源联合选择的干净增量。去掉该任务的诊断性均值仅改善43.18秒，约1.42%；该敏感性结果不替代全12任务主表。接收任务cifar10-2本身2179→2065秒，改善5.23%，且240–1200秒其他两个CIFAR进度不受损，这部分机制证据成立。

本轮未修改恢复语义、未补候选、未反事实重跑，不估计修正FO后的JCT。下一阶段若要做论文级FO/FULL比较，应首先使共享恢复例外对合法等速配置形状闭合，或预先定义GPU配置级恢复资格，再分别选择辅助配额；保留锁、真实代价和有限预算。该建议未实施。\n''')
print('Tail baseline semantics risk recorded')
