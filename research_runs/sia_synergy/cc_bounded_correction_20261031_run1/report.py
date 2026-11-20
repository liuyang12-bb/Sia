"""Compare independently reconstructed events; never impute incomplete JCT."""
import collections,csv,gzip,hashlib,json,time
from pathlib import Path
O=Path(__file__).resolve().parent;R=Path(r'D:/科研启航/Sia');OLD=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run5'
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 old=json.loads((OLD/'audit/raw_recomputed_metrics.json').read_text());new=json.loads((O/'audit/raw_recomputed_metrics.json').read_text())
 data={f'{version}_{m}':src['dual_medium_'+m] for version,src in [('old',old),('new',new)] for m in ('FO','FULL') if 'dual_medium_'+m in src}
 jobs={}
 for version,folder in [('old',OLD),('new',O)]:
  for row in csv.DictReader((folder/'audit/per_job.csv').open(encoding='utf8')):
   if row['cell'] in ('dual_medium_FO','dual_medium_FULL'):jobs.setdefault(version+'_'+row['cell'].removeprefix('dual_medium_'),{})[row['job']]=row
 comparisons={}
 for a,b in [('new_FO','old_FO'),('new_FULL','old_FULL'),('new_FULL','new_FO'),('old_FULL','old_FO')]:
  if a not in data or b not in data or data[a]['status']!='ALL_COMPLETE' or data[b]['status']!='ALL_COMPLETE':
   comparisons[a+'_vs_'+b]=dict(complete=False);continue
  delta={j:float(jobs[b][j]['jct'])-float(x['jct']) for j,x in jobs[a].items()}
  comps={app:dict(count=sum(x['application']==app for x in jobs[a].values()),saved=sum(v for j,v in delta.items() if jobs[a][j]['application']==app)) for app in ('cifar10','deepspeech2','bert')}
  top=max(delta,key=delta.get);net=sum(delta.values());y=data[a];z=data[b]
  comparisons[a+'_vs_'+b]=dict(complete=True,improvement_percent=100*(1-y['mean_jct']/z['mean_jct']),saved_mean_seconds=z['mean_jct']-y['mean_jct'],
   improved=sum(v>0 for v in delta.values()),worse=sum(v<0 for v in delta.values()),ties=sum(v==0 for v in delta.values()),by_application=comps,
   wait_saved=z['totals']['wait']-y['totals']['wait'],checkpoint_saved=z['totals']['checkpoint']-y['totals']['checkpoint'],training_saved=z['totals']['training']-y['totals']['training'],
   top=top,top_saved=delta[top],net_saved=net,diagnostic_excluding_top_saved_mean=(net-delta[top])/79,per_job_saved=delta)
 recoveries=[]
 for m in ('FO','FULL'):
  path=O/'cells'/('dual_medium_'+m)/'rounds.jsonl.gz'
  if not path.exists():continue
  last=None
  with gzip.open(path,'rt',encoding='utf8') as f:
   for line in f:
    d=json.loads(line)['decision_for_next_interval']
    if d['round']==last:continue
    last=d['round']
    for e in d['result']['audit'].get('tail_recoveries',[]):
     assert e['actual_startup']<=250 and e['cumulative_delay']<=500 and e['eta_after']<e['eta_before']
     recoveries.append(dict(method=m,round=last,**e))
 save('recovery_events.json',recoveries)
 interaction=None
 if len(data)==4 and all(v['status']=='ALL_COMPLETE' for v in data.values()):
  interaction=(data['new_FULL']['mean_jct']-data['new_FO']['mean_jct'])-(data['old_FULL']['mean_jct']-data['old_FO']['mean_jct'])
 save('comparison_summary.json',dict(metrics=data,comparisons=comparisons,recovery_interaction_seconds=interaction,
  interaction_sign='Negative enlarges FULL advantage over FO. Shared recovery gains are not uniquely joint gains.',recoveries=len(recoveries)))
 lines=['# 有限纠偏恢复：80作业FO/FULL共同规则对照','', '**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**','',
 '旧结果保持封存；新FO/FULL使用相同有限恢复规则。当前ready为空、多个已分配作业、目标已画像且有实际训练证据，固定其他当前分配后目标配置仍可行，预计完成时间含真实启动且严格改善。新恢复只开放一个现有比例点；不修改普通辅助优化。与旧单作业尾部恢复共用一次许可，用后固定GPU几何。', '',
 '余量见证证明无需驱逐peer即可落地；最终普通优化仍可调整其他peer，不声称其全程配置完全固定。不得读取未来到达。估计改善不是实际JCT保证。','',
 '|版本/方法|完成|平均JCT|P95|P99|Makespan|等待秒|训练秒|checkpoint秒|GPU-h|CPU-h|RAM-h|restart|',
 '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for k,v in data.items():
  t=v['totals'];vals=[k,f"{v['completed']}/80",v['mean_jct'],v['p95_jct'],v['p99_jct'],v['makespan'],t['wait'],t['training'],t['checkpoint'],t['gpu_hours'],t['cpu_core_hours'],t['ram_gb_hours'],t['restarts']]
  lines.append('| '+' | '.join(f'{x:.3f}' if isinstance(x,float) else str(x) for x in vals)+' |')
 lines+=['','## 配对变化','']
 for k,c in comparisons.items():
  if not c['complete']:lines.append(f'- {k}：未完成配对，不插补JCT。');continue
  lines.append(f"- {k}：均值改善{c['improvement_percent']:.3f}%；{c['improved']}改善/{c['worse']}退化/{c['ties']}持平。等待、checkpoint、训练总节省分别{c['wait_saved']}、{c['checkpoint_saved']}、{c['training_saved']}秒。")
 if 'new_FULL' in data and data['new_FULL']['status']=='ALL_COMPLETE':
  sia=old['dual_medium_SIA_P'];v=data['new_FULL']
  lines+=['',f"相同冻结场景历史SIA_P平均JCT为{sia['mean_jct']}秒；新FULL相对该未改动分支的平均改善{100*(1-v['mean_jct']/sia['mean_jct']):.3f}%。该Sia执行函数、输入和执行响应未改；Sia数据复用历史封存结果。"]
 lines+=['',f'恢复规则×联合选择交互：{interaction}秒/作业；负值为FULL对FO优势扩大。实际恢复事件{len(recoveries)}次，逐事件见recovery_events.json。',
  '', '## 验收与局限','',
  '348项测试（原335+新13）、真实unseeded BERT12轮链路、原始Sia3轮精确回归、新旧FO/FULL各3轮前缀桥接记录见software_acceptance.json。红测两项失败，捕获旧规则无法多作业恢复。初次扩展测试另有锁定peer零进度超过watchdog的测试输入错误，修正到180秒后重跑；失败日志保留。',
  '',f"独立原始事件审计违规总数：{sum(v['capacity_and_execution_violations'] for v in new.values())}。所有资源、锁、rho与互斥时间检查详见audit/。比较表来自原始完成事件和执行片段独立复算。",
  '', '单个80作业受控负载，响应依赖R0/E-legacy跨硬件合成及边界裁剪。不能推出所有负载获益；FO/FULL共享恢复的收益不能全部归为联合优化。求解墙钟不计入仿真JCT。预测ETA误差和少数长尾贡献仍须检查，主结果保留全部作业。',
  '', '## 可复现材料','', 'source_snapshot.zip、source_hashes.json、source_changes.patch、input_hashes.json、preregistration.json、execution_manifest.jsonl；历史源329项逐项核验不变。新树是完整源码副本，没有虚构Git commit。依赖版本固定在预注册environment。']
 if (O/'snapshot_and_matrix_receipt.json').exists():
  lines+=['','预注册从历史基线继承的source_snapshot_path和matrix两个说明字段仍指向旧路径/旧七格描述；未修改运行中的预注册。实际执行以cells中的FO/FULL两格、source_root和逐项源码/输入hash为准。重建使用本目录source_snapshot.zip，其哈希与预注册一致；路径及矩阵澄清见snapshot_and_matrix_receipt.json。']
 (O/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 save('artifact_hashes.json',{p.relative_to(O).as_posix():hashlib.sha256(p.read_bytes()).hexdigest().upper() for p in O.rglob('*') if p.is_file() and p.name not in ('pipeline.log','pipeline_start.json')})
 print('REPORT_COMPLETE',comparisons,flush=True)
if __name__=='__main__':main()
