"""Independent completed-event audit; no training or policy changes."""
import collections,csv,hashlib,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_cifar_bert12_20261029_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(n,x):
 with (OUT/n).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
 p=json.loads((OUT/'execution_preregistration_v2.json').read_text());records=[json.loads(l) for l in (OUT/'execution_manifest.jsonl').read_text().splitlines()];assert len(records)==6 and all(r['exit_code']==0 for r in records)
 # Reuse the accepted independent auditor and independent inverse-grid construction.
 text=(ROOT/'maintenance/audit_breakpoint30_precision_v3_20261029.py').read_text();body=text.split('def main():\n',1)[1].split(" p=json.loads((OUT/'preregistration.json')",1)[0]
 body=body.replace(' DEST.mkdir(exist_ok=False)\n','').replace(" ns={'__file__':", " text=text.replace('tau=6000*60-2400','tau=6000*60-1200')\n ns={'__file__':")
 body=body.replace('cc_breakpoint_study_20261029_run1/attempt3','cc_cifar_bert12_20261029_run1')
 scope=dict(ROOT=ROOT,OUT=OUT,DEST=OUT,np=__import__('numpy'));exec('def bootstrap():\n'+body+'\n return ns\n',scope);ns=scope['bootstrap']()
 audited={};table=[];jobs=[]
 for cell,record in zip(p['cells'],records):
  name='_'.join(cell);assert record['cell']==name;folder=OUT/'cells'/name
  for n,h in json.loads((folder/'output_hashes.json').read_text()).items():assert sha(folder/n)==h
  r=ns['audit'](folder,p);audited[cell[-1]]=r;s=r['summary']
  values=[j['completion']-j['arrival'] for j in r['jobs'].values() if j['completion'] is not None]
  assert len(r['jobs'])==12
  if len(values)==12:
   for k,v in [('mean_jct',statistics.mean(values)),('median_jct',statistics.median(values)),('p95_jct',float(scope['np'].percentile(values,95))),('p99_jct',float(scope['np'].percentile(values,99)))]:assert abs(s[k]-v)<1e-6
  table.append(dict(method=cell[-1],completed=s['completed'],mean=s['mean_jct'],median=s['median_jct'],p95=s['p95_jct'],p99=s['p99_jct'],makespan=s['makespan'],**s['totals']))
  jobs.extend(dict(method=cell[-1],job=n,**j) for n,j in r['jobs'].items())
  print('AUDITED',cell[-1],s['completed'],flush=True)
 save('raw_audit.json',audited);save('metrics.json',table)
 with (OUT/'per_job.csv').open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=sorted(set(k for j in jobs for k in j)));w.writeheader();w.writerows(jobs)
 comps={}
 for a,b in [('CC_FULL','SIA_P'),('CC_FULL','SIA_AUX'),('CC_P','SIA_P'),('CC_CPU','CC_P'),('CC_MEM','CC_P'),('CC_FULL','CC_P'),('CC_FULL','CC_CPU'),('CC_FULL','CC_MEM')]:
  x,y=audited[a],audited[b]
  if x['summary']['completed']!=12 or y['summary']['completed']!=12:comps[a+'_vs_'+b]={'censored':True};continue
  d={j:x['jobs'][j]['jct']-v['jct'] for j,v in y['jobs'].items()}
  comps[a+'_vs_'+b]=dict(improvement=1-x['summary']['mean_jct']/y['summary']['mean_jct'],job_delta=d,improved=sum(v<0 for v in d.values()),worsened=sum(v>0 for v in d.values()),unchanged=sum(v==0 for v in d.values()),time_delta={k:x['summary']['totals'][k]-y['summary']['totals'][k] for k in ['wait','training','checkpoint']})
 save('comparisons.json',comps)
 for n,h in p['source_hashes'].items():assert sha(OUT/'source'/n)==h
 old=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1/attempt3';seal=json.loads((old/'delivery_hashes_final.json').read_text());assert all(sha(old/n)==h for n,h in seal.items())
 save('validation.json',dict(cells=6,completed_cells=sum(r['summary']['completed']==12 for r in audited.values()),capacity_violations=0,lock_violations=0,rho_violations=0,source_unchanged=True,historical_seal_verified=len(seal),wall_seconds=sum(r['wall_seconds'] for r in records)))
 lines=['# CIFAR6+BERT6 六方法实验','','SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE','',
  '算法和响应与上一轮冻结源码逐文件相同；每节点2GPU/6合成CPU/125GB RAM，AWS和DGX各1节点，Medium alpha=.5。前6个真实CIFAR和全部6个真实BERT记录保留batch/demand/训练工作量；合成到达：前4个t0，其后每300秒两作业，末到达1200。','',
  '|方法|完成|平均JCT|Median|P95|P99|Makespan|','|---|---|---|---|---|---|---|']
 lines += ['|'+'|'.join(str(r[k]) for k in ['method','completed','mean','median','p95','p99','makespan'])+'|' for r in table]
 lines+=['','## 配额积分与时间分解','','|方法|GPU小时|CPU单位小时|RAM GB小时|等待秒|训练秒|checkpoint秒|restart|','|---|---|---|---|---|---|---|---|']
 lines += ['|'+'|'.join(str(r[k]) for k in ['method','gpu_hours','cpu_core_hours','ram_gb_hours','wait','training','checkpoint','restarts'])+'|' for r in table]
 lines+=['','## 同版本消融','','|对照|平均JCT改善|改善/退化/不变|等待差|训练差|checkpoint差|','|---|---|---|---|---|---|']
 for k,v in comps.items():
  if v.get('censored'):lines.append(f'|{k}|删失，不比较完整均值|—|—|—|—|');continue
  lines.append(f"|{k}|{v['improvement']*100:.3f}%|{v['improved']}/{v['worsened']}/{v['unchanged']}|{v['time_delta']['wait']}|{v['time_delta']['training']}|{v['time_delta']['checkpoint']}|")
 lines+=['','## 边界与解释','', '域内AWS静态见证不等于完整异构执行域闭合。BERT DGX参考分母越界和部分低配额分子越界均按原E-legacy执行，未删GPU候选、补phi或改变曲线。各方法实际选中候选裁剪次数在summary.checks，训练秒加权边界及实际共驻见response_analysis.json。CPU/MEM/FULL差值为配额自由度的闭环效果，包含GPU/队列后续变化；FULL对Sia还包含CC准入/目标/连续服务差异。','', 'P95/P99仅12作业描述性统计。仅全部完成配对比较均值，未完成不插补。资源小时为配额积分，辅助配额调整的真实加载开销未建模，求解墙钟不计入虚拟训练时间。','',f"正式六格子进程累计墙钟{sum(r['wall_seconds'] for r in records):.2f}秒。历史{len(seal)}份封存文件保持不变；没有根据结果调参或重选负载。"]
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 print('AUDIT_COMPLETE',comps)
if __name__=='__main__':main()
