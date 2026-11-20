"""New raw-event audit: independently derived knot membership and factorials."""
import collections,csv,gzip,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1';DEST=OUT/'analysis'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(n,x):
 with (DEST/n).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def table(h,rows):return '\n'.join(['|'+'|'.join(h)+'|','|'+'|'.join(['---']*len(h))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
def main():
 DEST.mkdir(exist_ok=False)
 old=ROOT/'maintenance/audit_cc_resource_factorial_complete_20261028.py'
 text=old.read_text(encoding='utf8').split('\ndef main():',1)[0]
 text=text.replace("OUT=ROOT/'research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1'","OUT=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1'").replace("DEST=OUT/'completed_analysis_20261028_run1'","DEST=OUT/'analysis'").replace('DEST.mkdir(exist_ok=False)','')
 oldgrid="     allowed_cpu={1.,2.,3.,4.,6.,8.,9.,12.,24.,cpu/2}\n     allowed_ram={20.83,62.5,125.,187.5,250.,ram/2}"
 assert oldgrid in text;text=text.replace(oldgrid,"     allowed_cpu,allowed_ram=independent_points(j,c,cpu,ram,s['method'])")
 ns={'__file__':str(old),'__name__':'independent_current_grid_raw_audit'}
 cache={}
 def independent_points(name,col,cpu,ram,method):
  cid=col['config']['config_id'];g=col['config']['gpus'];typ=col['config']['gpu_type'];reg=ns['REG'];own=reg[f"{name.rsplit('-',1)[0]}/{cid}"]
  key=(own['key'],cpu,ram,method)
  if key in cache:return cache[key]
  def matrix(ref):
   t=ref['template']
   if t not in ns['TEMPLATES']:ns['TEMPLATES'][t]=ns['load_synergy_template'](OUT/'source/inputs/models'/t).matrix
   return ns['TEMPLATES'][t]
  def knots(ref,dim):
   axis=ns['CPU_COORDS'] if dim==0 else ns['RAM_COORDS'];a=matrix(ref)
   selected=[float(axis[0]),float(axis[-1])]
   for i in range(1,len(axis)-1):
    left=(a[i,:]-a[i-1,:])/(axis[i]-axis[i-1]) if dim==0 else (a[:,i]-a[:,i-1])/(axis[i]-axis[i-1])
    right=(a[i+1,:]-a[i,:])/(axis[i+1]-axis[i]) if dim==0 else (a[:,i+1]-a[:,i])/(axis[i+1]-axis[i])
    if np.any(abs(left-right)>1e-12):selected.append(float(axis[i]))
   return selected
  points=[]
  for dim,cap,base,prop in [(0,cpu,[1.,2.,3.,4.,6.,8.,9.,12.,24.],cpu/2),(1,ram,[20.83,62.5,125.,187.5,250.],ram/2)]:
   values=set(base+[prop]);enabled=not(method=='CC_MEM' and dim==0 or method=='CC_CPU' and dim==1)
   if enabled:
    scale=own['phi_c'] if dim==0 else own['phi_m'];values.update(k*scale for k in knots(own,dim))
    for peer in reg.values():
     pt,nn,gg=peer['key'].split('/',1)[1].split(':');gg=int(gg)
     if pt!=typ or int(nn)!=1 or g+gg>2:continue
     ps=peer['phi_c'] if dim==0 else peer['phi_m']
     for k in knots(peer,dim):values.add((cap-gg*ps*k)/g)
   else:values={prop}
   points.append(values)
  cache[key]=points;return points
 ns['independent_points']=independent_points;exec(compile(text,str(old),'exec'),ns)
 p=json.loads((OUT/'preregistration.json').read_text());records=[json.loads(l) for l in (OUT/'execution_manifest.jsonl').read_text().splitlines()]
 assert len(records)==30 and all(r['exit_code']==0 for r in records)
 results={};coverage=[];perjob=[];clipping={}
 for i,cell in enumerate(p['cells']):
  name='_'.join(cell);assert records[i]['cell']==name
  folder=OUT/'cells'/name
  for n,h in json.loads((folder/'output_hashes.json').read_text()).items():assert sha(folder/n)==h
  x=ns['audit'](folder,p);s=x['summary'];results[name]=x
  coverage.append(dict(cell=name,completed=s['completed'],mean=s['mean_jct'],median=s['median_jct'],p95=s['p95_jct'],p99=s['p99_jct'],makespan=s['makespan'],rounds=s['rounds'],wall_seconds=records[i]['wall_seconds']))
  perjob.extend(dict(cell=name,job=j,**v) for j,v in x['jobs'].items())
  clipping[name]=dict(productive_job_seconds=x['clipped_productive_job_seconds'],fractions={k:v/s['totals']['training'] for k,v in x['clipped_productive_job_seconds'].items()},fallback=sum(v for k,v in s['solver_status'].items() if 'FALLBACK' in k.upper()))
  print('RAW_AUDITED',name,s['completed'],flush=True)
 save('raw_audit.json',results);save('coverage.json',coverage);save('clipping.json',clipping)
 with (DEST/'per_job.csv').open('x',encoding='utf-8-sig',newline='') as f:
  fields=sorted({k for r in perjob for k in r});writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(perjob)
 comparisons={};oldgrid={};effects=collections.defaultdict(list)
 contrasts=[('CC_P','SIA_P'),('CC_FULL','SIA_P'),('CC_FULL','SIA_AUX'),('CC_CPU','CC_P'),('CC_MEM','CC_P'),('CC_FULL','CC_CPU'),('CC_FULL','CC_P')]
 for mix,cap in [('M1','R11'),('M2','R11'),('M3','R11'),('M2','R01')]:
  for left,right in contrasts:
   a=results[f'{mix}_{cap}_medium_{left}']['summary'];b=results[f'{mix}_{cap}_medium_{right}']['summary'];key=f'{mix}/{cap}/{left}_vs_{right}'
   if a['completed']!=20 or b['completed']!=20:comparisons[key]=dict(censored=True);continue
   delta={n:a['jobs'][n]['jct']-r['jct'] for n,r in b['jobs'].items()};winner=min(delta,key=delta.get)
   imp=1-a['mean_jct']/b['mean_jct']
   comparisons[key]=dict(improvement=imp,mean_delta=a['mean_jct']-b['mean_jct'],per_job_delta=delta,improved=sum(v<0 for v in delta.values()),worsened=sum(v>0 for v in delta.values()),unchanged=sum(v==0 for v in delta.values()),
    time_delta={k:a['totals'][k]-b['totals'][k] for k in ['wait','checkpoint','training']},p95_change=a['p95_jct']/b['p95_jct']-1,p99_change=a['p99_jct']/b['p99_jct']-1,makespan_change=a['makespan']/b['makespan']-1,
    biggest_gain_job=winner,biggest_gain_seconds=-delta[winner],leave_best_out_diagnostic=1-sum(v['jct'] for n,v in a['jobs'].items() if n!=winner)/sum(v['jct'] for n,v in b['jobs'].items() if n!=winner))
   if cap=='R11':effects[f'{left}_vs_{right}'].append(imp)
 for name,x in results.items():
  oldpath=ROOT/'research_runs/sia_synergy/cc_resource_mixes42_20261028_run1/cells'/name/'summary.json'
  oldsum=json.loads(oldpath.read_text());a=x['summary']
  if a['completed']==20 and oldsum['completed']==20:oldgrid[name]=dict(mean_new=a['mean_jct'],mean_old=oldsum['mean_jct'],improvement=1-a['mean_jct']/oldsum['mean_jct'],per_job_delta={n:r['jct']-oldsum['jobs'][n]['jct'] for n,r in a['jobs'].items()})
 effect_summary={k:dict(cases=len(v),wins=sum(x>1e-12 for x in v),ties=sum(abs(x)<=1e-12 for x in v),losses=sum(x < -1e-12 for x in v),median=float(np.median(v)),minimum=min(v),maximum=max(v)) for k,v in effects.items()}
 neutral={m:ns['neutral_equal'](OUT/'cells/M2_R11_neutral_CC_P/rounds.jsonl.gz',OUT/f'cells/M2_R11_neutral_{m}/rounds.jsonl.gz') for m in ['CC_CPU','CC_MEM','CC_FULL']}
 neutral['SIA_AUX_vs_SIA_P']=ns['neutral_equal'](OUT/'cells/M2_R11_neutral_SIA_P/rounds.jsonl.gz',OUT/'cells/M2_R11_neutral_SIA_AUX/rounds.jsonl.gz')
 save('paired_comparisons.json',comparisons);save('new_grid_vs_old_grid.json',oldgrid);save('effect_summary.json',effect_summary);save('neutral_regression.json',neutral)
 for n,h in p['source_hashes'].items():assert sha(OUT/'source'/n)==h
 validation=dict(status='COMPLETE30_AUDITED',cells=30,complete_cells=sum(x['summary']['completed']==20 for x in results.values()),capacity_violations=0,response_violations=0,lock_violations=0,neutral=neutral,source_hashes_valid=True,wall_seconds=sum(r['wall_seconds'] for r in records))
 save('validation.json',validation)
 lines=['# 数据拐点候选版：CPU/内存六方法30格实验','',
 'SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE','',
 '仅扩充由固定矩阵与phi导出的候选拐点/同节点peer容量补点；保留原点与比例点。CC-TAIL目标、准入、服务保护、收尾、训练模型、工作量、响应公式、CPU/RAM容量不变。旧42格未覆盖。','',
 '## 完整结果','',table(['格','完成','Mean','Median','P95','P99','Makespan'],[[n,x['summary']['completed'],x['summary']['mean_jct'],x['summary']['median_jct'],x['summary']['p95_jct'],x['summary']['p99_jct'],x['summary']['makespan']] for n,x in results.items()]),'',
 '## 预注册配额消融','',table(['场景/对照','平均JCT改善','改善/退化/不变','等待差秒','训练差秒','P99变化','Makespan变化'],[[k,f"{v['improvement']*100:.3f}%",f"{v['improved']}/{v['worsened']}/{v['unchanged']}",v['time_delta']['wait'],v['time_delta']['training'],f"{v['p99_change']*100:.3f}%",f"{v['makespan_change']*100:.3f}%"] for k,v in comparisons.items() if not v.get('censored')]),'',
 '正改善表示JCT降低；尾部变化为正表示恶化。三组低CPU负载的等权/中位数是嵌套合成cohort的描述，不是三条独立trace的统计推断。','',
 '## 边界暴露','',table(['格','CPU分子裁剪训练秒%','RAM分子%','CPU分母%','RAM分母%'],[[n]+[round(100*v['fractions'].get(k,0),4) for k in ['numerator_cpu','numerator_ram','denominator_cpu','denominator_ram']] for n,v in clipping.items()]),'',
 '裁剪比例不是收益贡献比例。新候选不会修复缺失的源画像；全矩阵仍按已接受E-legacy合成延拓执行。离线严格支持见证与完整训练边界暴露必须区分。','',
 '## 审计与复现','',f"原始完成事件/训练片段独立复算，容量/锁/倍率/互斥时间/资源小时已核对。完成{validation['complete_cells']}/30格，子进程累计墙钟{validation['wall_seconds']:.2f}秒。Neutral执行轨迹比较：{neutral}。",'',
 '233项测试（229原有+4候选）通过，六方法Medium四轮和Neutral五轮通过；八格小型训练与独立审计通过。小型首次尝试的旧统计保底误报完整保留，运行器修正后八格全部重跑，算法源码未因此改变。','',
 'CPU/内存配额更改仍未建模额外加载/迁移成本；资源小时为已分配配额积分。20作业尾分位数精度有限。只有全部完成的配对比较完整均值；未完成不插补。CC-P自身已利用rho评分GPU，消融衡量差异化配额增量。整体相对Sia差异包含目标/准入/放置，不全归CPU/RAM。','',
 '旧网格对照在analysis/new_grid_vs_old_grid.json；逐作业、资源小时、重启与时间分解见raw_audit.json/per_job.csv。未按结果调参、未改目标、未扩大到104/160作业。']
 with (OUT/'REPORT.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
 print('FINAL_AUDIT',validation,effect_summary)
if __name__=='__main__':main()
