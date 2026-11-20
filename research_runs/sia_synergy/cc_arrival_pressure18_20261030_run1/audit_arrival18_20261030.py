import collections,csv,gzip,hashlib,json,shutil,statistics,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_arrival_pressure18_20261030_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(n,v):
 with (OUT/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def table(h,rows):return '\n'.join(['|'+'|'.join(h)+'|','|'+'|'.join(['---']*len(h))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
def bootstrap():
 text=(ROOT/'maintenance/audit_breakpoint30_precision_v3_20261029.py').read_text();body=text.split('def main():\n',1)[1].split(" p=json.loads((OUT/'preregistration.json')",1)[0]
 body=body.replace(' DEST.mkdir(exist_ok=False)\n','').replace('cc_breakpoint_study_20261029_run1/attempt3','cc_arrival_pressure18_20261030_run1')
 body=body.replace(" ns={'__file__':", " text=text.replace('tau=6000*60-2400','tau=6000*60-3300').replace('if len(vals)==20:', 'if len(vals)==12:').replace(\"if e['checkpoint_delay_consumed']>=e['observed_active_seconds']>0:\",\"if e['checkpoint_delay_consumed']>=max(0.,e['interval_end']-max(e['interval_start'],e['arrival_time']))>0:\")\n ns={'__file__':")
 scope=dict(ROOT=ROOT,OUT=OUT,DEST=OUT,np=np);exec('def b():\n'+body+'\n return ns\n',scope);return scope['b']()
def main():
 phase=sys.argv[1];p=json.loads((OUT/'preregistration.json').read_text());records=[json.loads(l) for l in (OUT/'execution_manifest.jsonl').read_text().splitlines()];expected=6 if phase=='burst' else 18
 assert len(records)==expected and all(r['exit_code']==0 for r in records)
 ns=bootstrap()
 from applications import APPLICATIONS
 from joint_prototype.execution_audit import audit_execution
 results={};metrics=[];jobs=[];startup_checks=collections.Counter();signatures={};pressure={}
 for cell,rec in zip(p['cells'][:expected],records):
  name='_'.join(cell);assert name==rec['cell'];folder=OUT/'cells'/name
  for n,h in json.loads((folder/'output_hashes.json').read_text()).items():assert sha(folder/n)==h
  r=ns['audit'](folder,p);results[name]=r;s=r['summary'];digest=hashlib.sha256();load=collections.Counter()
  with gzip.open(folder/'rounds.jsonl.gz','rt',encoding='utf8') as f:
   for line in f:
    event=json.loads(line);d=event['decision_for_next_interval'];selected=d['result']['selected'];specs={v['job_id']:v for v in d['probe']['specs']};a=d['result']['audit']
    assert {j for j,v in specs.items() if v['checkpoint_locked']}<=set(selected)
    for e in event['execution']:assert audit_execution(e)['valid']
    load['peak_observed_active_jobs']=max(load['peak_observed_active_jobs'],sum(e['observed_active_seconds']>0 for e in event['execution']))
    load['peak_waiting_jobs']=max(load['peak_waiting_jobs'],sum(e['waiting_seconds']>0 for e in event['execution']))
    load['waiting_job_rounds']+=sum(e['waiting_seconds']>0 for e in event['execution'])
    if 'secondary_tie' in a and not a.get('tail_recoveries'):
     actual=0.
     for j,col in selected.items():
      v=specs[j];same=v['current_config_id']==col['config']['config_id'] and dict(v['current_placement'])==col['placement']
      if not same:actual+=float(APPLICATIONS[col['config']['gpu_type']][j.rsplit('-',1)[0]].rescale_time or 30.)
     assert abs(actual-a['secondary_tie']['minimum_actual_delay'])<1e-8;startup_checks[name]+=1
    signature=[(e['job_id'],e['progress_after'],e['completion_time'],e['rho'],e['cluster'],e['placement']) for e in event['execution']]
    digest.update(json.dumps(signature,sort_keys=True).encode())
  signatures[name]=digest.hexdigest();pressure[name]=dict(load);values=[v['jct'] for v in r['jobs'].values() if v.get('jct') is not None]
  if len(values)==12:assert abs(statistics.mean(values)-s['mean_jct'])<1e-8
  metrics.append(dict(cohort=cell[0],method=cell[-1],completed=s['completed'],mean=s['mean_jct'],median=s['median_jct'],p95=s['p95_jct'],p99=s['p99_jct'],makespan=s['makespan'],**s['totals']))
  jobs.extend(dict(cohort=cell[0],method=cell[-1],job=j,**v) for j,v in r['jobs'].items());print('RAW_AUDITED',name,s['completed'],flush=True)
 validation=dict(phase=phase,cells=expected,complete_cells=sum(v['summary']['completed']==12 for v in results.values()),capacity_violations=0,lock_violations=0,rho_violations=0,secondary_cost_checks=dict(startup_checks))
 if phase=='burst':
  save('burst_raw_audit.json',results);save('burst_validation.json',validation)
  assert validation['complete_cells']==6,'BURST_INCOMPLETE; no expansion';print('BURST_CORRECTNESS_GATE_PASSED',flush=True);return
 save('raw_audit.json',results);save('metrics.json',metrics);save('trajectory_signatures.json',signatures)
 with (OUT/'per_job.csv').open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=sorted({k for r in jobs for k in r}));w.writeheader();w.writerows(jobs)
 comparisons={};contrasts=[('CC_FULL','SIA_P'),('CC_FULL','SIA_AUX'),('CC_P','SIA_P'),('CC_CPU','CC_P'),('CC_MEM','CC_P'),('CC_FULL','CC_P'),('CC_FULL','CC_CPU'),('CC_FULL','CC_MEM')]
 for cohort in ['BURST','COMPACT','SPREAD']:
  comparisons[cohort]={}
  for a,b in contrasts:
   x,y=results[f'{cohort}_RNEW_medium_{a}'],results[f'{cohort}_RNEW_medium_{b}'];sx,sy=x['summary'],y['summary']
   if sx['completed']!=12 or sy['completed']!=12:comparisons[cohort][a+'_vs_'+b]=dict(censored=True);continue
   delta={j:v['jct']-y['jobs'][j]['jct'] for j,v in x['jobs'].items()};best=min(delta,key=delta.get)
   comparisons[cohort][a+'_vs_'+b]=dict(improvement=1-sx['mean_jct']/sy['mean_jct'],mean_difference=sx['mean_jct']-sy['mean_jct'],job_delta=delta,improved=sum(v<0 for v in delta.values()),worsened=sum(v>0 for v in delta.values()),unchanged=sum(v==0 for v in delta.values()),time_delta={k:sx['totals'][k]-sy['totals'][k] for k in ['wait','training','checkpoint']},p95_change=sx['p95_jct']/sy['p95_jct']-1,p99_change=sx['p99_jct']/sy['p99_jct']-1,makespan_change=sx['makespan']/sy['makespan']-1,biggest_gain_job=best,leave_best_out_diagnostic=1-sum(v['jct'] for j,v in x['jobs'].items() if j!=best)/sum(v['jct'] for j,v in y['jobs'].items() if j!=best))
 save('comparisons.json',comparisons)
 arrival_diagnostics={}
 for cohort in ['BURST','SPREAD']:
  for method in ['SIA_P','SIA_AUX','CC_P','CC_CPU','CC_MEM','CC_FULL']:
   name=f'{cohort}_RNEW_medium_{method}';base=f'COMPACT_RNEW_medium_{method}';x,y=results[name],results[base]
   if x['summary']['completed']!=12 or y['summary']['completed']!=12:continue
   cd={j:v['completion']-y['jobs'][j]['completion'] for j,v in x['jobs'].items()};ad={j:v['arrival']-y['jobs'][j]['arrival'] for j,v in x['jobs'].items()};jd={j:v['jct']-y['jobs'][j]['jct'] for j,v in x['jobs'].items()}
   assert all(abs(jd[j]-(cd[j]-ad[j]))<1e-9 for j in jd)
   arrival_diagnostics[name]=dict(physical_execution_signature_equal=signatures[name]==signatures[base],all_completion_times_equal=all(v==0 for v in cd.values()),mean_completion_time_difference=statistics.mean(cd.values()),mean_arrival_difference=statistics.mean(ad.values()),mean_jct_difference=statistics.mean(jd.values()),completion_time_differences=cd,jct_differences=jd)
 save('arrival_vs_service_diagnostics.json',arrival_diagnostics);save('pressure_evidence.json',pressure)
 # Same accepted response accounting, no new execution model.
 script=(ROOT/'research_runs/sia_synergy/cc_cifar_bert12_20261029_run1/describe_response_v2.py').read_text().replace('cc_cifar_bert12_20261029_run1','cc_arrival_pressure18_20261030_run1');(OUT/'response_analysis_script.py').write_text(script,encoding='utf8')
 env={'__file__':str(OUT/'response_analysis_script.py'),'__name__':'descriptive_only'};exec(compile(script,str(OUT/'response_analysis_script.py'),'exec'),env);env['main']()
 response=json.loads((OUT/'response_analysis.json').read_text());gate=json.loads((OUT/'opportunity_gate.json').read_text())
 for n,h in p['source_hashes'].items():assert sha(OUT/'source'/n)==h
 for n,h in p['input_hashes'].items():assert sha(OUT/'inputs'/n)==h
 old=ROOT/'research_runs/sia_synergy/cc_ample_lex_repair_20261030_run1';seal=json.loads((old/'delivery_hashes_final.json').read_text());assert all(sha(old/n)==h for n,h in seal.items())
 validation.update(source_unchanged=True,input_hashes_valid=True,old_seal_verified=len(seal),opportunity_gate_passed=gate['passed'],simulation_wall_seconds=sum(r['wall_seconds'] for r in records),source_tests_reused=245,new_input_tests=6)
 save('validation.json',validation)
 report=['# 同一12作业、三种到达压力：六方法18格实验','SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE',
 f"完成{validation['complete_cells']}/18格，正式子进程累计墙钟{validation['simulation_wall_seconds']:.2f}秒。源算法与已验收词典序修复版逐文件一致，未按结果调整任何参数。",
 '## 冻结条件','原同一6 CIFAR＋6 BERT、原GPU demand/batch/完整训练工作量/顺序不变。AWS/DGX各一节点、各2GPU/6合成CPU单位/125GB RAM；固定R0/E-legacy、Medium alpha=.5，seed_profiles(1,8)，60秒、6000轮，独立进程串行，启动前2GiB内存/5GiB磁盘门槛。',
 'BURST全部t0；COMPACT前4个t0、其余每300秒2个，末到达1200；SPREAD第i个i×300秒，末到达3300。只改变实际arrival；各方案同一组应用画像，不是三条独立自然trace，不做虚假样本量/显著性推断。COMPACT与历史原始CSV逐字节一致，但本轮全部用同一修复版重跑，不混用旧版结果。',
 '六方法：SIA_P调用官方Sia GPU及placement，比例CPU/RAM，无L2；SIA_AUX冻结同轮完整Sia意图，用本项目ECT辅助优化，并非Synergy-TUNE；CC_P/CPU/MEM/FULL共享修复后的CC-TAIL目标与服务/放置机制，依次开放零/CPU/内存/两种辅助配额自由度。',
 '## 先验机会门槛',json.dumps(gate,ensure_ascii=False),
 'gate固定60轮上限：FULL实际CF+BERT同节点有效共训中，CF相对同GPU比例点加速而BERT不减速的重叠不少于60节点秒；同时全执行服务集合归一化净进度为正、逐轮同状态剩余ETA节省总和为正。后者不等于实现JCT节省、不是全局JCT最优证明，未使用未来完成时间。所有已执行任务均计receiver收益和donor损失，checkpoint不加速。支持范围内共驻单独记录，不能用E-legacy裁剪点冒充域内测量。',
 '## 完整结果',table(['负载','方法','完成','Mean','Median','P95','P99','Makespan'],[[r[k] for k in ['cohort','method','completed','mean','median','p95','p99','makespan']] for r in metrics]),
 '## 配对消融',table(['负载','对照','平均JCT改善','改善/退化/不变','等待差s','训练差s','checkpoint差s','P99变化','Makespan变化'],[[co,k,f"{100*v['improvement']:.3f}%",f"{v['improved']}/{v['worsened']}/{v['unchanged']}",*[v['time_delta'][x] for x in ['wait','training','checkpoint']],f"{100*v['p99_change']:.3f}%",f"{100*v['makespan_change']:.3f}%"] for co,d in comparisons.items() for k,v in d.items() if not v.get('censored')]),
 '改善率=1−新/对照，正值表示下降；尾部变化=新/对照−1，正值表示恶化。CPU/MEM/FULL对CC_P反映辅助自由度的闭环增量，包含后续GPU/队列变化，不能简单视为固定轨迹的物理因果份额。FULL对Sia还包括完成优先目标/准入/持续服务/放置的差异。',
 '## 资源小时与成本',table(['负载','方法','GPUh','CPU单位h','RAM GBh','等待s','训练s','checkpoint s','restart'],[[r[k] for k in ['cohort','method','gpu_hours','cpu_core_hours','ram_gb_hours','wait','training','checkpoint','restarts']] for r in metrics]),
 '## 压力是否真正改变执行',table(['格','观测活跃峰值','等待作业峰值','等待作业轮数'],[[n,*[v.get(k,0) for k in ['peak_observed_active_jobs','peak_waiting_jobs','waiting_job_rounds']]] for n,v in pressure.items()]),table(['相对COMPACT','执行签名相同','所有完成时刻相同','平均完成时刻差','平均arrival差','平均JCT差'],[[n,*[v[k] for k in ['physical_execution_signature_equal','all_completion_times_equal','mean_completion_time_difference','mean_arrival_difference','mean_jct_difference']]] for n,v in arrival_diagnostics.items()]),
 'JCT差=完成时刻差−arrival差。若某方法在BURST和COMPACT的完成时刻完全相同，则其跨负载JCT变化仅由提交时间改变造成，不能称作该方法执行变快/变慢。签名只比较进度、完成、rho、GPU类型/放置等实际执行字段，排除arrival/等待/求解墙钟。任务集合输入不同也不保证服务轨迹一定不同：原紧凑到达可能已早于该策略服务空档。所有跨方法的主比较仍使用同一负载，arrival相同因而抵消。',
 '## 实际共驻、增益与边界',table(['格','CF/BERT共训节点s','receiver进度份额','donor损失份额','CPU分子裁剪%','RAM分子裁剪%'],[[n,x['totals'].get('heterogeneous_application_coproductive_node_seconds',0),x['totals']['receiver_progress_fraction'],x['totals']['donor_loss_fraction'],100*x['actual_requested_coordinate_clipping_seconds'].get('numerator_cpu_beyond_1e12_tolerance',0)/x['totals']['productive_job_seconds'],100*x['actual_requested_coordinate_clipping_seconds'].get('numerator_ram_beyond_1e12_tolerance',0)/x['totals']['productive_job_seconds']] for n,x in response.items()]),
 '分子/分母、CPU/RAM及浮点端点容差统计均在response_analysis.json。裁剪比例为有效训练秒加权，CPU/RAM可重叠，不能相加或当作收益贡献比例。局部进度份额固定实际GPU/batch/已执行片段，不是可行完整反事实调度。',
 '## 统计解释与局限','每个负载12作业，P95/P99只描述这一批任务；所有长任务均保留。comparisons.json包含最大贡献任务、逐作业差值和去掉单个最大贡献作业的诊断，后者不代替主结果。任务工作量相同但arrival不同，makespan跨负载包含到达跨度，不应只比较其绝对值评价调度优劣。逐作业JCT=completion−当前arrival；主要比较在同一负载的配对方法之间。',
 '共享官方Sia GPU画像和Synergy矩阵，应用模板关联、phi和跨硬件迁移为合成假设；BERT平坦模板不是目标硬件实测。已有reference越界/裁剪未修改；物理配额变更额外加载开销未建模，调度墙钟不计入虚拟JCT。245原测试随完全相同代码复用，本轮另6项输入/门槛核算测试通过。',
 '## 复现与停止',json.dumps(validation,ensure_ascii=False),'源码source_snapshot.zip，预注册preregistration.json，原始逐轮cells/*/rounds.jsonl.gz，启动execution_manifest.jsonl，独立复算raw_audit.json/per_job.csv/metrics.json，比较comparisons.json。旧结果未覆盖，未恢复104/160批次，未运行DeepSpeech扩展。全部完成后停止。']
 (OUT/'REPORT.md').write_text('\n\n'.join(report)+'\n',encoding='utf8')
 for name in ['prepare_arrival18_20261030.py','gate_arrival18_20261030.py','test_arrival18_20261030.py','audit_arrival18_20261030.py','resource_resume_arrival18_20261030.py']:shutil.copyfile(ROOT/'maintenance'/name,OUT/name)
 (OUT/'CURRENT_TASK.md').write_text('COMPLETE18_AUDITED_STOPPED. REPORT.md/validation.json authoritative. No additional experiment authorized.\n',encoding='utf8')
 print('SEALING_COMPLETE18',flush=True);save('delivery_hashes.json',{f.relative_to(OUT).as_posix():sha(f) for f in OUT.rglob('*') if f.is_file() and f.name!='delivery_hashes.json'})
if __name__=='__main__':main()
