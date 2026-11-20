"""Independent raw-event review; no training, no edits to original cell outputs."""
from pathlib import Path
import collections,csv,gzip,json,math,sys,time,zipfile
import numpy as np
from experiment import OUT,METHODS,load,sha,save

def close(a,b):assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-5),(a,b)

def main():
    p=load();sys.path.insert(0,str(OUT/'source'))
    from joint_prototype.r0_liveness import R0Oracle
    from joint_prototype.execution_audit import audit_execution
    from run_stage2c_cell import independent_rho
    rows=list(csv.DictReader((OUT/'inputs/workload.csv').open(encoding='utf-8')))
    apps={r['name']:r['application'] for r in rows};arrivals={r['name']:float(r['time']) for r in rows}
    oracle=R0Oracle(OUT/'inputs/registry.json',OUT/'inputs/models',apps,'medium')
    results={};perjob=[]
    for method in METHODS:
      directory=OUT/'cells'/method
      if not (directory/'summary.json').exists():
        failure=json.loads((directory/'failure.json').read_text(encoding='utf-8')) if (directory/'failure.json').exists() else None
        results[method]=dict(status='FAILED' if failure else 'NOT_STARTED',failure=failure);continue
      s=json.loads((directory/'summary.json').read_text())
      totals=collections.defaultdict(collections.Counter);progress=collections.defaultdict(float);completion={}
      count=collections.Counter();profile=collections.defaultdict(collections.Counter);prev=None;decision_stream=[]
      with gzip.open(directory/'rounds.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
          r=json.loads(line)
          for e in r['execution']:
            assert audit_execution(e)['valid'];j=e['job_id'];close(progress[j],e['progress_before']);progress[j]=e['progress_after']
            assert e['progress_after']>=e['progress_before']-1e-8
            for key,field in [('wait','waiting_seconds'),('checkpoint','checkpoint_delay_consumed'),('training','training_seconds')]:totals[j][key]+=e[field]
            if e['completion_time'] is not None:completion[j]=e['completion_time']
            a=e['resource_allocation']
            if a:
              rho=independent_rho(oracle,j,a['config_id'],a['cpu_per_gpu'],a['ram_gb_per_gpu']);close(e['rho'],rho)
              for seg in e['training_segments']:
                close(seg['effective_rate'],seg['raw_rate_before_rho']*rho)
                close(seg['progress_after']-seg['progress_before'],seg['effective_rate']*seg['seconds'])
                count['training_segments']+=1
              dt=e['allocated_seconds']/3600
              totals[j]['gpu_hours']+=a['gpus']*dt
              for u in a['node_usage']:
                totals[j]['cpu_core_hours']+=u['cpu']*dt;totals[j]['ram_gb_hours']+=u['ram_gb']*dt
              profile[j][a['config_id']]+=e['training_seconds']
            count['execution_records']+=1
          d=r['decision_for_next_interval']
          if d['round']==prev:continue
          prev=d['round'];selected=d['result']['selected'];used=collections.defaultdict(lambda:[0.,0.,0.])
          for j,col in selected.items():
            close(col['rho'],independent_rho(oracle,j,col['config']['config_id'],col['cpu_per_gpu'],col['ram_gb_per_gpu']))
            if method in ['SIA_P','JOINT_ECT_P']:assert (col['cpu_per_gpu'],col['ram_gb_per_gpu'])==(6.,125.)
            for node,g in col['placement'].items():
              used[node][0]+=g;used[node][1]+=g*col['cpu_per_gpu'];used[node][2]+=g*col['ram_gb_per_gpu']
          for node,v in used.items():
            assert all(x<=y+1e-7 for x,y in zip(v,[2,12,250]));count['node_capacity_checks']+=1
          for spec in d['probe']['specs']:
            if spec['checkpoint_locked'] and spec['current_config_id']:
              col=selected[spec['job_id']]
              assert col['config']['config_id']==spec['current_config_id'] and col['placement']==dict(spec['current_placement'])
              close(col['cpu_per_gpu'],spec['current_cpu_per_gpu']);close(col['ram_gb_per_gpu'],spec['current_ram_gb_per_gpu']);count['lock_checks']+=1
          waiting=set(apps)-set(selected)-set(completion)
          if waiting:count['queued_rounds']+=1
          scores={v['column_id']:v for v in d.get('completion_scores',[])}
          if scores:
            total=0.
            for col in selected.values():
              v=scores[col['column_id']];total+=v['utility']
              if v['estimated_completion'] is not None:
                close(v['estimated_completion'],v['delay']+v['remaining_work']/(v['raw_rate']*v['rho']))
              count['completion_score_checks']+=1
            close(total,d['result']['scheduling_objective'])
          decision_stream.append(dict(round=d['round'],status=d['result']['status'],waiting=sorted(waiting),
            selected={j:dict(config=c['config']['config_id'],cpu=c['cpu_per_gpu'],ram=c['ram_gb_per_gpu'],
              placement=c['placement'],rho=c['rho'],completion_estimate=scores.get(c['column_id'])) for j,c in selected.items()}))
      for j,v in s['jobs'].items():
        if v['completed']:
          close(v['completion'],completion[j]);close(v['jct'],completion[j]-arrivals[j])
          close(v['jct'],totals[j]['wait']+totals[j]['checkpoint']+totals[j]['training'])
        for key,value in totals[j].items():close(v[key],value)
        perjob.append(dict(method=method,job=j,**v))
      if len(completion)==len(rows):close(s['mean_jct'],np.mean([completion[j]-arrivals[j] for j in apps]))
      with gzip.open(directory/'audited_decisions.jsonl.gz','xt',encoding='utf-8') as stream:
        for row in decision_stream:stream.write(json.dumps(row)+'\n')
      results[method]=dict(status=s['status'],valid=True,checks=dict(count),summary=s,
        productive_seconds_by_config=dict(profile),capacity_violations=0,
        hashes={f.name:sha(f) for f in directory.iterdir() if f.is_file()})
    comparisons={}
    if all(results.get(m,{}).get('summary',{}).get('completion_rate')==1 for m in METHODS):
      jo=results['JOINT_ECT']['summary']
      for baseline in ['SIA_P','SIA_ECT_AUX','JOINT_ECT_P']:
        base=results[baseline]['summary'];deltas={j:jo['jobs'][j]['jct']-base['jobs'][j]['jct'] for j in apps}
        largest=min(deltas,key=deltas.get);others=[j for j in apps if j!=largest]
        comparisons[baseline]=dict(improvement_rate=1-jo['mean_jct']/base['mean_jct'],job_jct_deltas=deltas,
          improved=sum(x<0 for x in deltas.values()),worsened=sum(x>0 for x in deltas.values()),unchanged=sum(x==0 for x in deltas.values()),
          total_jct_saving_seconds=-sum(deltas.values()),largest_beneficiary=largest,largest_job_saving=-deltas[largest],
          excluding_largest_diagnostic_improvement=1-sum(jo['jobs'][j]['jct'] for j in others)/sum(base['jobs'][j]['jct'] for j in others),
          time_deltas={k:jo['totals'][k]-base['totals'][k] for k in ['wait','checkpoint','training']})
    result=dict(status='ALL_COMPLETE_VALID' if len(comparisons)==3 else 'INCOMPLETE_MATRIX',
      results=results,comparisons=comparisons,full_wall_seconds=time.time()-p['created_unix'],
      warning=p['warning'],only_six_jobs=True,no_parameter_search=True)
    save(OUT/'independent_audit.json',result)
    save(OUT/'comparison_summary.json',dict(status=result['status'],comparisons=comparisons,
      methods={m:results[m].get('summary',{}) for m in METHODS}))
    with (OUT/'per_job.csv').open('x',encoding='utf-8',newline='') as f:
      fields=['method','job','jct','completion','completed','wait','checkpoint','training','restarts','gpu_hours','cpu_core_hours','ram_gb_hours','wasted_gpu_seconds']
      writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(perjob)
    lines=['# 六任务四策略完整训练对照报告','',p['warning'],'',
      '本轮按用户批准的六任务场景执行，保留相同工作量、Medium响应和既有算法版本。首次预注册之后未按结果调整参数。',
      '','六任务为CIFAR0/1/2与DeepSpeech8/9/12，同时到达；AWS与DGX各2 GPU、12合成CPU单位、250 GB RAM。比例点6CPU/125GB每GPU。',
      '','## 实际结果','','|策略|状态|完成|平均JCT(s)|Median|P95|P99|Makespan|等待(s)|训练(s)|checkpoint(s)|restart|',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in METHODS:
      s=results[m].get('summary')
      if not s:lines.append(f'|{m}|{results[m]["status"]}|—|—|—|—|—|—|—|—|—|—|');continue
      vals=[s.get(k) for k in ['mean_jct','median_jct','p95_jct','p99_jct','makespan']]
      fmt=lambda v:'N/A' if v is None else f'{v:.3f}'
      lines.append('|'+ '|'.join([m,s['status'],f'{s["completed"]}/6']+[fmt(v) for v in vals]+[fmt(s['totals'][k]) for k in ['wait','training','checkpoint','restarts']])+'|')
    lines+=['','## 配对与归因','']
    if comparisons:
      for baseline,v in comparisons.items():
        lines.append(f'- JOINT-ECT相对{baseline}平均JCT改善{v["improvement_rate"]*100:.3f}%；改善/退化/不变任务数{v["improved"]}/{v["worsened"]}/{v["unchanged"]}。净JCT节省{v["total_jct_saving_seconds"]:.3f}秒。最大受益任务{v["largest_beneficiary"]}贡献{v["largest_job_saving"]:.3f}秒；去掉该任务的诊断改善{v["excluding_largest_diagnostic_improvement"]*100:.3f}%，不替代主结果。')
        lines.append(f'  总等待/checkpoint/训练时间变化：{v["time_deltas"]}。')
    else:lines.append('矩阵未完整，不能宣布总体JCT优势。详见各策略failure.json或删失状态；未插补JCT。')
    lines+=['','C对D（JOINT-ECT对JOINT-ECT-P）共享目标、L2和GPU候选，识别辅助配额优化的增量。JOINT-ECT对Sia的差异还包含目标、准入和placement，不能全归因CPU。SIA-ECT-AUX冻结完整Sia GPU意图和placement，不加入L2。',
      '','## 逐任务JCT(s)','','|任务|'+'|'.join(METHODS)+'|','|---|'+'---:|'*4]
    for j in apps:
      vals=[results[m].get('summary',{}).get('jobs',{}).get(j,{}).get('jct') for m in METHODS]
      lines.append('|'+j+'|'+'|'.join('N/A' if v is None else f'{v:.3f}' for v in vals)+'|')
    lines+=['','## 资源、约束与模型边界','','|策略|GPU-h|CPU-unit-h|RAM-GB-h|低于比例进度点的选中次数|选中列数|截断计数|',
      '|---|---:|---:|---:|---:|---:|---|']
    for m in METHODS:
      s=results[m].get('summary')
      if s:
        t=s['totals'];lines.append(f'|{m}|{t["gpu_hours"]:.4f}|{t["cpu_core_hours"]:.4f}|{t["ram_gb_hours"]:.4f}|{s["below_proportional_selected"]}|{s["selected_columns"]}|{s["clipping_counts"]}|')
    lines+=['','CPU敏感性来自Synergy矩阵的合成跨应用/跨硬件迁移，CIFAR用AlexNet模板、DeepSpeech用DeepSpeech模板。两类都可能在低资源时减速，DeepSpeech不是绝对不敏感donor。',
      '','保留已知E-legacy分子及分母clip，CIFAR DGX2固定分母越界未修复或删除。计数以逐轮选中列为分母，CPU/RAM类别可重叠。任何收益均不能据此声称同平台测量或迁移模型已验证。',
      '','独立审计从原始完成事件复算JCT，从施加rho前raw及训练段复算倍率与进度，核对等待/训练/checkpoint互斥时间、完成后的资源释放、节点容量及完整锁。通过范围见independent_audit.json。',
      '','188项既有测试证据沿用上一轮已核对的同源码版本；本轮未修改算法，另有四策略构造验收。没有以构造验收替代实际逐轮审计。',
      '',f'本轮从预注册至审计墙钟{result["full_wall_seconds"]:.3f}秒；四子进程各自实测时间见execution_manifest.jsonl。每组仅6作业，P95/P99为描述性指标，没有总体尾延迟精度含义。',
      '','所有源码、输入、脚本与依赖冻结见preregistration.json。逐轮完整证据在cells/<method>/rounds.jsonl.gz；简明轨迹在audited_decisions.jsonl.gz；逐任务结果见per_job.csv。',
      '','本轮结束后停止。不按结果修改参数、不扩展负载、不恢复104格或160-job。']
    with (OUT/'REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    with zipfile.ZipFile(OUT/'source_inputs_scripts.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
      for root in [OUT/'source',OUT/'inputs']:
        for path in root.rglob('*'):
          if path.is_file():z.write(path,path.relative_to(OUT))
      for name in ['experiment.py','audit.py','CURRENT_TASK.md','preregistration.json','dependencies.lock.txt']:
        z.write(OUT/name,name)
    save(OUT/'delivery_hashes.json',{str(f.relative_to(OUT)):sha(f) for f in OUT.rglob('*') if f.is_file() and 'source' not in f.relative_to(OUT).parts and 'inputs' not in f.relative_to(OUT).parts})
    print(json.dumps(dict(status=result['status'],comparisons=comparisons),indent=2))

if __name__=='__main__':main()
