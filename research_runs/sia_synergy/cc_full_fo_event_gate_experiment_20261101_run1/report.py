"""Independent final description; only persisted complete events, never rollout."""
import json,gzip,hashlib,math,collections
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2]
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest().upper()
p=json.loads((O/'preregistration.json').read_text());T=Path(p['source']);stats=json.loads((O/'raw_audit.json').read_text());jobs=json.loads((O/'per_job.json').read_text());pairs=json.loads((O/'comparison_summary.json').read_text());gate=json.loads((O/'event_gate_result.json').read_text());parent=Path(p['corrected_gate']['evidence_parent']);old=json.loads((parent/'actual_prefix_comparison.json').read_text());old={r['time']:r for r in old['rows']};timelines={};prefix_check={};window={};boundary={}
for c in p['cells']:
 name=c['id'];prev={};seen=set();events=[];pc=[];checked=0;cuts=collections.Counter();trajectory=[];last_round=None;rhos=[];eight=[]
 for line in gzip.open(O/'cells'/name/'rounds.jsonl.gz','rt',encoding='utf8'):
  r=json.loads(line);d=r['decision_for_next_interval'];sel=d['result']['selected'];done={e['job_id']:e['completion_time'] for e in r['execution'] if e['completion_time'] is not None and e['job_id'] not in seen}
  fresh=d['round']!=last_round;last_round=d['round']
  if fresh and len(sel)==8:
   assert all(v['config']['gpus']==1 for v in sel.values())
   eight.append(r['interval_end'])
  for j,v in (sel.items() if fresh else []):
   cuts['queries']+=1;cuts['numerator_clipped']+=v['source_coordinates']['clipped'];cuts['denominator_clipped']+=v['baseline_source_coordinates']['clipped']
   rhos.append(v['rho'])
  trajectory.append(dict(time=r['interval_end'],progress={e['job_id']:e['progress_after'] for e in r['execution']},completed=done,configs={j:v['config']['config_id'] for j,v in sel.items()}))
  if done:
   nxt=sel if fresh else {}
   events.append(dict(time=r['interval_end'],actual_completions=done,released_previous_bundle={j:prev[j] for j in done if j in prev},next_allocations=nxt,survivor_changes={j:dict(before=prev[j],after=v) for j,v in nxt.items() if j in prev and prev[j]['column_id']!=v['column_id']}));seen.update(done)
  if c['workload']=='proposed_6job' and c['response']=='medium' and r['interval_end'] in old:
   checked+=1
   prior=old[r['interval_end']][c['method'].lower()];conf={j:v['config']['config_id'] for j,v in sel.items()};prog={e['job_id']:e['progress_delta'] for e in r['execution']}
   same=conf==prior['configs'] and prog.keys()==prior['progress'].keys() and all(math.isclose(v,prior['progress'][j],rel_tol=1e-10,abs_tol=1e-7) for j,v in prog.items())
   if not same:pc.append(dict(time=r['interval_end'],config_matches=conf==prior['configs'],progress_matches=all(math.isclose(v,prior['progress'].get(j,float('inf')),rel_tol=1e-10,abs_tol=1e-7) for j,v in prog.items())))
  prev=sel
 timelines[name]=events;window[name]=trajectory;boundary[name]=dict(**cuts,rho_min=min(rhos),rho_max=max(rhos),all8_single_gpu_rounds=len(eight),all8_start=min(eight) if eight else None,all8_last=max(eight) if eight else None);prefix_check[name]=dict(checked_rounds=checked,mismatches=pc,scope='same Medium6 prefix' if checked else 'not_applicable')
save('completion_release_events.json',timelines);save('progress_and_release_timeline.json',window);save('gate_prefix_reproducibility.json',prefix_check);save('boundary_counts.json',boundary)
attribution={}
for n,v in pairs.items():
 if 'full' not in v:continue
 b=v['full'];delta=v['per_job_saved'];ts=sorted({0.}|{float(x['completion']) for name in (n,b) for x in jobs[name].values()});areas=[]
 for t,u in zip(ts,ts[1:]):
  fo=sum(x['completion']>t for x in jobs[n].values());fu=sum(x['completion']>t for x in jobs[b].values())
  if fo!=fu:areas.append(dict(start=t,end=u,fo_unfinished=fo,full_unfinished=fu,job_seconds_saved=(fo-fu)*(u-t)))
 area=sum(x['job_seconds_saved'] for x in areas);assert math.isclose(area,sum(delta.values()),abs_tol=1e-6)
 attribution[n]=dict(gross_job_gains=sum(max(0,x) for x in delta.values()),gross_job_losses=-sum(min(0,x) for x in delta.values()),net_jct_sum_saved=area,completion_count_area=areas,first_release_fo=min(x['completion'] for x in jobs[n].values()),first_release_full=min(x['completion'] for x in jobs[b].values()))
save('completion_net_attribution.json',attribution)
integrity=dict(source_files=len(p['source_hashes']),source_mismatches=[n for n,h in p['source_hashes'].items() if sha(T/n)!=h],input_mismatches=[n for n,h in p['input_hashes'].items() if sha(Path(n))!=h],audit_errors={n:s['errors'] for n,s in stats.items()},complete_cells=sum(s['complete'] for s in stats.values()),cells=len(stats))
assert not integrity['source_mismatches'] and not integrity['input_mismatches'];assert all(not x for x in integrity['audit_errors'].values())
save('validation.json',integrity)
lines=['# 事件机会验收与原六格完整训练结果','', 'SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE','',
 '本轮仅修正机会验收定义，算法、目标、资源网格、恢复预算、执行响应、负载和观察参数未改。旧窗口、旧失败门槛及全部历史结果保留。新六格独立进程、独立目录，按冻结顺序串行运行。', '',
 '## 冻结条件与验收定义','', '主场景为官方workload-1前6个CIFAR记录同时到达；控制为前8个CIFAR同时到达。AWS4GPU/18合成CPU/125GB，RTX4GPU/12合成CPU/375GB。R0/E-legacy，Medium alpha=.5、Neutral alpha=0，60秒/6000轮，seed_profiles(1,8)。每格前执行2GiB内存/5GiB磁盘门槛。', '',
 '验收在ARRIVAL/COMPLETION/FIRST_VALID_CURRENT_MODEL等合法事件比较同状态、同服务集合下的实际FO/FULL决策，再要求所选GPU几何差异在两条实际前缀中连续执行至少6周期，并至少2个执行周期确实推进训练。重复重选不是持续受益的必要条件。此次是按用户批准对既有日志的重新定义与重分析，不是独立重复实验或未经观察数据的新假设检验。完整六格是在新定义冻结并通过后才启动，未以JCT作为验收条件。', '',
 '240秒的cifar10-2同状态比较为FO RTX1、FULL AWS1，实际差异持续240–1740秒，共25周期；窗口实际进度FO15145.72、FULL21624.80。关键任务比较点分子/分母受支持；同状态变化的cifar10-0在FO有裁剪，仍按已冻结E-legacy执行，风险不隐藏。通过的是合成模型内的机会门槛，不能宣称整个服务集合所有点均有原始数据支持。', '',
 '## 完整指标','', '|格|完成|平均JCT s|Median|P95|P99|Makespan|','|---|---:|---:|---:|---:|---:|---:|']
for n,s in stats.items():lines.append('|'+ '|'.join(str(x) for x in [n,s['completed'],s['mean'],s['median'],s['p95'],s['p99'],s['makespan']])+'|')
lines+=['','P95/P99分别仅来自6或8作业的经验分位，不能视为总体尾部的精确估计。FO是当前项目串行GPU→辅助优化对照，不能称为原始Sia。','', '## 配对净收益及受损作业','']
for n,v in pairs.items():
 lines += [f"### {n}",'',json.dumps(v,ensure_ascii=False,indent=2),'']
 if 'full' in v:
  b=v['full'];lines+=['|作业|FO JCT|FULL JCT|节省 s|等待节省|训练节省|checkpoint节省|','|---|---:|---:|---:|---:|---:|---:|']
  for j,a in jobs[n].items():
   z=jobs[b][j];lines.append('|'+ '|'.join(str(x) for x in [j,a['jct'],z['jct'],a['jct']-z['jct'],a['wait']-z['wait'],a['training']-z['training'],a['checkpoint']-z['checkpoint']])+'|')
  z=attribution[n];lines += ['',f"逐作业提前完成的总收益{z['gross_job_gains']:.3f}秒，推迟完成的总损失{z['gross_job_losses']:.3f}秒，净节省{z['net_jct_sum_saved']:.3f}作业秒。首个完成事件FO={z['first_release_fo']}秒，FULL={z['first_release_full']}秒；不能把某个同名任务的650秒提前误称为集群首次释放提前650秒。",'','完成数量曲线的净面积与完整JCT之和差逐项一致，详细区间见completion_net_attribution.json。这是整体完成节奏核对，不是将每次释放的独立因果效应相加。']
lines+=['','## 资源使用与执行审计','', '|格|GPU h|CPU h|RAM GB h|Restart|等待 s|训练 s|Checkpoint s|查询裁剪比例|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for n,s in stats.items():
 t=s['totals'];lines.append('|'+ '|'.join(str(x) for x in [n,t['gpu_hours'],t['cpu_core_hours'],t['ram_gb_hours'],t['restarts'],t['wait'],t['training'],t['checkpoint'],s['clipped_fraction']])+'|')
lines+=['','所有指标由原始完成事件、训练片段和分配时长独立复算，并与summary逐项核对；容量、锁、rho独立公式、单次倍率、互斥时间审计见raw_audit.json。分子与分母裁剪分别见boundary_counts.json，查询比例不是训练时间比例。', '',
 '## 资源释放如何计入结论','', 'completion_release_events.json逐事件保存完成者原配置、释放资源、下一轮存活任务资源与变化；progress_and_release_timeline.json保存各任务进度和完成顺序。分析必须把receiver提前与donor推迟同时纳入完整JCT；不能将一个receiver的提速当作全局净收益。两条轨迹在早期已经分叉，单次资源释放对最终JCT的精确因果份额无法仅由观察日志识别。', '',
 '本轮保持恢复能力等效并计入真实checkpoint与切换代价。若总平均JCT改善而个别任务退化，仍报告所有受损任务；若控制场景持平或退化，不重选任务或容量。Neutral用于区分执行响应耦合与GPU/准入本身差异；8作业控制仅在8任务共同服务时限制数量自由度，不能要求全轨迹强制持平。', '',
 '## 可复现与停止','', '完整344文件实际源码快照、逐文件SHA、输入、依赖和6格顺序在preregistration.json及runner_receipt.json；400项既有测试与Sia/FO/FULL精确三轮回归按完全相同源码哈希复用，新的5项事件验收测试通过。gate_prefix_reproducibility.json给出新Medium前60轮与旧真实前缀的配置/进度一致性核对；存在差异时逐条披露，不以接近均值替代。', '',
 '本轮不扩展其他场景，不调参，不自动改算法。']
(O/'REPORT.md').open('x',encoding='utf8').write('\n'.join(lines)+'\n')
save('final_artifact_hashes.json',{str(x.relative_to(O)):sha(x) for x in O.rglob('*') if x.is_file() and x.name!='final_artifact_hashes.json'})
print('REPORT_COMPLETE',integrity)
