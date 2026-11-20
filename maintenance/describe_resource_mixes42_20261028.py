"""Post-run descriptive execution-response accounting; not a new experiment."""
import collections,functools,gzip,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_resource_mixes42_20261028_run1'
sys.path.insert(0,str(OUT/'source/sia-simulator'))
from joint_prototype.joint_profile import load_synergy_template
def main():
 p=json.loads((OUT/'preregistration.json').read_text());reg={r['key']:r for r in json.loads((OUT/'source/inputs/registry.json').read_text())}
 matrices={t:load_synergy_template(OUT/'source/inputs/models'/t).matrix for t in ['alexnet_1.py','deepspeech_1.py']}
 work={j['name'].rsplit('-',1)[0]:j['work'] for c in json.loads((OUT/'execution_contracts.json').read_text()) for j in c['jobs']}
 @functools.lru_cache(maxsize=None)
 def rho(app,cid,c,m,alpha):
  r=reg[app+'/'+cid];matrix=matrices[r['template']]
  def f(x,y):return float(np.interp(x,[1,2,3,4,5,6,9,12,24],[np.interp(y,[20.83,62.5,125,187.5,250],v) for v in matrix]))
  return (f(c/r['phi_c'],m/r['phi_m'])/f(3/r['phi_c'],62.5/r['phi_m']))**alpha
 data={}
 for mix,cap,response,method in p['cells']:
  name='_'.join((mix,cap,response,method));cpu,ram=p['capacities'][cap];alpha=p['response_exponents'][response]
  out=collections.Counter();apps=collections.defaultdict(collections.Counter)
  with gzip.open(OUT/'cells'/name/'rounds.jsonl.gz','rt',encoding='utf8') as f:
   for line in f:
    r=json.loads(line);by_node=collections.defaultdict(list)
    for e in r['execution']:
     app=e['job_id'].rsplit('-',1)[0];a=e['resource_allocation']
     if not a:continue
     cid=a['config_id'];c,m=a['cpu_per_gpu'],a['ram_gb_per_gpu'];r00=rho(app,cid,cpu/2,ram/2,alpha)
     r10=rho(app,cid,c,ram/2,alpha);r01=rho(app,cid,cpu/2,m,alpha);r11=rho(app,cid,c,m,alpha)
     for z in e['training_segments']:
      assert math.isclose(z['rho'],r11,rel_tol=1e-12)
      w=z['raw_rate_before_rho']*z['seconds']/work[app]
      delta=w*(r11-r00)
      out['fixed_geometry_progress_fraction_delta']+=delta
      apps[app]['fixed_geometry_progress_fraction_delta']+=delta
      out['receiver_progress_fraction']+=max(0,delta);out['donor_loss_fraction']+=max(0,-delta)
      # Symmetric local response decomposition; not global JCT attribution.
      out['cpu_local_shapley_fraction']+=.5*w*((r10-r00)+(r11-r01))
      out['ram_local_shapley_fraction']+=.5*w*((r01-r00)+(r11-r10))
      out['productive_job_seconds']+=z['seconds']
      out['faster_than_proportional_job_seconds']+=z['seconds']*int(r11>r00+1e-10)
      out['slower_than_proportional_job_seconds']+=z['seconds']*int(r11<r00-1e-10)
      for node,g in a['placement'].items():
       if g==1:by_node[node].append((app,z['start_time'],z['start_time']+z['seconds']))
    # Epoch segments partition each task's productive timeline; intersections count node-seconds once.
    for node,spans in by_node.items():
     for i,(a,s,t) in enumerate(spans):
      for b,u,v in spans[i+1:]:
       if a==b:continue
       out['heterogeneous_application_coproductive_node_seconds']+=max(0,min(t,v)-max(s,u))
  assert math.isclose(out['cpu_local_shapley_fraction']+out['ram_local_shapley_fraction'],out['fixed_geometry_progress_fraction_delta'],abs_tol=1e-9)
  data[name]=dict(totals=dict(out),applications={a:dict(v) for a,v in apps.items()},
   interpretation='Actual executed training segments, compare response at proportional quota holding raw rate/GPU/batch/progress fixed. Not legal counterfactual scheduling when checkpoint locks quotas; not realized JCT saving; local CPU/RAM Shapley is response accounting only.')
  print('DESCRIBED',name,flush=True)
 dest=OUT/'analysis/execution_response_accounting.json'
 with dest.open('x',encoding='utf8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
 comp=json.loads((OUT/'analysis/paired_comparisons.json').read_text())
 def summary(l,r):
  vs=[(k,v['improvement']) for k,v in comp.items() if k.endswith('/'+l+'_vs_'+r) and 'improvement' in v]
  nums=[v for k,v in vs]
  return dict(cases=len(vs),wins=sum(v>0 for v in nums),ties=sum(v==0 for v in nums),losses=sum(v<0 for v in nums),median=float(np.median(nums)) if nums else None,min=min(nums) if nums else None,max=max(nums) if nums else None)
 effects={l+'_vs_'+r:summary(l,r) for l,r in [('CC_CPU','CC_P'),('CC_MEM','CC_P'),('CC_FULL','CC_CPU'),('CC_FULL','CC_P'),('CC_FULL','SIA_P'),('CC_FULL','SIA_AUX')]}
 with (OUT/'analysis/effect_summary.json').open('x',encoding='utf8') as f:json.dump(effects,f,ensure_ascii=False,indent=2)
 # Characterize RAM opportunity geometry from frozen data, without adding points.
 memory_limits=[]
 for cap,(cpu,ram) in p['capacities'].items():
  for app in ['cifar10','deepspeech2']:
   for typ in ['aws','dgx-ext']:
    for g in [1,2]:
     cid=f'{typ}:1:{g}';ref=reg[app+'/'+cid];base=rho(app,cid,cpu/2,ram/2,.5)
     feasible=[m for m in [20.83,62.5,93.75,125.,187.5,250.] if m*g<=ram+1e-9]
     best=max(rho(app,cid,cpu/2,m,.5) for m in feasible)
     memory_limits.append(dict(capacity=cap,application=app,config=cid,proportional_ram=ram/2,
      source_ram_at_proportional=ram/2/ref['phi_m'],mapped_62_5_ram_knot=62.5*ref['phi_m'],
      max_ram_only_rate_ratio=best/base,reference_supported=1<=3/ref['phi_c']<=24 and 20.83<=62.5/ref['phi_m']<=250,
      max_source_cpu_in_node=cpu/g/ref['phi_c']))
 with (OUT/'analysis/memory_opportunity_limits.json').open('x',encoding='utf8') as f:json.dump(memory_limits,f,ensure_ascii=False,indent=2)
 with (OUT/'REPORT.md').open('a',encoding='utf8') as f:
  f.write('\n\n## 六个受控场景的描述性收益汇总\n\n不作独立样本显著性检验；正改善率表示JCT降低。\n\n|对照|改善/持平/退化|改善率中位数|范围|\n|---|---|---|---|\n')
  for k,v in effects.items():
   if v['cases']:f.write(f"|{k}|{v['wins']}/{v['ties']}/{v['losses']}|{v['median']*100:.2f}%|{v['min']*100:.2f}% 至 {v['max']*100:.2f}%|\n")
  f.write('\nexecution_response_accounting.json进一步核查实际同节点异类任务的共同有效训练时长，以及在实际训练片段固定GPU/batch/raw率下，相对比例配额的接收方增量与让出方损失。局部响应CPU/RAM对称分解严格相加为总响应差，不能当作JCT的因果分解；尤其checkpoint时改变配额未必是合法调度反事实。总体JCT结论仍以同负载六策略完整完成事件配对为准。\n')
  f.write('\n## 感知与差异化配额的区别、内存机会的具体边界\n\nCC-P虽固定比例配额，GPU候选评分仍使用相同rho。因此CC-CPU/MEM/FULL对CC-P隔离的是差异化配额自由度，不能将其直接当作CPU/RAM感知本身对GPU选择的全部贡献。整体对Sia还包含目标、准入、连续服务与GPU/放置差异；本矩阵不能把它们逐项唯一分解。\n\n')
  f.write('对固定GPU和比例CPU枚举现有RAM网格：DeepSpeech全部四GPU配置，以及CIFAR的AWS单/双GPU配置，在93.75GB比例点已达到本可达CPU范围内的RAM平台，额外RAM的速率比为1。受支持的CIFAR/DGX单GPU可获得的RAM单独增益仅约5.24%（R11）或6.56%（R01）；DGX双GPU占满节点时没有提高每GPU RAM的容量空间，且其参考分母还涉及既有裁剪。详见memory_opportunity_limits.json。\n\n')
  f.write('因此增加DeepSpeech数量会增加潜在CF/DS组合，但没有新增一种能在现有网格上无损让出RAM的模板：从93.75降到62.5GB跨过DS的映射平台阈值，确实减速。已知连续无损互补点约103.28/84.22GB不在本轮网格。上述事实解释当前内存验证机会较窄，不证明它是全部JCT退化的唯一原因；当前局部目标、长任务让出代价与后续排队反馈仍会共同作用。本轮未加点、未改模型，也未新增仿真。\n')
if __name__=='__main__':main()
