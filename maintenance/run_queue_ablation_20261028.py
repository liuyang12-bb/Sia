"""Exactly eight frozen queue-factor cells and one official Sia reference."""
import argparse,collections,dataclasses,difflib,gzip,hashlib,importlib.metadata,itertools,json,math,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_runs/sia_synergy/cc_queue_mechanism_ablation_20261028_run1'
TREE=ROOT/'cc-joint-queue-ablation'
OLD=ROOT/'research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1'
CELLS=['Q111','SIA_P','Q000','Q001','Q010','Q011','Q100','Q101','Q110']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def enc(x):
 if dataclasses.is_dataclass(x):return dataclasses.asdict(x)
 if hasattr(x,'tolist'):return x.tolist()
 if hasattr(x,'item'):return x.item()
 raise TypeError(type(x).__name__)
def save(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False,default=enc)
def hashes(p):return {f.relative_to(p).as_posix():sha(f) for f in p.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc'}
def resources():
 import psutil
 r=dict(memory_gib=psutil.virtual_memory().available/2**30,disk_gib=psutil.disk_usage(str(OUT)).free/2**30)
 if r['memory_gib']<2 or r['disk_gib']<5:raise RuntimeError('RESOURCE_GATE '+str(r))
 return r
def freeze():
 assert 'Ran 10 tests' in (OUT/'green1.log').read_text() and '\nOK\n' in (OUT/'green1.log').read_text()
 t=(OUT/'full_tests.log').read_text();assert '\nOK\n' in t and 'Ran 239 tests' in t
 parent=json.loads((OUT/'parent_source_hashes.json').read_text())
 for n,h in parent.items():assert sha(TREE/n)==h and sha(OLD/'source'/n)==h,n
 shutil.copytree(TREE,OUT/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 save(OUT/'preregistration.json',dict(version='QUEUE_FACTOR_2X2X2_V1',cells=CELLS,
  source_hashes=hashes(OUT/'source'),runner_hash=sha(Path(__file__)),
  input_hashes={n:sha(TREE/n) for n in ['inputs/workload.csv','inputs/registry.json','inputs/config_symmetric.json','dependencies.lock.txt']},
  workload='Same existing controlled20 records:18 CIFAR10+2 DeepSpeech2; unchanged work/arrivals/order',
  nodes={'aws:0':dict(gpu=2,cpu=6.,ram=187.5),'dgx-ext:0':dict(gpu=2,cpu=6.,ram=187.5)},
  response='neutral',alpha=0.,rho=1.,aux='STRICT_NODE_PROPORTIONAL_3_CPU_93.75GB_PER_GPU',
  interval=60,max_rounds=6000,service_quantum=300.,lifetime_delay_bound=250.,seed_profiles=[1,8],
  factors={'A0':'Sia normalized goodput fairness utility including original restart proxy; NOT official Sia allocator',
           'A1':'CC reciprocal current physical remaining ETA including actual startup',
           'B0':'No mandatory FIFO head or drain; optional objective-driven admission; no individual starvation guarantee',
           'B1':'Existing mandatory FIFO head, existing services, drain when head cannot fit',
           'C0':'Existing300productive-second service segments, shared lifetime budget, no eviction if impossible to resume within remaining budget',
           'C1':'Current completion-continuity until actual completion'},
  shared=['Identical raw candidate domain/proportional resources,unknown-model bootstrap domain,actual training model,rho,workload',
          'Full true checkpoint lock and atomic capacity audit','Nonempty-start safety without objective bonus',
          '250 lifetime startup budget persists across all service segments; single original tail recovery remains unchanged',
          'Same geometry-event gate, known target/progress checks for held jobs, actual-startup tie then stable candidate tie'],
  interpretation='C is completion continuity versus bounded periodic service, NOT removal of all checkpoint/short-service protection; factorial results apply only to this comparator',
  outcomes=['mean/median/P95/P99 JCT,completion/censor,per-job timing,queue area,initial/later wait,training/checkpoint,resources/restarts,productive concurrency',
            'factor main effects averaged over other factors,leave-one-out contrasts,pair/three-way interactions,Shapley contributions if all8 complete'],
  missing='No imputed censored JCT; RMST at common357600s only if sufficient observation; complete-only JCT never compared as full mean',
  regression=dict(Q111_reference='R11_neutral_CC_P',SIA_P_reference='R11_neutral_SIA_P',strict='exact progress/allocation/restart/completion events'),
  failure='Safety gate stop before next cell; implementation/audit failure stops batch and retains evidence; no tuning/changed observation windows',
  packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','pandas','cvxpy','cylp','psutil']},
  warning='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE'))
 p=OUT/'DESIGN.md'
 p.write_text('''# 排队机制2×2×2预注册\n\n主比较不是更换CPU数据，而是隔离调度控制。全部使用原20作业、R11容量、Neutral与比例配额，因此CPU/RAM没有额外加速收益。\n\nA改变配置评分：Sia归一化Goodput公平效用/原restart proxy 对 CC倒数剩余ETA/实际startup。A0只是借用Sia评分，候选与放置仍为联合列，真正Sia另作为外部基线，不能把Q000冒称原Sia。\nB改变准入：FIFO队首强制+排空 对 目标驱动可选任务。共同保留checkpoint、已发服务段和非空启动安全；B0不声称单任务防饥饿。\nC改变服务：300秒有效训练段 对 持续至完成。checkpoint不算训练秒；两者均保留生命周期250秒上限、原事件门槛和一次旧TAIL许可，不能重置预算。若段到期后剩余预算不足再启动，则保持原服务，这是避免不可恢复状态的共同安全边界，单独统计。C0不是无保护任意抢占。\n\n预先假设：A可能促使小GPU共驻及快速完成；B可能帮助冷启动/公平服务，也可能因排空限制并发；C可能减少反复等待与checkpoint，也可能牺牲长任务速度。允许任一因素无收益、负收益、交互或删失。\n\n共8格，先Q111核验旧CC-P完全一致，再原Sia核验历史，然后Q000/001/010/011/100/101/110。每格独立Python进程串行、60秒、6000轮。A/B/C全部开启必须精确恢复旧CC-P分配/进度/完成和restart；否则停止，不运行后续格。输入、GPU画像、响应及基础网格不变。\n\n主效应以另外两因素四种组合上的平均差值报告；逐项拆除效应以Q111对Q011/Q101/Q110报告；存在交互时不能把三个百分比简单相加。若8格均完成，另给Shapley分摊Q000至Q111的净JCT差；这是单负载算法因素归因，不是跨负载统计证明。\n\n无新P99修复、无CPU响应修改、无104格或160作业实验。删失作业不插补JCT；性能不好不改参数。本轮新机制仅用于消融，旧CC-TAIL和所有历史版本不变。\n''',encoding='utf-8')
 save(OUT/'design_hash.json',dict(design=sha(p),preregistration=sha(OUT/'preregistration.json')))
 print('FROZEN',len(CELLS))
def setup():
 p=json.loads((OUT/'preregistration.json').read_text());assert p['runner_hash']==sha(Path(__file__))
 for n,h in p['source_hashes'].items():assert sha(OUT/'source'/n)==h,n
 for n,v in p['packages'].items():assert importlib.metadata.version(n)==v,n
 sys.path.insert(0,str(OUT/'source'));sys.path.insert(0,str(OUT/'source/sia-simulator'))
 from cc_factory import verify
 verify();return p
def cell(name):
 p=setup();assert name in CELLS;res=resources();d=OUT/'cells'/name;d.mkdir(parents=True,exist_ok=False)
 save(d/'start.json',dict(time=time.time(),resources=res,argv=sys.argv,preregistration_sha=sha(OUT/'preregistration.json')))
 from resource_study import build
 from joint_prototype.queue_factorial import install
 from run_stage2c_cell import Accumulator
 c=build('SIA_P' if name=='SIA_P' else 'CC_P','R11','neutral')
 if name!='SIA_P':install(c,*map(int,name[1:]))
 m=Accumulator(c);start=time.time();previous=None;new_decision=None;first_alloc={};first_train={};event_cursor=0
 try:
  with gzip.open(d/'rounds.jsonl.gz','xt',encoding='utf-8') as f:
   for i in range(p['max_rounds']):
    if c.all_complete():break
    c.step(60);r=c.joint_round_logs[-1]
    r['job_end_state']={j.name:dict(restarts=j.num_restarts or 0,wasted_gpu_seconds=j.wasted_gpu_seconds,completion=j.completion_time) for j in c.jobs}
    f.write(json.dumps(r,default=enc,separators=(',',':'))+'\n');f.flush()
    decision=r['decision_for_next_interval'];selected=decision['result']['selected'];audit=decision['result']['audit']
    if decision['round']!=new_decision:
     new_decision=decision['round'];assert audit['valid']
     specs={j['job_id']:j for j in decision['probe']['specs']};usage=collections.defaultdict(lambda:[0.,0.,0.])
     for j,col in selected.items():
      assert col['rho']==1. and col['cpu_per_gpu']==3. and col['ram_gb_per_gpu']==93.75
      if specs[j]['checkpoint_locked']:
       q=specs[j];assert col['config']['config_id']==q['current_config_id'] and col['placement']==dict(q['current_placement'])
       assert(col['cpu_per_gpu'],col['ram_gb_per_gpu'])==(q['current_cpu_per_gpu'],q['current_ram_gb_per_gpu'])
      for node,g in col['placement'].items():
       for k,v in enumerate([g,g*3.,g*93.75]):usage[node][k]+=v
     for v in usage.values():assert all(x<=y+1e-7 for x,y in zip(v,[2,6.,187.5]))
     if name!='SIA_P':
      assert decision['queue_factors']==dict(zip('ABC',map(int,name[1:])))
      for j,v in audit['service_delay_ledgers'].items():assert v['delay_charged']<=250+audit.get('tail_grants',{}).get(j,{}).get('extra_budget',0.)+1e-9
     else:assert decision['strict_full_intent']
    m.consume(c,r)
    c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
    if (i+1)%100==0:print(name,'ROUND',i+1,'COMPLETE',sum(j.completion_time is not None for j in c.jobs),flush=True)
  s=m.summary(c,dict(rmst_tau_seconds=357600));s.update(method=name,response='neutral',scenario='R11',total_wall_seconds=time.time()-start)
  save(d/'summary.json',s);save(d/'output_hashes.json',{x.name:sha(x) for x in d.iterdir() if x.is_file()})
  print('RESULT',name,s['completed'],s['mean_jct'],flush=True)
 except Exception as e:
  save(d/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=c.current_time,rounds=m.rounds,last_decision=c.policy.last_decision));raise
def trace_signature(r):
 d=r['decision_for_next_interval']
 return dict(execution=[(e['job_id'],e['progress_after'],e['completion_time'],e['cluster'],e['placement']) for e in r['execution']],
             selected={j:(c['config']['config_id'],c['placement'],c['cpu_per_gpu'],c['ram_gb_per_gpu']) for j,c in d['result']['selected'].items()},
             ends=r['job_end_state'])
def regression(name):
 ref=OLD/'cells'/('R11_neutral_CC_P' if name=='Q111' else 'R11_neutral_SIA_P')/'rounds.jsonl.gz'
 new=OUT/'cells'/name/'rounds.jsonl.gz';count=0
 with gzip.open(ref,'rt',encoding='utf-8') as a,gzip.open(new,'rt',encoding='utf-8') as b:
  for x,y in itertools.zip_longest(a,b):
   assert x is not None and y is not None,'LENGTH_DIFFERENCE'
   assert trace_signature(json.loads(x))==trace_signature(json.loads(y)),(name,count,'EXACT_REGRESSION_FAILED')
   count+=1
 save(OUT/(name+'_historical_regression.json'),dict(exact=True,rounds=count,reference=str(ref),reference_sha=sha(ref),current_sha=sha(new)))
def batch():
 setup()
 for name in CELLS:
  assert not (OUT/'cells'/name).exists(),'REFUSE_EXISTING_CELL'
  try:r=resources()
  except Exception as e:save(OUT/'safe_stop.json',dict(next_cell=name,error=repr(e)));print('SAFE_STOP',e,flush=True);return
  start=time.time()
  with (OUT/(name+'.log')).open('x',encoding='utf-8') as f:
   result=subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__)),'cell','--name',name],stdout=f,stderr=subprocess.STDOUT)
  with (OUT/'execution_manifest.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(cell=name,exit_code=result.returncode,wall_seconds=time.time()-start,resources=r))+'\n')
  print('CELL_END',name,result.returncode,flush=True)
  if result.returncode:return
  if name in ['Q111','SIA_P']:regression(name)
 print('ALL_NINE_FINISHED',flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','batch','cell']);a.add_argument('--name');args=a.parse_args()
 if args.action=='cell':cell(args.name)
 else:globals()[args.action]()
