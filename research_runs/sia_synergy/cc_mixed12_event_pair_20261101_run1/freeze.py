import hashlib,json,time,sys,importlib.metadata
from pathlib import Path
O=Path(__file__).resolve().parent;ROOT=O.parents[2];PRE=O.parent/'cc_mixed12_offline_preflight_20261101_run1';T=ROOT/'cc-joint-recovery-equivalent-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
p=json.loads((PRE/'preregistration.json').read_text(encoding='utf8'))
assert all(sha(T/n)==h for n,h in p['source_hashes'].items())
assert all(sha(n)==h for n,h in p['input_hashes'].items())
for n in ('config.json','workload.csv'):assert sha(O/'inputs'/n)==sha(PRE/'inputs'/n)
p.update(output=str(O),topology={'aws':4,'rtx':4},capacities=p['capacities']['capacity_scenarios']['main'],interval=60,max_rounds=6000,
 gate_rounds=40,gate_consecutive=6,gate_productive=2,
 gate_rule='Real cold prefix <=2400s. At legal ARRIVAL/COMPLETION/FIRST_VALID_CURRENT_MODEL compare FO/FULL on identical captured predecision state; same service set >=2, changed GPU and positive objective gap. Key geometry must match both actual paths for >=6 intervals and >=2 productive FULL intervals. Clipping disclosed, E-legacy unchanged. No JCT selection.',
 gate_event_types=['ARRIVAL','COMPLETION','FIRST_VALID_CURRENT_MODEL'],
 cells=[dict(id=m,method=m,capacity='config',workload='workload',response='medium') for m in ('FO','FULL')],
 scope='Real cold gate first; only if passes run independent complete FO then FULL, unchanged12 work/input/model; never tune; no other cohorts.',
 environment={'python':sys.version,'packages':{n:importlib.metadata.version(n) for n in ('numpy','scipy','cvxpy','cylp')}},
 frozen_unix=time.time(),previous_preregistration_sha256=sha(PRE/'preregistration.json'))
p['input_hashes'].update({str(O/'inputs'/n):sha(O/'inputs'/n) for n in ('config.json','workload.csv')})
with (O/'preregistration.json').open('x',encoding='utf8') as f:json.dump(p,f,indent=2,ensure_ascii=False)
with (O/'CURRENT_TASK.md').open('x',encoding='utf8') as f:f.write('本轮授权：12混合作业真实冷启动机会门槛，通过后完整FO/FULL。原算法/响应/容量/工作量；先gate40轮，事件选择后连续6周期至少2训练周期；不按结果延长或调参。运行入口pipeline.py，结果仅本目录。source=cc-joint-recovery-equivalent-v1，344文件hash及旧zip。停止于门槛失败、工程错误或完整配对报告。旧暂停104/160不恢复。\n')
print('FROZEN',sha(O/'preregistration.json'))
