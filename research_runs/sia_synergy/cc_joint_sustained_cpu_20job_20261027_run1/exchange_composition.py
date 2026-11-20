"""All Medium CC shared-node opportunities and realized exchange composition."""
import collections,csv,gzip,json,sys
from experiment import OUT,SOURCE,load,save

load();sys.path.insert(0,str(SOURCE))
from joint_prototype.r0_liveness import R0Oracle
from run_stage2c_cell import independent_rho
with (OUT/'inputs/workload.csv').open(encoding='utf-8') as f:apps={r['name']:r['application'] for r in csv.DictReader(f)}
oracle=R0Oracle(OUT/'inputs/registry.json',OUT/'inputs/models',apps,'medium')
composition=collections.Counter();exchanges=collections.Counter();last=None;first_cross=None;node_rounds=0
with gzip.open(OUT/'cells/medium/JOINT_CC/rounds.jsonl.gz','rt',encoding='utf-8') as f:
 for line in f:
    r=json.loads(line);d=r['decision_for_next_interval']
    if d['round']==last:continue
    last=d['round'];nodes=collections.defaultdict(list)
    for job,col in d['result']['selected'].items():
      rho0=independent_rho(oracle,job,col['config']['config_id'],6.,125.)
      row=dict(job=job,application=apps[job],cpu=col['cpu_per_gpu'],ram=col['ram_gb_per_gpu'],
        config=col['config']['config_id'],rho_ratio=col['rho']/rho0)
      for n in col['placement']:nodes[n].append(row)
    for node,rows in nodes.items():
      if len(rows)!=2:continue
      node_rounds+=1;label='+'.join(sorted(v['application'] for v in rows));composition[label]+=1
      for a in rows:
       for b in rows:
        if a['rho_ratio']>1+1e-9 and b['rho_ratio']<1-1e-9:
          exchanges[a['application']+' <- '+b['application']]+=1
          if a['application']!=b['application'] and first_cross is None:
            first_cross=dict(time=r['interval_end'],round=d['round'],node=node,receiver=a,donor=b)
save(OUT/'exchange_composition.json',dict(two_job_shared_node_rounds=node_rounds,composition=dict(composition),
    exchanges_by_receiver_donor=dict(exchanges),first_cross_application_exchange=first_cross,
    input_order='Controlled t0 CF0,DS8,CF1,DS9; subsequent CIFAR pairs every300s; FIFO service preserved.',
    inference='Co-residency is observed. Whether a different admission or placement would improve JCT is not isolated by this run.'))
print('EXCHANGE_COMPOSITION',dict(composition),dict(exchanges),flush=True)
