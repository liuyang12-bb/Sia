import collections,gzip,hashlib,json,math,sys
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];p=json.loads((OUT/'preregistration.json').read_text(encoding='utf-8'))
from audit import audit
def save(name,v):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def compare(base,new):
    assert set(base['jobs'])==set(new['jobs']);complete=base['completed']==base['total'] and new['completed']==new['total']
    pair=dict(baseline=base['method'],candidate=new['method'],complete_pair=complete)
    if not complete:
        pair.update(baseline_completion=base['completion_rate'],candidate_completion=new['completion_rate'],baseline_rmst=base['restricted_mean_jct'],candidate_rmst=new['restricted_mean_jct']);return pair
    diffs={j:new['jobs'][j]['jct']-base['jobs'][j]['jct'] for j in base['jobs']};net=sum(diffs.values());best=min(diffs,key=diffs.get);worst=max(diffs,key=diffs.get)
    delta={k:sum(v[k] for v in new['jobs'].values())-sum(v[k] for v in base['jobs'].values()) for k in ['wait','checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours','restarts','wasted_gpu_seconds']}
    assert math.isclose(net,sum(delta[k] for k in ['wait','checkpoint','training']),abs_tol=1e-5)
    pair.update(mean_delta=new['mean_jct']-base['mean_jct'],improvement_percent=100*(base['mean_jct']-new['mean_jct'])/base['mean_jct'],
        improved=sum(v<0 for v in diffs.values()),worse=sum(v>0 for v in diffs.values()),same=sum(v==0 for v in diffs.values()),per_job_delta=diffs,totals_delta=delta,
        best_job=best,best_delta=diffs[best],worst_job=worst,worst_delta=diffs[worst],best_share_of_net=diffs[best]/net if net else None,
        diagnostic_without_best=dict(baseline_mean=(sum(v['jct'] for v in base['jobs'].values())-base['jobs'][best]['jct'])/(len(diffs)-1),candidate_mean=(sum(v['jct'] for v in new['jobs'].values())-new['jobs'][best]['jct'])/(len(diffs)-1)),
        per_job_time_deltas={j:{k:new['jobs'][j][k]-base['jobs'][j][k] for k in ['wait','checkpoint','training','restarts','wasted_gpu_seconds']} for j in diffs})
    return pair
def main():
    results=[];valid=[];pairs=[]
    for f,h in p['hashes'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest().upper()==h,f
    for f,h in p['baseline_hashes'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest().upper()==h,f
    for c in p['reference_cells']:
        dest=OUT/'reference_audits'/f"{c['dataset']}_{c['response']}_{c['method']}";v=audit(c,c['folder'],dest)
        s=json.loads((Path(c['folder'])/'summary.json').read_text());s.update(dataset=c['dataset'],response=c['response'],method=c['method']);results.append(s);valid.append(v)
    for c in p['runs']:
        folder=OUT/c['id'];v=json.loads((folder/'validation.json').read_text());s=json.loads((folder/'summary.json').read_text());results.append(s);valid.append(v)
    for ds in ['original8','original20','holdout20']:
      for r in ['neutral','medium']:
        data={s['method']:s for s in results if s['dataset']==ds and s['response']==r}
        assert set(data)=={'sia_p','sia_aux','b3_free','joint_g'}
        for baseline,candidate in [('sia_p','b3_free'),('sia_aux','b3_free'),('joint_g','b3_free'),('sia_aux','joint_g'),('sia_p','sia_aux')]:
            pairs.append(dict(dataset=ds,response=r,**compare(data[baseline],data[candidate])))
    save('comparison_summary.json',dict(results=results,pairs=pairs))
    save('validation_summary.json',dict(valid=all(v['valid'] for v in valid),cells=len(valid),complete_cells=sum(v['all_complete'] for v in valid),counts=dict(sum((collections.Counter(v['counts']) for v in valid),collections.Counter())),source_input_hashes=len(p['hashes']),history_hashes=len(p['baseline_hashes']),cells_audit=valid))
    report=['# B3 版本桥接实验：结果表','','SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。','','本报告固定旧B3自由决策核心、当前R0与修复后执行引擎；不是历史旧配置重现。SIA-AUX为本项目辅助MIP，不是Synergy-TUNE。','',
      '|数据/响应|SIA-P|SIA-AUX|B3-FREE|JOINT-G|FREE对P|FREE对AUX|FREE对G|','|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v):return '删失' if v is None else f'{v:.3f}'
    for ds in ['original8','original20','holdout20']:
      for r in ['neutral','medium']:
        d={s['method']:s for s in results if s['dataset']==ds and s['response']==r}
        ratios=[next(x for x in pairs if x['dataset']==ds and x['response']==r and x['baseline']==b and x['candidate']=='b3_free').get('improvement_percent') for b in ['sia_p','sia_aux','joint_g']]
        report.append('|'+ds+'/'+r+'|'+'|'.join(fmt(d[m]['mean_jct']) for m in ['sia_p','sia_aux','b3_free','joint_g'])+'|'+'|'.join(fmt(x)+'%' if x is not None else '不比较' for x in ratios)+'|')
    report+=['','## 完整指标与完成率','','|数据/响应/策略|完成|median|P95|P99|makespan|GPU-h|CPU-h|RAM-h|等待s|CPs|训练s|restart|waste GPU-s|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in results:
        totals={k:sum(v[k] for v in s['jobs'].values()) for k in ['gpu_hours','cpu_core_hours','ram_gb_hours','wait','checkpoint','training','restarts','wasted_gpu_seconds']}
        report.append('|'+s['dataset']+'/'+s['response']+'/'+s['method']+f"|{s['completed']}/{s['total']}|"+'|'.join(fmt(s[k]) for k in ['median_jct','p95_jct','p99_jct','makespan'])+'|'+'|'.join(fmt(v) for v in totals.values())+'|')
    report+=['','## 逐作业JCT','','主分析保留全部作业；P95/P99每格仅8或20作业，不作总体尾部高精度估计。']
    for ds in ['original8','original20','holdout20']:
      for r in ['neutral','medium']:
        d={s['method']:s for s in results if s['dataset']==ds and s['response']==r}
        report+=['',f'### {ds}/{r}','','|Job|SIA-P|SIA-AUX|B3-FREE|G|FREE−AUX|FREE−G|','|---|---:|---:|---:|---:|---:|---:|']
        for j in d['sia_p']['jobs']:
            vals=[d[m]['jobs'][j]['jct'] for m in ['sia_p','sia_aux','b3_free','joint_g']]
            report.append('|'+j+'|'+'|'.join(fmt(v) for v in vals)+'|'+fmt(vals[2]-vals[1] if vals[2] is not None and vals[1] is not None else None)+'|'+fmt(vals[2]-vals[3] if vals[2] is not None and vals[3] is not None else None)+'|')
    with (OUT/'RESULT_TABLES.md').open('x',encoding='utf-8') as f:f.write('\n'.join(report)+'\n')
    print('COMPARISONS',len(results),len(pairs))
if __name__=='__main__':main()
