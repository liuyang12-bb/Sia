"""READ-ONLY, stdlib-only analysis of frozen logs. No simulator imports/calls."""
import ast, bisect, collections, gzip, hashlib, itertools, json, math, statistics, sys, time
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
OLD=OUT.parent/'cc_uncertainty_20261101_run1'
SOURCE=ROOT/'cc-joint-uncertainty-v4'

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for z in iter(lambda:f.read(1048576),b''):h.update(z)
    return h.hexdigest().upper()
def seal_check(root,manifest):
    m=read(manifest);bad=[]
    for name,value in m.items():
        p=root/name
        if not p.is_file() or sha(p)!=value:bad.append(name)
    return dict(root=str(root),manifest=str(manifest),count=len(m),mismatches=bad)
def rounds(method):
    with gzip.open(OLD/'cells'/method/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
        for n,l in enumerate(f,1):yield n,json.loads(l)
def pct(vals,q):
    if not vals:return None
    a=sorted(vals);x=(len(a)-1)*q;i=int(x);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
def numeric_summary(a):
    return dict(n=len(a),mean=statistics.mean(a) if a else None,median=pct(a,.5),p95=pct(a,.95),min=min(a) if a else None,max=max(a) if a else None)
def plan_geometry(cols):return tuple(sorted(cols.items()))

def scan(method):
    rows=[];unknown=[];proposals=[];selected_states={};last=None;final={};past=[]
    initial_selected=set()
    for line,r in rounds(method):
        final=r['job_end_state'];d=r['decision_for_next_interval']
        if last==d['round']:continue
        last=d['round'];a=d['result']['audit'];ev=a.get('release_completion',{})
        if not ev:continue
        t=r['interval_end'];selected=d['result']['selected'];spec={s['job_id']:s for s in d['probe']['specs']}
        unk=ev.get('baseline',{}).get('unknown_jobs',[]);ready=a.get('ready',[])
        locks=a.get('checkpoint_locks',[])
        row=dict(method=method,line=line,time=t,round=d['round'],active=len(spec),selected=len(selected),waiting=len(set(spec)-set(selected)),
                 ready=ready,locked=locks,unknown_jobs=unk,unknown_selected=sorted(set(unk)&set(selected)),status=ev['status'],
                 reason=ev.get('uncertainty',{}).get('reason'),plans=len(ev.get('plans',[])),candidate_count=d['probe']['joint_columns_count'],
                 selected_jobs=sorted(selected),required=a.get('required_jobs'),optional=a.get('optional_jobs'),drain=a.get('drain'),
                 raw_scorable_selected_proven_by_precheck=bool(unk and not(set(unk)&set(selected))))
        rows.append(row)
        for j in unk:
            state=d['probe']['job_state'][j]
            unknown.append(dict(method=method,time=t,job=j,selected=j in selected,used_gpu_seconds=state['used_gpu_seconds'],
                observed_progress=spec[j]['observed_progress'],sources=d['candidate_summary'][j]['goodput_sources'],
                config_ids=d['candidate_summary'][j]['config_ids'],checkpoint=spec[j]['checkpoint_locked']))
        if method=='NEW_FULL':
            criteria={
                'first_all_unknown_waiting':bool(unk and not set(unk)&set(selected) and selected),
                'first_checkpoint_and_waiting':bool(locks and set(spec)-set(selected)),
                'first_completion_and_waiting':bool(any(x['completion_time'] is not None and r['interval_start']<x['completion_time']<=t for x in r['execution']) and set(spec)-set(selected))}
            for key,ok in criteria.items():
                if ok and key not in selected_states:selected_states[key]=r
        plans=ev.get('plans',[])
        if plans:
            base=plans[0]
            for p in plans:
                e=p['evaluation'];different=plan_geometry(p['columns'])!=plan_geometry(base['columns'])
                z=dict(method=method,time=t,name=p['name'],different_from_legacy=different,status=e['status'],columns=p['columns'],
                       static_sum_eta=p['static_sum_eta'],legacy_static_sum_eta=base['static_sum_eta'],sum_completion=e.get('sum_completion'),
                       legacy_sum_completion=base['evaluation'].get('sum_completion'))
                if e['status']=='OK':z['predicted_gain_vs_legacy']=base['evaluation']['sum_completion']-e['sum_completion']
                proposals.append(z)
                if method=='NEW_FULL' and p['name']=='MIN_STATIC_SUM_ETA' and different and len([k for k in selected_states if k.startswith('different_eta_')])<3:
                    selected_states['different_eta_'+str(t)]=r
            if spec:
                past.append(dict(time=t,round=d['round'],predicted=base['evaluation'].get('completion_times',{})))
    errors=[]
    for z in past:
        for j,pred in z['predicted'].items():
            actual=final[j]['completion']-z['time']
            errors.append(dict(method=method,time=z['time'],job=j,predicted_remaining=pred,actual_remaining=actual,
                               error_seconds=pred-actual,ape=abs(pred-actual)/actual if actual>0 else None,
                               source='FROZEN_BASELINE_FORECAST_VS_LATER_EXISTING_COMPLETION;HINDSIGHT_VALIDATION_ONLY'))
    valid=[r for r in rows if r['active']]
    summary=dict(decisions=len(valid),status_counts=dict(collections.Counter(x['status'] for x in valid)),
        reason_counts=dict(collections.Counter(x['reason'] for x in valid)),
        unknown_all_waiting=sum(x['raw_scorable_selected_proven_by_precheck'] for x in valid),
        unknown_job_rounds=len(unknown),unknown_selected=sum(x['selected'] for x in unknown),
        unknown_with_effective_training=sum(x['used_gpu_seconds']>0 for x in unknown),
        unknown_sources=dict(collections.Counter(s for x in unknown for s in x['sources'])),
        unknown_first=min((x['time'] for x in valid if x['unknown_jobs']),default=None),
        unknown_last=max((x['time'] for x in valid if x['unknown_jobs']),default=None),
        distinct_proposal_rounds={name:sum(p['name']==name and p['different_from_legacy'] for p in proposals) for name in ['MIN_STATIC_SUM_ETA','CURRENT_ELIGIBLE_BUNDLE']},
        positive_proposed_gains=sum(p.get('predicted_gain_vs_legacy',0)>1e-7 for p in proposals),
        baseline_forecast_error=numeric_summary([x['error_seconds'] for x in errors]),
        baseline_forecast_ape=numeric_summary([x['ape'] for x in errors if x['ape'] is not None]))
    return summary,rows,unknown,proposals,errors,selected_states

CPU=[1,2,3,4,5,6,9,12,24];RAM=[20.83,62.5,125,187.5,250]
def matrix(template):
    tree=ast.parse((SOURCE/'inputs/models'/template).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Attribute) and node.targets[0].attr=='tput':
            return ast.literal_eval(node.value.args[0])
    raise ValueError('MISSING_MATRIX')
def interp(mat,x,y):
    def b(axis,z):
        z=min(max(z,axis[0]),axis[-1]);hi=bisect.bisect_right(axis,z)
        if hi==len(axis):return hi-1,hi-1,0.
        lo=hi-1;return lo,hi,(z-axis[lo])/(axis[hi]-axis[lo])
    i,k,u=b(CPU,x);j,l,v=b(RAM,y)
    return (mat[i][j]*(1-v)+mat[i][l]*v)*(1-u)+(mat[k][j]*(1-v)+mat[k][l]*v)*u
def response(mat,cfg,c,m):
    pc,pm=cfg['cpu_demand_scale'],cfg['memory_demand_scale'];x,y=c/pc,m/pm;bx,by=3/pc,62.5/pm
    rho=(interp(mat,x,y)/interp(mat,bx,by))**.5
    return dict(rho=rho,numerator_clipped=not(1<=x<=24 and 20.83<=y<=250),denominator_clipped=not(1<=bx<=24 and 20.83<=by<=250),
                coordinates=[x,y],baseline_coordinates=[bx,by])

def fixed_gpu_enumeration(r,labels):
    """Enumerate ONLY logged eligible quotas, no new GPU proposal/online model."""
    d=r['decision_for_next_interval'];a=d['result']['audit'];selected=d['result']['selected'];specs={s['job_id']:s for s in d['probe']['specs']}
    bynode=collections.defaultdict(list);data={};checks=[]
    for j,c in selected.items():
        spec=specs[j];cfg=c['config'];mat=matrix(spec['template_file']);placement=tuple(sorted(c['placement'].items()))
        assert len(placement)==1,'This offline enumerator handles observed single-node jobs only'
        current=(c['cpu_per_gpu'],c['ram_gb_per_gpu']);opts=set()
        elig=a.get('shared_eligibility',{}).get(j,{}).get('eligible_geometries',[])
        for text in elig:
            job,cid,p,cpu,ram=ast.literal_eval(text);p=ast.literal_eval(p) if isinstance(p,str) else p
            if cid==cfg['config_id'] and tuple(p)==placement:opts.add((cpu,ram))
        logged_member=any(abs(c-current[0])<1e-9 and abs(m-current[1])<1e-9 for c,m in opts)
        if spec['checkpoint_locked']:opts={current}
        else:opts.add(current) # selected audited incumbent is a valid option even if summary omits it
        opts=sorted(opts);rr=response(mat,cfg,*current);assert math.isclose(rr['rho'],c['rho'],rel_tol=1e-10,abs_tol=1e-10),(j,rr,c)
        options=[dict(cpu=x,ram=y,**response(mat,cfg,x,y)) for x,y in opts]
        data[j]=dict(job=j,config=cfg['config_id'],gpus=cfg['gpus'],placement=list(placement),locked=spec['checkpoint_locked'],
                    current=dict(cpu=current[0],ram=current[1],**rr),options=options,eligible_count=len(options),incumbent_in_eligible_log=logged_member)
        bynode[placement[0][0]].append(j)
    nodes=[]
    for node,jobs in sorted(bynode.items()):
        # actual frozen experiment has 2 GPUs, 6 CPU, 125 GB on each node
        g=sum(data[j]['gpus'] for j in jobs);assert g<=2
        candidates=itertools.product(*(data[j]['options'] for j in jobs));n=pareto=0
        best=None;best_value=-math.inf;best_pareto=None;bp=-math.inf;valid_supported=0
        for opts in candidates:
            cpu=sum(data[j]['gpus']*o['cpu'] for j,o in zip(jobs,opts));ram=sum(data[j]['gpus']*o['ram'] for j,o in zip(jobs,opts))
            if cpu>6+1e-7 or ram>125+1e-7:continue
            n+=1;ratios=[o['rho']/data[j]['current']['rho'] for j,o in zip(jobs,opts)];value=sum(ratios)
            supported=all(not(o['numerator_clipped'] or o['denominator_clipped']) for o in opts)
            valid_supported+=supported
            rec=dict(allocation={j:o for j,o in zip(jobs,opts)},cpu=cpu,ram=ram,relative_rates=dict(zip(jobs,ratios)),all_supported=supported,
                     equal_weight_relative_rate_sum=value,receiver_gain=sum(max(0,x-1) for x in ratios),donor_loss=sum(max(0,1-x) for x in ratios))
            if value>best_value+1e-10:best_value=value;best=rec
            if all(x>=1-1e-10 for x in ratios) and any(x>1+1e-9 for x in ratios):
                pareto+=1
                if value>bp+1e-10:bp=value;best_pareto=rec
        assert n>0
        nodes.append(dict(node=node,jobs=jobs,gpus=g,feasible_combinations=n,fully_supported_combinations=valid_supported,
                          strict_pareto_combinations=pareto,best_pareto=best_pareto,best_relative_speed=best))
    return dict(time=r['interval_end'],labels=labels,scope='FIXED_SERVICE_SET_GPU_PLACEMENT;LOGGED_QUOTA_DOMAIN;NO_TRAINING_OR_SOLVER',
                selected_jobs={j:{k:v for k,v in z.items() if k!='options'} for j,z in data.items()},nodes=nodes,
                any_strict_pareto=any(n['strict_pareto_combinations'] for n in nodes),
                interpretation='relative speed sum is descriptive, not an implemented objective, global progress measure, or JCT prediction; lock held quotas unchanged')

def calibration(method):
    with gzip.open(OLD/'cells'/method/'calibration_records.json.gz','rt',encoding='utf-8') as f:rows=json.load(f)
    result=[]
    for i,z in enumerate(rows):
        assert math.isclose(z['ratio'],z['observed_raw']/z['predicted_raw'],rel_tol=1e-12)
        past=[p for p in rows if p['key']==z['key'] and p['origin']<z['origin'] and p['observed_end']<=z['origin'] and p['observed_at']<=z['origin']]
        supported=len({p['origin'] for p in past})>=2
        low=min([1]+[p['ratio'] for p in past]) if supported else None
        high=max([1]+[p['ratio'] for p in past]) if supported else None
        result.append(dict(method=method,job=z['job'],origin=z['origin'],key=z['key'],ratio=z['ratio'],
             absolute_percentage_error=abs(z['predicted_raw']-z['observed_raw'])/z['observed_raw'],
             prior_labels=len(past),prior_origins=len({p['origin'] for p in past}),supported=supported,low=low,high=high,
             covered=(low-1e-12<=z['ratio']<=high+1e-12) if supported else None))
    summary=dict(rows=len(rows),strata=len({z['key'] for z in rows}),prequential_supported=sum(z['supported'] for z in result),
                 prequential_covered=sum(z['covered'] is True for z in result),ape=numeric_summary([z['absolute_percentage_error'] for z in result]),
                 warning='observed-config labels only; repeated correlated intervals, NOT unseen-candidate coverage or confidence calibration')
    return summary,result

def main():
    begin=time.perf_counter()
    seals=[seal_check(SOURCE,OLD/'source_hashes.json'),seal_check(OLD,OLD/'artifact_hashes.json')]
    assert all(not s['mismatches'] for s in seals),seals
    save('integrity_before.json',seals)
    summaries={};allrows=[];unknown=[];proposals=[];errors=[];states={};cal={};calrows=[]
    for method in ('NEW_FULL','NEW_FO'):
        s,r,u,p,e,ss=scan(method);summaries[method]=s;allrows+=r;unknown+=u;proposals+=p;errors+=e
        if method=='NEW_FULL':states=ss
        cal[method],cr=calibration(method);calrows+=cr
    save('availability_summary.json',summaries);save('availability_rounds.json',allrows);save('unknown_job_rounds.json',unknown)
    save('proposal_coverage.json',proposals);save('baseline_prediction_errors.json',errors)
    save('prequential_calibration_summary.json',cal);save('prequential_calibration_rows.json',calrows)
    unique={}
    for label,r in states.items():unique.setdefault(r['interval_end'],dict(frame=r,labels=[]))['labels'].append(label)
    assert len(unique)<=6
    save('selected_state_receipt.json',[dict(time=t,criteria=v['labels'],round=v['frame']['decision_for_next_interval']['round']) for t,v in sorted(unique.items())])
    enum=[]
    for t,v in sorted(unique.items()):enum.append(fixed_gpu_enumeration(v['frame'],v['labels']))
    save('fixed_gpu_quota_enumeration.json',enum)
    save('diagnostic_validation.json',dict(status='PASS',source_matches=True,prior_artifacts_match=True,
         algorithm_imports=False,simulation_calls=0,solver_calls=0,selected_states=len(enum),
         independent_rho_recalculation_matches_logged_selected=True,capacity_and_lock_assertions=True,
         elapsed_seconds=time.perf_counter()-begin))
    print(json.dumps(dict(availability=summaries,calibration=cal,states=[dict(time=x['time'],pareto=x['any_strict_pareto'],combos=sum(y['feasible_combinations'] for y in x['nodes'])) for x in enum],seconds=time.perf_counter()-begin),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
