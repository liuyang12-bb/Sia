"""Pre/post cleanup checks. Only bounded correctness probes, no experiments."""
import argparse,ast,csv,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
from inventory import ROOT,OUT,sha,save

CLEAN=ROOT/'cc-joint';SIM=CLEAN/'sia-simulator'
HISTORY=ROOT/'research_runs/sia_synergy/cc_joint_sustained_cpu_20job_20261027_run1'

def probe(args):
    sys.path.insert(0,str(CLEAN))
    if args.variant=='clean':
        from cc_factory import construct
        c=construct(args.method,args.response,OUT/'probe_inputs/inputs/workload.csv')
    else:
        spec=importlib.util.spec_from_file_location('frozen_experiment',HISTORY/'experiment.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.LIVE=HISTORY/'source';module.OUT=OUT/'probe_inputs'
        c=module.construct(args.method,args.response)
    sys.path.insert(0,str(CLEAN))
    from cc_metrics import metrics_class
    metrics=metrics_class()(c);states=[]
    for i in range(5):
        c.step(60);r=c.joint_round_logs[-1];metrics.consume(c,r)
        states.append(dict(time=c.current_time,allocations=c.allocations,
            resources={j:dict(config=col['config']['config_id'],placement=col['placement'],cpu=col['cpu_per_gpu'],ram=col['ram_gb_per_gpu'],rho=col['rho'])
                for j,col in c.policy.last_decision['result']['selected'].items()},
            jobs={j.name:dict(progress=j.progress,epoch=j.epoch,restarts=j.num_restarts,
                checkpoint=j.rescale_time,completion=j.completion_time,atomic_bsz=j.atomic_bsz,accum_steps=j.accum_steps) for j in c.jobs}))
    print('PROBE_JSON='+json.dumps(states,sort_keys=True))

def run_phase(phase):
    sys.path.insert(0,str(CLEAN));from cc_factory import verify
    verify();directory=OUT/phase;directory.mkdir(exist_ok=False)
    records=[]
    def call(label,command,cwd):
        log=directory/(label+'.log');started=time.time()
        with log.open('x',encoding='utf-8') as f:p=subprocess.run(command,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
        entry=dict(label=label,command=command,cwd=str(cwd),exit_code=p.returncode,seconds=time.time()-started,log_sha256=sha(log));records.append(entry)
        save(directory/(label+'.json'),entry)
        if p.returncode:
            save(directory/'failure.json',dict(records=records));raise RuntimeError('CHECK_FAILED:'+label)
        return log
    test=call('tests',[sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests','-p','test_*.py','-v'],SIM)
    assert 'Ran 212 tests' in test.read_text(encoding='utf-8') and '\nOK\n' in test.read_text(encoding='utf-8')
    official=[]
    for label,path in [('official',ROOT/'official-artifacts/sia-simulator'),('clean_sia',SIM)]:
        log=call(label,[sys.executable,'-X','utf8','-B',str(SIM/'check_stage1_legacy_regression.py'),'--probe',str(path)],ROOT)
        official.append(json.loads(next(l[11:] for l in log.read_text(encoding='utf-8').splitlines() if l.startswith('PROBE_JSON='))))
    assert official[0]==official[1]
    if phase=='before':
        # AST checks avoid claiming formatting extraction changed semantics.
        old=ast.parse((HISTORY/'baseline_runner.py').read_text(encoding='utf-8'))
        new=ast.parse((CLEAN/'cc_metrics.py').read_text(encoding='utf-8'))
        f1=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='metrics_class')
        f2=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='metrics_class')
        assert ast.dump(f1,include_attributes=False)==ast.dump(f2,include_attributes=False),'Metrics AST changed'
        pairs=[]
        for method in ['SIA_P','SIA_ECT_AUX','JOINT_CC_P','JOINT_CC']:
          for response in ['neutral','medium']:
            variants=[]
            for variant in ['frozen','clean']:
              label=method+'_'+response+'_'+variant
              command=[sys.executable,'-X','utf8','-B',str(Path(__file__)),'probe','--variant',variant,'--method',method,'--response',response]
              log=call(label,command,SIM)
              variants.append(json.loads(next(l[11:] for l in log.read_text(encoding='utf-8').splitlines() if l.startswith('PROBE_JSON='))))
            assert variants[0]==variants[1],(method,response)
            pairs.append(dict(method=method,response=response,exact=True,jobs=2,rounds=5))
        save(directory/'bounded_equivalence.json',dict(valid=True,training_experiment=False,pairs=pairs))
    save(directory/'summary.json',dict(valid=True,unchanged_tests=212,original_sia_bounded_exact=True,records=records))
    print('CHECKS_PASS',phase,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['before','after','probe']);p.add_argument('--variant');p.add_argument('--method');p.add_argument('--response');a=p.parse_args()
    if a.mode=='probe':probe(a)
    else:run_phase(a.mode)
