"""Independent checks, then report. Never starts a training process."""
import subprocess,sys,time
from experiment import OUT,save,sha

stages=['audit.py','cross_round_commit_audit.py','mechanism_analysis.py','exchange_composition.py',
        'outcome_analysis.py','observed_opportunities.py','allocation_opportunities.py','report.py']
records=[]
for stage in stages:
    command=[sys.executable,'-X','utf8','-B',str(OUT/stage)];started=time.time()
    log=OUT/(stage.replace('.py','')+'_analysis.log')
    with log.open('x',encoding='utf-8') as f:p=subprocess.run(command,cwd=OUT,stdout=f,stderr=subprocess.STDOUT)
    records.append(dict(stage=stage,command=command,exit_code=p.returncode,seconds=time.time()-started,log_sha256=sha(log)))
    print('ANALYSIS',stage,p.returncode,flush=True)
    if p.returncode:
      save(OUT/'analysis_failure.json',dict(records=records));raise SystemExit(p.returncode)
save(OUT/'analysis_manifest.json',dict(valid=True,training_run=False,records=records))
