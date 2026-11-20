"""Audit-only correctness repair; preserve failed attempt and rerun full pairs."""
import json,hashlib,zipfile,shutil
from pathlib import Path
OUT=Path(__file__).resolve().parent;p=json.loads((OUT/'preregistration.json').read_text());WORK=Path(p['tree'])/'sia-simulator'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest().upper()
def main():
    assert json.loads((OUT/'boundary_red_result.json').read_text())['exit_code']!=0
    assert json.loads((OUT/'boundary_green_all_result.json').read_text())['exit_code']==0
    assert sha(WORK/'job.py')==p['hashes'][str(WORK/'job.py')],'TRAINING_SEMANTICS_CHANGED'
    for f,h in p['baseline_hashes'].items():assert sha(f)==h,'BASELINE_CHANGED:'+f
    for f,h in p['hashes'].items():
        if Path(f).is_relative_to(WORK) and Path(f).name=='execution_audit.py':continue
        assert sha(f)==h,'UNAUTHORIZED_PARAMETER_OR_CODE_CHANGE:'+f
    dest=OUT/'attempt2';dest.mkdir(exist_ok=False)
    p['correctness_repair']='Requested interval vs completion-truncated observed interval in checkpoint audit; original Job.step/JCT unchanged. Segment ordering strengthened; bounded zero-duration epoch settlement independently reproduced by untouched official Job.'
    p['prior_attempt_directory']=str(OUT)
    for f in WORK.rglob('*.py'):p['hashes'][str(f)]=sha(f)
    for name in ['run.py','audit.py']:shutil.copyfile(OUT/name,dest/name)
    with zipfile.ZipFile(dest/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in WORK.rglob('*.py'):z.write(f,f.relative_to(WORK.parent))
    with (dest/'preregistration.json').open('x',encoding='utf-8') as f:json.dump(p,f,indent=2)
    with (dest/'repair_manifest.json').open('x',encoding='utf-8') as f:json.dump(dict(
        change='audit only, no new JOINT-G algorithm parameters',
        job_py_sha256=sha(WORK/'job.py'),
        first_attempt_files={str(v.relative_to(OUT)):sha(v) for name in ['n8_neutral_joint_g','n8_medium_joint_g'] for v in (OUT/name).iterdir() if v.is_file()},
        rule='rerun both8 then both20 in new directory; fixed same inputs and design; stop on failure'),f,indent=2)
    print('REPAIR_FROZEN; full pairs rerun, no algorithm retuning')
if __name__=='__main__':main()
