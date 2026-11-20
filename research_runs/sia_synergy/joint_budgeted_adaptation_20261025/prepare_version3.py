"""Keep the same scientific model; only verified invalid solver vectors retried."""
import ast,json,hashlib,zipfile,shutil
from pathlib import Path
OUT=Path(__file__).resolve().parent;P=OUT/'attempt2';p=json.loads((P/'preregistration.json').read_text());WORK=Path(p['tree'])/'sia-simulator'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest().upper()
def methods(source):
    t=ast.parse(source);cls=next(v for v in t.body if isinstance(v,ast.ClassDef) and v.name=='BundleAdmission')
    return {v.name:ast.dump(v,include_attributes=False) for v in cls.body if isinstance(v,ast.FunctionDef)}
def main():
    assert json.loads((OUT/'version3_all_result.json').read_text())['exit_code']==0
    with zipfile.ZipFile(P/'source_snapshot.zip') as z:old=methods(z.read('sia-simulator/joint_prototype/bundle_adaptation.py').decode())
    new=methods((WORK/'joint_prototype/bundle_adaptation.py').read_text())
    for name in ['eligible_held_columns','_secondary_tie']:assert old[name]==new[name],'SCIENTIFIC_POLICY_CHANGED'
    assert sha(WORK/'job.py')==p['hashes'][str(WORK/'job.py')]
    for f,h in p['baseline_hashes'].items():assert sha(f)==h
    dest=OUT/'attempt3';dest.mkdir(exist_ok=False)
    p['numerical_correctness_repair']='Validate every linear row of raw AND integer-rounded optimal result; if invalid re-solve identical fingerprinted model once with presolve=False. Never change equality/capacity/lock/objective/bounds. Stop if second result invalid or unresolved.'
    for f in WORK.rglob('*.py'):p['hashes'][str(f)]=sha(f)
    for name in ['run.py','audit.py']:shutil.copyfile(OUT/name,dest/name)
    with zipfile.ZipFile(dest/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in WORK.rglob('*.py'):z.write(f,f.relative_to(WORK.parent))
    with (dest/'preregistration.json').open('x',encoding='utf-8') as f:json.dump(p,f,indent=2)
    with (dest/'repair_manifest.json').open('x',encoding='utf-8') as f:json.dump(dict(
        protected_eligibility_and_primary_secondary_rules_ast_equal=True,
        prior_attempt_files={str(v.relative_to(OUT)):sha(v) for v in P.rglob('*') if v.is_file()},
        job_py_sha256=sha(WORK/'job.py'),gate='Full8 pair and full20 pair, no previous-version metrics mixed'),f,indent=2)
    print('VERSION3_FROZEN; policy functions AST identical; numerical guard only')
if __name__=='__main__':main()
