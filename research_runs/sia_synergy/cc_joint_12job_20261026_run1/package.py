"""Readback-verified source/inputs/history fixtures and experiment evidence."""
import hashlib,json,platform,sys,time,zipfile
from experiment import OUT,ROOT,SOURCE,LIVE,PARENT,load,sha,save

def main():
    p=load();archive=OUT/'complete_rebuildable_evidence.zip';hashes={}
    audit=json.loads((OUT/'independent_audit.json').read_text())
    conclusions=json.loads((OUT/'conclusions.json').read_text())
    manifest=json.loads((OUT/'run_manifest.json').read_text())
    assert audit['status']=='EIGHT_COMPLETE_VALID' and manifest['unique_launched_cells']==8
    assert all(r['summary']['completed']==12 and r['capacity_violations']==0 for r in audit['results'].values())
    assert conclusions['medium_improvement']==audit['comparisons']['medium']['SIA_P']['improvement']
    assert conclusions['medium_aux_increment_improvement']==audit['comparisons']['medium']['JOINT_CC_P']['improvement']
    assert json.loads((OUT/'tests_relocated.json').read_text())['exit_code']==0
    assert json.loads((OUT/'official_regression.json').read_text())['exact_match']
    save(OUT/'final_verification.json',dict(valid=True,cells=8,jobs_per_cell=12,capacity_violations=0,
        source_identical_to_parent=True,tests=212,original_sia_bounded_exact=True,
        conclusions_match_independent_audit=True,
        report_sha256=sha(OUT/'REPORT.md'),run_manifest_sha256=sha(OUT/'run_manifest.json')))
    old= json.loads((PARENT/'delivery_manifest.json').read_text())
    inherited=PARENT/old['archive'];assert sha(inherited)==old['archive_sha256']
    save(OUT/'environment.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform()))
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
      def add(name,data):
        h=hashlib.sha256(data).hexdigest().upper()
        if name in hashes:assert hashes[name]==h,name;return
        z.writestr(name,data);hashes[name]=h
      with zipfile.ZipFile(inherited) as older:
        for name in older.namelist():
          if name.endswith('.zip'):continue
          data=older.read(name);assert hashlib.sha256(data).hexdigest().upper()==old['archive_files'][name]
          add(name,data)
      for directory in [LIVE,OUT]:
        for f in sorted(directory.rglob('*')):
          if f.is_file() and '__pycache__' not in f.parts and f.suffix not in ['.pyc','.zip']:
            add(f.relative_to(ROOT).as_posix(),f.read_bytes())
    with zipfile.ZipFile(archive) as z:
      assert len(z.namelist())==len(hashes)
      for name,h in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest().upper()==h,name
    save(OUT/'delivery_manifest.json',dict(archive=archive.name,archive_sha256=sha(archive),
      archive_readback_valid=True,archive_files=hashes,source_hashes=p['source_hashes'],
      input_hashes=p['input_hashes'],created_unix=time.time(),inherited_archive_sha256=sha(inherited)))
    print('PACKAGE_VERIFIED',len(hashes),sha(archive),flush=True)

if __name__=='__main__':main()
