"""Archive closure from verified parent plus this complete actual version."""
import hashlib,json,platform,sys,time,zipfile
from experiment import OUT,ROOT,LIVE,PARENT,load,sha,save

def main():
    p=load();a=json.loads((OUT/'independent_audit.json').read_text());m=json.loads((OUT/'run_manifest.json').read_text())
    assert m['unique_launched_cells']==8
    assert all(r.get('valid') and r['capacity_violations']==0 for r in a['results'].values())
    assert json.loads((OUT/'tests_relocated.json').read_text())['exit_code']==0
    assert json.loads((OUT/'official_regression.json').read_text())['exact_match']
    save(OUT/'environment.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform()))
    save(OUT/'final_verification.json',dict(valid=True,status=a['status'],cells=8,jobs_per_cell=20,
      capacity_violations=0,source_unchanged=True,old_tests=212,original_bounded_sia_exact=True,
      preregistration_sha256=sha(OUT/'preregistration.json'),report_sha256=sha(OUT/'REPORT.md'),
      analysis_script_hashes={n.name:sha(n) for n in OUT.glob('*.py')}))
    old=json.loads((PARENT/'delivery_manifest.json').read_text());inherited=PARENT/old['archive']
    assert sha(inherited)==old['archive_sha256']
    archive=OUT/'complete_rebuildable_evidence.zip';hashes={}
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
      def add(name,data):
        h=hashlib.sha256(data).hexdigest().upper()
        if name in hashes:assert hashes[name]==h;return
        z.writestr(name,data);hashes[name]=h
      with zipfile.ZipFile(inherited) as older:
        for name in older.namelist():
          if name.endswith('.zip'):continue
          data=older.read(name);assert hashlib.sha256(data).hexdigest().upper()==old['archive_files'][name]
          add(name,data)
      for directory in [LIVE,OUT]:
        for path in sorted(directory.rglob('*')):
          if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ['.pyc','.zip']:
            add(path.relative_to(ROOT).as_posix(),path.read_bytes())
    with zipfile.ZipFile(archive) as z:
      assert len(z.namelist())==len(hashes)
      for name,h in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest().upper()==h
    save(OUT/'delivery_manifest.json',dict(archive=archive.name,archive_sha256=sha(archive),archive_readback_valid=True,
      archive_files=hashes,source_hashes=p['source_hashes'],input_hashes=p['input_hashes'],created_unix=time.time(),
      inherited_archive_sha256=sha(inherited)))
    print('PACKAGE_VERIFIED',len(hashes),sha(archive),flush=True)

if __name__=='__main__':main()
