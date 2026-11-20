"""Complete source/input/fixture snapshot and raw evidence; no source editing."""
import hashlib,json,platform,sys,time,zipfile
from pathlib import Path
from experiment import OUT,ROOT,LIVE,PREVIOUS,sha,save,load

def main():
    load()
    save(OUT/'environment.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform()))
    hashes={};archive=OUT/'complete_rebuildable_evidence.zip'
    inherited=OUT.parent/'joint_completion_objective_20261026_run1/rebuildable_source_inputs_complete.zip'
    prior=OUT.parent/'joint_completion_continuity_20261026_run1'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
      def add(path,name=None):
        name=name or path.relative_to(ROOT).as_posix()
        if name in hashes:return
        data=path.read_bytes();z.writestr(name,data);hashes[name]=hashlib.sha256(data).hexdigest().upper()
      # Historical test fixture dependencies, read-only official inputs, no
      # unrelated historical result archive or superseded executable source.
      with zipfile.ZipFile(inherited) as old:
        for name in old.namelist():
          if name.startswith('sia-joint-completion-time/') or '/joint_completion_objective_20261026_run1/' in name or name.endswith('.zip'):continue
          data=old.read(name);z.writestr(name,data);hashes[name]=hashlib.sha256(data).hexdigest().upper()
      for directory in [LIVE,OUT,prior]:
        for f in directory.rglob('*'):
          if f.is_file() and '__pycache__' not in f.parts and f.suffix not in ['.pyc','.zip']:
            add(f)
      for method in ['SIA_P','SIA_ECT_AUX','JOINT_ECT']:
        for name in ['rounds.jsonl.gz','summary.json']:add(PREVIOUS/'cells'/method/name)
      for name in ['preregistration.json','RESULTS_AND_CAUSES.md']:add(PREVIOUS/name)
    # Independent archive readback verifies byte preservation for every member.
    with zipfile.ZipFile(archive) as z:
      assert len(z.namelist())==len(hashes)
      for name,h in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest().upper()==h,name
    save(OUT/'delivery_manifest.json',dict(archive=archive.name,archive_sha256=sha(archive),
        archive_files=hashes,archive_readback_valid=True,created_unix=time.time(),
        current_source_hashes=json.loads((OUT/'source_changes.json').read_text())['source_hashes'],
        warning='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE'))
    print('PACKAGE_VERIFIED',len(hashes),sha(archive))

if __name__=='__main__':main()
