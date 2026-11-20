"""Replay ONE frozen cell to a new directory; never invoked by this study."""
import argparse,hashlib,importlib.util,json,shutil
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);p.add_argument('--dest',required=True);p.add_argument('--workload',required=True);p.add_argument('--name',required=True);a=p.parse_args()
 source=Path(a.artifact).resolve();dest=Path(a.dest).resolve()
 if dest.exists():raise RuntimeError('REFUSE_EXISTING_REPLAY_DESTINATION')
 prereg=json.loads((source/'preregistration.json').read_text())
 script=source/'rebuild_scripts/run_clean_ab_r2_20261030.py'
 assert hashlib.sha256(script.read_bytes()).hexdigest().upper()==prereg['runner_hash']
 assert [a.workload,a.name] in prereg['cells']
 dest.mkdir(parents=True);(dest/'cells').mkdir();shutil.copytree(source/'inputs',dest/'inputs');shutil.copy2(source/'preregistration.json',dest/'preregistration.json');shutil.copy2(source/'DESIGN.md',dest/'DESIGN.md')
 (dest/'replay_origin.json').write_text(json.dumps(dict(artifact=str(source),preregistration_sha256=hashlib.sha256((source/'preregistration.json').read_bytes()).hexdigest(),only_change='output destination; reads original frozen source'),indent=2),encoding='utf8')
 spec=importlib.util.spec_from_file_location('frozen_clean_runner',script);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.OUT=dest;m.SRC=source/'frozen_source'
 m.cell(a.workload,a.name)
if __name__=='__main__':main()
