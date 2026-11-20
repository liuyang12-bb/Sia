"""Legacy tests require workspace-level fixture paths; retain first failure log."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_runs/sia_synergy/cc_cifar_bert12_20261029_run1'
def main():
 first=json.loads((OUT/'tests.json').read_text());assert first['exit_code']!=0
 p=json.loads((OUT/'preregistration.json').read_text());canonical=ROOT/'cc-joint-breakpoint-precision'
 for n,h in p['source_hashes'].items():assert hashlib.sha256((canonical/n).read_bytes()).hexdigest().upper()==h,n
 text=(ROOT/'maintenance/gate_cifar_bert12_20261029.py').read_text()
 text=text.replace("'offline_boundary_coverage.json'","'offline_boundary_coverage_repeat.json'").replace("'tests.log'","'tests_canonical.log'").replace("'tests.json'","'tests_canonical.json'")
 text=text.replace("cwd=SRC/'sia-simulator'","cwd=ROOT/'cc-joint-breakpoint-precision/sia-simulator'")
 exec(compile(text,str(ROOT/'maintenance/gate_cifar_bert12_20261029.py'),'exec'),{'__file__':str(ROOT/'maintenance/gate_cifar_bert12_20261029.py'),'__name__':'__main__'})
if __name__=='__main__':main()
