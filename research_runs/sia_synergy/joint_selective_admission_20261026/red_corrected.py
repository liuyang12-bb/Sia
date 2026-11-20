"""Same corrected fixtures, explicitly exercise the preserved old admission."""
import sys,unittest,json,time
from pathlib import Path
OUT=Path(__file__).resolve().parent;WORK=OUT.parents[2]/'sia-joint-selective-admission/sia-simulator'
sys.path[:0]=[str(WORK),str(WORK/'tests')]
import test_selective_admission as test
test.SelectiveAdmission=test.BundleAdmission
start=time.perf_counter()
with (OUT/'red_corrected.log').open('x') as f:
    r=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test))
with (OUT/'red_corrected.json').open('x') as f:json.dump(dict(tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),seconds=time.perf_counter()-start),f)
assert len(r.failures)==4 and not r.errors
