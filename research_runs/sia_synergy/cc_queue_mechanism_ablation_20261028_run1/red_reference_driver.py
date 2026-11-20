"""Show that old CC semantics fail the newly required off-switch contracts."""
import sys,types,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[3]
SIM=ROOT/'cc-joint-queue-ablation/sia-simulator'
sys.path[:0]=[str(SIM/'tests'),str(SIM)]
import test_queue_factorial as t
from joint_prototype.tail_recovery import TailRecoveryAdmission
old=types.SimpleNamespace(QueueFactorAdmission=TailRecoveryAdmission)
suite=unittest.TestSuite(t.QueueFactorialTests(n) for n in [
 'test_fifo_factor_requires_oldest_even_when_later_has_higher_score',
 'test_no_continuity_releases_only_after_300_productive_seconds'])
with patch.object(t,'impl',return_value=old):
 result=unittest.TextTestRunner(verbosity=2).run(suite)
assert len(result.failures)==2 and not result.errors,'RED_REFERENCE_MUST_FAIL_BOTH_CONTRACTS'
print('EXPECTED_OLD_BEHAVIOR_FAILURES_CONFIRMED_2')
