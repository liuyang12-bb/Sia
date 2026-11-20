"""Select the original 61 tests without changing their source or assertions."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path.cwd()))
suite = unittest.defaultTestLoader.discover('tests', pattern='test_*.py')
def leaves(test):
    if isinstance(test, unittest.TestSuite):
        for child in test:
            yield from leaves(child)
    else:
        yield test
old = unittest.TestSuite(t for t in leaves(suite) if not t.id().startswith('test_stage1p1_correctness.'))
assert old.countTestCases() == 61
sys.exit(not unittest.TextTestRunner(verbosity=2).run(old).wasSuccessful())
