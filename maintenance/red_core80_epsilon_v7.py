"""Run the new contract against the preserved pre-repair backend."""
import importlib.util,sys,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];T=R/'cc-joint-core80-full-lex-repair-v7'
sys.path[:0]=[str(T),str(T/'sia-simulator'),str(T/'sia-simulator/tests')]
import engineering_repair
spec=importlib.util.spec_from_file_location('preserved_v6_numeric',R/'cc-joint-core80-full-lex-repair-v6/numeric_repair.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
with patch.object(engineering_repair,'canonical_bundle',old.canonical_bundle):
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_epsilon_lex_repair.EpsilonBackendRegression'))
sys.exit(not result.wasSuccessful())
