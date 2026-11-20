"""Prepare isolated correctness/freeze/audit entrypoints; does not launch."""
from pathlib import Path
R=Path(__file__).resolve().parents[1];M=R/'maintenance'
for name in ('bounded','accept','freeze_launch','audit','report','watch'):
 old=M/(name+'_core80_full_lex_v6.py');text=old.read_text(encoding='utf8').replace('full-lex-repair-v6','full-lex-repair-v7').replace('20261031_run3','20261031_run4').replace('_v6.py','_v7.py').replace('CORE80_V6_PRIMARY_BEFORE_SECONDARY_7PAIR','CORE80_V7_EXACT_STARTUP_BUDGET_7PAIR').replace('320','331')
 if name=='accept':
  text=text.replace("('FULL',45,36)","('FULL',45,41)").replace("cc_core80_full_lex_repair_20261031_run2","cc_core80_full_lex_repair_20261031_run3")
  text=text.replace("prior=frames(prior_root/'cells'/cid/'rounds.jsonl.gz',oldn)","prior=frames((prior_root/'tests/real80_FULL_45_lattice_primary/prefix.jsonl.gz') if method=='FULL' else (prior_root/'cells'/cid/'rounds.jsonl.gz'),oldn)")
  text=text.replace("new_tests=18,total_tests=331","new_tests=29,total_tests=331")
 if name=='freeze_launch':
  before="assert set(new)=={'integer_row_lattice.py','primary_band_presolve.py','sia-simulator/tests/test_rank_lattice_repair.py','sia-simulator/tests/fixtures/core80_full_rank_step19.json.gz','sia-simulator/tests/fixtures/core80_full_startup_step37.json.gz'},new"
  after="assert set(new)=={'integer_row_lattice.py','primary_band_presolve.py','capacity_cover.py','gpu_group_formulation.py','incumbent_complement.py','epsilon_lex_repair.py','sia-simulator/tests/test_rank_lattice_repair.py','sia-simulator/tests/test_capacity_cover_repair.py','sia-simulator/tests/test_epsilon_lex_repair.py','sia-simulator/tests/fixtures/core80_full_rank_step19.json.gz','sia-simulator/tests/fixtures/core80_full_startup_step37.json.gz','sia-simulator/tests/fixtures/core80_full_startup_step42.json.gz'},new"
  assert before in text;text=text.replace(before,after)
  text=text.replace('Only integer lattice row representation and directed weak-duality feasibility presolve of variables impossible at the existing primary band.', 'Exact lattice/capacity cover representation, directed weak-duality feasibility presolve, and equivalent startup budget/global-primary-bound certificates with bijective GPU group variables. The unused complement prototype is archived, never called.')
 if name=='report':
  text=text.replace('CC_CORE80_FULL_LEX_REPAIR_COMPARISON_20261031.md','CC_CORE80_FULL_LEX_EXACT_BUDGET_COMPARISON_20261031.md')
  # The new report must describe every captured blocker and its exact repair.
  anchor="add('## 2. 根因、证据与最小修复\\n')"
  text=text.replace(anchor,anchor+"\n add('本次最终修复链：旧第19轮per-job排名、V5第37轮启动次目标、V6第42轮启动次目标均保留原始失败。第42轮168184列，已验证启动可行值225，LP下界86.4117；不能把可行值当最优。V7将更优启动预算限制为220，在等价GPU分组和合法容量cover下最大化原主目标，零gap全局上界约2.5684724低于原主目标下界2.5701874，因而证明225全局最小。一般实现仅使用当前轮合法incumbent及整数gcd，预算由V-g推导，不写死225/220；超时或无证明会回到原后端/显式失败，不静默提交。\\n')")
 if name=='watch':
  text=text.replace("'prepare_core80_full_lex_v6.py'","'prepare_core80_full_lex_v7.py'")
  text=text.replace('only exact feasibility presolve and opt-in engineering-failure stopping','only exact feasibility presolve/equivalent epsilon startup proofs and opt-in engineering-failure stopping')
 target=M/(name+'_core80_full_lex_v7.py')
 with target.open('x',encoding='utf8') as f:f.write(text)
print('PIPELINE_PREPARED_NOT_LAUNCHED')
