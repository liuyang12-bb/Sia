"""Prepare isolated V8 acceptance/freeze/audit scripts, without launching."""
from pathlib import Path
R=Path(__file__).resolve().parents[1];M=R/'maintenance'
for name in ('bounded','accept','freeze_launch','audit','report','watch'):
 text=(M/(name+'_core80_full_lex_v7.py')).read_text(encoding='utf8').replace('full-lex-repair-v7','full-lex-repair-v8').replace('20261031_run4','20261031_run5').replace('_v7.py','_v8.py').replace('CORE80_V7_EXACT_STARTUP_BUDGET_7PAIR','CORE80_V8_EXACT_INTEGER_LEX_BUDGET_7PAIR').replace('331','335')
 if name=='accept':
  text=text.replace('cc_core80_full_lex_repair_20261031_run3','cc_core80_full_lex_repair_20261031_run4').replace('new_tests=29,total_tests=335','new_tests=33,total_tests=335')
 if name=='freeze_launch':
  old="'sia-simulator/tests/fixtures/core80_full_startup_step42.json.gz'},new";new="'sia-simulator/tests/fixtures/core80_full_startup_step42.json.gz','sia-simulator/tests/fixtures/core80_full_rank_step42.json.gz','sia-simulator/tests/test_integer_lex_epsilon_repair.py'},new";assert old in text;text=text.replace(old,new).replace('equivalent startup budget/global-primary-bound certificates','equivalent integer lex budget/global-primary-bound certificates retaining all earlier fixed rows')
 if name=='audit':
  text=text.replace("proof['incumbent_startup']","proof['incumbent_integer_cost']")
  old="if proof.get('proved') and not (proof['status']==0 and Decimal(proof['primary_global_upper'])<Decimal(proof['original_primary_lower_guard'])):errors.append('EPSILON_NO_GLOBAL_OPTIMALITY_CERTIFICATE')"
  new="if proof.get('proved'):\n       if proof['proof']=='EXACT_NONNEGATIVE_INTEGER_LEX_ATTAINED_ZERO':\n        if proof.get('valid_improved_integer_cost')!=0:errors.append('EPSILON_NO_ATTAINED_ZERO_WITNESS')\n       elif not (proof['status']==0 and Decimal(proof['primary_global_upper'])<Decimal(proof['original_primary_lower_guard'])):errors.append('EPSILON_NO_GLOBAL_OPTIMALITY_CERTIFICATE')"
  assert old in text;text=text.replace(old,new)
 if name=='report':
  text=text.replace('CC_CORE80_FULL_LEX_EXACT_BUDGET_COMPARISON_20261031.md','CC_CORE80_FULL_LEX_INTEGER_BUDGET_COMPARISON_20261031.md').replace('335项测试含原302项与累计新增29项','335项测试含原302项与累计新增33项')
  anchor="add('## 2. 根因、证据与最小修复\\n')"
  text=text.replace(anchor,anchor+"\n add('V7保存的真实第42轮故障已从启动阶段转移到后续规范排名：deepspeech2-10008，整数排名可行值8186、LP下界0，已有4组约束（物理/作业、主目标、启动225、此前作业掩码）。V8将同一预算证明扩展到一般非负整数词典序目标，全部保留前级启动及掩码；不混淆启动成本与排名编号。对预算8185，零gap主目标全局上界2.569973689748低于原下界2.570187416674，证明8186是最优排名。新增红测直接捕获旧入口返回None；没有丢弃前级约束或接受旧incumbent代替证明。\\n')")
 if name=='watch':text=text.replace('actual335-file snapshot','actual329-file snapshot').replace('actual327-file snapshot','actual329-file snapshot')
 with (M/(name+'_core80_full_lex_v8.py')).open('x',encoding='utf8') as f:f.write(text)
print('V8_PIPELINE_PREPARED_NOT_LAUNCHED')
