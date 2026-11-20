"""Post-run independent audit, response accounting, analysis and sealing."""
import csv,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_cifar_bert12_20261029_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def main():
 assert json.loads((OUT/'actual_opportunity_gate.json').read_text())['passed']
 records=[json.loads(l) for l in (OUT/'execution_manifest.jsonl').read_text().splitlines()];assert len(records)==6 and all(r['exit_code']==0 for r in records)
 with (OUT/'raw_audit_console.log').open('x',encoding='utf8') as f:r=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'maintenance/audit_cifar_bert12_20261029.py')],stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0,'AUDIT_FAILED; preserve logs'
 text=(ROOT/'maintenance/describe_breakpoint30_precision_v3_20261029.py').read_text()
 text=text.replace('cc_breakpoint_study_20261029_run1/attempt3','cc_cifar_bert12_20261029_run1')
 text=text.replace("['alexnet_1.py','deepspeech_1.py']","['alexnet_1.py','bert_1.py']")
 old="coverage=json.loads((OUT/'offline_coverage.json').read_text());work={r['application']:r['profiles'][0]['work'] for r in coverage}"
 assert old in text
 text=text.replace(old,"from applications import APPLICATIONS\n work={app:float(APPLICATIONS['aws'][app].get_progress(APPLICATIONS['aws'][app].max_epochs)) for app in ['cifar10','bert']}")
 text=text.replace("OUT/'analysis/execution_response_accounting.json'","OUT/'response_analysis.json'")
 script=OUT/'describe_response.py';script.write_text(text,encoding='utf8')
 with (OUT/'response_console.log').open('x',encoding='utf8') as f:r=subprocess.run([sys.executable,'-X','utf8','-B',str(script)],stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0,'RESPONSE_AUDIT_FAILED'
 response=json.loads((OUT/'response_analysis.json').read_text());audit=json.loads((OUT/'raw_audit.json').read_text());p=json.loads((OUT/'execution_preregistration_v2.json').read_text());comps=json.loads((OUT/'comparisons.json').read_text())
 with (OUT/'REPORT.md').open('a',encoding='utf8') as f:
  f.write('\n## 机会门槛与测试\n\n'+json.dumps(json.loads((OUT/'actual_opportunity_gate.json').read_text()),ensure_ascii=False,indent=2)+'\n\n完整241项回归在逐文件一致的canonical源码目录通过。离线覆盖核验8种物理应用/GPU配置，不删候选；5种比例点域内，3种继承参考边界缺口。正式六格条件与有界机会门槛相同，门槛结果不用于挑选或修改参数。\n')
  f.write('\n## 实际资源响应与共驻\n\n|方法|CF/BERT共同有效训练节点秒|比比例快的训练秒|比比例慢的训练秒|CPU分子越界训练秒%|RAM分子越界训练秒%|\n|---|---|---|---|---|---|\n')
  for cell in p['cells']:
   method=cell[-1];r=response['_'.join(cell)];t=r['totals'];cl=r['actual_requested_coordinate_clipping_seconds'];den=t['productive_job_seconds']
   f.write(f"|{method}|{t.get('heterogeneous_application_coproductive_node_seconds',0)}|{t.get('faster_than_proportional_job_seconds',0)}|{t.get('slower_than_proportional_job_seconds',0)}|{100*cl.get('numerator_cpu_beyond_1e12_tolerance',0)/den:.3f}|{100*cl.get('numerator_ram_beyond_1e12_tolerance',0)/den:.3f}|\n")
  f.write('\n比比例快/慢是固定实际GPU、batch、原始速率和训练片段的局部响应核算，不是完整合法反事实调度，也不能当作JCT因果份额。\n\n## 作业级增量\n\n|作业|FULL−Sia JCT秒|CPU−P|MEM−P|FULL−CPU|\n|---|---|---|---|---|\n')
  for name in audit['CC_P']['jobs']:
   f.write('|'+name+'|'+'|'.join(str(audit[a]['jobs'][name].get('jct','删失')-audit[b]['jobs'][name].get('jct',0)) if 'jct' in audit[a]['jobs'][name] and 'jct' in audit[b]['jobs'][name] else '删失' for a,b in [('CC_FULL','SIA_P'),('CC_CPU','CC_P'),('CC_MEM','CC_P'),('CC_FULL','CC_CPU')])+'|\n')
  f.write('\n测试首次在深目录触发既有fixture相对路径失败，完整日志保留；换至逐文件相同的既有canonical测试目录复核，未更改算法或测试内容。详情tests*.json/log。\n\n本轮已完成并停止，不自动扩大负载、改参数或恢复历史104/160批次。\n')
  f.write('\n## 完成配对的直接判断\n\n')
  for key in ['CC_FULL_vs_SIA_P','CC_FULL_vs_SIA_AUX','CC_CPU_vs_CC_P','CC_MEM_vs_CC_P','CC_FULL_vs_CC_CPU','CC_FULL_vs_CC_MEM']:
   x=comps[key]
   f.write(key+(': 存在删失，禁止完整均值比较。\n\n' if x.get('censored') else f": 平均JCT改善{x['improvement']*100:.3f}%，改善/退化/不变={x['improved']}/{x['worsened']}/{x['unchanged']}。\n\n"))
  f.write('独立CPU/内存/联合配额的效果必须使用同版本CC内部对照；CC-FULL相对Sia还含准入、GPU配置与完成导向目标差异。固定GPU离线见证的55.67%是速率提升，不是实测JCT改善；本表才是实际完成事件结果。\n')
 for script in ROOT.glob('maintenance/*cifar_bert12*20261029.py'):shutil.copyfile(script,OUT/script.name)
 with (OUT/'CURRENT_TASK.md').open('a',encoding='utf8') as f:f.write('\nCOMPLETE6_AUDITED_STOPPED. REPORT.md,validation.json,raw_audit.json and response_analysis.json authoritative. No further runs.\n')
 hashes={f.relative_to(OUT).as_posix():sha(f) for f in OUT.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc' and f.name!='delivery_hashes.json'}
 with (OUT/'delivery_hashes.json').open('x',encoding='utf8') as f:json.dump(hashes,f,ensure_ascii=False,indent=2)
 print('DELIVERED',len(hashes),comps)
if __name__=='__main__':main()
