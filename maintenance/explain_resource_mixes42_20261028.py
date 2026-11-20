"""Explain finished paired results, without changing inputs or running a model."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_resource_mixes42_20261028_run1'
def read(n):return json.loads((OUT/n).read_text(encoding='utf8'))
def main():
 p=read('analysis/paired_comparisons.json');c=read('analysis/clipping_and_resources.json');r=read('analysis/execution_response_accounting.json')
 lines=['## 收益与退化的具体证据','',
 '以下均为新减比例版本，负时间差表示节省，单位为任务秒；不把中间训练率收益当最终JCT。','',
 '|低CPU场景：CPU-only对P|ΣJCT差|等待差|训练差|checkpoint差|CIFAR ΣJCT差|DS ΣJCT差|','|---|---:|---:|---:|---:|---:|---:|']
 for mix in ['M1','M2','M3']:
  x=p[f'{mix}/R11/CC_CPU_vs_CC_P'];d=x['time_delta'];a=x['application_total_delta']
  lines.append(f"|{mix}|{sum(x['per_job_delta'].values())}|{d['wait']}|{d['training']}|{d['checkpoint']}|{a.get('cifar10',0)}|{a.get('deepspeech2',0)}|")
 lines+=['','M1的CPU增量同时包含等待和训练节省；M2主要是等待节省，训练总量时间只略降；M3训练和checkpoint时间反而增加，靠等待减少获得很小的净改善。长任务增多不自动扩大CPU优化增量；本数据不能把任务构成、长度与到达次序的影响分别唯一识别。','',
 '### 内存机会确实发生，但当前策略未稳定兑现JCT','']
 key='M2_R01_medium_CC_MEM';v=r[key]['totals'];x=p['M2/R01/CC_MEM_vs_CC_P']
 lines.append(f"作为中间混合比例的结构性诊断，M2/R01/MEM实际CF/DS同节点共同有效训练{v.get('heterogeneous_application_coproductive_node_seconds',0):.0f}节点秒。固定实际训练片段、GPU/batch/raw率，接收方响应增量为{v.get('receiver_progress_fraction',0):.6f}个归一化总工作单位，让出方损失为{v.get('donor_loss_fraction',0):.6f}，净{v.get('fixed_geometry_progress_fraction_delta',0):.6f}；这只是局部响应积分，不是可执行的完整调度反事实，未重新模拟完成边界/资源释放。实际平均JCT却增加{-100*x['improvement']:.2f}%，等待增加{x['time_delta']['wait']:.0f}任务秒、训练增加{x['time_delta']['training']:.0f}任务秒。因此不能用“没有遇到互补”解释全部失败，也不能从局部净推进推导总体JCT必降。")
 lines+=['','现有倒数ETA评分倾向较短剩余作业，允许牺牲长任务的辅助配额；内存粗网格又会将连续无损交换变成有损交换。这些是源码与执行账支持的机制限制。新矩阵中GPU、配额与后续队列路径会共同分叉，未进行额外隔离反事实，不能把所有JCT差值唯一归因于一次内存交换或某个代码分支。','',
 '### E-legacy边界暴露不能忽略','',
 '|低CPU场景|CPU-only分子CPU裁剪训练秒%|FULL分子CPU裁剪训练秒%|','|---|---:|---:|']
 for mix in ['M1','M2','M3']:
  vals=[100*c[f'{mix}_R11_medium_{m}']['fraction_productive_job_seconds']['numerator_cpu'] for m in ['CC_CPU','CC_FULL']]
  lines.append(f'|{mix}|{vals[0]:.3f}|{vals[1]:.3f}|')
 lines+=['','最大CPU增量所在M1/R11同时存在约24%的分子CPU裁剪训练时间暴露，不能忽略，也不能把该时间比例当成收益贡献比例。M2/M3暴露较低，CPU增量也更小；这只是共变，未做替代边界模型反事实，不能唯一归因。全部结果只在本次已接受的固定E-legacy假设下成立，尚不能宣称对低CPU外推方式稳健。','',
 '本轮保持算法、参数和网格不动，已经回答当前实现能兑现多少；没有通过改数据、延长窗口、重跑择优或调参把负结果消掉。后续若改进，应先针对资源交换的完成代价、数据导出的候选拐点与边界稳健性设计独立版本；本轮停止，不自动实施。']
 with (OUT/'analysis/INTERPRETATION.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines))
 with (OUT/'REPORT.md').open('a',encoding='utf8') as f:f.write('\n\n'+'\n'.join(lines)+'\n')
 print('INTERPRETATION_COMPLETE')
if __name__=='__main__':main()
