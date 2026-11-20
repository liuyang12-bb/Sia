"""Create a new, reproducible isolated source copy; never mutate old sources."""
import hashlib,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1'
OUT=ROOT/'research_runs/sia_synergy/cc_queue_mechanism_ablation_20261028_run1'
TREE=ROOT/'cc-joint-queue-ablation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(p,v):
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def main():
 assert not OUT.exists() and not TREE.exists(),'REFUSE_EXISTING_OUTPUT'
 hashes={p.relative_to(OLD/'source').as_posix():sha(p) for p in (OLD/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
 for n,h in hashes.items():assert sha(ROOT/'cc-joint-resource-study'/n)==h,n
 OUT.mkdir();shutil.copytree(OLD/'source',TREE,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for n,h in hashes.items():assert sha(TREE/n)==h,n
 save(OUT/'parent_source_hashes.json',hashes)
 save(OUT/'initial_state.json',dict(created=time.time(),parent=str(OLD/'source'),source_files=len(hashes),isolated_tree=str(TREE),git_repository=False,version_authority='FULL_SOURCE_SNAPSHOT_AND_SHA256',memory_disk_gates=[2,5]))
 (OUT/'CURRENT_TASK.md').write_text('''# 当前任务：排队机制独立消融\n\n状态：PREPARING。用户授权设计并执行完成导向配置、FIFO准入、持续服务的独立消融；当前CC-TAIL、官方源、历史30格全部保护。\n新源码：D:/科研启航/Sia/cc-joint-queue-ablation；仅本树新增模块。父版为30格source/完整282文件，复制后逐一hash核验。\n主实验仅原20作业、R11(每节点2GPU/6CPU/187.5GB)、Neutral rho=1、比例配额，60秒/6000轮。8个预注册因素组合加外部原Sia基线；不启动104格/160作业/Medium或新响应。\nA=完成导向倒数ETA vs Sia归一化Goodput评分；B=FIFO队首强制与排空 vs 自由任务选择；C=持续到完成 vs 300秒有效训练服务段。\n两种B均保护真实锁、当前承诺、容量与可行非空启动；B关闭不提供个体FIFO防饥饿保证。C各版本使用相同生命周期250秒预算，不能用服务段重置预算。旧CC-TAIL的唯一一次收尾恢复规则不改。\n本轮不优化P99，不改候选网格、CPU/RAM响应或工作负载。先红绿/原测试/短链回归，再冻结源码和矩阵；独立进程串行、2GiB内存/5GiB磁盘门槛，删失如实报告，无收益不调参。\n恢复：先读本文件和preregistration.json、execution_manifest.jsonl，禁止重复运行，已有目录永不覆盖。\n''',encoding='utf-8')
 print('COPIED_AND_VERIFIED',len(hashes),OUT)
if __name__=='__main__':main()
