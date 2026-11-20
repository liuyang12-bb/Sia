"""Final evidence and entry checks only. Does not start a training run."""
import json,shutil,subprocess,sys,time
from pathlib import Path
from inventory import ROOT,OUT,sha,save

before=json.loads((OUT/'before/summary.json').read_text(encoding='utf-8'))
after=json.loads((OUT/'after/summary.json').read_text(encoding='utf-8'))
relocation=json.loads((OUT/'relocation_validation.json').read_text(encoding='utf-8'))
assert before['valid'] and after['valid'] and relocation['valid']
sys.path.insert(0,str(ROOT/'cc-joint'));from cc_factory import verify
manifest=verify()

commands=[]
def run(label,args):
    log=OUT/(label+'.log');start=time.time()
    with log.open('x',encoding='utf-8') as f:p=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    commands.append(dict(label=label,command=args,exit_code=p.returncode,seconds=time.time()-start,log_sha256=sha(log)))
    assert p.returncode==0,(label,log.read_text(encoding='utf-8'))
    return log

log=run('final_entry_constructor',[sys.executable,'-X','utf8','-B',str(ROOT/'cc-joint/run_cc_joint.py'),'--check'])
content=log.read_text(encoding='utf-8');obj=json.loads(content[content.rfind('\n{')+1:])
assert obj['valid'] and obj['jobs']==20 and obj['simulated_seconds']==0
assert obj['all_job_progress_zero'] and obj['all_job_completion_none'] and obj['training_run'] is False
shell=shutil.which('pwsh') or shutil.which('powershell')
assert shell
restore=run('restore_dry_run',[shell,'-NoProfile','-File',str(OUT/'restore_legacy.ps1')])
assert 'DRY_RUN_ONLY' in restore.read_text(encoding='utf-8')
plan=json.loads((OUT/'relocation_plan.json').read_text(encoding='utf-8'))
assert all(not (ROOT/n).exists() for n in plan['names'])
history=ROOT/'research_runs/sia_synergy/cc_joint_sustained_cpu_20job_20261027_run1'
old=json.loads((history/'preregistration.json').read_text(encoding='utf-8'))
kept={n.removeprefix('sia-simulator/') for n in manifest['files'] if n.startswith('sia-simulator/')}
omitted=sorted(set(old['source_hashes'])-kept)
assert len(old['source_hashes'])==404 and len(kept)==232
assert all(sha(ROOT/'cc-joint/sia-simulator'/n)==old['source_hashes'][n] for n in kept)
omitted_py=[n for n in omitted if n.endswith('.py')]
assert len(omitted_py)==40
front={n:sha(ROOT/'cc-joint'/n) for n in ['cc_factory.py','cc_metrics.py','run_cc_joint.py','README.md','source_manifest.json']}
record=dict(valid=True,algorithm_changes=False,performance_experiment_started=False,
    canonical_root=str(ROOT/'cc-joint'),archived_trees=27,archived_working_files_unchanged=19693,
    git_states_preserved=22,original_source_files=404,retained_engine_test_data_files=232,
    omitted_from_current_tree=omitted,omitted_python=omitted_py,all_retained_engine_bytes_identical=True,
    before_tests=212,after_tests=212,original_sia_bounded_exact=True,
    bounded_equivalence=json.loads((OUT/'before/bounded_equivalence.json').read_text(encoding='utf-8')),
    constructor_check=obj,commands=commands,front_door_hashes=front,
    backup_manifest_sha256=sha(OUT/'backup_manifest.json'),relocation_validation_sha256=sha(OUT/'relocation_validation.json'),
    official_and_synergy_source_unchanged=True,historical_results_not_overwritten=True,
    note='No permanent file deletion. Only verified empty leftover directories removed during Windows Hidden .git recovery. Git path registrations updated; branches/HEAD/dirty states unchanged.')
save(OUT/'final_validation.json',record)
lines=['# CC-Joint代码整理验收','',
    '当前唯一开发目录：`D:/科研启航/Sia/cc-joint`；唯一运行入口`run_cc_joint.py`。',
    '', '## 已完成','',
    '- 原27个顶层`sia-*`旧版本目录移入`_archive/legacy_code_20261027/`，未永久删除源码。',
    '- 完整旧树备份ZIP约188.60MiB，20,058个文件逐成员读回校验；另存Git bundles及staged/unstaged patches。',
    '- 19,693个旧工作文件逐字节核对；22个Git工作区的HEAD、分支和dirty/untracked状态与整理前一致。',
    '- 新目录筛选保留232个引擎/测试/数据文件，均与原404文件冻结版对应字节一致；排除172个非必要文件，其中40个Python文件。',
    '- 额外集中现有配置、R0、模板和20作业输入；新factory/front door与metrics提取不改变调度机制。',
    '- 保留SIA-P、串行SIA-ECT-AUX、CC-P、CC四策略；旧命名模块若在继承链或回归中仍使用则保留。',
    '- 官方Sia/Synergy工作区源码、研究报告、原始运行结果、虚拟环境和未知tmp内容均保留。',
    '', '## 验证','',
    '- 整理前212/212测试通过，归档后212/212再次通过；没有删除或放松测试。',
    '- 原始Sia独立3轮分配/进度/restart/完成事件回归前后均精确一致。',
    '- 四策略×Neutral/Medium的8组、每组2作业5轮，与原冻结版逐状态精确一致。',
    '- 新入口实际构造20作业通过：仿真时间0、所有progress0、无完成事件；未启动20作业训练。',
    '- 恢复脚本dry-run通过；不覆盖任何已有目标。',
    '', '## 保留的失败证据','',
    '新入口首次--check只在JSON输出NumPy int32时失败；补default编码器后真实构造检查通过，未改引擎。',
    'PowerShell移动主仓库时，Hidden .git导致部分移动；26树已移动，主仓库剩两个源码子目录和空.git容器。'
    '在已校验备份前提下补齐移动，仅非递归移除两个已证实为空的容器，并用Git修复17个反向链接；最终字节与Git状态均校验。',
    '完整记录见constructor_output_failure.txt、main_relocation_failure.txt及relocation_journal.jsonl。',
    '', '## 当前使用与恢复','',
    '阅读`cc-joint/README.md`，只从新入口开展后续工作。生产配置为`cc-joint/inputs/config_symmetric.json`。'
    '引擎内旧JSON和少量旧模块属于回归/兼容依赖，不是另一个主实验入口。',
    '默认恢复预检：`maintenance/cc_joint_cleanup_20261027/restore_legacy.ps1`。'
    '只有另行需要恢复旧布局时，显式加`-Apply`；它拒绝目标已存在，不修改当前cc-joint。',
    '旧冻结脚本仍记录历史执行位置，不重写历史。需重跑旧版本时恢复旧布局，或依原可重建归档在空项目根恢复。',
    '', '## 边界','',
    '本轮是代码与目录整理，没有改变算法目标、FIFO/持续服务、checkpoint、250秒预算、Goodput、phi、clipping、资源容量或工作负载。'
    '没有重新做性能实验，没有提交或推送Git commit。官方仓库源码/HEAD未改，关联Git工作树路径登记因归档而更新。',
    '归档保留以确保可恢复，本轮主要减少开发区混乱而非释放磁盘。不要未经审查删除历史数据、Git归档或官方资料。',
    '', '新入口SHA见final_validation.json；全部保留文件SHA见cc-joint/source_manifest.json。']
with (OUT/'CLEANUP_REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
print('CLEANUP_COMPLETE_VALIDATED',flush=True)
