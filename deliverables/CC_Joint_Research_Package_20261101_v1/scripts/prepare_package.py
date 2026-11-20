"""Assemble three deliverables exclusively from preserved evidence; no simulator imports."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import io
import json
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
PAPER = PROJECT / 'research_notes/paper/cc_joint_draft_20261101_v1'
RUN = PROJECT / 'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run5'
SOURCE = PROJECT / 'cc-joint-core80-full-lex-repair-v8'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def write(name, text):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf-8')


def export_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    p = ROOT / 'evidence' / name
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


assert not (ROOT / '01_ALGORITHM_DESIGN_REPORT.md').exists(), 'Refusing to replace an existing delivery'
for folder in ['evidence', 'figures/equations', 'code_documentation']:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)
draft = (PAPER / 'CC_JOINT_PAPER_DRAFT.md').read_text(encoding='utf-8')
data = json.loads((PAPER / 'evidence/paper_results.json').read_text(encoding='utf-8'))
assert len(data['cells']) == 42 and all(c['completed'] == c['total'] for c in data['cells'])
for item in data['provenance'].values():
    assert sha(item['source_archive']) == item['source_archive_sha256']
for p in (PAPER / 'figures').iterdir():
    if p.is_file():
        shutil.copy2(p, ROOT / 'figures' / p.name)
for p in (PAPER / 'figures/equations').iterdir():
    if p.is_file():
        shutil.copy2(p, ROOT / 'figures/equations' / p.name)
for name in ['paper_results.json', 'evidence_inventory.md', 'algorithm_formalization.md']:
    shutil.copy2(PAPER / 'evidence' / name, ROOT / 'evidence' / name)

algorithm = draft[draft.index('## 3 问题定义'):draft.index('## 5 实现与实验方法')]
algorithm = re.sub(r'^(#{2,3}) ([34])(?=[. ])', lambda m: m[1]+' '+str(int(m[2])-2), algorithm, flags=re.M)
algorithm = algorithm.replace('第4.3节', '第2.3节')
algorithm = algorithm.replace('图1', '图1').replace('论文', '本报告')
algorithm_intro = '''# CC-Joint 算法设计报告

**交付1｜完整机制、流程图、数学公式与伪代码**

## 阅读说明

本报告对应已完成80作业七方法实验的`cc-joint-core80-full-lex-repair-v8`，实际329文件源码快照，而非裸Git HEAD。最新恢复等效版本单列于实验报告，未冒充v8复跑。算法未因本次交付而修改。

研究目标是在异构GPU集群中，同时组织任务服务与GPU/CPU/RAM配置，以降低平均JCT。保留Sia合法异构GPU、弹性数量、batch自适应和Goodput模型；CPU/RAM响应借鉴Synergy公开矩阵，联合有限列MIP是本项目实现，并非直接移植Synergy-TUNE。

主流程为：当前状态 → 合法联合候选 → 性能与ETA查询 → checkpoint/服务约束 → 联合词典序求解 → 独立审计 → 原子提交 → 真实训练与完成反馈。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。

'''
algorithm_end = '''
## 3 输入、输出与状态契约

|对象|内容|用途|
|---|---|---|
|Job当前状态|arrival、源行、进度、epoch、batch、已观测profile、当前分配|合法GPU与ETA，不读取未来完成时间|
|节点容量|GPU数、合成CPU单位、RAM GB|逐节点物理容量约束|
|服务账本|waiting队列、held、真实productive秒数、启动损失预算|有序启动、持续服务与排空|
|checkpoint锁|当前完整GPU、placement、CPU/RAM与剩余倒计时|完整等式保护，不能当可选列|
|Oracle|R0 registry、矩阵、phi、参考点、alpha、版本|统一响应查询，分子/分母裁剪分别记账|
|输出|完整allocation、候选来源、ETA、rho、求解状态、审计|只有整体合法才原子提交|

## 4 冷启动、失败和保证边界

未画像ETA不伪造正训练率；候选使用共享比例bootstrap，首次服务还依赖可行队首强制准入。主80作业统一seed_profiles(1,8)，不是完全无画像冷启动实验。

在队列有限、每任务单独可行、真实训练率有正下界、工作量有限、启动损失有界、求解成功及资源不撤销时，已承诺任务完成并释放资源，排空保证被阻塞队首最终可以接纳。零训练率、不可行锁、无限过载和求解失败不在该保证内。

求解无可用解不等于可合法提交空分配。回退必须重建当前轮查询，并独立证明完整容量、候选资格与锁合法；不能证明时停止并记录诊断。不能放松锁来继续运行。

## 5 核心代码地图与复杂度

|文件|职责|
|---|---|
|core80.py|v8组装入口、服务/资源因素、实际Cluster适配|
|completion_time.py|声明工作量、当前raw训练率与ETA|
|completion_continuity.py|完成效用、有效服务账本和条件活性|
|clean_ablation.py|共同候选资格、队首准入、强制集合与稳定择解|
|joint_integration.py / joint_scheduler.py|Sia候选桥接、联合列、placement、容量和原子审计|
|joint_profile.py / r0_liveness.py|Synergy矩阵、固定phi与响应|
|tail_recovery.py|单任务有界收尾恢复|
|job.py / applications.py / goodput.py / sia.py|真实训练推进及继承的Sia模型|
|engineering_repair.py / numeric_repair.py等|正式v8等价数值求解与验收覆盖入口|

候选规模约为任务数×GPU配置数×placement数×辅助资源点数；有限列MIP最坏组合复杂度高。v8主实验求解P99为90.77秒，超过60秒周期；当前仿真没有将计算墙钟注入JCT，不能据此宣称在线部署已满足实时性。

精确函数、行号及版本差异见附带`evidence/algorithm_formalization.md`。主80源码SHA-256为B0DCAEB4A5A205B6CA27893F3ACFE1F43C3F850CA8AF6E5BC79B15029518D6EB。

## 6 当前实现的适用范围

有序启动与持续服务主要控制重复等待；联合资源配置利用差异化响应，但倒数ETA不等价全局平均JCT最优，部分长任务可能受损。CPU/RAM自身配额调整没有额外加载费用。主实验是每GPU类型一个节点、单节点任务候选，不能外推成大规模跨节点验证。

Sia已有重配置惩罚和checkpoint保护，本方法不是补上“原Sia完全没有”的能力。FIFO、服务保护与CPU感知也都存在相关先例，贡献在具体协调机制与独立消融证据。
'''
write('01_ALGORITHM_DESIGN_REPORT.md', algorithm_intro + algorithm + algorithm_end)

experiment = draft[draft.index('## 5 实现与实验方法'):draft.index('## 8 结论')]
experiment = re.sub(r'^(#{2,3}) ([567])(?=[. ])', lambda m: m[1]+' '+str(int(m[2])-4), experiment, flags=re.M)
experiment = experiment.replace('本文', '本报告')
experiment = experiment.replace('图6：左侧为', '图6：上部为').replace('右侧将不同证据组各自的时间差', '下部将不同证据组各自的时间差')
exp_intro = '''# CC-Joint 完整实验报告

**交付2｜Baseline对照、完整指标表、曲线与逐作业数据**

## 范围与结论

本报告完整收录论文采用的四套已完成证据，共42个实验格：clean_ab 15、arrival18 18、core80 7、最新mixed12 2。不是声称项目历史所有尝试只有42格；未完成、旧失败和其他探索结果仍保存在原research_runs。本次没有启动或重跑训练仿真。

主80作业FULL对官方Sia决策适配基线平均JCT下降29.62%，P99/makespan恶化18.29%/14.50%；对串行FO的均值增量仅1.09%。差异化CPU/RAM在固定合成模型中有效，联合GPU相对串行额外收益依赖负载。后期12作业仅0.336%增量及所有DeepSpeech退化均保留。

所有42格已从逐作业completion-arrival重算聚合，并核对waiting+training+checkpoint=JCT。不同源码组不合并为同一版本的统一效应。不将单次确定性实验的作业当独立重复。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。

'''
groups = defaultdict(list)
for c in data['cells']:
    groups[(c['dataset'], c['cohort'])].append(c)
appendix = '\n## 4 全部42格完成结果\n\n下表单位为秒；每格均为全部作业完成，分位数采用逐作业经验分布的线性插值。\n'
for (dataset, cohort), cells in groups.items():
    appendix += f'\n### {dataset} / {cohort}\n\n|方法|完成|平均JCT|中位数|P95|P99|Makespan|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for c in cells:
        appendix += '|'+ '|'.join([c['method'],f"{c['completed']}/{c['total']}"]+[f"{c[k]:.2f}" for k in ['mean_jct','median_jct','p95_jct','p99_jct','makespan']])+'|\n'
appendix += '''
## 5 完整数据、复算与原始证据入口

`evidence/all_cells.csv`包含42格的JCT、等待/训练/checkpoint、资源小时、重启和fallback；`evidence/all_jobs.csv`包含每格每个作业的arrival、completion、JCT、等待、训练、checkpoint和资源积分；`evidence/all_comparisons.csv`包含配对差值和高影响作业诊断；完整机器数据为`evidence/paper_results.json`。

原始逐轮日志仍在项目的research_runs，未复制全部约16.4 GiB历史数据，也未改写。每格的原始summary路径、源码快照hash和预注册来源包含在机器数据中。若脱离原项目分享，这些相对独立的数据表可复算表格，原始逐轮日志仍需按来源目录提供。

被纳入图表的曲线由逐作业记录重新绘制，没有复制其他论文的图像或借用其测量数据作为本系统结果。PDF内图号沿用原研究稿，缺图号不表示实验缺失。

本报告重新核验完成事件和时间分解；容量、checkpoint及单次rho零违规结论对应既有独立原始日志审计，未再次全量扫描全部压缩日志。42格的实验条件按各组冻结，不能将三种到达变体当独立生产trace。
'''
write('02_EXPERIMENT_REPORT.md', exp_intro + experiment + appendix)

cell_rows, job_rows, comparison_rows = [], [], []
for c in data['cells']:
    row = {k:c[k] for k in ['id','dataset','cohort','method','completed','total','rounds','status','mean_jct','median_jct','p95_jct','p99_jct','makespan','fallback_count','source_summary']}
    row.update(c['totals'])
    row['solver_status'] = json.dumps(c['solver_status'], ensure_ascii=False)
    cell_rows.append(row)
    for job, values in c['jobs'].items():
        row = dict(cell=c['id'], dataset=c['dataset'], cohort=c['cohort'], method=c['method'], job=job, **values)
        if 'completed' not in values:
            row['completed'] = values.get('completion') is not None
            row['completed_evidence'] = 'derived_from_recorded_completion_event'
        else:
            row['completed_evidence'] = 'explicit_original_field'
        job_rows.append(row)
for c in data['comparisons']:
    comparison_rows.append({k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in c.items()})
export_csv('all_cells.csv',cell_rows)
export_csv('all_jobs.csv',job_rows)
export_csv('all_comparisons.csv',comparison_rows)

structure = '''# CC-Joint 核心代码与结构说明

本包冻结的是论文主版本v8的完整329文件，不是从裸HEAD重建，也没有删除看似旧名的真实继承依赖。主版本80作业报告、后期恢复等效12作业数字不能混用。

## 目录

```text
workspace/
  cc-joint-core80-full-lex-repair-v8/  329个原始源码/输入/测试文件
    core80.py                       构造七方法及共同执行模型
    run_core80.py                   单格正式运行器（不会自动运行）
    engineering_repair.py           v8正式求解上下文
    numeric_repair.py 等            等价求解与数值证明辅助
    inputs/                        冻结80作业、R0、矩阵和容量配置
    sia-simulator/
      sia.py, job.py, goodput.py, applications.py
      joint_prototype/             核心策略、响应、服务、锁、原子审计
      tests/                       原样保留的测试
provenance/                        原预注册、源码hash、输入hash和环境
fixtures/                          原external_test_dependencies.zip
tools/                             原验收脚本ZIP与离线hash核对脚本
STRUCTURE.md, REPRODUCTION.md       本包新增说明
```

## 调用关系

`run_core80.main → core80.build → make_cluster → install_completion / install_tail / install80 → Cluster.step → optimize_policy → 候选与服务约束 → 联合MIP → 原子提交 → Job.step → 完成事件与指标`。

正式v8会进入engineering_repair.repair_context，不能只调用旧run_cc_joint.py后声称复现了80作业结果。原源码README源自早期统一入口，保留原字节；运行语义以本说明、core80.build和原预注册为准。

## 模块

|模块|职责|
|---|---|
|completion_time|当前模型预测raw rate、声明剩余工作量和ETA|
|completion_continuity|完成导向目标、真实有效服务与连续性|
|clean_ablation|共享候选、队首与held强制集合、确定性择解|
|joint_integration / joint_scheduler|真实Sia候选、联合CPU/RAM/placement、容量与原子提交|
|joint_profile / r0_liveness|固定R0与Synergy矩阵查询|
|tail_recovery|单任务有界收尾恢复|
|job / applications / goodput / sia|实际训练和Sia异构/弹性/batch模型|
|engineering_repair及求解辅助|主值/启动/词典序的等价数值验收|

所有源码文件与历史source_hashes逐项一致。参考算法报告和evidence/algorithm_formalization.md可定位具体函数及行号。
'''
write('PROJECT_STRUCTURE.md',structure)
reproduction = '''# 复现说明

本交付只打包，不运行仿真。源码zip包含工作量、GPU画像、R0、矩阵和原依赖锁；不包含500MB虚拟环境或16.4GiB原始实验日志。环境为Python3.10.10、NumPy1.23.5、SciPy1.10.1、CVXPY1.3.2、CyLP0.91.6等，以provenance/environment.json和源码dependencies.lock.txt为准。CBC本地库还受平台安装约束，不能保证pip单条命令可重建二进制环境。

1. 解压到新目录；不要覆盖当前项目。先运行`python -B tools/verify_package.py`，只读校验所有329个源码文件。
2. 原测试有外部fixture依赖。解压fixtures/external_test_dependencies.zip到workspace/，保留其内部路径；不能把外部数据缺失当成算法测试失败。
3. 原验收记录和小型回归见provenance/software_acceptance.json、provenance/REBUILD.md、tools/maintenance_scripts.zip；正式主版本已有335项测试证据。本次没有重新运行它们。
4. 原preregistration.json保存的是历史绝对路径和已关闭输出目录，不能直接启动。需另行预注册新的源码位置、输入位置和全新输出目录，仅迁移路径、不改内容hash或算法参数。校验run_core80.py的--manifest/--cell接口后，才可运行单格；不要直接运行watch/launch脚本。

历史命令接口（仅示意，不在本交付中执行）：

`python -B workspace/cc-joint-core80-full-lex-repair-v8/run_core80.py --manifest NEW_FROZEN_MANIFEST.json --cell dual_medium_FULL`

内存至少2GiB、磁盘至少5GiB；60秒周期、6000轮上限；不减少训练工作量、不改响应、不覆盖旧输出。原始source_hashes.json中的key相对源码根；input_hashes.json中的原绝对路径需按新布局对应核对。某些原测试和历史脚本还使用项目布局，跨机器完整测试需逐项恢复fixture，不能把包描述成无需环境准备的一键部署。

原始源码快照SHA-256：B0DCAEB4A5A205B6CA27893F3ACFE1F43C3F850CA8AF6E5BC79B15029518D6EB。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。
'''
write('code_documentation/REPRODUCTION.md',reproduction)
verification_script = '''from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
hashes=json.loads((root/'provenance/source_hashes.json').read_text(encoding='utf-8'))
source=root/'workspace/cc-joint-core80-full-lex-repair-v8'
for name,h in hashes.items():
 p=source/name
 assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest().upper()==h.upper(),name
print('PASS:',len(hashes),'original source/input/test files match historical hashes; no simulation started.')
'''
archive = ROOT / '03_CC_JOINT_CODE_AND_STRUCTURE.zip'
source_hashes = json.loads((RUN/'source_hashes.json').read_text(encoding='utf-8'))
with zipfile.ZipFile(RUN/'source_snapshot.zip') as old, zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    assert len(old.namelist()) == len(source_hashes) == 329
    for name in old.namelist():
        content=old.read(name)
        assert hashlib.sha256(content).hexdigest().upper()==source_hashes[name].upper(),name
        assert sha(SOURCE/name)==source_hashes[name].upper(),name
        z.writestr('workspace/cc-joint-core80-full-lex-repair-v8/'+name,content)
    for name in ['preregistration.json','source_hashes.json','input_hashes.json','environment.json','dependencies.lock.txt','software_acceptance.json','REBUILD.md','source_changes.patch','external_test_dependency_hashes.json','freeze_receipt.json']:
        z.write(RUN/name,'provenance/'+name)
    z.write(RUN/'external_test_dependencies.zip','fixtures/external_test_dependencies.zip')
    z.write(RUN/'maintenance_scripts.zip','tools/maintenance_scripts.zip')
    z.writestr('tools/verify_package.py',verification_script)
    z.writestr('STRUCTURE.md',structure)
    z.writestr('REPRODUCTION.md',reproduction)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name,h in source_hashes.items():
        assert hashlib.sha256(z.read('workspace/cc-joint-core80-full-lex-repair-v8/'+name)).hexdigest().upper()==h.upper()
write('evidence/code_package_receipt.json',json.dumps({'status':'PASS','source_files':329,'source_snapshot_sha256':sha(RUN/'source_snapshot.zip'),'zip_sha256':sha(archive),'zip_bytes':archive.stat().st_size,'all_source_files_match_frozen_and_live':True,'new_simulations':0},ensure_ascii=False,indent=2))

builder=(PAPER/'scripts/build_draft_pdf.py').read_text(encoding='utf-8')
builder=builder.replace("ROOT=Path(__file__).resolve().parents[1]", "import argparse\nap=argparse.ArgumentParser();ap.add_argument('source');args=ap.parse_args()\nSOURCE=Path(args.source).resolve()\nROOT=SOURCE.parent")
builder=builder.replace("(ROOT/'CC_JOINT_PAPER_DRAFT.md').read_text", "SOURCE.read_text")
builder=builder.replace("pdf=ROOT/'CC_JOINT_PAPER_DRAFT.pdf'", "pdf=SOURCE.with_suffix('.pdf')")
builder=builder.replace("'CC-Joint 研究论文初稿 v0.1'", "'CC-Joint 研究交付报告'")
builder=builder.replace("(ROOT/'evidence/pdf_build.json')", "(ROOT/'evidence'/(SOURCE.stem+'_pdf_build.json'))")
builder=builder.replace("title='CC-Joint 面向异构GPU集群的有序启动与持续服务多资源联合调度'", "title=SOURCE.stem")
write('scripts/build_reports.py',builder)
write('README.md','''# CC-Joint 三项交付

1. `01_ALGORITHM_DESIGN_REPORT.pdf` / `.md`：完整算法、流程图、10个公式、伪代码、输入输出、约束、失败与活性边界。
2. `02_EXPERIMENT_REPORT.pdf` / `.md`：Baseline、四组42格完整结果、资源与尾部、曲线、逐作业数据及负结果。这里“完整”指所列四套证据，不代表历史所有探索和失败只有42格。
3. `03_CC_JOINT_CODE_AND_STRUCTURE.zip`：原样329文件v8源码、输入/测试/依赖、外部fixture包、预注册和结构/复现说明。不是仅几段核心函数。

PROJECT_STRUCTURE.md可在不解压ZIP时阅读；evidence/*.csv可用表格软件打开；figures/提供PNG、SVG、PDF原图。v8原源码README沿用早期版本说明，本包STRUCTURE/REPRODUCTION明确实际入口与版本，不改原源码字节。

本次没有运行仿真或改算法。主80作业、干净消融、arrival18和后期12作业分别冻结，不混算版本。原始历史文件与17GB级日志保持原位置。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。
''')
print(json.dumps({'status':'PREPARED','cells':len(cell_rows),'job_rows':len(job_rows),'code_files':329,'zip_MiB':round(archive.stat().st_size/2**20,2)},ensure_ascii=False))
