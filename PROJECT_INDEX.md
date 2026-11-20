# CC-Joint 项目导航与保留依据

整理日期：2026-11-01。当前活动是论文准备，正式实验保持暂停。

## 1. 论文与可复查结果

|用途|入口|注意事项|
|---|---|---|
|论文、13篇文献、11篇公开PDF、6张原创图|[论文目录说明](research_notes/paper/cc_joint_draft_20261101_v1/README.md)|初稿；附42格历史记录复算，不是一轮新实验|
|主80作业七方法完整实验|[run5报告](research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run5/REPORT.md)|对应v8；FULL对Sia均值改善29.62%，同时披露尾部退化|
|有序/持续服务及干净资源消融|[干净消融产物](research_runs/sia_synergy/cc_clean_ab_20261030_run3/)|A组公平评分、B组完成评分；不可混算贡献|
|三种到达下六方法资源开关|[arrival18产物](research_runs/sia_synergy/cc_arrival_pressure18_20261030_run1/)|CPU、内存与联合开关证据|
|最新恢复等效12作业完整FO/FULL|[REPORT_v2](research_runs/sia_synergy/cc_mixed12_event_pair_20261101_run1/REPORT_v2.md)|0.336%均值增量与DeepSpeech退化均保留；旧失败报告没有删除|
|版本、来源hash及结论边界|[论文证据清单](research_notes/paper/cc_joint_draft_20261101_v1/evidence/evidence_inventory.md)|优先以冻结run目录为准，不以文件夹名字推断最新结论|

## 2. 保留的源码目录

|目录|定位|核心文件|
|---|---|---|
|[cc-joint-core80-full-lex-repair-v8](cc-joint-core80-full-lex-repair-v8/)|论文主系统与80作业完成版|completion_time.py、completion_continuity.py、clean_ablation.py、joint_scheduler.py、tail_recovery.py|
|[cc-joint-recovery-equivalent-v1](cc-joint-recovery-equivalent-v1/)|最新恢复等效修正版|对FO/FULL恢复资格进行几何闭包，不能继承旧版80作业数字|
|[cc-joint-clean-ab-r3](cc-joint-clean-ab-r3/)|两条贡献的干净消融|共同候选/目标/准入契约及开关|
|[cc-joint-resource-study](cc-joint-resource-study/)|资源开关与研究基础版本|比例、CPU、RAM和完整资源对照|
|[cc-joint-breakpoint-precision](cc-joint-breakpoint-precision/)|资源响应候选与数值精度基础|变化点候选、响应/容量审计|
|[cc-joint](cc-joint/)|早期整理后的四方法统一入口|run_cc_joint.py；2026-10-27版本，不代表论文主系统|

各核心模块通常位于该目录的`sia-simulator/joint_prototype/`。执行器为`sia-simulator/job.py`；Sia GPU与batch建模由`sia.py`、`applications.py`、`goodput.py`等提供。精确函数和行号见[机制到源码映射](research_notes/paper/cc_joint_draft_20261101_v1/evidence/algorithm_formalization.md)。

没有将后续修复自动合并回`cc-joint/`。本轮只整理，不改变算法版本或实验可比性。

## 3. 必须保留的数据和环境

- `official-artifacts/`：官方Sia、GPU画像、Saturn traces和原始论文。已有未跟踪缓存原样保留。
- `research_sources/Synergy/`：官方矩阵与TUNE算法参照；其他AFS、Lazer、Tiresias资料也是研究溯源。
- `research_runs/`：约16.4 GiB的原始结果，是论文复算和失败追踪依据，未删除或改写。
- `research_notes/`：报告、文献、图表和脚本，未修改既有文件。
- 两个`.venv-*`：保留全部环境和依赖版本。仅删除有对应`.py`源码的`__pycache__/*.pyc`。
- `tmp/`：仍有早期试验结果与论文截图，尚不能确认全部为冗余，保留原路径。

## 4. 中间版本与恢复

其余根目录`cc-joint-*`中间源码完整归档到：

`_archive/cc_intermediate_sources_20261101_v1.zip`

逐文件原路径、SHA-256和归档hash见[archive_manifest.json](maintenance/project_organization_20261101_v1/archive_manifest.json)。没有永久销毁旧源码或选择性删除失败版本。中间版本原命令需要先恢复对应目录；历史run自带的独立源码快照保持原位置。

恢复脚本默认只检查、不写入：

```powershell
& .\maintenance\project_organization_20261101_v1\restore_intermediate_sources.ps1
```

完整恢复时添加`-Execute`。脚本校验归档hash，拒绝覆盖已存在的目录，并逐文件核对恢复后hash。只想查看单个文件时可直接打开ZIP，无需恢复所有版本。

`maintenance/`中的旧冻结、验收和分析脚本可能是结果复现依赖，保留原路径。使用新的工作区浏览日常研究内容，可避免被历史脚本干扰。

## 5. 约定

论文结论必须对应明确版本、输入和执行响应；不要将最新源码套用旧实验数字。没有新实验授权时不启动历史watch/launch/resume脚本。所有CPU/RAM结果均为合成跨硬件迁移证据，不是目标Saturn硬件实测。
