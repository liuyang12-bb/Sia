# Sia＋Synergy：CC-Joint联合调度研究

建议打开 [CC_Joint_Research.code-workspace](CC_Joint_Research.code-workspace)，按“论文、主实现、修正版、消融、实验证据、官方来源”浏览。完整入口见 [PROJECT_INDEX.md](PROJECT_INDEX.md)。

## 先看这些

- **论文初稿与文献**：[交付说明](research_notes/paper/cc_joint_draft_20261101_v1/README.md)、[初稿PDF](research_notes/paper/cc_joint_draft_20261101_v1/CC_JOINT_PAPER_DRAFT.pdf)。
- **论文主实现**：[cc-joint-core80-full-lex-repair-v8](cc-joint-core80-full-lex-repair-v8/)，对应已完成80作业七方法实验。
- **最新恢复等效修正版**：[cc-joint-recovery-equivalent-v1](cc-joint-recovery-equivalent-v1/)，对应后期12作业FO/FULL证据；不能冒充80作业已在此版本复跑。
- **干净机制消融**：[cc-joint-clean-ab-r3](cc-joint-clean-ab-r3/)，区分有序启动、持续服务与资源因素。
- **早期统一开发入口**：[cc-joint/README.md](cc-joint/README.md)，这是2026-10-27版本。其文档中的“当前唯一”仅对应当时的整理，不能替代上述后续版本。

## 保存规则

- `research_runs/`：完整实验日志、失败证据、预注册和源码快照，原位置保留。
- `research_notes/`：既有报告、论文、文献、图表与复算脚本，原位置保留。
- `official-artifacts/`、`research_sources/`：官方算法、GPU画像、Synergy矩阵与论文，原位置保留。
- `.venv-sia-baseline/`、`.venv-synergy/`：独立依赖环境，保留安装包；仅清除可重建字节码。
- `_archive/`：早期归档及本轮中间源码压缩包。不是废纸篓，不应直接删除。
- `maintenance/project_organization_20261101_v1/`：本轮归档清单、逐文件hash、恢复脚本和验证记录。

2026-11-20 已建立 GitHub 私有备份。最新搜索优化源码和本次验证通过独立提交保存，历史实验继续对应各自的冻结版本与逐文件 SHA-256。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。
- **最新搜索优化版**：[cc-joint-search-v1](cc-joint-search-v1/)，策略入口为 core80.py，包含 LAZY_FO 恢复预判和可选 LOCAL。12 作业规模验证与 20 作业对照见[完整报告](research_runs/sia_synergy/cc_search_validation_20261120/TWO_STUDIES_REPORT.md)。
