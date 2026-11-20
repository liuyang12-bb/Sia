# Sia＋Synergy联合调度研究

当前唯一开发入口：[cc-joint/README.md](cc-joint/README.md)。

- **当前实现**：`cc-joint/`，CC-Joint、比例消融与必要的Sia对照。
- **论文与报告**：`research_notes/`；最新实验原始证据在`research_runs/`，不清理历史结果。
- **官方参照**：`official-artifacts/`与`research_sources/Synergy/`，保留官方源码与响应数据。
- **旧代码归档**：`_archive/legacy_code_20261027/`，移出当前开发区，未永久删除。
- **清理与恢复**：`maintenance/cc_joint_cleanup_20261027/`，含完整备份、Git bundles、逐文件校验和恢复脚本。
- **环境**：继续使用`.venv-sia-baseline/`；不升级或删除现有环境。

不要再从旧`sia-*`目录开始修改策略。当前算法核心为
`cc-joint/sia-simulator/joint_prototype/completion_continuity.py`，统一运行入口为
`cc-joint/run_cc_joint.py`。整理不改变算法或实验结论，不自动启动任何新实验。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。
