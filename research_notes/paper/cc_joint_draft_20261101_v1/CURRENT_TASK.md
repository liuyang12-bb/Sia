# CC-Joint 文献调研与论文初稿

授权：2026-11-01用户要求调研CPU/主机内存联合资源优化和异构GPU调度论文，在项目中保存文献，吸收行文和图表形式，并结合有序启动、持续有效服务、CPU/RAM联合优化与已完成实验编写中文初稿。

本轮只读历史源码/结果，下载公开论文，独立复算已有指标、绘图与撰稿。不修改算法、不运行仿真、不恢复旧批次、不覆盖旧报告、不推送。新成果全部位于本目录。

主要系统版本：cc-joint-core80-full-lex-repair-v8，329文件源码快照；完整80作业7方法结果在research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run5。Clean A/B和arrival18是独立机制证据；最新mixed12恢复等效版本是边界案例，不合并冒充一个最终版本。

恢复入口：evidence/paper_results.json（42格逐作业重算及来源hash）、evidence/algorithm_formalization.md（源码公式）、literature/*sources.json（主来源书目）、scripts/plot_results.py与scripts/build_draft_pdf.py。目标输出LITERATURE_REVIEW.md、CC_JOINT_PAPER_DRAFT.md/.pdf、参考文献BibTeX、原创实验图及数据、README与完整hash清单。检查图文数字/版本/引用及PDF版面后停止，不自动补实验。

强制披露：SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。不宣称首次组合异构GPU与辅助资源；不宣称Sia没有checkpoint/重配成本；不将源矩阵覆盖等同目标硬件实测。论文是研究初稿，不保证接收。
