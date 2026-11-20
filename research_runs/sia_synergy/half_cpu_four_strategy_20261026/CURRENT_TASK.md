# 半 CPU 四策略对照：已完成并通过独立验证

范围：版本桥接的 original20 与 holdout20，各 20 作业；Medium alpha=0.5；半 CPU，GPU/RAM/R0/E-legacy/到达/工作量不变。顺序 SIA-P、SIA-AUX、B3-FREE、JOINT-G，每组各四格，共八格。60 秒、最多 600 轮。达到上限作为删失保留，不插补 JCT。每格独立 Python 进程和全新子目录。原始 104 格实验继续暂停。

先运行 `prepare.py` 生成 `preregistration.json` 并核对哈希。只有可用内存≥2 GiB、磁盘≥5 GiB，才能运行 `run_all.py`；策略异常即停止、保留部分目录，不覆盖。完成后运行独立结果审计、比较并报告；不按 JCT 调整参数。

首次检查：可用内存 1.731 GiB，低于冻结门槛，未启动任何策略进程。历史桥接结果与预检 attempt2 均保留。

最终状态：八格全部完成 20/20，逐轮独立验证八格 `valid=true`。首个正式运行中第四格子进程因瞬时内存门槛停止，未产生该格的部分输出；经核验已完成前缀及源/输入 hash 后，在 `attempt2` 和 `attempt4` 的全新目录串行完成剩余格。`attempt3` 为预启动低内存记录，未启动策略。分析见 `comparison_summary.json`、`mechanism_summary.json` 和 `D:/科研启航/Sia/research_notes/synergy/HALF_CPU_FOUR_STRATEGY_COMPARISON_20261026.md`。104 格实验仍暂停；本轮不继续扩展。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。
