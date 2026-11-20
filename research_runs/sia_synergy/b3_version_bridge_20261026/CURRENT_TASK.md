# 当前授权：版本桥接，不继续 JOINT-S

最终状态（优先）：8新增格＋16复用对照24/24全部完成并经原始片段复算；146测试通过。会话48317与75418已退出，不启动任何恢复。原20 Medium P/AUX/FREE/G=3477.15/3112.70/3200.85/3157.15；独立20=3730.50/3872.10/3618.60/3624.75。6同状态、原始末状态计数、首轮同分几何核验完成。主报告research_notes/synergy/B3_VERSION_BRIDGE_REPORT.md，FINAL_REVIEW为入口。seal.py核验全部hash并生成最终DELIVERY_MANIFEST后停止，不改变算法或继续S/104实验。工具启动前错误/汇总父目录错误见TOOLING_FAILURES，没有重跑/择优实验格。执行源f64b64ea...，prereg cd9e2fdc...；不以裸2d5 HEAD冒充实际版本。

历史关键证据：historical_context.json和historical_admission_evidence.json。旧w10late B3对B4净收益90.077%来自DS159；旧B1/B4中该任务382个活跃决策轮有316次GPU意图存在却未辅助准入，316*60=18960记录等待；当前SIA-P完整意图＋main容量与旧CPU admission不同。不可将跨数据/容量/phi的旧26%与新结果当作单因素算法退化。

四方法：官方 Sia 决策＋比例资源（SIA-P）、固定官方 GPU 意图＋本项目辅助 MIP（SIA-AUX）、保留自由联合选择的正确性修复 B3（B3-FREE）、JOINT-G。

只用已冻结的原8/20与独立20作业，Neutral/Medium、main容量、R0/E-legacy、seed_profiles(1,8)、60秒、600轮。先8作业正确性关，再20。旧 B3 不加入 L2/bonus/新活性；空分配或饥饿如实记录，不能修到获胜。B3-FREE 是旧决策核心在当前固定数据/执行模型的移植，不冒充历史旧配置的逐字节复现。

隔离树 sia-joint-version-bridge / exp/b3-version-bridge。基线2d5f9db加parent_actual_source.zip实际源码；parent_tracked.patch及hash保留。原树、官方源、旧产物不变；104格持续PAUSED。

8个预定新格：原8 N/M B3-FREE；原20 N/M B3-FREE；独立20 N/M B3-FREE与G。其余16个历史对照仅在输入/源/依赖/执行口径核验一致时复用。运行前最终manifest冻结；独立进程串行，2GiB内存/5GiB磁盘门槛。无新参数搜索或自动扩大。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。
