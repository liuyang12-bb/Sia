# 可重建与复跑

1. 保留本目录所有旧日志不动。rebuild_source.zip含完整284文件源码、GPU数据/CPU矩阵、输入与dependencies.lock.txt，逐文件哈希为preregistration.json/source_hashes。source_additions.patch只记录本轮新增模块/测试，父282文件未改。
2. 在全新项目根目录还原source为cc-joint-queue-ablation，按原依赖锁创建独立环境；测试路径依赖原项目层级及官方Synergy矩阵，须保留research_sources/Synergy/simulator/models的相同只读内容；本轮没有安装/升级任何依赖。
3. 复制rebuild_scripts到新根目录maintenance，复制本轮preregistration.json及source到同名research_runs/sia_synergy/cc_queue_mechanism_ablation_20261028_run1。不得复制旧cells到目标，否则独占创建拒绝运行；必须使用不同根目录，不能覆盖本项目证据。
4. 使用相同Python/依赖依次调用run_queue_ablation_20261028.py cell --name Q111/SIA_P/Q000/Q001/Q010/Q011/Q100/Q101/Q110；每格独立进程，启动前2GiB内存/5GiB磁盘检查。脚本verify不允许随意变更冻结源或依赖。
5. 历史逐轮回归需要原30格Q111/Sia参考日志，路径和hash见两个historical_regression.json。与新训练独立的复核入口audit_queue_ablation_20261028.py从raw完成事件与训练片段复算；应放在新的输出根目录，analysis独占创建。

所有复跑是操作说明，本轮交付后不自动执行。
