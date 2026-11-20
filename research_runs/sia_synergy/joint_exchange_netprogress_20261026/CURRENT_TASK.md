# 资源交换与净进度优化实现及小型机制实验

用户已授权落实理论优化。基于实际 version-bridge 源码建立隔离树 sia-joint-exchange-netprogress，保护官方源和全部历史结果。

仅实现两个可独立开关：放开逐任务 proportional progress floor（物理最低配额仍保留），以及固定 H=300 秒窗口的真实启动损失主评分。保留现有 FIFO/L2/Q=300/D=250、完整 checkpoint 锁和准入集合；不实现第三项准入改变。p=-0.5、lambda_n=1.1、rho响应与原执行成本保持；窗口评分替代旧 restart factor，不能叠乘。

先新增失败测试，再实现；运行全部既有测试及真实 unseeded BERT 集成。预注册1–3作业场景：资源互补、GPU排序翻转、弹性配置CPU瓶颈、高启动代价，以及neutral/充足资源反例。合成机制数据与真实 Sia/R0 集成分开说明；固定观察上限，不因结果延长。每次模拟前内存2GiB/磁盘5GiB。104格正式实验仍暂停，不运行20/160作业新实验。

交付新源码、完整快照、测试及独立枚举证据、小型实验结果和限制，不推送。

完成：156项测试全过；7状态×7方法共49个静态结果与独立枚举一致；8格真实Sia三作业均3/3完成且独立原始日志审计valid。G10在真实小负载未改变结果，G01/G11较G00平均JCT退化2.28%，已保留。报告在research_notes/synergy/JOINT_EXCHANGE_NETPROGRESS_IMPLEMENTATION_20261026.md。停止于此，不按结果调参或扩大实验。历史正式104格保持暂停。
