最新状态：30格执行和独立原始日志审计已完成，实验已停止。全部20/20完成：True。最终报告：completed_analysis_20261028_run1/COMPLETED_REPORT.md；最终验证：该目录final_validation.json。旧REPORT.md和pause_summary.json保留为原9格历史，不代表最新状态。不得自动继续调参、扩展负载或恢复104格。

当前恢复会话：95425；已启动resume_20261028_run1，仅监测现有进程，禁止重复启动恢复脚本。最终独立分析将写入completed_analysis_20261028_run1/，不覆盖旧9格报告。

最新状态：RESUMING_USER_AUTHORIZED。用户已明确授权续跑未完成21格。原9格与冻结源码/输入哈希已核验，原暂停记录保留为历史。原顺序、独立进程串行、2GiB/5GiB门槛不变。
恢复入口：maintenance/resume_cc_resource_factorial_20261028.py；恢复证据：resume_20261028_run1/。不得重复启动。

# CPU 内存资源压力与六方法对照

最新状态：USER_PAUSED，用户明确要求停止。已完成9/30且均20/20、独立审计通过；21格未运行。禁止自动恢复，详见USER_PAUSE_20261027.md。
会话4481已经在第10格启动前因内存0.887GiB安全退出，当前无本轮运行器可见活动进程。下一未启动格R10_medium_CC_CPU。
先前“已启动30格”指冻结计划和串行批次启动，不代表30格完成。

状态：G0通过，224旧测试+5适配测试通过，官方Sia独立3轮精确一致；30格已冻结并启动。
运行器maintenance/run_cc_resource_factorial_20261027.py，当前执行会话4481。只监测现有进程，不重复启动、不改冻结源码。
执行顺序：R00/R10/R01/R11各Medium六方法，再R11 Neutral六方法。启动清单在execution_manifest.jsonl，各格原始日志在cells。
结束后执行maintenance/audit_cc_resource_factorial_20261027.py独立复算并报告，随后封装源码与哈希。安全门槛停止则保留现场并报告，不静默跳格。

用户已授权：机会预检、六方法适配测试；通过后冻结30格（4容量x6方法Medium，加R11x6方法Neutral）。不修改CC-TAIL目标、基础候选网格、响应、工作负载和观察窗。
原CC与CC-TAIL保留；新隔离目录cc-joint-resource-study，父279文件hash见parent_source_hashes.json。
G0规则在计算前固定：取历史250GB CC-TAIL日志中，各GPU类型首个CF/DS各1GPU共驻、均已画像且非checkpoint的状态，最多2个。固定服务集合/GPU/batch，使用该状态已记录预测率和声明工作量，完整计算接收者与donor的净归一工作推进。只把分子/分母均在支持域的正净收益原网格方案计作关键机会；CPU压力/RAM压力场景分别至少需一个这样的CPU/内存见证。连续点只用于诊断覆盖，不加入候选。
若CPU或内存关键机会仅在网格外、缺少必要数据或适配测试失败，停止在预检/测试，不冻结或运行30格；不得改点位、补phi或调参。无损donor不是必须条件，但必须完整记录donor损失和静态ETA代价，不把净进度等同JCT保证。
运行保持20作业、60秒6000轮、Medium alpha=.5；Sia基线不加L2，四CC使用同一CC-TAIL控制器，只收缩CPU/RAM自由度。每格独立进程串行、2GiB内存5GiB磁盘门槛。
报告需区分全域legacy裁剪和支持域见证，不能删配置以隐藏覆盖缺口。负收益/删失保留，不能因JCT不好更换参数。禁止160作业或恢复104格。
