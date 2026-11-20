# 当前授权：根据8/20实验定位原因并完善联合调度

最终状态COMPLETE（优先）：最终attempt3四格全部完成，141测试通过，新旧原始事件/轨迹复核通过，数值异常5个子问题均在同一1860秒被拦截并合法重求，无已提交违规。8/M新G1385.5比AUX好6.15%；20/M3157.15仍比AUX差1.43%，比旧R改善3.66%。没有继续调参，104仍PAUSED，所有实验进程已退出。主报告JOINT_GLOBAL_EXCHANGE_REPORT.md＋FINAL_REVIEW.md。封存以source_version.json/delivery_manifest为准；旧冻结JSON branch字段继承标签已在FINAL_REVIEW明确更正，不改原manifest。不要重启下文任何历史session或继续下一阶段。

最新恢复（覆盖下文旧状态）：attempt2的最后20/M在1860秒返回物理audit真但必选集合不完整的向量，被拒绝未提交。新模式数值hook检查全部约束，只有“status0声称最优但原/取整x违反约束”才将同一fingerprint问题presolve=False重求一次，其他失败仍停止；原科学候选及主/次目标AST与attempt2相同。新增故障红绿，最终141全测通过；最终新目录attempt3，active manifest=attempt3/preregistration.json，session29311正在attempt3/run.py串行4格，不可重复启动/改运行代码。当前8/N已完成同JCT1606.875，8/M正在运行；8两格audit过才20。最终只用attempt3数据，旧两次失败保持。完成后baseline_reaudit.py、report.py（已指向attempt3）、完整前置快照/sha封存。原始Sia及Synergy源不改；104PAUSED。若此版本仍数值/审计失败则报告明确阻塞，不再无限追加版本或按JCT调参。

最新恢复状态（优先于下文历史进度）：首次8任务N完成、M在t720触发原执行审计假阳性，原始失败和v1快照全部保留。未修改Job.step或算法；未改动官方Job真实复现了checkpoint结束后0秒取整结算。新增4个边界红绿测试后139全测通过，execution_audit严格区分请求时间与完成截断时间并检查segment不得在checkpoint内。新目录attempt2，active manifest=attempt2/preregistration.json。session7232正在串行attempt2/run.py（不是87678）；只监控，不重复启动、不修改运行源码。8两格audit通过后20两格，仍原600轮和完全相同参数。最终仅用attempt2单一版本；v1结果不混合。完成后独立复核旧12格在新audit仍通过、N旧/新轨迹是否完全相同，生成报告/完整快照/hash，104仍PAUSED。

恢复入口：最终采用JOINT-G（整组交换，而不是仅删除短回本门槛）。原单任务速率不降/剩余Q回本会禁止必要的peer让渡；同状态离线原目标复原匹配，整组放宽才有部分状态收益。方案DESIGN.md，源码bundle_adaptation.py；旧类只新增no-op二级tie hook，新模式同分才最小真实启动成本。135全测通过，preregistration.json已冻结848源码/输入hash、84旧产物hash。当前session87678运行run.py，严格只4个新格，不重复启动；8两格audit通过才20。目标/响应/Q/D都未改，不能按新JCT反复设计到获胜。104继续暂停。

不恢复104格、不改变响应/容量/工作量、不按JCT寻优。隔离树sia-joint-budgeted-adaptation，分支exp/joint-budgeted-adaptation；基线2d5f9db + 前一轮11份未提交源码完整复制，旧树/旧产物保持。

先从不超过6个预定义旧日志状态重建合法候选、在线画像和真实锁，固定同一任务集合离线比较短服务回本过滤；用代码证据决定最小结构修改，不靠调大Q/预算/bonus。拟保持真实checkpoint、FIFO服务资格、累计实际延迟≤250和正训练条件，将重复的短服务回本硬门槛与原公平目标中的重配置成本区分。

完成红绿测试/原125回归、冻结后，最多新策略8和20各Neutral/Medium共4格，仍原始输入与600轮、独立进程串行、2GiB/5GiB门槛。8两格正确性完成才进入20，不按性能输赢决定扩展。旧12格仅在源/输入hash不变时复用；无第二轮参数调整。若最小方案失败如实交付，不优化到胜出。完成报告/完整快照/hash后停止。
