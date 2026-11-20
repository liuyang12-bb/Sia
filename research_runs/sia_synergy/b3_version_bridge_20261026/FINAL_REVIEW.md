# 版本桥接最终入口

主报告：D:/科研启航/Sia/research_notes/synergy/B3_VERSION_BRIDGE_REPORT.md。

8新格＋16旧格全完成、原始审计通过；146测试。原20 Medium Sia/AUX/FREE/G=3477.15/3112.70/3200.85/3157.15；独立20=3730.50/3872.10/3618.60/3624.75。G在6组负载/响应中4组均值优于FREE，自由版未稳定恢复旧大幅收益。不要据此宣称G普遍优于AUX。

旧w10late B3对B4的净收益90.077%来自DS159。旧B1中316次GPU意图存在却被二次CPU admission拒绝，完整意图违规均为RTX CPU8>6；记录等待18960秒。当前main12/24/24 CPU及完整Sia意图不再是该场景，另有到达/phi/工程修复/辅助目标差异，不能把旧26%与当前均值当作单因素算法退化。

6同状态离线比较显示自由任务集合可提高即时公平效用，但由训练时间节省到JCT收益之间存在等待/重启抵消：原20 FREE−G，训练−2581、等待+2520、CP+935；独立20训练−5183、等待+4080、CP+980。

额外原始转移核验：Medium两组FREE停跑25/23次，G为0；非空GPU切换反而FREE41/43对G44/45。额外重启不能误归因为非法checkpoint迁移或未画像目标（观察切换全direct profile）。Neutral首轮两组20的目标都为2.0，却各14个GPU几何不同，存在版本同分规则混杂，不能把Neutral差异全归L2。

独立20 G对AUX的净收益由bert-118贡献超过100%，其他作业合计更慢；FREE对AUX同一任务贡献79.19%。主分析保留全部作业，诊断性去除不代替主结果。

执行版本以execution_source.zip和preregistration哈希为准，非裸2d5f9db HEAD。父G105源码不变，新桥接只增加3文件；官方Sia14方法AST一致。完整checkpoint/候选/原子提交/独立执行审计保留，自由版没有加入L2或掩盖负效用全空问题。

7次数值重求包含5旧＋2新，均未提交非法向量。根目录TOOLING_FAILURES保存非科学记录工具错误，没有重跑/择优策略结果。DELIVERY_MANIFEST为最终封存入口，requirements及快照可在新目录恢复。

JOINT-S未改、未跑；104保持暂停，无后续自动实验、不提交不推送。SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。
