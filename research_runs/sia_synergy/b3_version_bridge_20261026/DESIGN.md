# 版本桥接预定义

问题：在相同执行物理条件下，自由联合选择相比 JOINT-G 的 L2/适配限制如何改变 JCT？不是复原旧容量、动态phi或历史缺陷，也不是让新算法获胜。

|方法|GPU决策|辅助/放置|任务集合|锁与fallback|
|---|---|---|---|---|
|SIA-P|真实官方Sia MIP＋placement|3CPU/62.5GB每GPU|官方意图，无二次准入|当前完整checkpoint锁和原子检查|
|SIA-AUX|固定同一官方GPU意图|本项目辅助MIP，不是TUNE|不得丢弃官方意图|失败时完整合法比例回退|
|B3-FREE|真实在线Sia多GPU候选|旧B3联合有限列MIP|全部可选，只有真实checkpoint必选|Stage1.1当前候选审计/刷新回退；无L2|
|JOINT-G|相同原始GPU候选＋已验收适配过滤|整组联合交换|FIFO服务资格、排空、Q300/D250|完整锁与当前数值保护|

四者p=-.5/lambda_n=1.1不变；B3-FREE字面调用JointSiaPolicy.optimize/_b3_result，不复制改写公平目标。普通作业sum(x)<=1，真实checkpoint sum(x)=1；可合法全空，不添加sum(x)>=1、轮转或bonus。旧求解无次目标，G保留当前startup与geometry同分规则；这是明确的版本差异，不将全部差异归因于L2。

B3-FREE仅继承当前R0固定phi和执行环境，不向在线Goodput灌入R0吞吐。原B3候选Pareto剪枝保留；证据日志另列完整网格。数值保护直接复用G的同模型guard，不增加约束或改变系数；只有非法最优向量才同模型关闭presolve重求。无可用解按Stage1.1刷新现轮资源的fallback，不复用旧rho。

复用既有原8/20、独立20的N/M数据及16格hash封存结果，补8格。原8是正确性关，不按性能胜负决定进入20；如合法删失，保留并报告，不延长600轮。硬性实现/资源/锁审计错误则停止。每格独立Python串行，2GiB/5GiB。不跑160或恢复104。

主比较：B3-FREE对SIA-P/AUX/G；G对AUX；Neutral→Medium的差分。主结果保留全部作业，按相同任务逐一分解等待、checkpoint、训练与重启。完整均值仅用于全完成配对；删失用完成率/RMST，不插补。最长尾贡献和去掉单个高影响作业仅诊断，不代替主表。

限制：即便桥接发现L2使收益变小，也不能量化容量/动态phi/到达/旧锁缺陷各自历史贡献。这需要另行控制实验；不恢复有缺陷旧版本、也不根据结果改变本轮数据。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。
