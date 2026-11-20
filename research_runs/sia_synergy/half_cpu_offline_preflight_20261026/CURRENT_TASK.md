# 当前授权：半 CPU 离线预检，禁止启动仿真

基于已完成版本桥接。只读sia-joint-version-bridge、冻结config/R0/模型与原20、独立20 CSV及完成日志，新增本目录脚本/结果和独立报告。不得创建Cluster/Job、调用step/optimize、启动策略进程、恢复104格、改响应/phi/容量定义/工作负载或覆盖旧文件。

范围：main与既有half容量（AWS CPU12→6，DGX/RTX24→12；GPU/RAM不变）；Neutral和固定alpha=.5 Medium；参考点始终3CPU/62.5GB，不随容量重新归一化。

预定义离线分析：完整55条R0来源验证，相关3应用×11物理GPU配置，40实际job的历史已观察合法候选联合覆盖；逐点物理可行/进度floor/分子和分母裁剪分类。应用级rho查询以unit Goodput仅求响应，不假造绝对训练速率。

每个20作业负载从已完成G/Medium日志选2状态：首决策；首次所有三类应用均已有有效训练且当轮没有checkpoint锁。只取首个满足者；未找到则报告NO_MATCH。固定该状态所选GPU/placement和全任务集合，不改准入；比较同节点比例CPU、CPU-only精确DP、CPU+RAM离线整数模型。全作业rho不得低于当容量比例点。优化只最大化相对各自baseline的倍率和，证明无损互补是否存在，不冒充调度目标或JCT反事实。最多8个小离线整数模型，30秒上限；不靠结果修改状态规则或档位。

先验证半CPU比例包始终可行，区分真实模型内速度瓶颈与由clipping形成的无损让渡。只有明确覆盖、合法且有资源竞争/互补证据才建议进入四策略对照；边界不确定性如实标注，不修改曲线。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。
