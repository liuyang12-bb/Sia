# 当前任务：优化联合调度的准入决策

最终状态：ENGINEERING_VALIDATED_EXPERIMENT_BLOCKED_MEMORY。新增JOINT-S与154项测试通过、旧G八轮逐分配/执行轨迹完全一致；静态诊断3个匹配状态＋1个NO_MATCH完成。正式小规模验证第一格在导入依赖后的2GiB门槛停止，未构造模拟器、无策略结果；事后可用内存1.800GiB。不得把该格当作删失JCT或算法失败。无运行会话需要续接。等待资源满足后须在全新attempt目录按相同冻结manifest启动，不能覆盖根目录已存在log。104保持暂停。报告JOINT_SELECTIVE_ADMISSION_REPORT.md、封存delivery_manifest.json。

基于已完成JOINT-G，用户授权继续改进。104格完整160作业正式实验保持暂停，不重启。保护旧工作树、官方源、全部历史产物。

隔离树sia-joint-selective-admission，分支exp/joint-selective-admission。先恢复2d5f9db基线＋前轮全部未提交源码并逐文件核对hash，保存完整旧源码快照与patch，不能仅凭HEAD标识执行源码。

本轮只检验一个机制：FIFO队首＋未兑现服务承诺必选；其余未承诺队列作业交由相同原公平效用选择，替代先贪心最大可行并发再优化。checkpoint完整锁、有效服务Q=300、累计真实启动D=250、60秒周期、目标p/lambda/bonus、网格、响应R0/E-legacy、phi、clipping、硬件容量、画像初始化不变。保留旧G，新增opt-in JOINT-S。队首不能与承诺并存时保持排空，不接纳后来者绕过。

先结构性选择不超过4个旧日志状态静态复原、比较；红测证实缺陷后最小实现；完整141项旧测试＋新机制测试和真实unseeded BERT链路。冻结后最多10个小规模运行：原8/20各Neutral/Medium新S四格；独立workload-5按同一轮转规则20作业，P/AUX/S各Neutral/Medium六格。独立进程顺序运行；8两格完成与审计通过才20；所有600轮，不按结果延长。内存2GiB/磁盘5GiB，不满足安全停止。独立输入在任何新结果出现前冻结；不按JCT修改策略、数据或参数。

所有成功单元从原始执行片段复核资源、rho、checkpoint、服务和JCT；删失不插补均值。保留失败版本，不在本轮追加第二种策略到获胜为止。交付改动、红绿、诊断、完整快照与结果报告后停止。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。
