# 三策略小型机制验证：SIA-P、SIA-AUX、JOINT-O

## 结论

六组固定短测试全部完成3/3，无删失、无fallback、无资源或执行审计违规。**固定官方Sia GPU意图后进行CPU/RAM优化，可以利用已有模型中的资源互补：Medium SIA-AUX平均JCT比SIA-P降低3.7055%，三个作业均改善。完整JOINT-O也实际使用了响应加速，但平均JCT比SIA-P增加24.6709%。**

JOINT-O的CIFAR10、DeepSpeech2完成更早，BERT明显变慢，导致平均JCT和makespan方向相反。本轮不能写成联合方案胜出，也不能把失败归因于没有敏感性机会。

完整候选日志使同状态反事实首次可以重建：固定同一Sia服务任务集合和真实checkpoint锁时，B（固定GPU＋辅助优化）和C（联合GPU＋辅助优化）的公平效用都优于A或持平，且C三个状态均高于B。但这是瞬时预测，不能替代运行JCT。另有真实服务承诺挡住更高预测效用列的证据。

**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。** 只有一个人工组织的三作业机制负载，每单元一次运行；没有对总体性能或真实硬件做统计推断。正式104格批次继续暂停。

## 版本、预注册及负载

基线 `2d5f9dbfcb7e7bfc2bfd9dc194b1838674a227c2`，新隔离树 `D:/科研启航/Sia/sia-joint-small-mechanism`、分支 `exp/sia-aux-small-mechanism`。仅新增三个源码/测试文件；未修改历史算法文件，没有推送。完整Python源码快照和逐文件hash分别为source_snapshot.zip、source_hashes.json；本轮新增文件是未提交源码，不能把基线HEAD冒称含有新实现的commit。重建时从基线创建新树，再使用本目录源码快照并核对hash；其余输入由preregistration.json逐文件固定。

preregistration.json在第一组运行前写入。Neutral再Medium，各按SIA-P→SIA-AUX→JOINT-O串行、独立进程运行。60秒调度周期，统一最多120轮（7200秒），完成即提前停止；没有根据结果延长。每次启动均检查2GiB可用内存。六次策略进程墙钟合计 **194.782秒**，不含先行全套测试和事后分析；测试用时48.998秒。

按workload-1原始行顺序分别取首个BERT、CIFAR10、DeepSpeech2记录，并统一设为0时刻到达，保留原始arrival和source_row_index。该到达控制在运行前固定，用来产生互补机会，不是原始trace重放。GPU demand、batch、应用对应训练总工作量均保留。

| Job | 原arrival | 原始行index（0起） | GPU demand | Batch | 控制arrival |
|---|---:|---:|---:|---:|---:|
| bert-28 | 4931 | 28 | 12 | 384 | 0 |
| cifar10-0 | 401 | 0 | 12 | 4096 | 0 |
| deepspeech2-8 | 1468 | 8 | 6 | 320 | 0 |

L2仍按真实控制arrival和稳定source_row_index执行FIFO，而不是按表格呈现顺序。本轮没有人为新增“纯RAM敏感”模板：CIFAR10、DeepSpeech2在当前矩阵中均有CPU和RAM响应，程度不同；BERT平坦。原映射、完整55配置R0 registry、E-legacy、alpha=0/.5、比例参考(3,62.5)不变。

容量沿用main：AWS4GPU/12CPU/250GB，DGX8/24/500，RTX8/24/500。CPU为合成单位。可行互补包已在前一诊断离线证明，例如DGX上BERT4GPU用1CPU/20.83GB每GPU，CIFAR4GPU用4CPU/62.5GB，每节点8GPU/20CPU/333.32GB，CIFAR rho约1.12077、BERT仍1。该包仅证明资源机会存在，没有强制实际调度器选用。

## 实现和共享契约

新增 `sia-simulator/joint_prototype/small_mechanism.py`：

- `EvidenceMixin`（47行）记录当前完整JobSpec、全部GPU候选原始/归一化Goodput及来源、固定phi、可行辅助列、被progress-floor剔除的物理可行列、每配置restart factor、age/used/wasted、前一资源配置、L2准入前后队列及完整服务承诺。几何/容量可行域可从这些输入和冻结网格重建。容量不合法点不作为候选，物理过滤由节点容量与GPU/配额乘积可确定复算。
- `AuxBase.optimize`（74行）实际调用 `SiaPolicy.optimize`，固定当轮GPU类型/数量/节点意图；在该配置下枚举辅助网格和合法placement。当前每GPU类型只有一个节点，所以没有额外节点选择差异。
- `fixed_set_solve`（29行）强制官方Sia已选任务每任务一列，未选任务零列，保留真实checkpoint完整几何/配额；目标使用当前B3的p=-.5、lambda_n=1.1、restart factor及两位小数变换，bonus为0。它不是Synergy-TUNE，也不加入L2。
- 求解失败仅允许全量、通过当前独立审计的比例fallback；无法证明安全则显式报错。GPU意图提交前再次核对；不丢任务、不部分提交。
- ProbeSia/ProbeJoint只增加日志；JOINT-O使用已有L2 FIFO、Q=300有效训练秒、D_bar=250、排空和重新入队，目标、候选、tie-break保持原样。

执行仍使用Job.step的真实当前batch/epoch训练模型，raw rate乘rho一次；checkpoint及有效服务秒数不乘rho。四类因素没有被偷偷统一：SIA-P/AUX保留官方Sia准入，JO保留L2，因此完整运行包含GPU、placement与准入差异。它们的纯GPU/辅助作用在同状态固定集合分析中隔离。所有模式都调用seed_profiles(1,8)，不把R0吞吐注入在线Goodput。

源码 `sia.py` 与official-artifacts逐字节相同，SHA256 `0DF97EBB660B9E5527EAAF1D2F5D6EA6634281203CD28050C7E0E79B80C4D432`。FullMetrics沿用已有执行审计，但SIA-AUX的solve_seconds只记辅助子问题，不含官方Sia求解和候选证据采集，不能用它与其他方法宣称调度开销优势；可用启动退出墙钟和round_wall字段检查总开销。

## 测试与独立复核

已有103项测试及新增4项共 **107项全通过**。新测试覆盖负效用下固定服务集合不被清空、固定GPU配置/容量、空意图不额外准入、无候选拒绝和MILP未决显式失败。真实六格又核对SIA-AUX每轮意图与官方Sia保持一致。

Neutral SIA-P与SIA-AUX共46轮的GPU意图、逐作业progress、epoch、完成事件、训练及checkpoint秒数逐项精确相同；配额可不同，但rho严格1。原始Sia核心未变化，已有原始CLI、真实未seed BERT链路等回归随全套测试运行。

analyze.py独立核验逐节点GPU/CPU/RAM、checkpoint完整锁、直接raw×rho、训练进度和互斥时间，并从原始片段积分资源小时。bert_trace.py从原始完成事件重新计算六格平均JCT，均与summary一致。无容量违规、非法部分提交、倍率重复或checkpoint加速。final_validation.json记录源码/输入hash终检、Neutral精确回归与旧中断日志/manifest保持原hash；artifact_hashes.json封存全部产物。

运行入口：`prepare_run.py`负责先测试、冻结、独立进程顺序六格；每格完整命令在对应_launch.json中。源码入口run_small_mechanism.py。重建时请使用新输出目录，不覆写本目录。

## 六格运行结果

所有完成率均3/3、fallback和异常均0。P95/P99为三个样本的线性插值，仅作这三个作业的描述。

| Response | 方法 | 轮数 | 平均JCT | Median | P95 | P99 | Makespan |
|---|---|---:|---:|---:|---:|---:|---:|
| Neutral | SIA-P | 46 | 1367.33 | 794 | 2535.50 | 2690.30 | 2729 |
| Neutral | SIA-AUX | 46 | 1367.33 | 794 | 2535.50 | 2690.30 | 2729 |
| Neutral | JOINT-O | 48 | 1856.67 | 2049 | 2754.60 | 2817.32 | 2833 |
| Medium | SIA-P | 46 | 1367.33 | 794 | 2535.50 | 2690.30 | 2729 |
| Medium | SIA-AUX | 45 | 1316.67 | 764 | 2473.10 | 2625.02 | 2663 |
| Medium | JOINT-O | 43 | 1704.67 | 2049 | 2509.80 | 2550.76 | 2561 |

| Response/方法 | GPU-h | CPU单位-h | RAM GB-h | 等待秒 | checkpoint秒 | 训练秒 | restart | wasted GPU-s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N/P | 6.7556 | 20.2667 | 422.2222 | 180 | 610 | 3312 | 7 | 2735 |
| N/AUX | 6.7556 | 6.7556 | 140.7182 | 180 | 610 | 3312 | 7 | 2735 |
| N/JO | 8.8944 | 8.8944 | 185.2713 | 180 | 535 | 4855 | 5 | 2335 |
| M/P | 6.7556 | 20.2667 | 422.2222 | 180 | 610 | 3312 | 7 | 2735 |
| M/AUX | 6.6133 | 19.1756 | 403.7008 | 180 | 610 | 3160 | 7 | 2735 |
| M/JO | 7.6044 | 20.0200 | 368.1145 | 180 | 510 | 4424 | 4 | 2035 |

所有方法总等待相同，不存在“靠更低等待必然改善JCT”的解释。JO共发17/16次Neutral/Medium服务，分别14/13次Q到期重新入队，无DRAIN事件。Medium JO有8个CPU饱和且GPU空闲节点轮事件，表示辅助配额占用实际起作用；本负载未触发队首排空，不把未发生事件当通过证据。

## 逐作业归因和执行兑现

| Job | Neutral P/AUX | Neutral JO | Medium P | Medium AUX | Medium JO |
|---|---:|---:|---:|---:|---:|
| bert-28 | 794 | 2049 | 794 | 764 | 2049 |
| cifar10-0 | 579 | 688 | 579 | 523 | 504 |
| deepspeech2-8 | 2729 | 2833 | 2729 | 2663 | 2561 |

Medium AUX相对P三者分别改善30/56/66秒。JO相对AUX，CIFAR改善19秒、DeepSpeech改善102秒，BERT却退化1285秒，净差1164秒，即平均JCT多388秒。JO的makespan比P短168秒，但平均JCT更差，不能只报告尾作业加速。

**BERT损失不是更多checkpoint。** P/AUX/JO的BERT都等待60秒、checkpoint360秒、restart2次；JO有效训练用了1629秒，AUX只用344秒，差值正好1285秒。BERT模板rho恒1，变化来自GPU选择及其时机。Sia初轮把三个任务均放RTX1；JO初轮把BERT放AWS1、CIFAR放DGX1、DeepSpeech放RTX1。初轮这些候选都标记unprofiled_sia_fallback，归一化Goodput为1，不能凭离线真实GPU性能推定决策器当时已知道速度差异。

JO的BERT先在AWS1有效训练300秒，累计原始进度约271.29；随后RTX4训练300秒，最后RTX8训练1029秒。SIA-AUX则在RTX1训练60秒、DGX4训练180秒、DGX8训练104秒。快GPU被响应敏感任务占用、冷启动同分选择、服务承诺保持和后续模型更新共同作用，构成已观测的GPU配置路径差异。不能从这一次运行分解出每项单独的因果比例，更不能为修复结果直接灌入离线R0吞吐。

Medium AUX的733训练秒rho>1、训练秒加权rho约1.03345；JO为979秒、加权rho约1.29973，最高rho分别2.46143/3.18007。它们都确实执行了CPU/RAM加速，且JO瞬时响应增益更大仍可能有更差JCT。BERT在AUX中rho始终1却提前30秒，是其他任务加速后GPU配置路径变化带来的间接收益，不可写成BERT的CPU加速。

分子clip按选中列轮次计，Neutral P/AUX/JO约19.12%/83.82%/78.49%，Medium约19.12%/18.46%/16.47%。这些是已选点比例，不等于整个网格覆盖率，也不能由低clip推断模型已校准；分母、坐标和phi的逐列证据仍在压缩日志中。

## 同状态A/B/C：不混淆准入或未来信息

选取Medium SIA-P轨迹第一次三任务活跃（60秒）、第一次初轮后重配置（120秒）、第一次已有训练且无真实checkpoint锁（180秒）。规则基于结构，未按预测收益大小选择。每个状态恢复完整当前JobSpec与全部列，重新用冻结Oracle生成候选几何和rho并核对；不推进任何Job，不读取未来完成时间。

A使用当前官方Sia已选GPU＋比例列。B保持同一任务集合、GPU配置及真实锁，优化辅助配额。C保持相同任务集合和真实锁，允许全部当前合法GPU配置及辅助列。三者使用完全相同的p/lambda、当前restart factor和父候选归一化；没有给Sia加入L2。B/C最大化同一公平效用，**不是最大化下表Goodput总和**，也没有使用R0 T0替代在线GPU估计。

| 时刻 | A效用/预测Goodput | B效用/预测Goodput | C效用/预测Goodput |
|---|---|---|---|
| 60 | .30 / 3.0000 | .95 / 5.4345 | 1.03 / 6.1531 |
| 120 | .89 / 6.9317 | .91 / 6.9978 | 1.30 / 12.7014 |
| 180 | 1.07 / 7.5697 | 1.08 / 7.6366 | 1.19 / 13.4295 |

60秒时A三个RTX1各用3CPU/62.5GB；B把BERT降到1/20.83，CIFAR用12/125、DeepSpeech9/125，总RTX22CPU/270.83GB，rho分别1/2.46143/1.97307。C再把CIFAR放到DGX1取得24CPU/125GB的rho3.18007，把平坦BERT放AWS1，释放RTX。该状态证明GPU类型选择与辅助配额存在共同机会，但同分冷启动BERT的预测没有辨认AWS的实际慢速，正是外推预测需要谨慎的例子。

120秒B保留CIFAR DGX8，无法增加其辅助配额；只给RTX4 DeepSpeech每GPU4CPU，使rho约1.041。C可改为CIFAR DGX1、DeepSpeech RTX1，利用更高配额。180秒C选择CIFAR DGX2/每GPU12CPU和125GB，而非A的DGX8比例点。每列都有原始/有效Goodput、配额和实际restart factor，见same_state_ABC.json与原始probe字段。

预测Goodput数值是各任务归一化后的模型值，不可当作相同单位的真实训练样本吞吐；效用包含重配置惩罚，表中预测Goodput未额外扣除具体启动秒数。跨配置时，BERT/CIFAR/DeepSpeech的模型penalty分别沿用120/30/25秒规则；实际执行延迟继续用原应用参数。本轮未增加辅助配额变更的数据加载开销。

## 单独量化L2承诺

从Medium JO轨迹取最早两个存在旧服务承诺的状态（120、180秒），固定同一个已选任务集合。比较H：保持已有服务几何/配额，F：仅在离线求解中解除服务承诺的几何限制；真实checkpoint锁继续完整保留。

- 120秒有BERT真实checkpoint锁，服务限制挡住290列，但H/F效用均1.34、预测Goodput均12.7146。更多列不自动构成有效机会。
- 180秒真实checkpoint已结束，服务限制挡住404列；H效用1.34、Goodput12.7146，F效用1.39、Goodput18.3445。F选择CIFAR DGX2，H仍保持初次DGX1服务。这证实服务承诺在这个状态限制了更优预测配置。

F只是一项同状态约束松弛诊断，没有作为执行策略、没有删除L2或承诺保护。它不提供新的饥饿保证，也不能把静态提升44%左右写成JCT提升。完整轨迹里JO仍能反复提供服务，三个作业均完成。

## 下一步判断

本轮已经证实：辅助资源优化有机会、能执行兑现；GPU联合选择也有同状态预测机会；当前完整JO的JCT并未因此改善。直接扩大到20作业性能对照还不是优先项。

建议下一步先审查两个研究语义：①冷启动GPU估计同分时，长服务承诺是否应锁定该GPU配置；②是否只冻结真实checkpoint和最低有效训练服务，而允许经过证明的服务中配置调整。任何新执行方案都必须保留强制首次服务、有限前序负载下重复服务、原子资源与真实锁保护，不能直接删除L2，也不能从本轮JCT选择Q或灌入离线GPU真值。

可先利用已经保存的完整候选做固定集合与固定配额/几何的离线分解，再预注册一个新的小型活性方案。SIA-AUX可作为保留的严格串联对照；现有TUNE仍作为外部参考，不被改名替换。未经新授权，不扩展20作业、九cohort或恢复104格。
