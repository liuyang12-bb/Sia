# Sia＋Synergy 数据机会与联合策略聚焦诊断

## 直接判断

**现有数据能检验合成机制下的CPU/RAM敏感性调度，但目前不足以有效检验“完整联合策略是否优于Sia”。数据中有可行机会，当前实现的搜索与准入又确实限制了机会；两者不能相互替代作为解释。**

固定R0的55个应用/GPU配置中，32个存在节点内可行且预测加速超过1%的辅助配额。存在敏感任务与平坦BERT配额互补的具体资源包。SIA-TUNE已在三个DeepSpeech2作业的实际训练片段兑现响应收益。然而Medium JOINT-O没有运行，不能据此评价完整联合优化是否发现了这些机会。Neutral JOINT-O确实降低CPU/RAM分配，却没有降低JCT；该现象不能说明非Neutral优化无价值。

可信范围：单个完整workload-1、六次完整运行、六个静态状态、固定R0/E-legacy。没有新增训练仿真。**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。**

## 现场、版本与保护

- 当前树：`D:/科研启航/Sia/sia-joint-full160`，HEAD `2d5f9dbfcb7e7bfc2bfd9dc194b1838674a227c2`。
- 实验冻结：`../sia_r0_full160_comparison/attempt2/source_snapshot.zip`。逐字节核对全部799文件，**0差异**。Git status的既有M不能被误当成本轮算法变化；执行字节与冻结相同。版本证据：`review2/version_check.json`。
- 工作区及相关子树未发现适用AGENTS.md。已读CURRENT_TASK和USER_PAUSE_20261025；自动续跑状态仍为PAUSED。
- 完整证据仅六格。Medium JP只读原始压缩流中78条完整轮次，4680秒、15/160完成；仍保留EOF与原hash，不计算总体JCT。Medium JO没有结果。
- 六格均从完成事件、片段、placement和配额重新积分，独立核对summary；容量、响应、锁、互斥时间、进度审计均通过。`verification.json`记录静态解容量和固定几何/承诺检查；所有冻结输入hash再核验通过。
- 六份初次原始审计保留于本目录，最终静态分析位于review2。新诊断脚本曾把Neutral历史承诺错误按Medium过滤，形成空候选；已修正诊断定义，首次错误说明保存在DIAGNOSTIC_ATTEMPT1.md。实验源码和历史结果没有被修改。

## 数据验收与响应依据

| 层次 | 来源与验收 | 能支持的判断 / 限制 |
|---|---|---|
| GPU训练性能 | Sia Artifact traces和Job.step模型；固定R0 registry有原文件、行号、microbatch、median和hash | 来源可复核；离线固定吞吐不是当前batch/epoch的联合硬件测量 |
| CPU×RAM矩阵 | Synergy `simulator/models/{alexnet,bert,deepspeech,res50,mobilenet}_1.py` 的9×5 tput字面值 | 官方仓库画像数据；本轮未验证测量设备、噪声及目标平台可迁移性 |
| 应用映射 | cifar10→alexnet，bert→bert，deepspeech2→deepspeech，imagenet→res50，yolov3→mobilenet | 全部为固定合成映射；名称相似也不构成同平台实测匹配 |
| phi | R0完整11配置域、每GPU参考吞吐相对域内最小值；CPU clip到[.5,4]，RAM为平方根再clip到[.5,2] | 策略无关、不会随获胜轨迹改变；GPU吞吐推CPU需求、平方根推RAM需求均为假设 |
| 覆盖 | 5应用×11配置=55；主容量内1240个离散单作业点，缺失拒绝 | 单独可行不等于多个作业能同时取得这些点 |
| 边界 | 6/55参考分母配置越界；508/1240可行网格分子点越界（40.97%） | 数值可查询，但边界平台区可能由clipping制造，不能当测量饱和证据 |
| 构成 | workload-1：134 CIFAR10、10 DeepSpeech2、7 YOLOv3、6 BERT、3 ImageNet | 83.75%同一敏感模板，平坦模板仅3.75%；不是均衡敏感性混合 |
| 可行加速 | 32/55配置最高rho>1.01；CPU单独31/55、RAM单独31/55 | 证明合成模型有独立CPU/RAM响应；不是31种真实硬件测量 |

定义 `r0=(T0/GPU)/min_domain(T0/GPU)`，`T0=GPU×b_ref/median(step_time)`。固定参考为每GPU `(3,62.5)`，执行

`rho(c,m)=[R_clip(c/phi_c,m/phi_m)/R_clip(3/phi_c,62.5/phi_m)]^alpha`，Medium alpha=.5、Neutral alpha=0。

矩阵轴CPU为1/2/3/4/5/6/9/12/24，RAM为20.83/62.5/125/187.5/250；执行候选CPU网格没有5，详见冻结config。双线性插值前分子/分母均clip到源轴范围，正值缺失不补1。比例点rho=1由分子分母相同导致，**不是没有CPU瓶颈的实验证据**。CPU单位是合成单位，旧字段cpu_core_hours沿用名字，不代表真实核心。

矩阵原值范围：BERT恒1；AlexNet .1–2.94，DeepSpeech .1–1，Res50 .1–1.15，MobileNet .1–1.55。CPU方向无负增量，但AlexNet和MobileNet存在边际斜率回升（分别7、4个相邻斜率比较）；不能声称所有曲线严格收益递减。RAM低端跳变及后续平台也非平滑凹函数。

phi随GPU类型/数量变化但不随在线训练变化。例如DGX CIFAR10的phi_c在1/2/4/8GPU为2.731/3.189/2.968/2.845，通信与所选microbatch可使其不单调；它是每GPU需求假设，任务总需求还乘GPU数。BERT DGX四种配置phi_c均被clip到4，CPU坐标3/4低于源下界；全平坦矩阵仍保持rho=1，不能由此反证缩放正确。

### 任务/配置证据

以下是单作业节点可行点，速度相对同GPU配置的(3,62.5)。最优点可能占尽整节点CPU，不能当作所有任务可同时获得的需求。55配置全表见response_config_table.csv；workload-1全部160作业×11配置的1760行证据见task_config_evidence.csv。原始GPU画像源行保存在JSON source_rows。

| 应用/模板 | GPU配置 | 比例CPU/RAM | 可行替代CPU/RAM（每GPU） | rho | CPU/RAM总增量 | 信度 |
|---|---|---|---|---:|---|---|
| CIFAR10/AlexNet | aws:1:1 | 3/62.5 | 12/125 | 2.4772 | +9 / +62.5 | 合成迁移 |
| CIFAR10/AlexNet | dgx-ext:1:1 | 3/62.5 | 24/125 | 3.1801 | +21 / +62.5 | 合成迁移 |
| CIFAR10/AlexNet | rtx:1:1 | 3/62.5 | 24/187.5 | 3.1069 | +21 / +125 | 合成迁移 |
| DeepSpeech2/DeepSpeech | aws:1:1 | 3/62.5 | 6/125 | 1.3705 | +3 / +62.5 | 合成迁移 |
| ImageNet/Res50 | dgx-ext:1:1 | 3/62.5 | 9/125 | 1.6528 | +6 / +62.5 | 合成迁移 |
| YOLOv3/MobileNet | dgx-ext:1:1 | 3/62.5 | 24/125 | 2.6354 | +21 / +62.5 | 合成迁移 |
| BERT/BERT | 所有11配置 | 3/62.5 | 1/20.83 | 1 | 每GPU−2 / −41.67 | 平坦官方矩阵＋合成映射 |
| 任一敏感应用 | aws4 / dgx8 / rtx8 | 3/62.5 | 网格内单独最优仍3/62.5 | 1 | 0/0 | 当前容量导致无增加空间 |

有具体互补包：DGX节点BERT4GPU用1CPU/20.83RAM每GPU，CIFAR10 4GPU用4CPU/62.5RAM每GPU，总8GPU/20CPU/333.32GB，容量24CPU/500GB内；CIFAR10 rho=1.120772，BERT不慢。比较两个都比例分配时24CPU/500GB，该包既节约资源又提供合成加速。但真实日志是否同时有这两个可选配置，仍受在线候选、锁、服务承诺限制，不能称作已发生收益。

## 实际决策链和限制

调用路径：FullCluster→R0Cluster.optimize_policy→JointCluster/Cluster调度→FullJointPolicy→R0Policy→NeutralL2Policy。候选 `_raw_candidates` 调用Sia `_compute_goodputs_v2`，保留direct/projected/unprofiled来源、弹性min/max及scale-unit；父域先归一化。R0Policy随后替换固定phi。FactorScheduler枚举合法GPU配置、节点placement、辅助网格；按逐节点GPU/CPU/RAM和比例进度floor过滤。训练执行与候选Goodput不同：前者由Job.step的实际当前batch/epoch速率推进，后者是逐作业归一化预测模型。

列效用为 `u_jc=lambda_n−min(round((max(G_jc*rho_jc*f_restart,ZERO_ALLOC_GAIN))^p,2),clip)`（p=−.5、lambda_n=1.1）。restart factor使用当前age、restart及used/wasted状态；不是简单固定0.9：原型build_joint_columns的临时效用最终被factor_columns覆写。锁定作业保留GPU配置、节点/placement和CPU/RAM配额，原子容量审计后提交。

**L2首先确定任务集合，而非把所有任务放进最终效用优化共同选择。** 已服务任务在Q=300有效训练秒内固定完整几何与配额；真实checkpoint与该承诺分开记录。按真实arrival/source index FIFO加入队列；服务到期排到队尾；队首无法与已有承诺共存则排空，避免无限回填。每次可行性MILP都令指定任务 `sum_c x_jc=1`，其余0；遍历ready按能否加入集合决定admission，最后才在固定集合上最大化上述效用并词典序tie-break。

因此lambda_n在固定集合上是常数，不能为排队敏感任务创造额外机会；已有服务承诺在约五个60秒轮次内又只有原配额列。中途空出的CPU/RAM可能无法及时调给承诺任务。JO仍能改变新发服务的GPU与配额，也能通过不同资源包改变可行准入集合，所以不是完全没有联合选择，但其收益范围显著窄于“每轮自由联合调度所有任务”。

比例进度floor要求每个配置rho不低于该配置比例值。它可避免牺牲单个任务的瞬时速度，却排除了“某任务小幅变慢、释放资源后其他任务显著变快”的Pareto权衡。不能由此直接建议删floor；需先决定是否允许这种权衡及公平性保障。

SIA-TUNE实际调用官方Sia优化/放置，再固定GPU意图；按FIFO尝试模板需求×phi，再试比例，必要时把未锁peer全部降到比例重排。没有遍历中间配额，也通常不会主动把平坦peer压低到比例以下。需求点超过节点CPU时即退比例，会漏掉4/6/9等可行中间点。此为适配搜索范围限制，不能冒充官方TUNE实现错误。

目标与JCT偏差包括：归一化公平效用没有剩余总工作量/完成事件目标；rho预测可能受clip平台影响；物理服务秒数固定而非固定训练进度；restart factor是历史状态近似；300秒服务到期即重排可能增加实际checkpoint；辅助配额调整本身没有额外加载成本模型。服务到期、锁及放置与GPU候选共同改变轨迹，不能只归因于CPU算法。

## 六个同状态诊断

选择规则在求解前写入脚本，分别取第一次出现以下结构的不同状态：JO Neutral排空、服务承诺且有等待者；TUNE Medium重配置、至少两任务运行、敏感任务且CPU闲置、有等待者且使用比例点。未按收益大小选状态。逐列输入和结果见review2/state_1.json到state_6.json、same_state_comparisons.json。

对TUNE日志，A是该同状态真实Sia GPU意图＋比例辅助资源（保留真实锁）；B固定相同GPU、任务集合和placement，在既有网格/进度floor及逐节点容量下最大化预测归一化Goodput。B是静态机会诊断，不是声称已经运行的B4或其公平效用。两种Neutral JO状态来自决策后的承诺快照，A必须保持这些配额，**不能称为已重建的官方Sia反事实**，仅诊断锁对可调空间的限制。

| 结构/来源 | 模拟时刻 | A预测Goodput | B预测Goodput | A→B CPU / RAM总量 | 等待 | 解释 |
|---|---:|---:|---:|---|---:|---|
| JO Neutral首次排空 | 3000 | 40.3202 | 40.3202 | 20→20 / 416.60→416.60 | 1 | 11个服务承诺、20GPU；辅助资源空闲不等于还能准入GPU任务 |
| JO Neutral承诺＋等待 | 3060 | 43.4091 | 43.4091 | 20→20 / 416.60→416.60 | 1 | 同样11承诺；当前几何和配额锁定 |
| TUNE Medium首次重配置 | 60 | 1 | 3.1069 | 3→24 / 62.5→187.5 | 0 | RTX1GPU；TUNE需求约26.42CPU超24，直接退比例；中间网格可行 |
| TUNE Medium首次两任务 | 180 | 5.9374 | 8.0444 | 27→48 / 562.5→687.5 | 0 | 同9GPU且节点容量合法；给空闲辅助资源有预测收益 |
| TUNE Medium敏感＋CPU闲置 | 1080 | 38.5832 | 40.8740 | 54→58 / 1125→1250 | 0 | 历史实际为38.9537，介于A/B；TUNE已选一部分机会 |
| TUNE Medium比例＋等待 | 3000 | 49.5370 | 52.0986 | 60→60 / 1250→1250 | 1 | 20GPU；总辅助资源不增加仍有重分配收益，比例分配非最优 |

六个静态MILP均OPTIMAL且复核容量/固定GPU/placement/锁合法。A/B的GPU和placement不变，没有新增GPU重启代价；辅助重配开销沿现模型未计，不能直接折成现实净收益。Neutral两个状态保持实际alpha0，其余alpha.5，没有改变历史响应。

**C完整联合GPU反事实不能精确重建。** 日志candidate_summary/factor_candidate_keys只保存未选配置ID、几何或配额，没有全部候选当前Sia Goodput、SpeedupFunction参数、age/restart factor及batch状态。为避免用离线R0冒充在线GPU模型，本轮不补造这些值，不调用训练回放。C在机读表明确NOT_IDENTIFIABLE；也不拿另一策略不同轨迹上的决策充作同状态C。最小补充是未来授权测试时持久化当前全部候选查询及重启因子，不需未来信息。

## 六组完成结果与执行兑现

| 响应/方法 | 平均JCT | P95 | Makespan | 等待秒 | checkpoint秒 | productive秒 | restart |
|---|---:|---:|---:|---:|---:|---:|---:|
| N SIA-P | 8323.225 | 18223.00 | 79039 | 734387 | 56765 | 540564 | 699 |
| N SIA-TUNE | 8323.225 | 18223.00 | 79039 | 734387 | 56765 | 540564 | 699 |
| N JOINT-P | 8337.775 | 29161.70 | 81643 | 590807 | 77145 | 666092 | 1097 |
| N JOINT-O | 8337.775 | 29161.70 | 81643 | 590807 | 77145 | 666092 | 1097 |
| M SIA-P | 8323.225 | 18223.00 | 79039 | 734387 | 56765 | 540564 | 699 |
| M SIA-TUNE | 8003.100 | 18258.00 | 79501 | 689807 | 53255 | 537434 | 674 |

Neutral JO较SIA-P：平均JCT差+14.55秒（+0.175%）、P95约+60%、makespan约+3.29%。等待减少143580秒，但productive增加125528、checkpoint增加20380，三者净增加2328秒，恰为160×14.55。低等待不能单独证明训练效率改善。restart多398次；这与服务轮转/配置变化相容，但因GPU/placement/admission同时不同，不能把398次全部断言由L2导致。

Neutral JO/JP逐作业JCT一致。JO分配CPU小时397.735、RAM小时8284.814，相比JP的1193.204、24858.420约为三分之一；GPU小时均397.735，SIA-P为390.411。Neutral资源下降是配额下降，不是加速。JO选中点94.66%分子clip，JP约2.25%；边界平台明显影响资源解释。

Medium SIA-TUNE相对SIA-P平均JCT改善3.846%，但makespan增加462秒，P95增加35秒。实际加速仅：deepspeech2-8为250训练秒、-12为35秒、-21为35秒，最高rho1.3705118；全部训练时间加权rho约1.00017088。第一处分叉发生1140秒的deepspeech2-8：同raw rate1.223323593、同35训练秒，rho从1升到1.370511790，有效率从1.223323593升到1.676579407。其CPU4.3632、RAM75.3741每GPU，分母.532395588、分子1，`sqrt(1/.532395588)=1.370511790`，未clip。证明响应确实作用于训练，并非只改目标分数。

3.846%的JCT差不能等同于直接训练平均快3.846%。此后在线画像、队列和Sia配置分叉，长尾/完成次序放大或抵消初始变化。逐作业为71改善、61退化、28相同；主要正贡献作业包括cifar10-82/144/76/141/120，各降低约9160–10640秒，这五项合计49202秒，占净51220秒收益的96.06%（包含其余任务正负抵消，不是占所有正收益的比例）。去掉最大一项后的诊断性平均改善为255.22秒，主结果仍保留全部160作业。完整正负差见paired_metrics.json。SIA-TUNE的GPU小时387.613、CPU1162.872、RAM24221.946；比SIA-P略低，不代表所有任务更快。

已有Stage2C20-job报告可作旁证：Medium JO比JP平均JCT差2.49%，但GPU/RAM小时减少；该实验的固定单提案不是官方Sia，容量也不同，不能合并当前160-job对照。Medium JO本轮未执行，JP中断，完整联合CPU/RAM收益仍待隔离验证。

## 问题、代码证据与最小建议（本轮不实施）

代码路径均相对 `D:/科研启航/Sia/sia-joint-full160/sia-simulator/`。

| 严重度 | 证据文件/行 | 影响机制 | 最小建议与验收 |
|---|---|---|---|
| High | `joint_prototype/neutral_liveness.py:87-97,108,143-158,166-170` | 先FIFO定服务集合、承诺固定完整配额；效用不能选择未准入敏感任务 | 先补完整列/held原因日志；用同状态离线对照量化集合和配额承诺损失，再审定共同救援规则，禁止直接删L2 |
| High | `joint_prototype/joint_profile.py:179-204,263-276`；`r0_liveness.py:19-48` | 合成缩放和分子/分母clip可能制造敏感性/饱和，目标硬件真实性不足 | 明确边界内覆盖验收与clip计数，保留E-legacy；新边界模型另预注册，不能按JCT挑phi |
| High | `joint_prototype/joint_integration.py:544-606` | 未保存全部在线候选和重启状态，C反事实不可复算 | 持久化每候选规范化前后Goodput、rho、restart factor、过滤/锁原因与状态hash；验收从日志重建相同MIP，不执行训练 |
| Medium | `joint_prototype/synergy_adapter.py:94-120,143-196`；`full160.py:66-102` | demand/fair跳跃，peer仅降到fair，遗漏中间点及平坦资源回收 | 保留TUNE外部基线；新增“固定官方Sia GPU＋有限列辅助优化”作为单独方法，验证GPU意图严格固定 |
| Medium | `joint_prototype/joint_scheduler.py:326-342,699-704` | 每配置比例进度floor拒绝任务小幅让速换整体收益 | 先报告被floor删掉的可行列及潜在机会；若要放宽需预注册公平/饥饿保障，不能默认通过 |
| Medium | `neutral_liveness.py:74-83,218-221`；`joint_integration.py:381-392`；`job.py:565-607` | 300物理训练秒轮转、非JCT目标与历史重启因子，增加checkpoint可能抵消效益 | 记录服务到期→实际重配置→损失链，固定Q不调参；同状态对照改变任务选择与否分别审计 |
| Medium | `joint_profile.py:309-310`；`job.py:431-442,495-543` | 辅助配额变更无额外数据加载成本；预测归一化Goodput与执行率不同 | 报告分别命名两类速率，保留直接raw记录；若增加开销须独立定义来源和边界测试 |

未发现本轮实际rho重复施加、非法容量提交或checkpoint训练推进的证据。缺少完整Medium JO结果属于证据不足，不属于算法已失败。

## 下一轮最小方案与准入条件

建议先做三作业确定性机制测试，再做一个预注册20-job对照；此次不启动。核心三方法为①真实官方Sia＋比例执行，②相同官方Sia GPU意图冻结＋辅助有限列优化，③完整联合优化；TUNE保留为外部串联参考。它们使用相同Sia画像初始化、到达/工作量、R0及E-legacy、容量、restart和执行模型。辅助有限列方法必须单独命名，不冒称原Synergy-TUNE。

负载预先按离线响应类别定义：一个平坦任务、一个CPU敏感、一个RAM敏感，先分别证明在同节点内有可共存的资源包；类别选择根据矩阵轴向变化与可行容量，不看JCT。源记录保持真实应用/工作量，CPU/RAM映射继续明确合成。负载平衡是额外机制场景，保留真实trace自然构成的外部有效性对照，不能替换旧trace结果。

活性不能仅删除L2。建议评审“正常阶段保持当前公平效用，超时FIFO队首进入强制救援”的备选：checkpoint和已发服务承诺完整保留；救援要求队首单独可行、正执行率、有界启动和有限同时排在其前的工作；阻挡时停止新回填直到可发至少Q有效训练服务，延迟不计Q，到期更新服务债务。这样提供有条件的首次/重复服务保障；有限工作量、持续正速率和有界前序工作才推出最终完成，过载到达下不声称统一完成界。具体超时/债务参数需预注册，不由当前收益选择。

隔离L2的方法是：同一快照、同一完整候选域、相同mandatory checkpoint集合，分别离线比较L2固定服务集合与共同救援mandatory集合，先量化被准入挡住的候选；随后在授权的小规模受控三方法中统一采用同一个准入规则，命名为“共同准入适配”系列。真实原Sia继续外部基线，不能把施加共同规则后的版本叫官方Sia。已有L2系列与新系列完整配对，不混用历史指标。

最小优先事项只有三项：①补齐可离线重建的全部候选/准入/承诺证据；②添加严格固定官方GPU意图的辅助优化对照，保留TUNE不替换；③在有节点内互补包的预注册小负载上验证模型与共同活性规则。phi/边界争议单独解决，先不改旧模型。

验收：候选/过滤原因全覆盖；锁、服务承诺、原子性/故障注入通过；同状态B不改GPU意图；执行raw×rho独立校验且checkpoint不加速；CPU/RAM边界计数分子分母分开；有限负载服务等待界按状态机逐步核验。20-job必须完整配对或明确删失，平均JCT只比较全完成对照；同时报告median/P95/P99/makespan、逐作业正负差、等待/训练/checkpoint、restart和资源小时。单状态预测收益只作为机制证据，不替代JCT结果。扩展九组前需完成上述单cohort gate，仍需新的用户运行授权。

## 复算入口

本目录diag脚本仅读取冻结文件和日志。首次diagnose.py完成六格原始审计；finish.py复用这些审计并在review2生成最终静态结果（最多六次10秒MILP）；execution_detail.py追踪实际rho片段；paired_metrics.py导出逐作业差和任务/配置表；verify.py检查独立审计、静态容量/锁、799源码及全部输入hash。复算请使用新的输出目录，现有脚本用独占创建避免覆盖。正式实验及sia-104保持暂停。
