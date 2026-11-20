# JOINT-R：有界配置适应的联合调度改进

## 结论

本轮实现并验证了一个针对决策限制的改进：**保证任务继续获得有效服务，但不再把服务承诺等同于300秒的完整配置冻结。** 保留真实checkpoint锁、FIFO、Q=300、D_bar=250、既有公平目标和R0/E-legacy，仅改变服务中的候选资格与实际延迟记账。

在原样复用的三个作业上，Neutral/Medium两次新运行都完成3/3，122项测试通过。Medium新策略平均JCT为1335秒，较旧JOINT-O改善21.69%，较SIA-P改善2.36%；仍比SIA-AUX差1.39%。Neutral平均JCT较旧JO改善26.57%，与SIA-P接近，但P95/P99和makespan略差。**改进有效缓解了已观测问题，没有证明完整联合策略全面优于串联，也没有调整到获胜为止。**

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。

## 现场、隔离与冻结

- 新树：`D:/科研启航/Sia/sia-joint-adaptive-service`；分支：`exp/joint-adaptive-service`。
- 基线HEAD：`2d5f9dbfcb7e7bfc2bfd9dc194b1838674a227c2`。此前小型机制的3个未提交源码文件完整复制到新树，不遗漏其实现。新功能以新增文件隔离，历史tracked策略文件未改。没有提交或推送；请按源码hash/快照识别这次执行版本，不能仅引用基线HEAD。
- 复用前核对91份旧Python源码和55份旧小型产物，hash均一致。旧六格因此只作为同输入、同旧实现的对照，没有择优重跑。复用检查见baseline_reuse_check.json。
- 预注册先于新运行写入preregistration.json；96份Python源码保存为source_snapshot.zip、source_hashes.json。重建可从旧完整基线bundle/commit建立新树，再恢复该Python快照；输入/配置/矩阵/registry由manifest中的完整路径及hash固定。
- 旧104格manifest及中断压缩流保持原hash，自动续跑sia-104仍PAUSED。official-artifacts、Synergy官方源、旧树和历史报告未改。

输入仍是workload-1原始首个BERT、CIFAR10、DeepSpeech2记录：bert-28、cifar10-0、deepspeech2-8，同时到达0，保留原始arrival/index、GPU demand、batch、总训练工作量。容量main不变：AWS4GPU/12CPU/250GB，DGX8/24/500，RTX8/24/500；CPU为合成单位。Neutral alpha0、Medium alpha.5，同一固定R0和E-legacy。每轮60秒、固定最多120轮，完成即停；不延长、无删失。seed_profiles(1,8)不变，不使用R0 GPU吞吐作为在线画像。

新两格独立进程严格Neutral→Medium顺序，启动检查2GiB内存。进程墙钟合计60.776秒；编排含完整测试合计114.828秒。单进程内部总墙钟分别约24.80/25.80秒，进程启动和导入时间另计。

## 具体根因和修改

旧 `neutral_liveness.py` 的held循环只保留上一完整列。旧Medium JO日志中，240秒时BERT已在线预测DGX4/8明显快于AWS1，仍被服务承诺固定AWS1。敏感peer的高CPU配额同样被固定，可能挡住其他任务使用空闲GPU。

新增源码：

1. `joint_prototype/adaptive_service.py`：AdaptiveAdmission，保留原L2的FIFO、每任务等式准入、到期排队、排空、原MIP目标与词典序tie-break。服务中任务继续强制选中，不因效用为负而清空。真实checkpoint精确保护GPU配置、placement、CPU和RAM。
2. `joint_prototype/adaptive_probe.py`：显式opt-in JOINT-R，收集真实应用的启动延迟和当前direct/projected来源，沿用完整候选证据。限制在已验证的一节点/类型拓扑、最多3个活跃任务；不是通过删除规模保护扩大运行。
3. `run_adaptive_cell.py`：固定短测试入口，保存逐轮gzip、预算、当前列、过滤原因、异常和资源结果。
4. `tests/test_adaptive_service.py`、`tests/test_adaptive_integration.py`：状态机、故障、预算和真实Job/Cluster链路验收。

没有改p、lambda_n、Goodput公式、辅助网格、phi、clipping、容量、工作量或原Sia求解器。CPU/GPU配置联合作用保留；没有用每任务最大原始GPU Goodput作硬下限，避免把“更少GPU＋更多CPU”的有效方案排除。

## 决策规则

每段服务记录：issued_time、productive_base、delay_charged、last_geometry_productive、geometry_changes及当前列。服务Q目标仍按物理有效训练秒计算，不乘rho。

真实checkpoint中：只允许当前完整列。

解锁后：

- 同GPU几何的CPU/RAM调整可进行，但每任务预测有效速率不得低于当前模型刷新后的原配额点。现模型没有辅助加载代价，故这种更新不增加启动预算；这是合成执行假设，不代表真实系统零开销。
- GPU类型/数量/placement变化必须在上次几何变更之后已有正的有效训练；目标候选必须来自当前direct/projected profile，不接受未画像占位当作已知性能。
- 费用d由实际 `Job.applications[target].rescale_time or30` 提供。BERT为120秒、CIFAR为50秒、DeepSpeech为25秒。CIFAR不能错误使用公平目标里的30秒代理惩罚。
- 初次发服务也计费。累计费用满足 `delay_charged+d <= 250`，不在变更后重置时钟或有效训练基准。
- 令H为Q减去本服务已兑现的有效训练秒，候选还必须满足 `(H-d)*G_new > H*G_current`。G为当前在线模型的原始归一化Goodput乘同一rho，不读未来工作量。公平目标仍额外使用原有restart factor。
- 当前列始终刷新并保留；无收益/同分不会靠字典顺序迁移到另一个GPU。预算耗尽后保持GPU几何，直到服务到期或任务完成；辅助配额仍可在保护规则下重优化。

例如BERT初始120＋一次120秒纠正＝240，允许；再一次120拒绝。CIFAR按50累计最多到250。ImageNet初始250会耗尽本服务GPU适应预算，属于显式保守限制。新服务若继续原GPU、没有实际rescale，则启动费用0，不能重复计费。

## 活性与原子性

服务中的任务仍满足Σx_j=1；配置适应不影响该服务身份、FIFO序号、issued_time或productive_base。全部启动延迟累计≤D_bar，持续分配且累计有效训练到Q后退出。因此，在正执行率、稳定资源、当前列始终合法、求解有限可决的假设下，服务不能因反复配置变化无限延长；保守墙钟界为Q+D_bar+2Δ。代码沿用更严格的Q+D_bar+Δ运行期检查，两次实际最大已观察服务年龄480秒，均满足610秒检查。

排空时不无限发新承诺，有限先行服务有界退出，队首在单独始终可行的条件下最终获得服务；FIFO尾插带来有限前序负载下的重复服务。最终完成还需要有限总工作量和有效进度正下界。没有宣称过载无限到达下有统一JCT上界。

队列与预算采用深拷贝事务；求解未决、容量/候选/锁审计失败时不提交。预算先在完整解中记录，实际GPU调整事件与费用在独立事后审计逐项相符。配置或锁真实不可行仍显式失败，不假设无解等于全等待。

回本条件只保证按当前速率预测、在旧配置尚需H秒的参考观察时刻，新配置的预期进度更高；它不保证实际JCT改善。预测误差、peer资源机会与任务完成顺序仍会影响结果。

## 测试和失败证据

全套 **122/122通过，49.311秒**：原107项保持，加13项状态机测试与2项真实集成测试。包括：

- 同一真实MIP单元里旧L2仍固定AWS，新机制在已有训练/已画像后能调整；负效用依然服务真实FIFO。
- checkpoint完整保护；未画像、无有效训练、无回本、预算不足目标拒绝。
- BERT120＋120边界；CIFAR实际50累计250；平坦peer释放CPU、敏感peer受益而服务计数不重置。
- 求解失败和全量审计失败原子回滚；继续相同GPU不重复收费。
- 固定24个状态步、最多3任务的大gang排空和重复服务；真实未seed BERT最多10轮完整调度链路；真实CIFAR应用费用来源。

test_development.log保存第一次测试的1失败/1错误；为fixture缺少CPU4点、故障注入提前打断单列刷新，修正限测试文件。test_development2.log为13项通过。没有为过测放松策略或资源校验。

analyze.py从原始完成事件、训练片段和placement配额独立复算JCT和资源小时，校验每个selected rho、直接raw×rho、checkpoint冻结、服务时钟不重置。每次GPU变更均核对在线画像、回本、实际rescale/migrate费用；所有服务累计费用最高240秒，没有超过250。全部容量审计、锁、单次倍率、预算均通过，异常/fallback为0。

## 完整结果

旧对照使用hash一致的上一轮六格，新方法只增加两格，没有重复运行并选择较好的值。P95/P99只有三个作业，不能解释为总体尾延迟估计。

| Response | 方法 | Avg JCT | Median | P95 | P99 | Makespan |
|---|---|---:|---:|---:|---:|---:|
| Neutral | SIA-P | 1367.33 | 794 | 2535.50 | 2690.30 | 2729 |
| Neutral | SIA-AUX | 1367.33 | 794 | 2535.50 | 2690.30 | 2729 |
| Neutral | 旧JOINT-O | 1856.67 | 2049 | 2754.60 | 2817.32 | 2833 |
| Neutral | 新JOINT-R | **1363.33** | 894 | 2583.30 | 2733.46 | 2771 |
| Medium | SIA-P | 1367.33 | 794 | 2535.50 | 2690.30 | 2729 |
| Medium | SIA-AUX | **1316.67** | **764** | 2473.10 | 2625.02 | 2663 |
| Medium | 旧JOINT-O | 1704.67 | 2049 | 2509.80 | 2550.76 | 2561 |
| Medium | 新JOINT-R | 1335.00 | 1064 | **2363.60** | **2479.12** | **2508** |

Neutral新方法比旧JO改善26.57%，比SIA-P改善0.29%，但median/P95/P99/makespan均略差。Medium新方法比旧JO改善21.69%、比SIA-P改善2.36%，比SIA-AUX增加1.39%。平均与尾部存在明确权衡，不能挑选单一获胜指标。

| 作业JCT | N/P或AUX | N/旧JO | N/新R | M/P | M/AUX | M/旧JO | M/新R |
|---|---:|---:|---:|---:|---:|---:|---:|
| bert-28 | 794 | 2049 | 894 | 794 | 764 | 2049 | 1064 |
| cifar10-0 | 579 | 688 | 425 | 579 | 523 | 504 | 433 |
| deepspeech2-8 | 2729 | 2833 | 2771 | 2729 | 2663 | 2561 | 2508 |

Medium相对SIA-P为2改善/1退化：CIFAR提前146秒，DeepSpeech提前221秒，BERT慢270秒，净提前97秒、平均32.33秒。相对SIA-AUX仍是2改善/1退化，净慢55秒；没有掩盖BERT剩余损失。

| 新方法 | GPU-h | CPU单位-h | RAM GB-h | 等待 | checkpoint | 训练 | Restart | Wasted GPU-s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Neutral | 7.0389 | 7.0389 | 146.6201 | 180 | 560 | 3350 | 6 | 2535 |
| Medium | 6.1783 | 18.3378 | 344.6710 | 180 | 705 | 3120 | 8 | 3345 |

Medium SIA-P为6.7556GPU-h、20.2667CPU-h、422.2222RAM-h，SIA-AUX为6.6133/19.1756/403.7008；新方法资源小时更低，但checkpoint/restart比二者多，不能声称所有开销下降。新Neutral/Medium物理等待均180秒，无队首排空发生；大gang排空证据来自预注册的有界状态机测试，不能伪称本三作业轨迹出现。

## 决策行为是否按设计改变

Medium共5次服务内GPU适应：

| 时刻 | 任务 | 新配置 | 当前服务累计延迟 | 新/原预测有效速率 |
|---|---|---|---:|---|
| 180 | CIFAR10 | DGX2 | 100 | 14.8422 / 9.2123 |
| 240 | BERT | RTX4 | 240 | 5.3187 / 1.0000 |
| 480 | DeepSpeech2 | RTX2 | 25 | 2.8456 / 2.5022 |
| 540 | DeepSpeech2 | DGX8 | 50 | 5.6733 / 4.1643 |
| 780 | BERT | DGX8 | 240 | 79.6176 / 12.5181 |

这证明新机制不只更改类名：CIFAR在180秒就能由DGX1改成DGX2辅助资源包；BERT在240秒取得首次在线模型后离开AWS。旧策略BERT在AWS训练300有效秒，新策略只训练60秒即纠正。新BERT后续RTX4训练240秒、RTX8训练60秒、DGX8训练164秒，总训练524秒；旧JO为1629秒。BERT JCT因此从2049降至1064，但仍没有达到Sia串联的764秒。

新Medium703个训练秒rho>1，物理训练秒加权rho约1.28588；旧JO为979秒、1.29973。新方法较低的平均rho仍有更低JCT，支持“应纠正GPU配置和服务限制，而不是单纯追求高rho”。新候选分子CPU/RAM clipping为14/65（各21.54%），分母CPU10/65（15.38%），RAM0；Neutral分子各56/67，分母CPU5/67。固定E-legacy的边界不确定性仍在，不因机制改进而消失。

## 局限与下一步

这个版本保持当前目标，只做候选资格/服务保证的结构改进，已经显著缓解旧损失；还不能证明其普遍优于Sia或SIA-AUX。预算可能阻止长期任务的第二次高收益改配；逐任务配额不降速约束可能拒绝全局更高效的让渡；在线projected模型可能误差大；公平目标不是JCT目标。初轮完全未知GPU性能的同分选择没有被伪造画像修正。

本轮没有按结果再试不同Q、预算、bonus或映射。下一步若研究者继续授权，建议先用更多预注册的不同到达/资源组合的小型机制负载检验非回归，再做单20-job对照；当前不恢复104格。新策略和SIA-AUX都应保留，不能用本小例宣布完整联合已优于串联。

复现入口：prepare_run.py（全测试、旧结果hash核验、冻结、两格启动），每格精确命令在neutral/medium_launch.json；analyze.py独立审计；DESIGN.md数学语义和假设；source_snapshot.zip＋source_hashes.json保存执行Python；final_manifest.json保存最终源码/输入/输出hash。需要在全新目录重建，不覆盖本轮结果。
