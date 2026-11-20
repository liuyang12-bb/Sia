# 固定 R0 响应下的 w10 late 八单元实验结果

## 1 结论

**八次实验全部完成20/20作业，技术验收通过，但辅助资源优化没有带来额外的平均JCT收益。** Neutral 下联合GPU单元相对固定单提案单元的平均JCT降低44.05%；Medium 下 U-JO 相对 U-FP、U-FO 分别降低51.69%、52.93%，但比 U-JP **增加2.49%**。不能把这一结果写成“GPU与辅助资源联合优化全面优于仅联合GPU选择”。

本轮比较的是共享L2语义的 **U-FP／U-FO／U-JP／U-JO**，不是历史B1/B4/B3-PROP/B3。F是GPU-only单提案，不是原始Sia MIP；B5/SY不在本轮矩阵中。因此也不能据此声称优于官方Sia或Synergy。

**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。** 仅一个冻结cohort、每单元一次运行；20个作业不是20次独立实验。较大平均收益集中于四个长作业，且E-legacy存在大量坐标clipping，限制了性能结论的外推。

## 2 授权范围与冻结版本

协调对话 `01a1719a-a726-7100-a163-0c806c8a4e60` 中，用户在明确的合并方案之后授权自动修复、验收并完成首轮八次实验。授权原文与任务恢复边界记录在 [CURRENT_TASK.md](D:/科研启航/Sia/research_runs/sia_synergy/stage2c_r0_w10late/CURRENT_TASK.md)。本轮没有扩展九cohort、160-job、Low/High、E-linear执行或参数搜索。

- 基线：Stage2B `bd6105a37c6662336236a135eb7ad610ca6e16de`。
- 新分支：`exp/b3-stage2c-r0-w10late`；新工作树：`D:\科研启航\Sia\sia-joint-stage2c-r0`。
- 八次共同执行commit：`141138d95fc51755e2f71c15f01964b4c20757da`，仅本地提交。
- 新产物：[stage2c_r0_w10late/attempt1](D:/科研启航/Sia/research_runs/sia_synergy/stage2c_r0_w10late/attempt1)。没有覆盖历史结果；本轮没有运行期失败或选择性重跑。
- Python3.10.10，NumPy1.23.5、SciPy1.10.1、CVXPY1.3.2、CyLP0.91.6。HiGHS/CBC引擎版本未独立获取，不猜测。新单元使用SciPy MILP，`time_limit=30s, mip_rel_gap=0`。

| 冻结对象 | SHA256 |
|---|---|
| preregistration.json | 545CEC26AF8F271AFC581C894F5D270EA34CB07B3E0EFE195B2FFDE0E3175F57 |
| frozen_config_6000.json | 667D7717CF8FF09EEE6A82C9979180615B58B92670BF91A5D4E5728F4218DE03 |
| workload.csv | 719267711948607055CAD9BA3F5966D63C65B4913693375D21B5F4E1F3675949 |
| registry.json | 78F676E7695CBE09AA9A3DB71F539FE2A2CEF1DBC243E57CC064E4E294DCF1C9 |

启动前逐项核对145个输入文件hash、387个模拟器执行子树文件hash、commit及依赖版本。完整仓库另由Git bundle和最终794文件字节快照保存；不要把387文件的执行子树快照误称为整个仓库。

## 3 不变条件和模型

workload-10按到达时间稳定排序后的第141–160条、job140–159；首个到达归零，最后到达8010秒。CSV与源trace有序抽取逐单元格相同。16个CIFAR10、2个YOLOv3、1个ImageNet、1个DeepSpeech2，工作量及batch不改。集群为AWS4GPU/4CPU/500GB、DGX-ext8/8/1000、RTX8/6/1000；这些CPU/RAM容量是合成规格。

每轮60秒，上限6000轮；作业全部完成即停止。依次Neutral FP/FO/JP/JO、Medium FP/FO/JP/JO，独立进程串行运行。八次启动内存为2.463–3.271GiB，均满足2GiB门槛。策略进程总墙钟782.26秒。

沿用冻结配置中的官方Sia `seed_profiles(1,8)` 设置，八次一致；**不把R0吞吐或phi作为在线Goodput样本**。五应用真实未seed路径单独通过短测试，但本20-job矩阵不应被称作全未画像启动实验。

### 固定参考响应

复用Stage2B的完整55配置R0域，而非当前选中列或获胜策略轨迹。令

`T0(a,g)=GPU数量×b_ref(a)/median(step_time)`，
`r0(a,g)=(T0/GPU数量)/min_完整配置域(T0/GPU数量)`，
`phi_c=clip(r0,0.5,4)`，`phi_m=clip(sqrt(r0),0.5,2)`。

原始文件、物理行号、microbatch、聚合及源hash随registry保留。55项严格hash验证；缺失配置、错误模板或非registry尺度直接拒绝，不补phi=1。CPU/RAM响应为

`rho = [ R_Elegacy(CPU_per_GPU/phi_c, RAM_per_GPU/phi_m) / R_Elegacy(3/phi_c, 62.5/phi_m) ]^alpha`。

E-legacy分子分母均使用原双线性插值与边界clipping。Neutral的alpha=0，Medium=0.5；同一配置在(3,62.5)参考配额下rho=1。执行以Job.step直接记录的raw rate乘rho一次；checkpoint倒计时及有效训练服务秒数不乘rho。预测Goodput仍是Sia归一化候选模型，不把它的数值单位冒称为实际Job.step执行率。

### 四单元契约

| 单元 | GPU候选 | CPU/RAM候选 |
|---|---|---|
| U-FP | 当前合法且单独比例可行的GPU-only最佳单提案 | 比例点及既有最低floor |
| U-FO | 同一单提案规则 | 原网格与比例点，保留proportional进度floor |
| U-JP | 全部当前合法GPU候选 | 同一比例点/floor |
| U-JO | 全部当前合法GPU候选 | 同一优化网格与进度floor |

四单元都优化placement，故本实验**不单独识别placement因素**。共享父候选归一化、p=-0.5、lambda_n=1.1、restart factor、同一词典序末级tie-break和L2准入。F按Sia normalized Goodput×restart factor排名，不按CPU/RAM收益选GPU，不调用原Sia MIP。

L2按真实arrival/稳定源行顺序FIFO，checkpoint与服务承诺分离；服务几何/配额在Q=300有效训练秒内保持，D_bar=250秒；延迟不计Q。到期退队尾，队首不能共存时排空，禁止无限回填。求解器未决或非法承诺显式停止，不用空分配掩盖错误。原历史模式及Stage2B三作业/Neutral保护保持。

## 4 八次实际结果

JCT与makespan单位秒。每项均完成20/20、无删失。P95/P99采用NumPy线性插值，样本仅20，不是高精度总体尾延迟估计。

| 响应 | 单元 | 轮数 | 平均JCT | Median | P95 | P99 | Makespan |
|---|---|---:|---:|---:|---:|---:|---:|
| Neutral | FP | 867 | 6452.75 | 982.5 | 27807.30 | 42008.66 | 51980 |
| Neutral | FO | 867 | 6452.75 | 982.5 | 27807.30 | 42008.66 | 51980 |
| Neutral | JP | 586 | 3610.05 | 1125.5 | 14621.70 | 25910.74 | 35154 |
| Neutral | JO | 586 | 3610.05 | 1125.5 | 14621.70 | 25910.74 | 35154 |
| Medium | FP | 947 | 7757.60 | 763.5 | 35773.20 | 47434.64 | 56771 |
| Medium | FO | 953 | 7961.80 | 961.5 | 36132.85 | 47788.97 | 57124 |
| Medium | JP | 600 | 3656.85 | 719.0 | 13920.40 | 26420.88 | 35967 |
| Medium | JO | 635 | 3747.90 | 816.5 | 14294.30 | 28191.66 | 38087 |

Neutral的FP/FO与JP/JO各自20项JCT逐作业全等，而非只比较平均值。Medium JO比JP的mean、median、P95、P99、makespan均差；没有以个别指标获胜掩盖这一事实。

| 响应/单元 | GPU-h | CPU-core-h | RAM-GB-h | 等待秒 | checkpoint秒 | productive秒 | restart | wasted GPU-s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N/FP | 113.298 | 113.298 | 14162.222 | 70106 | 9615 | 49334 | 78 | 66735 |
| N/FO | 113.298 | 113.298 | 2359.993 | 70106 | 9615 | 49334 | 78 | 66735 |
| N/JP | 99.126 | 99.126 | 12390.764 | 866 | 4045 | 67290 | 37 | 13895 |
| N/JO | 99.126 | 99.126 | 2064.797 | 866 | 4045 | 67290 | 37 | 13895 |
| M/FP | 123.424 | 123.424 | 15428.056 | 92006 | 11630 | 51516 | 96 | 83235 |
| M/FO | 123.167 | 127.450 | 15395.833 | 96206 | 11710 | 51320 | 97 | 83875 |
| M/JP | 88.928 | 88.928 | 11115.972 | 746 | 3735 | 68656 | 33 | 11405 |
| M/JO | 83.784 | 107.558 | 10473.056 | 746 | 3950 | 70262 | 34 | 12385 |

等待、checkpoint、productive互斥且逐作业之和等于JCT；checkpoint列包括源模拟器的启动/重配置延迟。资源小时积分包含分配期间的延迟，周期中途完成后的时间不计。迁移和restart计数沿用源语义，另记录GPU配置变化，不能仅凭migration计数为0推断没有跨GPU类型切换。

Neutral优化单元选择较低RAM配额，rho仍为1，所以RAM小时下降不代表训练加速。Medium JO相比JP节省GPU/RAM小时，却使用更多CPU小时、增加checkpoint与restart、整体完成更晚；这是实测模拟代价，不是容量违规。

## 5 因子效应及长尾归因

令`Y_FP/Y_FO/Y_JP/Y_JO`为平均JCT，降低为收益。条件效应为`Y_FP−Y_FO`、`Y_FP−Y_JP`、`Y_FO−Y_JO`、`Y_JP−Y_JO`；交互收益

`I = Y_FO + Y_JP − Y_FP − Y_JO`。

I>0表示两因素共同改变后的降幅超出相加预测，**不等于辅助资源因子本身有正收益**。主效应是另一个因素两个水平上条件效应的平均。

| 响应 | 辅助优化给定F | GPU联合给定P | GPU联合给定O | 辅助优化给定J | 交互I | GPU主效应 | 辅助主效应 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Neutral | 0 | 2842.70 | 2842.70 | 0 | 0 | 2842.70 | 0 |
| Medium | -204.20 | 4100.75 | 4213.90 | -91.05 | 113.15 | 4157.325 | -147.625 |

Medium中JO比FP/FO/JP的平均JCT改善率分别为51.69%/52.93%/**-2.49%**。JO相对FP的优势从Neutral44.05%扩大到51.69%，但这并非证明辅助优化有效：Medium相对Neutral，FP、FO、JP、JO的平均JCT分别增加20.22%、23.39%、1.30%、3.82%。固定单提案对响应下降更敏感，是相对差距扩大的一个直接表现。

| JO对照 | 改善/退化/相同作业数 | DS159占净总JCT收益 | 去掉DS159后的诊断均值收益 |
|---|---|---:|---:|
| Neutral FP或FO | 9 / 6 / 5 | 16.87% | 2487.53秒 |
| Medium FP | 15 / 5 / 0 | 20.23% | 3366.74秒 |
| Medium FO | 16 / 4 / 0 | 19.34% | 3577.79秒 |
| Medium JP | 12 / 8 / 0 | 净收益为负，不解释为正贡献百分比 | -147.47秒 |

**并非只有DeepSpeech2驱动收益，但仍高度依赖少数长作业。** 相对FP，YOLO149、YOLO153、ImageNet154、DeepSpeech159合计占Neutral净收益96.32%、Medium净收益96.32%。其余16个作业的净贡献很小。所有20个作业均保留为主分析；删除作业只用于上述诊断。

Medium JO对JP，DS159快981秒、YOLO149快1190秒，但ImageNet154慢2120秒、CIFAR150慢1071秒等，最终总JCT增加1821秒。去掉ImageNet154后的诊断平均收益仅15.74秒，不能用该删后结果替代主结果。逐作业全部配对差值、完成时间和等待见 [review2_aggregate_summary.json](D:/科研启航/Sia/research_runs/sia_synergy/stage2c_r0_w10late/attempt1/review2_aggregate_summary.json)。

### DeepSpeech2 159 的实际路径

arrival均为8010秒、最终progress均为28536、epoch均为80，工作量未减少。

| 响应/单元 | JCT | 等待 | checkpoint | productive | restart | 首分配/首训练 |
|---|---:|---:|---:|---:|---:|---|
| N/FP或FO | 13993 | 11070 | 225 | 2698 | 8 | 8400 / 8425 |
| N/JP或JO | 4402 | 30 | 75 | 4297 | 2 | 8040 / 8065 |
| M/FP | 21857 | 16830 | 350 | 4677 | 13 | 8340 / 8365 |
| M/FO | 21931 | 16950 | 350 | 4631 | 13 | 8220 / 8245 |
| M/JP | 6612 | 30 | 75 | 6507 | 2 | 8040 / 8065 |
| M/JO | 5631 | 30 | 50 | 5551 | 1 | 8040 / 8065 |

FP/FO的配置路径为AWS1→RTX4→DGX8，争用完整DGX配置导致大量等待；Neutral联合单元使用RTX1→2→4，训练秒数反而更多，但等待/重启少得多，JCT更短。

Medium JP沿用RTX1→2→4，rho分别约1.08069、0.82232、0.58574；JO选择DGX1→2。JO的DGX1使用每GPU3或4CPU、125GB，rho约1.16501或1.35343；DGX2每GPU3CPU、125GB，rho约1.09569。这确实体现GPU选择与配额之间的交互。DS159的JO GPU-hours为2.845、CPU-hours为8.635，对比JP的6.563/6.563：以CPU换GPU并提前981秒完成，但不能将该个例外推到全体。

完整逐轮config、quota、placement、直接raw/effective执行率及rho见 [review2_ds159_trace.json](D:/科研启航/Sia/research_runs/sia_synergy/stage2c_r0_w10late/attempt1/review2_ds159_trace.json)。

## 6 审计与边界风险

独立重读6041轮JSONL：8983次固定phi校验、7382次服务几何保持、629次checkpoint几何保持、22978个直接执行率片段、7801次从placement与quota重新求和的逐节点容量检查，**违规均为0**。资源独立求和不只相信求解器的audit布尔字段。进度单调、无训练片段时无进度、时间互斥、完成边界均检查。

| 响应/单元 | rho范围 | 分子CPU clipping / 已选列 | 分母CPU clipping | CPU饱和且GPU闲置节点轮 | DRAIN轮 | 服务签发数 |
|---|---|---|---:|---:|---:|---:|
| N/FP | 1–1 | 993/993 | 0 | 0 | 405 | 162 |
| N/FO | 1–1 | 993/993 | 0 | 0 | 405 | 162 |
| N/JP | 1–1 | 1200/1200 | 31 | 36 | 2 | 229 |
| N/JO | 1–1 | 1200/1200 | 31 | 36 | 2 | 229 |
| M/FP | 0.55041–1.27183 | 1065/1065 | 0 | 0 | 514 | 169 |
| M/FO | 0.55041–1.52031 | 971/1066 | 0 | 95 | 536 | 168 |
| M/JP | 0.58574–1.33626 | 1219/1219 | 40 | 0 | 0 | 237 |
| M/JO | 0.73995–1.92585 | 908/1247 | 31 | 307 | 0 | 245 |

Medium所有已选点分子RAM、分母RAM都未越界；Neutral FO/JO采用20.83配额，分子RAM分别993/1200次clipping，执行rho仍为1。这些是**已选列轮次**，不是全域覆盖比例；完整55配置和未选网格支持域沿用Stage2B离线预检。E-strict删除全部可行比例点的结论没有被本轮clipping成功执行“修复”。

Medium JO仍有72.81%的已选列触发分子CPU clipping。固定phi去除了在线策略路径依赖，却没有验证边界外真实物理响应。E-legacy只是冻结的兼容性假设；不能把计算合法、八次完成等同于该边界模型实测正确。

全部求解结果为OPTIMAL_L2或DRAIN_FOR_HEAD，零fallback、零异常退出。求解子阶段累计3.08–5.29秒/单元，P99约7.76–36.54毫秒；这仅是L2求解/相关审计计时，不含全候选构建、全部训练推进和进程启动，不能冒称端到端调度开销。各次策略墙钟78.91–126.46秒。

CPU饱和/GPU闲置是逐节点逐轮事件，不是作业等待秒数。JO的Medium等待总量746秒但有307个此类节点事件，两者不矛盾：作业已获得少GPU、多CPU的合法服务时，节点仍可能GPU未满。

最终作业完成后Cluster跳过下一次优化，因此COMPLETED_SERVICE日志为19，而真实完成事件为20；最后一个服务随全局终止结束。统计完成率依据Job.completion_time，不依据服务释放事件数。结果中的旧序列化policy字符串`L2_NEUTRAL`由共享父类保留；实验响应以启动manifest、response_exponent及shared_contract为准，Medium没有按该字符串切换为Neutral。

预注册删失规则：若6000轮未完成，只报告完成率、真实已完成JCT和删失时刻；全体mean/median/tail/makespan为空。共同tau=360000−8010=351990秒，经验restricted mean为平均min(JCT,tau)，不插补无穷大。本轮全部完成且JCT低于tau，因此restricted mean恰等于完整均值。

## 7 实现与红绿证据

commit仅四文件，406行新增、5行删除；历史策略文件、Job.step和旧88项测试未改。

| 文件 | 变化与证据 |
|---|---|
| [neutral_liveness.py](D:/科研启航/Sia/sia-joint-stage2c-r0/sia-simulator/joint_prototype/neutral_liveness.py:38) | 增加可继承的范围/Neutral门槛；默认仍≤3/Neutral；原L2目标、约束、队列及tie-break不变 |
| [r0_liveness.py](D:/科研启航/Sia/sia-joint-stage2c-r0/sia-simulator/joint_prototype/r0_liveness.py:23) | R0Oracle严格55域与hash；R0Policy替换尺度而不替换Goodput；新模式≤20；执行率检查；事件增量日志 |
| [run_stage2c_cell.py](D:/科研启航/Sia/sia-joint-stage2c-r0/sia-simulator/run_stage2c_cell.py:28) | 独立一维插值复算rho，配额/floor/时间/容量审计，流式JSONL和明确完整/删失统计 |
| [test_stage2c_r0.py](D:/科研启航/Sia/sia-joint-stage2c-r0/sia-simulator/tests/test_stage2c_r0.py) | 新增6项；五应用各≤10×60秒真实未seed测试，固定phi/参考点/缺失拒绝/范围隔离/四单元包含/Medium故障原子性 |

红测red1为新接口尚未存在的ImportError，不将其冒称调度算法反例；green1初始五项通过。接入运行摘要器后green2捕获扁平列schema误用、green3捕获Cluster属性名误用，均保留。修正仅涉及新统计器，不更改模型/调度参数。full2最终 **94/94通过**，18.29秒墙钟。真实原始Sia一作业三轮状态与官方checkout完全相同，见 [legacy1/comparison.json](D:/科研启航/Sia/research_runs/sia_synergy/stage2c_r0_w10late/legacy1/comparison.json)。

旧全空反例保持为历史行为测试。新模式活性由服务准入修复，不通过调大lambda、p或增加bonus；未把新活性偷偷加入历史B1/B4/B5/SY。

## 8 交付与下一步判断

可重建资料及命令见 [REBUILD.md](D:/科研启航/Sia/research_runs/sia_synergy/stage2c_r0_w10late/REBUILD.md)。每策略目录包含start、summary、output_hashes及rounds.jsonl.gz；每次进程的文本日志在attempt1根目录。`commands.json`、`execution_manifest.jsonl`记录完整命令、退出码、内存和墙钟；`review2_raw_validation.json`是增强的独立容量复算版本，第一版分析亦保留。源码完整快照、bundle、依赖锁、diff及全部产物hash在根目录交付索引中。

工程上已具备继续预注册复验的基础：八单元完成、输入一致、原始回归和关键约束通过。但**尚不能得出辅助资源优化带来普遍增量收益**，更不能直接宣称SOTA或真实集群收益。下一步最有价值的是在审查后检验此负增量是否跨cohort持续，以及比较预注册响应边界的敏感性，而不是按本次JCT调参。大量clipping、单cohort、固定单提案而非原Sia对照、四长作业主导，都是直接影响结论的限制。

本轮已停止，未自动启动九cohort或新的响应档位。
