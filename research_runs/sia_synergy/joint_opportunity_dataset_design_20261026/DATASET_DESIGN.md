# 联合调度机会验证数据设计

## 结论

现有六作业数据不是没有机会。它已经触发持续排队、GPU 排名翻转和资源交换，但当前 G112 仍比同输入 SIA-P 的平均 JCT 高 13.08%。因此不能把不获胜简单归因于数据，也不能承诺换一组数据就能获胜。

新设计分开检验两个必要条件：固定 GPU 后是否存在可兑现的资源交换；在辅助资源已优化后，改变 GPU 配置是否仍有增量价值。已生成三份源记录提取文件，完成少量离线枚举，没有启动训练仿真、修改策略或恢复104格实验。

**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**

## 现有数据能说明什么

六作业 G112 前60轮中，54轮选中了 GPU 排名翻转配置，17轮发生实际资源互补，机会门槛已通过。6个作业完成后，平均 JCT 为5825.33秒，G111为5900.67秒，SIA-P为5151.33秒。G112对G111改善1.28%，但只改善2个作业、退化4个。

该场景把6个作业同时投到5GPU，既改变辅助资源分配，也触发服务保护、排队和频繁改配。它适合压力验证，却不适合单独归因。最新运行530个已选 job-round 配置中，285个CPU分子坐标发生截断。先前的低配额BERT donor也依赖边界截断；不能把这种收益描述为目标主机实测。

依据：[六作业完整报告](D:/科研启航/Sia/research_notes/synergy/JOINT_SERVICE_HORIZON_PAIRED_REPORT_20261026.md)，以及 [run4产物](D:/科研启航/Sia/research_runs/sia_synergy/joint_service_horizon_trial_run4_20261026)。历史结果原样保留。

## 数据来源与固定规则

作业仍来自官方 workload-1.csv。保留原 name、application、num_replicas、batch_size；训练总工作量继续由原 Sia application 模型定义，不自行填入运行时长或缩短训练。仅为机制场景把到达设为0，同时保留 original_arrival 和 source_row_index。它们不是原始自然到达 trace。

num_replicas 是源 workload 参数，不等于弹性策略必须一次获得这么多GPU；实际候选由节点拓扑和原合法配置规则决定。

响应沿用现有 R0、E-legacy、Medium alpha=0.5、映射和辅助网格，不生成新曲线：CIFAR10映射AlexNet，BERT映射BERT，DeepSpeech2映射DeepSpeech。这些是合成关联。官方矩阵单元和Sia GPU画像属于原始数据；跨应用映射、跨硬件phi和两类性能相乘属于模型假设。

固定参考点为每GPU 3合成CPU单位和62.5GB。对配置g：

`rho = [M(c/phi_c(g), m/phi_m(g)) / M(3/phi_c(g), 62.5/phi_m(g))]^0.5`。

M沿用既有双线性插值和边界截断。任何新拓扑的proportional点都不自动令rho=1。只有CPU3/RAM62.5参考点如此。所有数据文件、映射、registry、原始模板和查询代码的SHA-256见 [manifest](D:/科研启航/Sia/research_runs/sia_synergy/joint_opportunity_dataset_design_20261026/offline_v1/manifest.json)。

## 场景一 固定 GPU 的 CPU 资源交换

作业：bert-28、cifar10-0。合成AWS节点：2GPU、8CPU单位、125GB。静态诊断固定每个作业1GPU，固定放置和服务集合。

|作业|比例CPU和RAM|交换后CPU和RAM|比例rho|交换rho|预测速度变化|
|---|---|---|---:|---:|---:|
|BERT|4，62.5GB|2，62.5GB|1|1|0%|
|CIFAR10|4，62.5GB|6，62.5GB|1.143918|1.378214|+20.48%|

两方案均占8CPU、125GB和2GPU；分子、分母坐标均在官方矩阵范围内。这是没有借助clipping的CPU互补证据。它主要检验“Sia＋辅助优化”应具备的能力，不能单独证明联合GPU选择优于串联。当前实际GPU决策若不选择同一服务集合，则需分开报告，不能把静态规定冒充原Sia策略。

输入：[D1](D:/科研启航/Sia/research_runs/sia_synergy/joint_opportunity_dataset_design_20261026/offline_v1/D1_auxiliary_exchange.csv)。

## 场景二 GPU 配置选择与辅助资源耦合

作业：cifar10-0、deepspeech2-8。两台合成节点：AWS 1GPU/6CPU/125GB，DGX-ext 1GPU/3CPU/125GB。较快的CIFAR GPU配较少CPU，但没有使用低于官方响应坐标范围的候选来构造结果。

静态诊断强制相同两个作业都服务，枚举两种1GPU映射及全部既有可行CPU/RAM点，共270个组合。GPU参考速率取R0 T0，**不是当前在线Sia Goodput，也不是一次真实Sia MIP回放**。为避免把不同应用吞吐直接相加，每个作业除以其两类型中较慢的单GPU T0，再比较归一化总速率和 `-sum(v^-0.5)`；这两个指标在本例选择一致。

|分配|CIFAR GPU及配额|CIFAR参考有效速率|DeepSpeech GPU及配额|DeepSpeech参考有效速率|
|---|---|---:|---|---:|
|只看原始GPU参考速率，再比例分配|DGX 1GPU，3CPU，125GB|1194.576|AWS 1GPU，6CPU，125GB|29.382|
|冻结上行GPU，再穷举优化辅助资源|同上|1194.576|同上|29.382|
|联合穷举GPU与辅助资源|AWS 1GPU，6CPU，125GB|1378.884|DGX 1GPU，3CPU，125GB|27.487|

不同应用的两列速率单位不可直接相加。联合方案使CIFAR预测速度增加15.43%，DeepSpeech下降6.45%；归一化总速率从2.967836增至3.125891，即+5.33%。公平代理效用从−1.645430变为−1.619608。入选的两套方案均没有分子或分母截断。

这里单纯优化CPU不能挪用另一节点的CPU，GPU映射必须改变才获得增量。这比“某个作业GPU排名发生翻转”更强：固定GPU的辅助最优也已检查。但它仍不是JCT改进保证，尤其不能忽略受损作业、实际在线模型和迁移成本。

Neutral静态对照中，联合和原始GPU参考速率选择的最优效用严格相同。输入：[D2](D:/科研启航/Sia/research_runs/sia_synergy/joint_opportunity_dataset_design_20261026/offline_v1/D2_gpu_auxiliary_coupling.csv)。建议先在这一小拓扑上验证真实在线调用链，不立刻扩大任务数量。它只隔离GPU类型选择，尚不检验GPU数量弹性。

## 场景三 弹性配置和排队训练候选

作业为官方源行最早的3个CIFAR10和3个DeepSpeech2：cifar10-0/1/2、deepspeech2-8/9/12。不按历史JCT选择。六个作业同时到达，保留源行稳定顺序。

|合成节点|GPU|CPU单位|RAM GB|
|---|---:|---:|---:|
|AWS|2|12|250|
|DGX-ext|2|6|250|

这将场景二的每GPU资源比按节点容量加倍，提供1GPU和2GPU配置以及排队，不另行搜索容量。主场景Medium；两个固定对照是相同输入Neutral，以及两节点CPU都为12、GPU/RAM不变的Medium。后者只去掉CPU/GPU比例错配，不宣称消除所有CPU争用。

未来比较SIA-P、SIA-AUX、G112；沿用原目标、完整checkpoint锁、当前活性规则、共同执行响应与seed_profiles(1,8)。不把L2加入Sia。60秒周期、600轮上限；未完成报告删失，不延长或插补。原始Sia逻辑基线的CPU适配应沿用已有审计版本，不拿静态R0代理替代。

输入：[D3](D:/科研启航/Sia/research_runs/sia_synergy/joint_opportunity_dataset_design_20261026/offline_v1/D3_queued_training_proposal.csv)。本场景目前是训练提案，不是已通过完整训练验收的数据集。

### 不能隐藏的响应覆盖问题

枚举D3两类应用、两GPU类型、1/2GPU和所有节点内可行辅助点，得到8个配置、168个配置资源点。其中76点的分子发生截断；9点的分母发生截断，均来自CIFAR10的DGX-ext 2GPU配置。该配置 `3/phi_c=0.940805<1`，增加容量也不能修复归一化分母越界。全部比例配置满足物理容量，但这不等于响应全部有测量支持。

全registry的55个应用配置中，共6个分母越界：BERT DGX的1/2/4/8GPU、BERT RTX 1GPU、CIFAR DGX 2GPU。未删除任何合法配置、补phi=1或更换边界模型。

D1和D2的具体机制证据已避开截断；D3若继续沿用E-legacy，只能检验该明确边界假设下的算法。不能称为全域无截断实验。若要求全域范围内证据，D3应暂停，先独立审定边界数据方案，不以修改参数让G112获胜。

## 内存与持续收益的验收

同一CIFAR AWS单GPU、6CPU下，RAM62.5→125GB使rho从1.378214→1.843770，即预测速度+33.78%，两点都未截断。这证明现有模型包含内存敏感性；但它多用了62.5GB，**不证明共享预算中存在无损RAM donor**。本轮D1主要证明CPU交换，D2主要证明GPU与CPU供给耦合，不把它们包装成已经充分验证内存竞争的场景。

若换配置新增延迟D、稳定速度收益g，则至少需要 `H > D(1+g)/g` 才能让单任务窗口训练量增加。例：20.48%收益且额外250秒延迟，需要约1470.6秒稳定共存；不是发现某一轮有收益就足够。D1仅CPU变化在现模型中无额外GPU重启，但真实CPU/RAM重配成本仍未测量。实际训练应使用每条迁移/缩放路径的真实增量延迟，不能把250统一当实际值。

未来运行前60轮固定记录在线候选，不按JCT筛选数据：

1. 在相同当前状态、服务集合和容量下比较原Sia意图比例点、冻结其GPU后的辅助最优、联合候选；保留被checkpoint和承诺阻挡的原因。
2. 机会门槛为至少连续3轮存在实际可行的资源交换或联合增益，且receiver提升≥5%、donor损失≤10%；记录每次净量收益与真实切换损失。门槛与旧机制尺度一致，不用策略胜负筛选。
3. 分开记录“机会存在但G112未选”和“选中但净收益未兑现”。某策略未选择机会不是停止并换数据的理由。
4. 若在线Goodput不支持R0参考排名，报告代理与在线模型差异；不灌入R0、预填Goodput或换作业。若机会不持续，也保留失败场景。
5. 通过门槛后才解释完整JCT；仍须报告全部作业、完成率、等待/训练/checkpoint互斥时间、重启、资源小时、clipping和逐作业损益。Sia与G112的准入/放置差异属于混杂，不把差异全部归于CPU优化。

## 复算与产物

离线脚本只导入响应查询模块，不导入或运行Cluster、Job训练循环。输出目录拒绝复用。脚本中的源字段保持、容量、无截断证据、完整枚举和Neutral最优一致性断言全部通过，退出码0；这不等于完整调度器测试通过。

```powershell
& 'D:\科研启航\Sia\.venv-sia-baseline\Scripts\python.exe' -X utf8 -B `
  'D:\科研启航\Sia\research_runs\sia_synergy\joint_opportunity_dataset_design_20261026\prepare_offline.py' `
  --output '<另一个不存在的绝对输出目录>'
```

[offline_evidence.json](D:/科研启航/Sia/research_runs/sia_synergy/joint_opportunity_dataset_design_20261026/offline_v1/offline_evidence.json) 保存所有查询坐标、来源、rho、270组合的分组最优及枚举数量、全域边界统计、场景与输入哈希。其SHA-256为 `669A3A9CE6F06CAE7E11D61A9D867EB35C2DDDE8FF213BA3839B67D328150C9C`。

当前树裸HEAD不代表全部执行源码；manifest记录了dirty状态和关键源文件哈希，历史run4保留完整source_snapshot.zip。正式训练前仍须单独冻结执行源码、配置与依赖，不能直接把本设计当已经开跑的预注册批次。

下一步优先验证D2的两任务真实调度链，再考虑D3；本轮不执行。即使新数据具备可利用机会，G112仍可能因准入、局部目标或重启而输。这样的结果应推动策略诊断，而不是再寻找必胜数据。
