# 补充文献核验与写作借鉴

检索日期：2026-11-01。以下为主来源核验后的精选补充文献，不包含主文献组 Sia、Pollux、Synergy、Gavel。不是穷尽性系统综述。期刊/会议原文、出版方页面及其登记的 Crossref 元数据优先；未取得全文的项目明确标记。参考文献元数据及可用 PDF 地址见 `supplemental_sources.json`。

## 一、直接关联与贡献边界

| 文献 | 机制及与本研究的关系 | 是否优化主机 CPU 和 RAM 配额 | 原文核验范围 |
|---|---|---|---|
| Tiresias，NSDI 2019 | 以已获服务量及离散优先级缓解排队和抢占代价；提醒我们必须区分排队收益与训练加速 | 否；主要为 GPU 服务次序与放置 | USENIX 摘要、正文 §3.2.3、图7/12/13、表4 |
| Optimus，EuroSys 2018 | 在线拟合资源与训练进度，动态配置 worker/parameter server；可借鉴“模型—分配—放置”论证 | 有 CPU/GPU 角色资源建模；并非 Synergy 式 CPU×主机 RAM 响应优化 | 作者公开 PDF 首文、机构书目；全文提取因12MB工具限制不完整 |
| Gandiva，OSDI 2018 | 利用迭代重复性作时间切片、迁移及资源伸缩 | 否；其 GPU 显存/计算共享不能等同主机 RAM 配额 | USENIX 原文，图11/12/13 |
| Themis，NSDI 2020 | 完成时间公平与放置竞价；公平指标不同于最小平均 JCT | 否；GPU 分配与放置敏感性 | USENIX 原文 §6.2.1、图9–14、表5 |
| AntMan，OSDI 2020 | 协同调度与训练框架，动态共享 GPU 计算和显存并保护保证型作业 | 否；“memory scaling”主要指 GPU 显存，不能写成主机 RAM 联合调度 | USENIX 原文，图9/12 |
| CoorDL，PVLDB 2021 | 用 CPU 预处理与缓存分析区分 prep/fetch stall，减少输入数据停顿 | 分析 CPU 与主机内存敏感性，但核心是数据装载库而非多租户配额调度器 | 正式 VLDB PDF §4.3、图4/5、表1 |
| tf.data，PVLDB 2021 | 输入流水线的并行、预取和缓存影响端到端训练；给出大规模数据供给动机 | 调整输入流水线资源，非跨作业集群联合调度 | 正式 VLDB PDF，图1及引言 |
| RAP，ASPLOS 2024 | 将输入预处理与训练映射到可用 GPU 资源，协同优化图映射与共运行 | 不等同主机 CPU/RAM 配额分配；对象为推荐模型输入流水线 | 作者机构 PNNL 摘要及出版方登记元数据；未核读全文图表 |
| Nexus，ICCC 2024 | 已明确研究异构 GPU 与辅助资源敏感性，和本研究问题直接重叠 | 出版方摘要明确讨论 CPU/内存辅助资源；具体优化变量及约束须读全文确认 | IEEE 摘要、正式会议程序、出版方 Crossref 元数据；未取得开放全文 |

**必须调整的新颖性表述：**不能声称首次把异构 GPU 与 CPU/内存敏感性结合，Nexus 已直接覆盖该问题。也不能声称首次用服务次序降低排队或首次考虑抢占代价；Tiresias 等已有明确设计。当前论文更合适的贡献是：在继承 Sia Goodput 与弹性配置能力的执行模型内，协调有序启动、有效服务保护与主机辅助资源配置，并用受控消融分离收益来源和边界。相对 Nexus 的确切差异仍需全文核读，不能根据摘要推定其缺少某一机制。

## 二、可借鉴的行文及图表组织

以下为我们对已核验论文组织方式的归纳，属于写作设计建议；不复制原图、数据或措辞。

| 原文锚点 | 图表所回答的问题 | 本稿可采用的形式 |
|---|---|---|
| Tiresias 图7（PDF第7页，正文标号7） | 相同作业下，服务次序怎样改变完成时刻 | 小例子的 GPU×时间甘特图；明确示意而非新实验 |
| Tiresias 表4、图12（PDF第11页） | 排队下降与放置带来的训练加速分别贡献多少 | 真实 waiting/productive/checkpoint 堆叠图，旁列均值与尾部；不要全部归因 CPU |
| Tiresias 图13（PDF第12页） | 平均值之外的 JCT 分布变化 | 逐作业经验 CDF；小样本不伪造置信区间 |
| Gandiva 图11/12/13（PDF第11–12页） | 伸缩行为和迁移开销能否落地 | GPU 配置时间线＋checkpoint 时间；不能只展示吞吐评分 |
| Themis 图9/10/13、表5（PDF第12页） | 总体指标与特定短/长作业为何不同 | 分布图＋长尾作业表；区分平均改善与长作业损失 |
| AntMan 图9（PDF第7页） | 统计反馈与调度执行如何闭环 | 本项目“状态—候选—优化—审计—推进”架构图 |
| AntMan 图12（PDF第12页） | 资源共享是否让接收方获益而伤害保证型任务 | 同一节点多作业资源和有效进度时间线，显示 donor/receiver |
| CoorDL 图4（正式PDF第5页） | 总时间中 GPU 计算和输入停顿如何分解 | 绘制我们的可直接观测时间分解；不能把仿真训练时间强行叫硬件 stall |
| CoorDL 图5（正文§4.3.3） | 增加 CPU 是否仍带来边际吞吐收益 | 现有模板响应曲线，标明源矩阵与合成迁移后的曲线不同 |
| tf.data 图1（正式PDF第1页） | 数据供给是否值得优化 | 作为动机引用；我们没有百万作业数据，不能仿作相同规模证据 |

推荐结构：**现象与限制 → 可核验机制假设 → 联合资源模型 → 服务状态机和优化器 → 同版本对照 → 干净消融 → 个体/尾部反例 → 合成迁移与外部有效性限制**。尽量让每张图只回答一个研究问题。资源优化图和排队机制图应分组，不将不同版本的数值混画成同一次实验。

## 三、精确参考文献及来源

1. GU J, CHOWDHURY M, SHIN K G, et al. Tiresias: A GPU Cluster Manager for Distributed Deep Learning[C]//NSDI 2019. 485–500. [官方页面](https://www.usenix.org/conference/nsdi19/presentation/gu)，[开放PDF](https://www.usenix.org/system/files/nsdi19-gu.pdf)。
2. PENG Y, BAO Y, CHEN Y, WU C, GUO C. Optimus: An Efficient Dynamic Resource Scheduler for Deep Learning Clusters[C]//EuroSys 2018. Article 3, 1–14. DOI: [10.1145/3190508.3190517](https://doi.org/10.1145/3190508.3190517)，[作者PDF](https://i.cs.hku.hk/~cwu/papers/yhpeng-eurosys18.pdf)。
3. XIAO W, BHARDWAJ R, RAMJEE R, et al. Gandiva: Introspective Cluster Scheduling for Deep Learning[C]//OSDI 2018. 595–610. [官方页面](https://www.usenix.org/conference/osdi18/presentation/xiao)，[开放PDF](https://www.usenix.org/system/files/osdi18-xiao.pdf)。
4. MAHAJAN K, BALASUBRAMANIAN A, SINGHVI A, et al. Themis: Fair and Efficient GPU Cluster Scheduling[C]//NSDI 2020. 289–304. [官方页面](https://www.usenix.org/conference/nsdi20/presentation/mahajan)，[开放PDF](https://www.usenix.org/system/files/nsdi20-paper-mahajan.pdf)。
5. XIAO W, REN S, LI Y, et al. AntMan: Dynamic Scaling on GPU Clusters for Deep Learning[C]//OSDI 2020. 533–548. [官方页面](https://www.usenix.org/conference/osdi20/presentation/xiao)，[开放PDF](https://www.usenix.org/system/files/osdi20-xiao.pdf)。
6. MOHAN J, PHANISHAYEE A, RANIWALA A, CHIDAMBARAM V. Analyzing and Mitigating Data Stalls in DNN Training[J]. Proceedings of the VLDB Endowment, 2021, 14(5): 771–784. DOI: [10.14778/3446095.3446100](https://doi.org/10.14778/3446095.3446100)，[正式开放PDF](https://vldb.org/pvldb/vol14/p771-mohan.pdf)。
7. MURRAY D G, ŠIMŠA J, KLIMOVIC A, INDYK I. tf.data: A Machine Learning Data Processing Framework[J]. Proceedings of the VLDB Endowment, 2021, 14(12): 2945–2958. DOI: [10.14778/3476311.3476374](https://doi.org/10.14778/3476311.3476374)，[正式开放PDF](https://www.vldb.org/pvldb/vol14/p2945-klimovic.pdf)。
8. WANG Z, WANG Y, DENG J, ZHENG D, LI A, DING Y. RAP: Resource-aware Automated GPU Sharing for Multi-GPU Recommendation Model Training and Input Preprocessing[C]//ASPLOS 2024, Volume 2. 964–979. DOI: [10.1145/3620665.3640406](https://doi.org/10.1145/3620665.3640406)，[作者机构页面](https://www.pnnl.gov/publications/rap-resource-aware-automated-gpu-sharing-multi-gpu-recommendation-model-training-and)。
9. GAO X, ZENG C, HUANG D, DU L, XIAO N. Resource-Aware Heterogeneous Cluster Scheduling System for Deep Neural Network Training Tasks[C]//2024 10th International Conference on Computer and Communications (ICCC). 1228–1234. DOI: [10.1109/ICCC62609.2024.10942295](https://doi.org/10.1109/ICCC62609.2024.10942295)，[IEEE页面](https://ieeexplore.ieee.org/document/10942295/)，[正式会议程序，第61页，IC282](https://iccc.org/ICCC2024-program.pdf)。Nexus 为正文系统名。全文未取得，未核验图号或详细算法；作者与页码已从 IEEE 登记的 Crossref 元数据核验。

## 四、不宜写入稿件的推断

- 不将 GPU 显存调度论文全部归类为主机内存调度。
- 不把 CoorDL/tf.data 的真实输入瓶颈测量当成本项目 Saturn 的实测画像。
- 不把优于 Sia 当成优于上述全部方法，也不引用其最大倍数作横向数值排名。
- 不用“已有方法忽略排队和切换开销”概括 Sia、Tiresias、Gandiva。应写成本项目在特定弹性与资源竞争条件下发现了目标与平均 JCT 的偏差。
- 不虚构 2025/2026 新论文，也不把网页抓取日期当出版年份。Nexus 会议年份2024、IEEE入库2025，两者应区分。

## 五、检索的局限

本次为面向初稿的定向综述；没有声称覆盖全部最新相关工作。Nexus 和 RAP 的正式元数据、原作者/出版方摘要已核验，但全文图表未核读。Optimus 可公开的作者PDF约12MB，网页工具无法完整提取，故不列未核验图号。投稿前应优先补齐 Nexus 全文，精确确认与本工作服务保护和弹性配置机制的交集；这不阻碍先写有边界的初稿。
