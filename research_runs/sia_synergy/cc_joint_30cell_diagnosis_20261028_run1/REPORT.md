# CC-TAIL：30格已完成实验的平均JCT、P99与双重受限退化诊断

**结论：平均JCT收益是真实完成事件上的排队面积减少；长尾主要是长任务低GPU规模和让出辅助资源后训练变慢；R11负结果是短任务优先的局部配额选择与后续GPU/排队路径、服务承诺及恢复门槛共同作用。未发现本轮所查的重复倍率、容量超配或指标计算错误。**

**最小优先修复建议：把现有“一次有界收尾恢复”扩展到当前队列为空、可保持所有其他作业完整分配不变的资源隔离恢复，保留checkpoint、原子性和一次许可边界。不要先盲目抬高预算或把目标改成每轮ETA和最小。**

本轮仅只读日志、源码和输入，新增本目录离线脚本与报告；没有修改算法、参数、响应或历史产物，没有调用Job.step/Cluster/调度器，没有训练仿真。静态比较仅枚举已有日志中的有限候选。

**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**

## 1. 版本、范围与证据

证据根目录：D:/科研启航/Sia/research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1。四容量×六方法Medium共24格，以及R11六方法Neutral共6格，全部20/20。负载固定为18个CIFAR10、2个DeepSpeech2，真实Saturn作业参数及工作量、受控到达；4GPU，合成CPU/RAM容量。
冻结执行源码source/的282文件逐项验证，当前cc-joint-resource-study与之逐字节一致。两个实验副本没有.git，因此本报告以实际完整快照与SHA-256为版本依据，不把某个裸HEAD当作完整实现。全部30格output_hashes、原始压缩日志与summary的哈希已核对，诊断结束再次验证未改变。
复用已有独立原始日志审计，本轮额外核验全部30格JCT/时间互斥分解、未完成作业面积与等待面积恒等式，并独立从首次选择和首次有效训练事件重算first_allocation/first_training。
[integrity_before.json](integrity_before.json)；[scan_validation.json](scan_validation.json)；[diagnostic_validation.json](diagnostic_validation.json)。
Sia基线为官方GPU决策逻辑的统一辅助资源执行适配，比例配额；SIA-AUX为固定Sia意图的ECT辅助优化，不是完整官方Synergy-TUNE。四CC共享同一CC-TAIL控制器，仅CPU/RAM配额自由度不同。

## 2. 平均JCT收益从哪里来

对每个已完成作业，JCT=等待+startup/checkpoint+有效训练；整个负载的ΣJCT=∫N_未完成(t)dt。下面均为FULL减Sia，负值表示减少，单位为任务秒。

|容量场景|平均JCT改善|ΣJCT差|等待差|首次分配前等待差|后续等待差|训练差|checkpoint差|首次等待占等待节省|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|R00|46.97%|-69123|-76320|-73560|-2760|7747|-550|96.38%|
|R10|42.98%|-81253|-89940|-85620|-4320|10037|-1350|95.20%|
|R01|45.28%|-67621|-75000|-71940|-3060|8054|-675|95.92%|
|R11|39.40%|-75587|-85680|-81240|-4440|11493|-1400|94.82%|

以R00为例，净节省69123任务秒=等待节省76320+checkpoint节省550−训练增加7747。等待节省中的96.38%发生在首次分配前。不是训练总工作量减少，也不是全部作业的训练本身变快。
CC选择较小GPU配置让多个作业共驻，并维持服务至完成，更多CIFAR较早获得资源并完成释放资源；长任务因此承担更低GPU规模/辅助配额和更长训练时间。GPU小时仍可降低：更多单GPU并发可减少总体排队与资源占用，而任务训练墙钟之和增加。

|R00方法|4作业同时占有GPU的精确秒数|4作业同时有效训练秒数|等待任务秒|
|---|---:|---:|---:|
|SIA_P|600|200|118080|
|CC_P|7729|6629|49740|
|CC_FULL|7125|5875|41760|

这些时长按到达、完成和训练片段边界精确积分，排除了周期内完成后的占用；因此与旧报告仅按60秒边界分类的600/7620秒等数值不可混用。
结构见证S1：R00/Sia第14行、t=840，两个DeepSpeech各2GPU占满4GPU，已有CIFAR等待；S2：R00/FULL第13行、t=780，四作业各1GPU共驻。它们是各自真实轨迹的结构见证，**不是同一训练状态上的严格反事实**。源码上Sia优化归一化Goodput公平目标，CC改用倒数ETA，叠加FIFO和持续服务；本矩阵不能把这些组件的贡献各自唯一识别。
CC-P在Neutral没有辅助响应增益仍比Sia均值低32.00%；Medium各容量也已有较大收益。因此整体39.40%–46.97%不能全部归因于CPU/内存。四CC配额消融才支持辅助自由度的增量归因，且包含后续GPU/队列反馈。
代码：Sia目标 [sia.py:949](D:/科研启航/Sia/research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1/source/sia-simulator/sia.py:949)；CC最终评分 [completion_continuity.py:136](D:/科研启航/Sia/research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1/source/sia-simulator/joint_prototype/completion_continuity.py:136)；FIFO/承诺 [adaptive_service.py:254](D:/科研启航/Sia/research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1/source/sia-simulator/joint_prototype/adaptive_service.py:254)。
[summary_decomposition.json](summary_decomposition.json) 给全部配对与逐作业；[timeline_statistics.json](timeline_statistics.json) 给精确面积/并发。

## 3. 为什么平均JCT更低而P99更差

四个Medium/FULL中最高两个JCT都由两个DeepSpeech占据；Sia最高两个则为晚到的cifar10-22和cifar10-20。20个样本的线性分位数：P99=0.19×第19顺位+0.81×第20顺位，P95=0.95×第19顺位+0.05×第20顺位。P99尤其受最慢一个作业支配，不是高精度总体尾延迟估计。

|场景|Sia第19/20顺位JCT|FULL第19/20顺位JCT|P99增加秒数|
|---|---|---|---:|
|R00|cifar10-22=9507 / cifar10-20=9589|deepspeech2-9=9910 / deepspeech2-8=11257|1427.65|
|R10|cifar10-22=13057 / cifar10-20=13081|deepspeech2-9=12284 / deepspeech2-8=13960|565.12|
|R01|cifar10-22=9777 / cifar10-20=9840|deepspeech2-8=10282 / deepspeech2-9=11215|1209.70|
|R11|cifar10-22=13293 / cifar10-20=13328|deepspeech2-8=12969 / deepspeech2-9=14550|928.26|

|场景/长任务|JCT增加|等待变化|训练增加|checkpoint增加|
|---|---:|---:|---:|---:|
|R00/deepspeech2-8|4294|60|4109|125|
|R00/deepspeech2-9|3467|-420|3712|175|
|R10/deepspeech2-8|6211|60|6026|125|
|R10/deepspeech2-9|4701|-720|5246|175|
|R01/deepspeech2-8|3319|60|3159|100|
|R01/deepspeech2-9|4772|-420|5067|125|
|R11/deepspeech2-8|5181|60|5021|100|
|R11/deepspeech2-9|6937|-780|7567|150|

**主要是训练速度/配置损失，不是未获得服务。**R00的DS8只多等待60秒，却多训练4109秒；DS9等待反而减少420秒，却多训练3712秒。不能靠进一步缩短首次等待解决这两个长任务。

|场景/方法/任务|有效训练秒|单GPU训练时间占比|时间加权GPU数|实际平均有效进度率|
|---|---:|---:|---:|---:|
|R00_medium_SIA_P/deepspeech2-8|6753|2.30%|1.977|4.2280|
|R00_medium_SIA_P/deepspeech2-9|5708|2.72%|1.973|5.0056|
|R00_medium_CC_P/deepspeech2-8|7884|100.00%|1.000|3.6205|
|R00_medium_CC_P/deepspeech2-9|10618|81.23%|1.188|2.6911|
|R00_medium_CC_FULL/deepspeech2-8|10862|69.19%|1.308|2.6304|
|R00_medium_CC_FULL/deepspeech2-9|9420|100.00%|1.000|3.0299|
|R11_medium_SIA_P/deepspeech2-8|7578|2.05%|1.980|3.7649|
|R11_medium_SIA_P/deepspeech2-9|6518|2.38%|1.976|4.3769|
|R11_medium_CC_FULL/deepspeech2-8|12599|100.00%|1.000|2.2650|
|R11_medium_CC_FULL/deepspeech2-9|14085|77.14%|1.229|2.0251|

这些均值是不同真实路径的描述，不把它们伪称固定epoch/固定batch的因果速度比。固定点上的辅助响应差异另在第5节独立比较。
R00中Sia的DS8/DS9约97%–98%训练时间使用双GPU，FULL中分别约69%和100%使用单GPU；R11中FULL/DS8全程单GPU。持续服务只保证被选中和持续有正训练，不保证最低相对速度。
Makespan和P99不矛盾：R00/Sia最后完成的是2400秒到达的cifar10-22，完成11907、JCT9507；FULL最后是0秒到达的DS8，完成/JCT11257。总体更早清空，但早到长任务的个人JCT更大。
重配置也有代价，但上述长任务增加100–175秒的checkpoint远小于3159–7567秒的训练增加，不能把P99主要归因于重启。
代码：持续服务不再到期 [completion_continuity.py:76](D:/科研启航/Sia/research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1/source/sia-simulator/joint_prototype/completion_continuity.py:76)；移除每轮不减速保护 [bundle_adaptation.py:66](D:/科研启航/Sia/research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1/source/sia-simulator/joint_prototype/bundle_adaptation.py:66)。

## 4. R11完整优化为何比仅CPU优化差1.38%

CPU_ONLY平均5734.20，FULL平均5813.35，净增加1583任务秒，即均值增加79.15秒。13作业变慢、7作业变快；CIFAR合计+790，DeepSpeech合计+793。等待+1200、训练+433、checkpoint−50，恒等式严格成立。新增等待全部发生在首次分配前。
这不是某一个长尾独占：即使诊断性去掉最大退化DS8的499秒，剩余19作业仍净增加1084任务秒；主结果始终保留全部20作业。

### 4.1 首次分歧本来是一个有利、可落地的内存机会

t=120两方法进入决策的进度、epoch、GPU状态相同；GPU、CPU和选中任务也相同。FULL把独占DGX单GPU的CF0从6CPU/93.75GB改为6CPU/125GB，利用空闲RAM，rho增加6.56%，候选ETA从1461.46降1374.54秒。另一作业配额不变。实际CF0完成提前42秒、CF1提前18秒。不能说最初就没有机会或第一步一定分错。
进度在180秒首先分叉；选中集合、GPU与CPU首次在1020秒分叉。因此后续两条完整轨迹不能再按“同状态”直接对比。[first_divergence_R11_CPU_FULL.json](first_divergence_R11_CPU_FULL.json)。

### 4.2 中期RAM交换的局部评分偏向短任务，忽略长期让出代价

S4：R11/FULL第17行、t=1020，固定实际服务集合、GPU与placement，用同一日志中的候选独立枚举。DGX上CF1和DS9各1GPU，总预算6CPU/187.5GB。两种方案都合法、都在原始矩阵支持域内，均不裁剪：

|作业|仅CPU候选|FULL实际选择|候选ETA变化|
|---|---|---|---:|
|CF1|4CPU/93.75GB，ETA352.33|4CPU/125GB，ETA332.61|−19.72秒|
|DS9|2CPU/93.75GB，ETA24137.74|2CPU/62.5GB，ETA27570.44|+3432.69秒|
FULL让CF1预测速率提高5.93%，DS9降低12.46%；已知任务ETA和反而增加3412.98秒，但归一化后的倒数ETA目标从0.760347升到0.802708，因此求解器正确地按现有目标选择了它。CF2当轮ETA未知且点不变，差值来自两个已知作业，不能把包含未知CF2的数值称为全部作业完成时间预测。
这一比较是同状态、同任务、同GPU，仅改变配额，排除了跨路径状态差异，也排除了裁剪；但候选ETA是假设当前速率持续的近似，不是实际JCT。后续任务可重新分配资源，不能将3433秒当作实际损失。S5在4980秒也出现CF11 ETA减少37.09秒、DS8增加1402.30秒的类似取舍。
数学上，u=1/(D+W/r)=r/(W+Dr)，故∂u/∂r=W/(W+Dr)^2；D很小时约为1/W，剩余工作越少越容易取得高边际分数。而∂(W/r)/∂r=−W/r²，长任务的完成代价没有被同样度量。当前目标也没有显式预测某次提前完成会释放多少资源、给等待队列节省多少面积。
不能简单改成每轮ΣETA最小或要求其不增加：S2/R00已有CPU交换同样增加已知ETA和约4105.81秒，但CPU消融在完整训练里平均JCT改善9.55%。硬ΣETA门槛会拒绝一部分已验证有队列收益的交换。应比较含当前等待队列的完成事件/释放收益，不是仅换一个局部指标。
代码：ETA [completion_time.py:17](D:/科研启航/Sia/research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1/source/sia-simulator/joint_prototype/completion_time.py:17)；最终倒数ETA [completion_continuity.py:143](D:/科研启航/Sia/research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1/source/sia-simulator/joint_prototype/completion_continuity.py:143)。[same_state_quota_comparisons.json](same_state_quota_comparisons.json) 给全部6个状态的静态枚举和支持域对照。

### 4.3 小幅速度变化经GPU路径和60秒边界放大为排队

CF3在CPU_ONLY于2516秒完成，FULL于2544秒完成，相差28秒，跨过2520秒调度边界，后继CF5首次服务从2520延至2580。CF13是更明显的路径分歧：

|CF13阶段|CPU_ONLY|FULL|
|---|---|---|
|首次分配|4800|4920|
|首次进入DGX单GPU|4860|5520|
|中期配置|DGX1、1CPU/93.75GB|AWS1、4CPU/93.75GB|
|总有效训练时间|1001|1205|
|完成时间|5901|6225|
其JCT增加324秒=等待120+训练204；CF15的首次服务随后从5940推迟到6240（300秒）。后面的部分CIFAR训练反而更快，但不足以抵消排队增加。
FULL在4920秒将DS8迁入DGX后用尽250秒生命周期预算；4980秒CF13首次模型可用时，不能再像CPU_ONLY那样把占据DGX的长任务换回AWS。S5静态GPU互换证明：在E-legacy全域下，这类交换容量合法且提高主目标，但DS8被SERVICE_DELAY_BUDGET拒绝。**然而支持域内的同类互换不提高主目标**，CPU_ONLY中CF13的1CPU/DGX路径发生源CPU坐标裁剪。故不能把这条GPU路径差异全部归为预算错误，也不能无视边界模型对CPU_ONLY优势的影响。
[S5_GPU_swap_diagnostic.json](S5_GPU_swap_diagnostic.json)；[selected_job_phases.json](selected_job_phases.json)；[observed_R11_event_chain.json](observed_R11_event_chain.json)。

### 4.4 长任务恢复延迟又影响另一个长任务

DS8在两方法都100%单GPU训练；DGX1期间时间加权rho由CPU_ONLY的0.9811降到FULL的0.8961，平均有效率由3.0099降到2.7341。其总体训练增加499秒，完成由12470推迟至12969。
DS9随后取得整个DGX双GPU的时间由12480推迟到13020（540秒）；FULL在末段需要的时间反而少246秒，最终仍晚294秒完成。此链路解释DS8的慢进度如何推迟同类型整节点配置的释放。没有重跑反事实，因此不把所有差值唯一分配给某一次RAM交换。

## 5. 真正未被利用的机会：多作业排空阶段的资源隔离恢复

仅统计预算拒绝的列数容易夸大损失：很多候选与其他承诺冲突或根本不能缩短ETA。本轮普查四个FULL的全部预算拒绝列，保持其他选中作业完整分配不变，重新独立检查容量及源坐标，既计入有利候选也计入不利候选。下面使用每个场景**第一个按时间出现的**队列为空、剩两个作业的有效状态，再列出该状态内最佳目标；没有按后续JCT挑状态。

|场景|首个状态/任务|当前配置及ETA|合法目标及ETA（含启动）|预测节省|
|---|---|---|---|---:|
|R00|t=8340/deepspeech2-8|aws:1:2，4CPU/125GB每GPU，4112.45s|dgx-ext:1:1，6CPU/125GB每GPU，3449.04s|663.40s|
|R10|t=11400/deepspeech2-9|dgx-ext:1:1，6CPU/125GB每GPU，949.80s|dgx-ext:1:2，3CPU/125GB每GPU，675.46s|274.34s|
|R01|未发现符合本普查条件的正收益状态|—|—|—|
|R11|t=12000/deepspeech2-8|dgx-ext:1:1，6CPU/93.75GB每GPU，1045.75s|dgx-ext:1:2，3CPU/93.75GB每GPU，740.81s|304.94s|

R11/t12000：DS8当前DGX1，6CPU、93.75GB；目标DGX2，每GPU3CPU/93.75GB，总6CPU/187.5GB，当前节点容量正好满足。DS9的AWS2及其全部配额不变。预测有效率4.4141→6.4487；25秒启动已计入，ETA1045.75→740.81。当前没有checkpoint，确有COMPLETION事件且两端响应坐标均支持。被拒原因是DS8已花250秒，而TAIL只允许活动作业数为1。
因此既不是缺少候选，也不必从另一个任务拿CPU；是已生成的有利候选被策略可行域挡住。R00、R10也有类似无损恢复窗口，R01未发现，不能把所有预算拒绝都叫错失机会。
这些是当前状态的可行性和预测严格改进证据，**没有计算反事实真实完成时间或JCT，也不保证未知未来到达不受影响**。
代码：预算拒绝 [adaptive_service.py:113](D:/科研启航/Sia/research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1/source/sia-simulator/joint_prototype/adaptive_service.py:113)；单作业激活门槛 [tail_recovery.py:25](D:/科研启航/Sia/research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1/source/sia-simulator/joint_prototype/tail_recovery.py:25)。[restriction_census.json](restriction_census.json)；[nonstealing_recovery_witnesses.json](nonstealing_recovery_witnesses.json)。

## 6. 前置准入到底限制了多少联合决策

|FULL场景|选中作业×轮|其中必须服务作业×轮|可选作业被选中次数|
|---|---:|---:|---:|
|R00|615|615|0|
|R10|797|797|0|
|R01|628|628|0|
|R11|839|839|0|

四场景中实际选中作业100%属于承诺/队首强制集合。其余作业虽然在代码中可选，本数据里没有兑现自由任务选择。原因可以从代码确认：尚未训练的作业即使seed了吞吐画像，grad_params仍未知，ETA为None、效用为0；主目标并列时先最小化启动成本，加入额外未知作业只增加正启动成本，因此被排除，最终依靠FIFO队首逐个启动。
持续服务又使已经启动的作业直到完成都为必须服务。因此当前收益应描述为“完成导向GPU配置/配额与连续FIFO服务共同减少排队”，不能描述为优化器自由挑选全局最佳任务集合。
这是能力/研究表述限制，不是此次活性失败：FIFO已经使全部作业完成；不能直接删除FIFO或持续服务。若下一阶段要兑现联合任务选择，需要单独定义冷启动准入价值与启动成本的词典序，保留锁和防饥饿对照。
代码：模型未知判断 [job.py:230](D:/科研启航/Sia/research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1/source/sia-simulator/job.py:230)；未知ETA置零 [completion_time.py:112](D:/科研启航/Sia/research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1/source/sia-simulator/joint_prototype/completion_time.py:112)；启动成本先于可选presence并列处理 [adaptive_service.py:225](D:/科研启航/Sia/research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1/source/sia-simulator/joint_prototype/adaptive_service.py:225)、[bundle_adaptation.py:88](D:/科研启航/Sia/research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1/source/sia-simulator/joint_prototype/bundle_adaptation.py:88)。

## 7. 数据、预测和执行问题是否是主因

|FULL场景|训练秒加权源CPU裁剪比例|源RAM裁剪|参考分母CPU/RAM裁剪|
|---|---:|---:|---|
|R00|0.00%|0.00%|0.00% / 0.00%|
|R10|8.79%|0.00%|0.00% / 0.00%|
|R01|0.00%|0.00%|0.00% / 0.00%|
|R11|8.19%|0.00%|0.00% / 0.00%|

应用映射、R0跨GPU缩放及执行乘法仍为合成假设。S4/S5的RAM取舍与第5节恢复目标都在支持域内；R00/R01完整轨迹没有所统计的执行坐标裁剪，所以不能把所有收益/失败归因于裁剪。另一方面，R10/R11约8%–9%训练任务秒涉及CPU裁剪，CF13的廉价DGX点也是例子，不能忽略其不确定性或宣称已验证目标硬件性能。

预测率严格用决策记录continuity_scores.raw_rate与下一周期Job.step直接捕获的raw_rate_before_rho比较，不用归一化Goodput冒充进度率，也不用effective/rho反推raw。R11/FULL中DS8和DS9的训练秒加权预测/实际首段raw比约0.9583与1.0630；存在预测误差，但不足以把所有数千秒训练增加解释为倍率/单位错误。状态变化、epoch边界和模型误差仍可能影响临界决策，未据此做校准。
所有30格的互斥时间、完成事件、队列面积和首次服务时间检查通过；既有原始审计无超配、无重复rho。没有发现需要修改执行引擎的证据。[timeline_statistics.json](timeline_statistics.json) 的prediction_ratio/epoch_aligned保留逐任务分布。

## 8. 问题清单与最小修复建议（本轮均未实施）

|级别|问题与证据|影响|最小建议及边界|
|---|---|---|---|
|High：设计限制|预算与仅单任务TAIL门槛；R11/t12000、R00/t8340、R10/t11400|存在合法、不改变任何peer的有利GPU恢复却不可选|优先新增独立模式：允许队列为空时的资源隔离恢复，仍只发一次实际成本许可，不一般性提高预算|
|High：目标对齐|S4倒数ETA目标改善而已知ETA和增加3412.98秒；最终R11净损失1583任务秒|低估长任务让出代价，局部收益与队列释放错配|RAM提案以同状态CPU-only为锚，增加包含当前队列的有界完成事件评估；先做影子风险记录，不盲目改权重或强制ΣETA不增|
|High：服务保护不等于速度保护|DS8/9训练增加数千秒，等待很少增加；service_expired=False|长作业可以持续慢跑并占据部分节点，拖长尾及后续整节点可用时间|先修无损恢复；有donor的长任务保护另行设计，不能直接禁止所有让出，也不能直接删除持续服务|
|Medium：任务选择自由未兑现|四场景可选选中次数0；unknown ETA=0与次级最小启动成本|性能选择基本限于强制服务集合；当前叙事容易夸大joint admission|先修正表述。若需要新准入，另做冷启动/词典序消融；不默认加入bonus或真实未来时长|
|Medium：响应支持域|R11 CPU裁剪8.19%，CF13的CPU1/DGX点越界|部分路径收益依赖E-legacy边界模型|保留现有结果，报告支持域与裁剪路径；后续响应边界方案需独立冻结，不能在本轮补phi或偷偷删配置|
|Medium：外推与统计|同一20作业、18/2构成；P99由最高两项插值|不能推断多负载普遍收益或真实硬件尾分布|先验证最小修复，再预注册跨负载；保留全部长任务，不伪造样本量|

### 优先修复的精确定义

建议新模式保持CC-TAIL原目标、网格和响应，保留历史默认。仅将单任务TAIL激活条件扩展为以下充分条件：当前ready为空；目标作业已有有效训练且无checkpoint；目标配置已观测；把该作业替换为目标列后，所有其他作业的完整列保持不变；独立审计GPU/CPU/RAM及所有锁；计入真实启动后ETA严格更小；每作业最多一次许可，额度仍为一次实际启动且不超过既有D_bar；成功后固定该作业GPU几何。求解/审计失败不消耗许可、不部分提交。
这保留防饥饿与有限重配置边界，针对本轮最强的无损机会。它不能保证修复R11全部均值退化，也不保证一般在线未来到达不受影响；修复后的JCT必须重新做隔离对照，不能把候选ETA节省直接当性能结论。
第二项RAM决策修改属于研究语义变化，不应与第一项一并悄悄实施。倒数ETA已有显著队列收益，替换整个目标或加硬ΣETA门槛可能毁掉已验证优势；应先在固定任务/GPU的受控反例中比较有界完成事件面积，同时计入接收方、donor和当前等待者。

## 9. 下一阶段验收要求，仅设计不执行

|验收|精确输入/条件|预期|
|---|---|---|
|两任务无损恢复红绿|R11/t12000日志快照：DGX2GPU/6CPU/187.5GB，DS8当前1GPU/6CPU/93.75GB、预算已250；DS9 AWS2GPU/6CPU/187.5GB保持；目标DS8 DGX2、每GPU3CPU/93.75GB、实际启动25|旧版拒绝；新模式可提出一次许可；peer逐字段不变，总资源合法。候选必须来自真实当前模型/记录，不能seed假Goodput|
|锁与队首|同一快照把目标或peer设为真实checkpoint，或ready存在队首|不得放松锁，不得以隔离名义窃取队首资源；不可证明安全时拒绝|
|一次性与失败原子性|许可已用、目标未知/非正率、超容量、求解失败或当前轮版本不匹配|拒绝或明确失败，不消费许可、不提交旧rho/旧列|
|边界与回归|生命周期额外成本仍≤一次实际启动≤250，源候选不加点；历史默认不开新模式|旧CC-TAIL行为逐轮保持，Sia基线不变|
|小型完整训练|2–3作业，运行前固定观察窗与到达，不读取未来|先确认机会实际发生，再用完成事件评价JCT/尾部/重启；接受无收益或退化|
|相同20作业隔离对照|仅修改恢复激活规则，原20作业、容量、Medium与6000轮全部不变，新目录同版本成对运行|报告所有作业、均值/尾部/资源小时与代价；若仍不改善，不自动继续调到获胜|

本轮没有执行以上任何修复或训练测试。当前实验继续保持停止。下一阶段最少需要决定：是否先实现资源隔离恢复；RAM风险评估与冷启动准入是否保持为后续独立研究因子。

## 10. 产物与复核入口

- [summary_decomposition.json](summary_decomposition.json)：各配对逐作业JCT/等待/训练/checkpoint、首次等待、应用组、P95/P99顺位及留一诊断。
- [timeline_statistics.json](timeline_statistics.json)：30格精确并发面积、GPU驻留、实际速率、预算过滤和预测校准。
- slim_timelines/：30格压缩精简逐轮轨迹，原始行号和时间可回查旧日志。
- [selected_states.json](selected_states.json) 与states/：按结构规则选择的6个原始状态快照及哈希。
- [same_state_quota_comparisons.json](same_state_quota_comparisons.json)：固定服务集合/GPU的独立配额枚举，含支持域与不同局部目标对照。
- [restriction_census.json](restriction_census.json)、[nonstealing_recovery_witnesses.json](nonstealing_recovery_witnesses.json)：全部拒绝列普查及首个无损恢复状态。
- [observed_R11_event_chain.json](observed_R11_event_chain.json)、[selected_job_phases.json](selected_job_phases.json)、[S5_GPU_swap_diagnostic.json](S5_GPU_swap_diagnostic.json)：负结果的实际事件链、阶段和受预算限制的静态互换。
- [diagnostic_validation.json](diagnostic_validation.json)：0仿真、0调度器调用、0算法变更及30格历史保护。
- scan.py、investigate.py、recovery_details.py、swap_diagnostic.py、validate_and_chain.py、make_report.py：离线可复算脚本，输出采用独占创建，重跑应复制到新的诊断目录。

结论可信范围：同一受控20作业、固定R0/E-legacy合成响应与既定候选下的真实模拟轨迹，以及这些状态上的静态候选数学证据。对修复后的真实JCT、不同到达负载、未观测硬件、未来到达或全局最优性均不作外推。
