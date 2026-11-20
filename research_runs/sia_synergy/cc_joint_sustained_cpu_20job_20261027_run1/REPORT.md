# CC-Joint：20任务持续CPU互补场景实验

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE

状态：EIGHT_COMPLETE_VALID。本轮算法源码404文件逐字节不变；只改变预注册的负载组成与到达时序。

## 结论与主要指标

|响应|策略|完成|平均JCT(s)|Median|P95|P99|Makespan|Restart|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|neutral|SIA_P|20/20|11075.75|12291.50|15546.25|15580.45|17989.00|50|
|neutral|SIA_ECT_AUX|20/20|11075.75|12291.50|15546.25|15580.45|17989.00|50|
|neutral|JOINT_CC_P|20/20|7627.05|6689.50|13883.75|17619.15|18553.00|38|
|neutral|JOINT_CC|20/20|7627.05|6689.50|13883.75|17619.15|18553.00|38|
|medium|SIA_P|20/20|7357.45|8542.50|9511.10|9573.42|11907.00|42|
|medium|SIA_ECT_AUX|20/20|7491.45|8523.00|9785.15|9878.63|12179.00|49|
|medium|JOINT_CC_P|20/20|4313.45|4068.00|8390.45|10464.49|10983.00|33|
|medium|JOINT_CC|20/20|3984.25|3309.00|10060.30|12344.86|12916.00|36|

|响应|CC-Joint相对对照|均值改善率|平均差(CC−对照)|改善/退化/不变|最大贡献作业|排除该作业诊断改善率|
|---|---|---:|---:|---:|---|---:|
|neutral|SIA_P|31.14%|-3448.70|15/5/0|cifar10-4|28.79%|
|neutral|SIA_ECT_AUX|31.14%|-3448.70|15/5/0|cifar10-4|28.79%|
|neutral|JOINT_CC_P|0.00%|0.00|0/0/20|cifar10-0|0.00%|
|medium|SIA_P|45.85%|-3373.20|16/4/0|cifar10-3|43.67%|
|medium|SIA_ECT_AUX|46.82%|-3507.20|16/4/0|cifar10-3|44.67%|
|medium|JOINT_CC_P|7.63%|-329.20|16/4/0|cifar10-20|6.58%|

Medium完整版相对SIA-P的平均JCT改善45.85%，相对串行辅助方案改善46.82%。关键配额消融CC相对CC-P改善7.63%（负数表示退化）。因此本场景支持差异化辅助配额的增量JCT收益。
这不是跨负载普适结论，更不是目标集群实测。正改善率表示更快；保留全部20个作业。

配额消融的任务级归因：18个CIFAR合计JCT变化-10173秒，2个DeepSpeech合计变化3589秒，净变化-6584秒。CIFAR组均值改善15.18%，DeepSpeech组变化改善率-18.66%（负值为退化）。这属于开启配额自由度的闭环总效应，包含后续GPU/放置/排队反馈，不等于固定GPU时CPU加速率。最大受益cifar10-20占净节省20.43%；诊断性去除后仍改善6.58%，主结果仍保留全部任务。
必须保留长任务损失：deepspeech2-8等作业可能明显延后，后文给全部逐作业数据、P95/P99及makespan。本研究主目标平均JCT可改善，但不能描述为所有作业都更快或所有资源指标都更好。

## 预注册输入与公平性

从官方workload-1依原始行序选前18条CIFAR10、前2条DeepSpeech2。t0固定交错CF0、DS8、CF1、DS9；剩余CIFAR每300秒两条，到2400秒。原batch、GPU demand和完整应用训练工作量不改。合成到达时间与原始时间、行号均另列。source_row_index为新场景的稳定队列顺序，original_source_row_index保留原始出处。

|Job|受控Arrival|原Arrival|原GPU demand|Batch|原行(0-based)|
|---|---:|---:|---:|---:|---:|
|cifar10-0|0.0|401.0|12|4096|0|
|deepspeech2-8|0.0|1468.0|6|320|8|
|cifar10-1|0.0|528.0|12|4096|1|
|deepspeech2-9|0.0|1747.0|8|320|9|
|cifar10-2|300.0|723.0|8|2048|2|
|cifar10-3|300.0|1077.0|6|2048|3|
|cifar10-4|600.0|1174.0|12|4096|4|
|cifar10-5|600.0|1225.0|4|2048|5|
|cifar10-6|900.0|1413.0|8|4096|6|
|cifar10-10|900.0|1894.0|4|2048|10|
|cifar10-11|1200.0|2126.0|10|4096|11|
|cifar10-13|1200.0|2257.0|10|4096|13|
|cifar10-14|1500.0|2793.0|16|4096|14|
|cifar10-15|1500.0|2892.0|16|4096|15|
|cifar10-17|1800.0|3102.0|12|4096|17|
|cifar10-18|1800.0|3125.0|16|4096|18|
|cifar10-19|2100.0|3362.0|10|4096|19|
|cifar10-20|2100.0|3535.0|10|4096|20|
|cifar10-22|2400.0|3685.0|12|4096|22|
|cifar10-24|2400.0|4465.0|4|2048|24|

AWS、DGX-ext各一节点，每节点2GPU、12合成CPU单位、250GB RAM。比例点6CPU/125GB每GPU；固定参考3CPU/62.5GB，R0/E-legacy、alpha0/.5、原网格及既有CPU8、seed_profiles(1,8)、batch自适应不变。60秒周期，6000轮共同观察窗，八格独立进程串行；逐格内存2GiB、磁盘5GiB门槛。
SIA-P实际调用官方Sia GPU逻辑，不加CC准入；SIA-ECT-AUX固定完整Sia任务集合/GPU意图/放置优化辅助配额。CC-P/CC共享完成优先评分、GPU候选、FIFO持续服务、checkpoint保护和250秒终身启动预算，仅差辅助配额自由度。CC-P仍使用响应预测GPU配置速度，不是完全CPU盲基线。CC相对Sia还同时改变目标、准入与放置，不能把总收益都归因CPU。

## 机会预检：在JCT实验之前完成

固定一GPU、原声明batch、逐epoch官方公式的参考耗时，取较快GPU类型，CIFAR占44.58%。若固定AWS为71.03%，固定DGX为44.58%。参考占比依GPU和batch定义而变，不声称在所有策略实际路径下都占同样比例。该离线计算未向在线任务注入未来性能/完成信息，真实训练仍按原有自适应batch执行。

|GPU|CF收益(s)|DS损失(s)|平均净节省(s)|CF/DS交换CPU|支持范围|
|---|---:|---:|---:|---|---|
|aws|687.04|226.06|230.49|8/4 vs6/6|分子分母均支持|
|dgx-ext|118.85|69.00|24.93|8/4 vs6/6|分子分母均支持|

上述解析任务对在CF完成后将CPU还给DS，计入50/25秒真实启动；是全任务平均速率近似，不是执行器精确完成仿真或全局最优证明。AWS的GPU耦合见证比较DS2→CF2串行与CF1+DS1→DS2，计入25秒缩放，主见证分子/分母均在支持域。DGX2分母已知裁剪，因此DGX GPU耦合数字不作为无裁剪主证据。预检证明机会可落地而非保证调度器获胜；实际触发与完整JCT要分别查验。
80个job×GPU配置均覆盖；全可行点边界计数{'queries': 2200, 'numerator_cpu': 608, 'numerator_ram': 520, 'denominator_cpu': 270, 'denominator_ram': 0}。全域沿用旧E-legacy裁剪，未偷偷删配置；主见证不依赖边界裁剪。映射、phi、参考、插值均属合成跨硬件假设，源矩阵并非Saturn测量。

## 实际竞争、交换与时间分解

观察到的两作业同节点构成：{'cifar10+deepspeech2': 153, 'cifar10+cifar10': 55, 'deepspeech2+deepspeech2': 55}；资源交换构成：{'cifar10 <- cifar10': 54, 'cifar10 <- deepspeech2': 136}。
首次跨应用交换：{"time": 780, "round": 12, "node": "dgx-ext:0", "receiver": {"job": "cifar10-1", "application": "cifar10", "cpu": 9.0, "ram": 125.0, "config": "dgx-ext:1:1", "rho_ratio": 1.2153748728708473}, "donor": {"job": "deepspeech2-9", "application": "deepspeech2", "cpu": 3.0, "ram": 125.0, "config": "dgx-ext:1:1", "rho_ratio": 0.7840564530389538}}。
实际完整训练区间内的支持域机会检查：{"SIA_P": {}, "SIA_ECT_AUX": {}, "JOINT_CC_P": {"full_productive_mixed_node_intervals": 166, "supported_exchange_intervals": 166, "positive_net_declared_work_intervals": 166, "nonpositive_net_declared_work_intervals": 0}, "JOINT_CC": {"full_productive_mixed_node_intervals": 110, "supported_exchange_intervals": 110, "positive_net_declared_work_intervals": 110, "nonpositive_net_declared_work_intervals": 0}}。这里固定GPU/batch和直接记录的raw执行率，比较6/6与8/4，双方按各自声明总工作量归一。正净工作分数包含donor损失，但依旧不等于完整运行的反事实JCT收益。
不局限跨应用共驻的全节点同状态CPU枚举：{"SIA_P": {"allocated_node_intervals": 387, "excluded_partial_or_checkpoint_intervals": 68, "supported_fixed_set_node_intervals": 317, "positive_net_normalized_work_node_intervals": 41, "single_job_idle_CPU_opportunity_intervals": 0, "actual_above_same_state_proportional_intervals": 0, "actual_below_same_state_proportional_intervals": 0, "excluded_source_boundary_intervals": 2}, "SIA_ECT_AUX": {"allocated_node_intervals": 395, "excluded_partial_or_checkpoint_intervals": 75, "supported_fixed_set_node_intervals": 320, "positive_net_normalized_work_node_intervals": 33, "single_job_idle_CPU_opportunity_intervals": 0, "actual_above_same_state_proportional_intervals": 0, "actual_below_same_state_proportional_intervals": 33}, "JOINT_CC_P": {"allocated_node_intervals": 336, "excluded_partial_or_checkpoint_intervals": 72, "excluded_source_boundary_intervals": 6, "supported_fixed_set_node_intervals": 258, "positive_net_normalized_work_node_intervals": 180, "single_job_idle_CPU_opportunity_intervals": 0, "actual_above_same_state_proportional_intervals": 0, "actual_below_same_state_proportional_intervals": 0}, "JOINT_CC": {"allocated_node_intervals": 379, "excluded_partial_or_checkpoint_intervals": 76, "supported_fixed_set_node_intervals": 303, "positive_net_normalized_work_node_intervals": 148, "single_job_idle_CPU_opportunity_intervals": 1, "actual_above_same_state_proportional_intervals": 144, "actual_below_same_state_proportional_intervals": 4}}。枚举全部原有CPU点，固定GPU/batch/placement与完整共驻任务集合，RAM固定125；容量12CPU。未省略任何donor，主见证分子分母均支持；边界点和checkpoint/部分区间另记排除数。该离线诊断不修改配置或启动仿真，也不是按JCT选参数。

|单元|CIFAR训练任务秒占比|初始等待|后续等待|Checkpoint|有效训练|GPU-h|CPU单位-h|RAM-GB-h|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|neutral/SIA_P|63.53%|170160.00|5820.00|3300.00|42235.00|19.23|115.38|2403.75|
|neutral/SIA_ECT_AUX|63.53%|170160.00|5820.00|3300.00|42235.00|19.23|20.90|435.28|
|neutral/JOINT_CC_P|46.63%|91440.00|0.00|2400.00|58701.00|18.39|110.36|2299.13|
|neutral/JOINT_CC|46.63%|91440.00|0.00|2400.00|58701.00|18.39|20.06|417.85|
|medium/SIA_P|52.43%|115320.00|2760.00|2875.00|26194.00|12.66|75.94|1582.01|
|medium/SIA_ECT_AUX|53.16%|116880.00|3120.00|3225.00|26604.00|12.91|64.19|1613.68|
|medium/JOINT_CC_P|45.99%|49740.00|0.00|2275.00|34254.00|10.88|65.28|1360.03|
|medium/JOINT_CC|38.34%|41760.00|0.00|2300.00|35625.00|11.94|65.80|1492.40|

CC相对SIA_P的总时间差（任务秒，负值为减少）：{'wait': -76320.0, 'checkpoint': -575.0, 'training': 9431.0}。

CC相对SIA_ECT_AUX的总时间差（任务秒，负值为减少）：{'wait': -78240.0, 'checkpoint': -925.0, 'training': 9021.0}。

CC相对JOINT_CC_P的总时间差（任务秒，负值为减少）：{'wait': -7980.0, 'checkpoint': 25.0, 'training': 1371.0}。

Medium CC相对SIA_P的Median/P95/P99/Makespan变化率（正数为变差）：{'median_jct': -61.26426690079016, 'p95_jct': 5.774305811104941, 'p99_jct': 28.94932009668434, 'makespan': 8.474006886705299}。平均值若改善而尾部/整体结束变差，不能隐去冲突。

Medium CC相对SIA_ECT_AUX的Median/P95/P99/Makespan变化率（正数为变差）：{'median_jct': -61.175642379443865, 'p95_jct': 2.811913971681612, 'p99_jct': 24.965303893353607, 'makespan': 6.05139995073487}。平均值若改善而尾部/整体结束变差，不能隐去冲突。

Medium CC相对JOINT_CC_P的Median/P95/P99/Makespan变化率（正数为变差）：{'median_jct': -18.65781710914455, 'p95_jct': 19.901793110023892, 'p99_jct': 17.96905534813451, 'makespan': 17.599927160156614}。平均值若改善而尾部/整体结束变差，不能隐去冲突。

完整CC实际训练墙钟中，相对同GPU比例点更快/更慢/相同的任务秒：{'above_proportional_training_seconds': 10569.0, 'below_proportional_training_seconds': 10775.0, 'equal_proportional_training_seconds': 14281.0}。资源交换的决策级统计：{'selected_columns': 642, 'nonproportional_ram': 0, 'decisions': 215, 'accelerated_job_decisions': 136, 'slowdown_job_decisions': 134, 'mixed_exchange_decisions': 134, 'mixed_shared_node_decisions': 134}。若这些机会被选中却未降低JCT，应区分候选/评分、放置、准入、重配置和任务间损失，而不是继续只增加任务数。

等待、checkpoint、有效训练互斥且总和逐作业等于JCT。执行训练片段保存raw_before_rho、rho、effective rate与epoch边界残差。同状态比例时间等价变化只作局部诊断，不能当作真实反事实JCT节省。

|单元|排队轮|最大等待作业|CPU饱和GPU空闲节点轮|Fallback|容量违规|所选列裁剪|
|---|---:|---:|---:|---:|---:|---|
|neutral/SIA_P|271|16|0|0|0|{'numerator_cpu': 0, 'numerator_ram': 0, 'denominator_cpu': 5, 'denominator_ram': 0}|
|neutral/SIA_ECT_AUX|271|16|0|0|0|{'numerator_cpu': 750, 'numerator_ram': 750, 'denominator_cpu': 5, 'denominator_ram': 0}|
|neutral/JOINT_CC_P|226|14|0|0|0|{'numerator_cpu': 0, 'numerator_ram': 0, 'denominator_cpu': 19, 'denominator_ram': 0}|
|neutral/JOINT_CC|226|14|0|0|0|{'numerator_cpu': 1009, 'numerator_ram': 1009, 'denominator_cpu': 19, 'denominator_ram': 0}|
|medium/SIA_P|180|16|0|0|0|{'numerator_cpu': 0, 'numerator_ram': 0, 'denominator_cpu': 4, 'denominator_ram': 0}|
|medium/SIA_ECT_AUX|185|16|0|0|0|{'numerator_cpu': 0, 'numerator_ram': 0, 'denominator_cpu': 2, 'denominator_ram': 0}|
|medium/JOINT_CC_P|136|12|0|0|0|{'numerator_cpu': 0, 'numerator_ram': 0, 'denominator_cpu': 11, 'denominator_ram': 0}|
|medium/JOINT_CC|121|11|2|0|0|{'numerator_cpu': 0, 'numerator_ram': 0, 'denominator_cpu': 0, 'denominator_ram': 0}|

|单元|求解状态计数|优化总墙钟(s)|优化P99(s)|
|---|---|---:|---:|
|neutral/SIA_P|{'ORIGINAL_SIA_OPTIMAL': 299}|4.813|0.022|
|neutral/SIA_ECT_AUX|{'OPTIMAL_FIXED_OFFICIAL_INTENT': 299}|0.425|0.002|
|neutral/JOINT_CC_P|{'OPTIMAL_COMPLETION_CONTINUITY': 101, 'DRAIN_FOR_HEAD': 208}|3.567|0.039|
|neutral/JOINT_CC|{'OPTIMAL_COMPLETION_CONTINUITY': 101, 'DRAIN_FOR_HEAD': 208}|6.957|0.073|
|medium/SIA_P|{'ORIGINAL_SIA_OPTIMAL': 198}|3.203|0.021|
|medium/SIA_ECT_AUX|{'OPTIMAL_FIXED_OFFICIAL_INTENT': 202}|0.422|0.003|
|medium/JOINT_CC_P|{'OPTIMAL_COMPLETION_CONTINUITY': 65, 'DRAIN_FOR_HEAD': 118}|2.190|0.036|
|medium/JOINT_CC|{'OPTIMAL_COMPLETION_CONTINUITY': 112, 'DRAIN_FOR_HEAD': 103}|5.764|0.090|

优化器墙钟没有额外推进仿真时间，四策略统计一致，不能据此声称真实计算零成本。辅助配额变化的额外数据加载开销未建模，沿用原执行假设。

## 逐作业JCT与退化

### neutral

|Job|SIA_P|SIA_ECT_AUX|JOINT_CC_P|JOINT_CC|
|---|---:|---:|---:|---:|
|cifar10-0|1423.00|1423.00|1445.00|1445.00|
|deepspeech2-8|8488.00|8488.00|18553.00|18553.00|
|cifar10-1|1423.00|1423.00|1600.00|1600.00|
|deepspeech2-9|8436.00|8436.00|13638.00|13638.00|
|cifar10-2|9669.00|9669.00|2620.00|2620.00|
|cifar10-3|9799.00|9799.00|2740.00|2740.00|
|cifar10-4|12592.00|12592.00|3760.00|3760.00|
|cifar10-5|11991.00|11991.00|3880.00|3880.00|
|cifar10-6|12739.00|12739.00|4900.00|4900.00|
|cifar10-10|13384.00|13384.00|5020.00|5020.00|
|cifar10-11|10341.00|10341.00|6040.00|6040.00|
|cifar10-13|13588.00|13588.00|6160.00|6160.00|
|cifar10-14|10424.00|10424.00|7219.00|7219.00|
|cifar10-15|13900.00|13900.00|8231.00|8231.00|
|cifar10-17|14128.00|14128.00|9029.00|9029.00|
|cifar10-18|14704.00|14704.00|10095.00|10095.00|
|cifar10-19|14944.00|14944.00|10875.00|10875.00|
|cifar10-20|15544.00|15544.00|11955.00|11955.00|
|cifar10-22|15589.00|15589.00|12102.00|12102.00|
|cifar10-24|8409.00|8409.00|12679.00|12679.00|
### medium

|Job|SIA_P|SIA_ECT_AUX|JOINT_CC_P|JOINT_CC|
|---|---:|---:|---:|---:|
|cifar10-0|786.00|786.00|810.00|723.00|
|deepspeech2-8|6963.00|6963.00|8254.00|12916.00|
|cifar10-1|786.00|786.00|963.00|1050.00|
|deepspeech2-9|6443.00|6443.00|10983.00|9910.00|
|cifar10-2|6963.00|6963.00|1323.00|1224.00|
|cifar10-3|8540.00|8712.00|1503.00|1604.00|
|cifar10-4|7623.00|8071.00|1863.00|1730.00|
|cifar10-5|8629.00|9108.00|2043.00|2117.00|
|cifar10-6|8545.00|8334.00|2403.00|2237.00|
|cifar10-10|8929.00|9145.00|3003.00|2657.00|
|cifar10-11|6733.00|6851.00|3318.00|2777.00|
|cifar10-13|7265.00|6973.00|3918.00|3197.00|
|cifar10-14|8593.00|8792.00|4218.00|3421.00|
|cifar10-15|8929.00|9204.00|4818.00|3936.00|
|cifar10-17|8929.00|9126.00|5118.00|4128.00|
|cifar10-18|9289.00|9542.00|5718.00|4608.00|
|cifar10-19|9245.00|9486.00|6018.00|4790.00|
|cifar10-20|9589.00|9902.00|6618.00|5273.00|
|cifar10-22|9507.00|9779.00|6533.00|5453.00|
|cifar10-24|4863.00|4863.00|6844.00|5934.00|

## 测试、版本与局限

212旧测试通过；原始Sia独立3轮分配/进度/重启/完成事件回归一致；独立输入验收通过。Neutral内部轨迹对照：{'SIA_P_vs_SIA_ECT_AUX': {'exact_gpu_progress_completion_restart_trajectory_equal': True, 'first_different_interval_index': None, 'round_counts': [300, 300]}, 'JOINT_CC_P_vs_JOINT_CC': {'exact_gpu_progress_completion_restart_trajectory_equal': True, 'first_different_interval_index': None, 'round_counts': [310, 310]}}。完整容量、锁、有效服务、预算、目标/近整数解、响应单次倍率审计见independent_audit.json。
构造验收未完成时曾提前调用批次，入口在缺少验收文件处拒绝，未启动任何训练；证据prebatch_ordering_stop.md。所有正式运行仅启动一次，安全门槛或其他停止均保留在manifest。没有按结果调参或挑选重跑。
源码fingerprint：21CA3574775F3A10030473BD22183802CC5DED06D2ED57EDE35F3D1A332F5A57。版本是实际404文件快照，不冒称裸HEAD能重建。输入/脚本/包版本见preregistration.json，依赖见dependencies.lock.txt；可重建归档及逐文件hash见delivery_manifest.json。
八格训练进程总墙钟1440.58秒。单一有意构造的场景、20作业不是20次独立重复。P95/P99只作描述，不能作为精确总体尾延迟。辅助配额变化没有额外加载成本、跨硬件响应有合成假设、ETA是当前速率近似，均限制推广。与先前12作业不同负载和到达，不能把两轮均值直接相减作因果效果。本轮不自动扩展。
