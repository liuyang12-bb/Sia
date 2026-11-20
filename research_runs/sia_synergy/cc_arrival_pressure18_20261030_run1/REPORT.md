# 同一12作业、三种到达压力：六方法18格实验

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE

完成18/18格，正式子进程累计墙钟1059.82秒。源算法与已验收词典序修复版逐文件一致，未按结果调整任何参数。

## 冻结条件

原同一6 CIFAR＋6 BERT、原GPU demand/batch/完整训练工作量/顺序不变。AWS/DGX各一节点、各2GPU/6合成CPU单位/125GB RAM；固定R0/E-legacy、Medium alpha=.5，seed_profiles(1,8)，60秒、6000轮，独立进程串行，启动前2GiB内存/5GiB磁盘门槛。

BURST全部t0；COMPACT前4个t0、其余每300秒2个，末到达1200；SPREAD第i个i×300秒，末到达3300。只改变实际arrival；各方案同一组应用画像，不是三条独立自然trace，不做虚假样本量/显著性推断。COMPACT与历史原始CSV逐字节一致，但本轮全部用同一修复版重跑，不混用旧版结果。

六方法：SIA_P调用官方Sia GPU及placement，比例CPU/RAM，无L2；SIA_AUX冻结同轮完整Sia意图，用本项目ECT辅助优化，并非Synergy-TUNE；CC_P/CPU/MEM/FULL共享修复后的CC-TAIL目标与服务/放置机制，依次开放零/CPU/内存/两种辅助配额自由度。

## 先验机会门槛

{"passed": true, "methods": {"SIA_P": {"totals": {"net_normalized_progress": 0.0, "sum_round_same_state_eta_savings": 0.0, "productive_job_seconds": 9362.0, "productive_coplace_node_seconds": 652}, "per_job": {"cifar10-2": {"net_normalized_progress": 0.0, "receiver_fraction": 0, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 0.0}, "cifar10-3": {"net_normalized_progress": 0.0, "receiver_fraction": 0, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 0.0}, "cifar10-0": {"net_normalized_progress": 0.0, "receiver_fraction": 0, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 0.0}, "cifar10-1": {"net_normalized_progress": 0.0, "receiver_fraction": 0, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 0.0}, "bert-28": {"net_normalized_progress": 0.0, "receiver_fraction": 0, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 0.0}, "bert-37": {"net_normalized_progress": 0.0, "receiver_fraction": 0, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 0.0}, "bert-54": {"net_normalized_progress": 0.0, "receiver_fraction": 0, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 0.0}}, "scope": "Same executed GPU/batch/raw rate; proportional counterfactual changes only quotas. Normalized progress and round ETA sums are diagnostic, not realized JCT savings. All served jobs counted; delays unchanged by rho."}, "CC_CPU": {"totals": {"net_normalized_progress": 0.5410845114896541, "sum_round_same_state_eta_savings": 54182.734251683185, "productive_job_seconds": 12372.0, "productive_coplace_node_seconds": 5242, "fully_supported_coplace_node_seconds": 120, "receiver_faster_donor_not_slower_node_seconds": 5092}, "per_job": {"cifar10-0": {"net_normalized_progress": 0.15178305500958314, "receiver_fraction": 0.15178305500958314, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 3792.417419259923}, "cifar10-1": {"net_normalized_progress": 0.14953145918742153, "receiver_fraction": 0.1496698619998945, "donor_fraction_loss": 0.00013840281247295023, "same_state_eta_saving_sum": 17670.288042967444}, "bert-28": {"net_normalized_progress": 0.0, "receiver_fraction": 0, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 0.0}, "bert-37": {"net_normalized_progress": 0.0, "receiver_fraction": 0, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 0.0}, "cifar10-2": {"net_normalized_progress": 0.14747384340767963, "receiver_fraction": 0.14747384340767963, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 14676.05799746486}, "bert-54": {"net_normalized_progress": 0.0, "receiver_fraction": 0, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 0.0}, "cifar10-3": {"net_normalized_progress": 0.09229615388496985, "receiver_fraction": 0.09229615388496985, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 18043.97079199098}, "bert-99": {"net_normalized_progress": 0.0, "receiver_fraction": 0, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 0.0}, "cifar10-4": {"net_normalized_progress": 0.0, "receiver_fraction": 0, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 0.0}}, "scope": "Same executed GPU/batch/raw rate; proportional counterfactual changes only quotas. Normalized progress and round ETA sums are diagnostic, not realized JCT savings. All served jobs counted; delays unchanged by rho."}, "CC_MEM": {"totals": {"net_normalized_progress": 0.8960478381450822, "sum_round_same_state_eta_savings": 61340.88795364615, "productive_job_seconds": 12321.0, "productive_coplace_node_seconds": 5161, "fully_supported_coplace_node_seconds": 120, "receiver_faster_donor_not_slower_node_seconds": 5021}, "per_job": {"cifar10-0": {"net_normalized_progress": 0.235451254580177, "receiver_fraction": 0.235451254580177, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 4425.324948559917}, "cifar10-1": {"net_normalized_progress": 0.2277761921150539, "receiver_fraction": 0.22882728944601743, "donor_fraction_loss": 0.0010510973309635378, "same_state_eta_saving_sum": 18157.123828630476}, "bert-28": {"net_normalized_progress": 0.0, "receiver_fraction": 0, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 0.0}, "bert-37": {"net_normalized_progress": 0.0, "receiver_fraction": 0, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 0.0}, "cifar10-2": {"net_normalized_progress": 0.23024944667772454, "receiver_fraction": 0.23024944667772454, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 19473.584810224223}, "cifar10-3": {"net_normalized_progress": 0.20257094477212673, "receiver_fraction": 0.20257094477212673, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 19284.85436623154}, "bert-54": {"net_normalized_progress": 0.0, "receiver_fraction": 0, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 0.0}, "bert-99": {"net_normalized_progress": 0.0, "receiver_fraction": 0, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 0.0}, "cifar10-4": {"net_normalized_progress": 0.0, "receiver_fraction": 0, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 0.0}}, "scope": "Same executed GPU/batch/raw rate; proportional counterfactual changes only quotas. Normalized progress and round ETA sums are diagnostic, not realized JCT savings. All served jobs counted; delays unchanged by rho."}, "CC_FULL": {"totals": {"net_normalized_progress": 1.5707869556678262, "sum_round_same_state_eta_savings": 94524.83642533839, "productive_job_seconds": 12236.0, "productive_coplace_node_seconds": 4240, "fully_supported_coplace_node_seconds": 60, "receiver_faster_donor_not_slower_node_seconds": 4150}, "per_job": {"cifar10-0": {"net_normalized_progress": 0.3890341909876853, "receiver_fraction": 0.3890341909876853, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 6756.341126936614}, "cifar10-1": {"net_normalized_progress": 0.37659094985317954, "receiver_fraction": 0.37764204718414307, "donor_fraction_loss": 0.0010510973309635378, "same_state_eta_saving_sum": 24737.069133652607}, "bert-28": {"net_normalized_progress": 0.0, "receiver_fraction": 0, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 0.0}, "bert-37": {"net_normalized_progress": 0.0, "receiver_fraction": 0, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 0.0}, "cifar10-2": {"net_normalized_progress": 0.38294129065127597, "receiver_fraction": 0.38294129065127597, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 28551.03260238415}, "bert-54": {"net_normalized_progress": 0.0, "receiver_fraction": 0, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 0.0}, "cifar10-3": {"net_normalized_progress": 0.38404482621884684, "receiver_fraction": 0.38404482621884684, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 15590.844262265271}, "bert-99": {"net_normalized_progress": 0.0, "receiver_fraction": 0, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 0.0}, "cifar10-4": {"net_normalized_progress": 0.038175697956839186, "receiver_fraction": 0.038175697956839186, "donor_fraction_loss": 0, "same_state_eta_saving_sum": 18889.549300099763}}, "scope": "Same executed GPU/batch/raw rate; proportional counterfactual changes only quotas. Normalized progress and round ETA sums are diagnostic, not realized JCT savings. All served jobs counted; delays unchanged by rho."}}, "thresholds_frozen_before_runs": true, "warning": "E-legacy clipped exposure distinguished from supported points; fixed-state ETA sums not whole-run JCT proof. No tuning on failure."}

gate固定60轮上限：FULL实际CF+BERT同节点有效共训中，CF相对同GPU比例点加速而BERT不减速的重叠不少于60节点秒；同时全执行服务集合归一化净进度为正、逐轮同状态剩余ETA节省总和为正。后者不等于实现JCT节省、不是全局JCT最优证明，未使用未来完成时间。所有已执行任务均计receiver收益和donor损失，checkpoint不加速。支持范围内共驻单独记录，不能用E-legacy裁剪点冒充域内测量。

## 完整结果

|负载|方法|完成|Mean|Median|P95|P99|Makespan|
|---|---|---|---|---|---|---|---|
|BURST|SIA_P|12|4891.583333333333|4765.0|8234.0|8383.6|8421|
|BURST|SIA_AUX|12|4554.833333333333|4493.0|7488.049999999999|7747.21|7812|
|BURST|CC_P|12|4914.166666666667|5111.5|8073.75|8089.15|8093|
|BURST|CC_CPU|12|4556.166666666667|4511.0|7604.05|7660.81|7675|
|BURST|CC_MEM|12|4430.666666666667|4386.0|7461.95|7466.79|7468|
|BURST|CC_FULL|12|3994.3333333333335|3871.5|7253.4|7584.280000000001|7667|
|COMPACT|SIA_P|12|4613.833333333333|4887.0|7157.35|7193.87|8403|
|COMPACT|SIA_AUX|12|4411.333333333333|4183.5|7430.599999999999|7627.72|8129|
|COMPACT|CC_P|12|4414.166666666667|4661.5|6873.75|6889.15|8093|
|COMPACT|CC_CPU|12|4056.1666666666665|4145.0|6404.05|6460.81|7675|
|COMPACT|CC_MEM|12|3930.6666666666665|4119.5|6261.95|6266.79|7468|
|COMPACT|CC_FULL|12|3494.3333333333335|3421.5|6218.4|6417.280000000001|7667|
|SPREAD|SIA_P|12|3364.25|3155.5|5225.6|5229.12|8522|
|SPREAD|SIA_AUX|12|3168.5|3414.0|4874.65|4908.53|7840|
|SPREAD|CC_P|12|3647.4166666666665|3367.0|6697.799999999997|7949.160000000002|8718|
|SPREAD|CC_CPU|12|3458.5|3088.5|6532.799999999997|7916.160000000002|8562|
|SPREAD|CC_MEM|12|3308.3333333333335|3002.0|6257.999999999998|7666.000000000002|8318|
|SPREAD|CC_FULL|12|2737.5833333333335|2365.0|5267.5999999999985|6467.920000000002|7668|

## 配对消融

|负载|对照|平均JCT改善|改善/退化/不变|等待差s|训练差s|checkpoint差s|P99变化|Makespan变化|
|---|---|---|---|---|---|---|---|---|
|BURST|CC_FULL_vs_SIA_P|18.343%|8/4/0|-11580.0|1983.0|-1170.0|-9.534%|-8.954%|
|BURST|CC_FULL_vs_SIA_AUX|12.306%|8/4/0|-9480.0|3754.0|-1000.0|-2.103%|-1.856%|
|BURST|CC_P_vs_SIA_P|-0.462%|8/4/0|-4560.0|6021.0|-1190.0|-3.512%|-3.895%|
|BURST|CC_CPU_vs_CC_P|7.285%|8/3/1|-2700.0|-1546.0|-50.0|-5.295%|-5.165%|
|BURST|CC_MEM_vs_CC_P|9.839%|9/2/1|-3540.0|-2212.0|-50.0|-7.694%|-7.723%|
|BURST|CC_FULL_vs_CC_P|18.718%|9/2/1|-7020.0|-4038.0|20.0|-6.241%|-5.264%|
|BURST|CC_FULL_vs_CC_CPU|12.331%|8/2/2|-4320.0|-2492.0|70.0|-0.999%|-0.104%|
|BURST|CC_FULL_vs_CC_MEM|9.848%|7/3/2|-3480.0|-1826.0|70.0|1.574%|2.665%|
|COMPACT|CC_FULL_vs_SIA_P|24.264%|8/4/0|-14040.0|1526.0|-920.0|-10.795%|-8.759%|
|COMPACT|CC_FULL_vs_SIA_AUX|20.787%|7/5/0|-12000.0|2156.0|-1160.0|-15.869%|-5.683%|
|COMPACT|CC_P_vs_SIA_P|4.328%|6/6/0|-7020.0|5564.0|-940.0|-4.236%|-3.689%|
|COMPACT|CC_CPU_vs_CC_P|8.110%|8/3/1|-2700.0|-1546.0|-50.0|-6.218%|-5.165%|
|COMPACT|CC_MEM_vs_CC_P|10.953%|9/2/1|-3540.0|-2212.0|-50.0|-9.034%|-7.723%|
|COMPACT|CC_FULL_vs_CC_P|20.838%|9/2/1|-7020.0|-4038.0|20.0|-6.849%|-5.264%|
|COMPACT|CC_FULL_vs_CC_CPU|13.851%|8/2/2|-4320.0|-2492.0|70.0|-0.674%|-0.104%|
|COMPACT|CC_FULL_vs_CC_MEM|11.101%|7/3/2|-3480.0|-1826.0|70.0|2.401%|2.665%|
|SPREAD|CC_FULL_vs_SIA_P|18.627%|9/2/1|-4860.0|-990.0|-1670.0|23.690%|-10.021%|
|SPREAD|CC_FULL_vs_SIA_AUX|13.600%|9/2/1|-2640.0|-911.0|-1620.0|31.769%|-2.194%|
|SPREAD|CC_P_vs_SIA_P|-8.417%|7/5/0|2400.0|2908.0|-1910.0|52.017%|2.300%|
|SPREAD|CC_CPU_vs_CC_P|5.179%|10/1/1|-1080.0|-1187.0|0.0|-0.415%|-1.789%|
|SPREAD|CC_MEM_vs_CC_P|9.297%|11/1/0|-2160.0|-2029.0|120.0|-3.562%|-4.588%|
|SPREAD|CC_FULL_vs_CC_P|24.945%|11/1/0|-7260.0|-3898.0|240.0|-18.634%|-12.044%|
|SPREAD|CC_FULL_vs_CC_CPU|20.845%|10/2/0|-6180.0|-2711.0|240.0|-18.295%|-10.441%|
|SPREAD|CC_FULL_vs_CC_MEM|17.252%|10/2/0|-5100.0|-1869.0|120.0|-15.628%|-7.814%|

改善率=1−新/对照，正值表示下降；尾部变化=新/对照−1，正值表示恶化。CPU/MEM/FULL对CC_P反映辅助自由度的闭环增量，包含后续GPU/队列变化，不能简单视为固定轨迹的物理因果份额。FULL对Sia还包括完成优先目标/准入/持续服务/放置的差异。

## 资源小时与成本

|负载|方法|GPUh|CPU单位h|RAM GBh|等待s|训练s|checkpoint s|restart|
|---|---|---|---|---|---|---|---|---|
|BURST|SIA_P|8.394444444444446|25.18333333333334|524.652777777778|34500.0|20769.0|3430.0|30|
|BURST|SIA_AUX|7.852777777777778|18.490794019252405|396.92392923073595|32400.0|18998.0|3260.0|28|
|BURST|CC_P|8.313888888888894|24.94166666666667|519.6180555555553|29940.0|26790.0|2240.0|16|
|BURST|CC_CPU|7.970555555555559|19.532669653720998|498.1597222222222|27240.0|25244.0|2190.0|15|
|BURST|CC_MEM|7.735555555555559|23.206666666666667|398.0918168028675|26400.0|24578.0|2190.0|15|
|BURST|CC_FULL|7.577500000000005|14.67244370949176|317.8741605632872|22920.0|22752.0|2260.0|15|
|COMPACT|SIA_P|8.76361111111111|26.290833333333335|547.7256944444446|30960.0|21226.0|3180.0|32|
|COMPACT|SIA_AUX|8.645833333333336|19.646276616904707|428.34087761767285|28920.0|20596.0|3420.0|34|
|COMPACT|CC_P|8.313888888888894|24.94166666666667|519.6180555555553|23940.0|26790.0|2240.0|16|
|COMPACT|CC_CPU|7.970555555555559|19.532669653720998|498.1597222222222|21240.0|25244.0|2190.0|15|
|COMPACT|CC_MEM|7.735555555555559|23.206666666666667|398.0918168028675|20400.0|24578.0|2190.0|15|
|COMPACT|CC_FULL|7.577500000000005|14.67244370949176|317.8741605632872|16920.0|22752.0|2260.0|15|
|SPREAD|SIA_P|8.416388888888893|25.249166666666675|526.0243055555557|17400.0|19701.0|3270.0|31|
|SPREAD|SIA_AUX|8.153611111111113|18.470494127288227|406.85985900542767|15180.0|19622.0|3220.0|30|
|SPREAD|CC_P|9.274722222222215|27.824166666666642|579.6701388888893|19800.0|22609.0|1360.0|4|
|SPREAD|CC_CPU|8.986388888888884|15.0056424453557|561.6493055555559|18720.0|21422.0|1360.0|4|
|SPREAD|CC_MEM|8.718611111111104|26.15583333333331|327.19195707477644|17640.0|20580.0|1480.0|5|
|SPREAD|CC_FULL|7.694166666666663|13.476839860912266|290.2097183326405|12540.0|18711.0|1600.0|6|

## 压力是否真正改变执行

|格|观测活跃峰值|等待作业峰值|等待作业轮数|
|---|---|---|---|
|BURST_RNEW_medium_SIA_P|12|12|575|
|BURST_RNEW_medium_SIA_AUX|12|12|540|
|BURST_RNEW_medium_CC_P|12|12|499|
|BURST_RNEW_medium_CC_CPU|12|12|454|
|BURST_RNEW_medium_CC_MEM|12|12|440|
|BURST_RNEW_medium_CC_FULL|12|12|382|
|COMPACT_RNEW_medium_SIA_P|12|9|516|
|COMPACT_RNEW_medium_SIA_AUX|12|9|482|
|COMPACT_RNEW_medium_CC_P|12|8|399|
|COMPACT_RNEW_medium_CC_CPU|12|8|354|
|COMPACT_RNEW_medium_CC_MEM|11|7|340|
|COMPACT_RNEW_medium_CC_FULL|11|7|282|
|SPREAD_RNEW_medium_SIA_P|9|6|290|
|SPREAD_RNEW_medium_SIA_AUX|8|5|253|
|SPREAD_RNEW_medium_CC_P|9|6|330|
|SPREAD_RNEW_medium_CC_CPU|8|5|312|
|SPREAD_RNEW_medium_CC_MEM|8|5|294|
|SPREAD_RNEW_medium_CC_FULL|8|5|209|

|相对COMPACT|执行签名相同|所有完成时刻相同|平均完成时刻差|平均arrival差|平均JCT差|
|---|---|---|---|---|---|
|BURST_RNEW_medium_SIA_P|False|False|-222.25|-500|277.75|
|BURST_RNEW_medium_SIA_AUX|False|False|-356.5|-500|143.5|
|BURST_RNEW_medium_CC_P|True|True|0|-500|500|
|BURST_RNEW_medium_CC_CPU|True|True|0|-500|500|
|BURST_RNEW_medium_CC_MEM|True|True|0|-500|500|
|BURST_RNEW_medium_CC_FULL|True|True|0|-500|500|
|SPREAD_RNEW_medium_SIA_P|False|False|-99.58333333333333|1150|-1249.5833333333333|
|SPREAD_RNEW_medium_SIA_AUX|False|False|-92.83333333333333|1150|-1242.8333333333333|
|SPREAD_RNEW_medium_CC_P|False|False|383.25|1150|-766.75|
|SPREAD_RNEW_medium_CC_CPU|False|False|552.3333333333334|1150|-597.6666666666666|
|SPREAD_RNEW_medium_CC_MEM|False|False|527.6666666666666|1150|-622.3333333333334|
|SPREAD_RNEW_medium_CC_FULL|False|False|393.25|1150|-756.75|

JCT差=完成时刻差−arrival差。若某方法在BURST和COMPACT的完成时刻完全相同，则其跨负载JCT变化仅由提交时间改变造成，不能称作该方法执行变快/变慢。签名只比较进度、完成、rho、GPU类型/放置等实际执行字段，排除arrival/等待/求解墙钟。任务集合输入不同也不保证服务轨迹一定不同：原紧凑到达可能已早于该策略服务空档。所有跨方法的主比较仍使用同一负载，arrival相同因而抵消。

## 实际共驻、增益与边界

|格|CF/BERT共训节点s|receiver进度份额|donor损失份额|CPU分子裁剪%|RAM分子裁剪%|
|---|---|---|---|---|---|
|BURST_RNEW_medium_SIA_P|2403|0|0|43.94048822764697|0.0|
|BURST_RNEW_medium_SIA_AUX|1736|1.1884889897978448|0.07276694892412357|53.705653226655436|47.70502158121908|
|BURST_RNEW_medium_CC_P|9000|0|0|33.68421052631579|0.0|
|BURST_RNEW_medium_CC_CPU|8427|0.869635817617835|0.00013840281247295023|55.73205514181588|0.0|
|BURST_RNEW_medium_CC_MEM|8118|1.3146115952654716|0.0010510973309635376|35.91016356090813|57.677597851737325|
|BURST_RNEW_medium_CC_FULL|5876|2.2998276019963546|0.0010510973309635376|70.82014767932489|70.82014767932489|
|COMPACT_RNEW_medium_SIA_P|0|0|0|45.533779327240175|0.0|
|COMPACT_RNEW_medium_SIA_AUX|307|0.7948332967800077|0.3643874248698579|56.60322392697611|51.91299281413867|
|COMPACT_RNEW_medium_CC_P|9000|0|0|33.68421052631579|0.0|
|COMPACT_RNEW_medium_CC_CPU|8427|0.869635817617835|0.00013840281247295023|55.73205514181588|0.0|
|COMPACT_RNEW_medium_CC_MEM|8118|1.3146115952654716|0.0010510973309635376|35.91016356090813|57.677597851737325|
|COMPACT_RNEW_medium_CC_FULL|5876|2.2998276019963546|0.0010510973309635376|70.82014767932489|70.82014767932489|
|SPREAD_RNEW_medium_SIA_P|1507|0|0|45.76925029186336|0.0|
|SPREAD_RNEW_medium_SIA_AUX|963|1.121569505704664|0.38377838135954795|57.10427071654266|54.658036897360105|
|SPREAD_RNEW_medium_CC_P|5717|0|0|33.57954796762351|0.0|
|SPREAD_RNEW_medium_CC_CPU|5248|0.7101525349779243|0|73.30314629819812|0.0|
|SPREAD_RNEW_medium_CC_MEM|4820|1.1257557442361257|0|37.80855199222546|70.26239067055394|
|SPREAD_RNEW_medium_CC_FULL|4180|2.2795383000187583|0|72.35316124205013|71.07049329271551|

分子/分母、CPU/RAM及浮点端点容差统计均在response_analysis.json。裁剪比例为有效训练秒加权，CPU/RAM可重叠，不能相加或当作收益贡献比例。局部进度份额固定实际GPU/batch/已执行片段，不是可行完整反事实调度。

## 统计解释与局限

每个负载12作业，P95/P99只描述这一批任务；所有长任务均保留。comparisons.json包含最大贡献任务、逐作业差值和去掉单个最大贡献作业的诊断，后者不代替主结果。任务工作量相同但arrival不同，makespan跨负载包含到达跨度，不应只比较其绝对值评价调度优劣。逐作业JCT=completion−当前arrival；主要比较在同一负载的配对方法之间。

共享官方Sia GPU画像和Synergy矩阵，应用模板关联、phi和跨硬件迁移为合成假设；BERT平坦模板不是目标硬件实测。已有reference越界/裁剪未修改；物理配额变更额外加载开销未建模，调度墙钟不计入虚拟JCT。245原测试随完全相同代码复用，本轮另6项输入/门槛核算测试通过。

## 复现与停止

{"phase": "final", "cells": 18, "complete_cells": 18, "capacity_violations": 0, "lock_violations": 0, "rho_violations": 0, "secondary_cost_checks": {"BURST_RNEW_medium_CC_P": 135, "BURST_RNEW_medium_CC_CPU": 128, "BURST_RNEW_medium_CC_MEM": 125, "BURST_RNEW_medium_CC_FULL": 127, "COMPACT_RNEW_medium_CC_P": 135, "COMPACT_RNEW_medium_CC_CPU": 128, "COMPACT_RNEW_medium_CC_MEM": 125, "COMPACT_RNEW_medium_CC_FULL": 127, "SPREAD_RNEW_medium_CC_P": 146, "SPREAD_RNEW_medium_CC_CPU": 142, "SPREAD_RNEW_medium_CC_MEM": 137, "SPREAD_RNEW_medium_CC_FULL": 127}, "source_unchanged": true, "input_hashes_valid": true, "old_seal_verified": 384, "opportunity_gate_passed": true, "simulation_wall_seconds": 1059.8210084438324, "source_tests_reused": 245, "new_input_tests": 6}

源码source_snapshot.zip，预注册preregistration.json，原始逐轮cells/*/rounds.jsonl.gz，启动execution_manifest.jsonl，独立复算raw_audit.json/per_job.csv/metrics.json，比较comparisons.json。旧结果未覆盖，未恢复104/160批次，未运行DeepSpeech扩展。全部完成后停止。
