# 资源充足档词典序修复与六方法完整复验

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE

六格完成数=6/6；完整245项测试通过，原始Sia一作业三轮逐状态一致。六方法子进程累计668.69秒。算法主目标、CPU/RAM模型、quota网格、资源容量、工作量、初始化画像、solver参数均未改变。

## 根因与证据

真实CC_CPU和CC_MEM均在第240秒复现。失败不是已经证明的数值误判：511列/263列模型的实际最优启动成本为220秒，旧代码把它记为270秒，导致新增二级等式与其他约束矛盾。CC_CPU带对象身份的快照进一步显示已选CIFAR列在候选表中被同一对象重复引用。

上游来源：joint_scheduler.py:365用:g构造展示ID，不区分非常接近的配额；bundle_adaptation.py:68–72用ID重建held/reintroduced候选，可以形成重复引用。真正引发本次错误的代码是原bundle_adaptation.py:98扫描候选表所有位置，用any(v is c for v in candidate)累加成本，而非每个已选变量计费一次。

数学定义保持C(x)=Σ_i d_i x_i。旧记账错误相当于把某个已选列的d_i额外乘以对象在候选表中的出现次数m_i。修复按已选candidate列表逐项计费，并用identity查取其固有单列成本；主效用、二级成本定义、词典序顺序及全部约束保持不变。重复列及展示ID本轮未清理，避免改变候选顺序/可行域。它们仍带来冗余和日志歧义，后续若清理须单独证明几何候选集合等价。

|方法|失败列数|旧错误启动成本下界|独立最小成本|修复模型状态|
|---|---|---|---|---|
|CC_CPU|511|269.999999999|220.0|0|
|CC_MEM|263|269.999999999|220.0|0|

完整失败模型CC_CPU_failed_model.json.gz/CC_MEM_failed_model.json.gz、带identity的CPU快照、独立证明independent_model_proof.json均保留。新增4测试在旧代码上2失败/2通过（重复2次错误100、重复10次错误500），修复后4/4。真实CPU/MEM各8轮越过旧故障点、经过下一次到达/真实调度；无mock替换下一次调度或修改Goodput。

## 修改与回归

{'changed': ['source_manifest.json', 'sia-simulator/joint_prototype/bundle_adaptation.py'], 'added': ['sia-simulator/tests/test_ample_lex_identity.py']}

运行代码为本目录source和完全相同的隔离目录cc-joint-ample-lex-repair。source_diff.patch、baseline_source_snapshot.zip、repaired_source_snapshot.zip和依赖锁支持重建。source_manifest仅更新hash及修复说明；官方源和历史目录未改。原Sia有界回归日志和original_sia_regression.json记录进度、分配、restart、完成事件等精确一致。

## 六方法结果

|方法|完成|Mean|Median|P95|P99|Makespan|
|---|---|---|---|---|---|---|
|SIA_P|12|3095.8333333333335|3236.5|4944.75|5206.55|6172|
|SIA_AUX|12|3095.8333333333335|3236.5|4944.75|5206.55|6172|
|CC_P|12|2648.9166666666665|2443.5|5106.049999999999|5303.610000000001|6553|
|CC_CPU|12|2648.9166666666665|2443.5|5106.049999999999|5303.610000000001|6553|
|CC_MEM|12|2648.9166666666665|2443.5|5106.049999999999|5303.610000000001|6553|
|CC_FULL|12|2648.9166666666665|2443.5|5106.049999999999|5303.610000000001|6553|

## 资源与成本

|方法|GPUh|CPU单位h|RAM GBh|等待s|训练s|checkpoint s|restart|
|---|---|---|---|---|---|---|---|
|SIA_P|6.076944444444443|232.53489914261587|1356.4571959035302|19620.0|14190.0|3340.0|31|
|SIA_AUX|6.076944444444443|91.08681820338978|664.2410648812686|19620.0|14190.0|3340.0|31|
|CC_P|6.350000000000004|242.98339783335015|1417.40693414795|11340.0|18237.0|2210.0|14|
|CC_CPU|6.350000000000004|56.37995574183565|1417.4069341472575|11340.0|18237.0|2210.0|14|
|CC_MEM|6.350000000000004|242.98339783194857|429.4756184915665|11340.0|18237.0|2210.0|14|
|CC_FULL|6.350000000000004|56.37995574183565|429.4756184915665|11340.0|18237.0|2210.0|14|

资源为每节点2GPU、76.53020404150541合成CPU单位、446.42738083375554GB RAM，AWS/DGX各一节点。容量来自先前预注册8种配置的最大每GPU饱和需求×2，未因本轮结果调整。Medium alpha=.5，固定R0/E-legacy，60秒/6000轮，原BASE12作业完整原工作量与seed_profiles(1,8)。比例点沿用既有round9和最低资源契约；容量审计沿用原1e−7容差。

旧成功格逐作业比较见previous_successful_comparison.json；旧失败格不补JCT、不与新结果挑选最佳一次。源码已变化，本次全部六格统一重跑，只有本次六格形成完整主配对。

## 独立审计

{"cells": 6, "completed_cells": 6, "capacity_violations": 0, "lock_violations": 0, "rho_violations": 0, "secondary_count_checks": {"CC_P": 109, "CC_CPU": 109, "CC_MEM": 109, "CC_FULL": 109}, "source_and_inputs_frozen": true, "historical_seals_valid": true, "tests": 245, "original_sia_bounded_exact": true, "simulation_wall_seconds": 668.694988489151}

从原始完成事件和训练片段独立复算JCT/资源小时/互斥时间；独立R0插值核对rho，直接记录raw_rate_before_rho验证倍率只乘一次、checkpoint不加速。全量检查locked集合必须被选中及其完整分配；每个非tail override轮次独立核算已选作业实际启动成本与secondary_tie.minimum_actual_delay一致。响应边界与固定轨迹进度差见response_analysis.json，裁剪比例不能解释为JCT贡献比例。

## 下一轮仅设计

真正差异负载见NEXT_WORKLOAD_DESIGN.md和proposed_workloads.json，本轮未运行它们。通过改变实际submission_time确保调度输入状态不同；不靠换trace名称、无效CSV duration或被自适应忽略的GPU/batch声明制造样本。候选应用扩展另有画像覆盖门槛，未经验证不加入正式矩阵。

## 范围与停止

这是固定合成跨硬件响应下的模拟证据，未校准真实目标硬件CPU/RAM性能；BERT借用模板平坦、已有分母越界和clipping假设保留。辅助配额变更加载开销未建模，solver墙钟不计入虚拟JCT；P95/P99仅12作业描述。未扩大工作负载或自动运行新实验，完成后停止。
