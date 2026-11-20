# 原FULL：预测可用性、候选机会与误差来源离线诊断

**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**

## 1. 结论与本轮边界

本轮没有修改任何调度器、响应模型或实验参数，没有调用训练推进、调度求解或新增仿真；只读取已有完整日志、源码和分叉结果，新增本目录中的独立分析脚本及证据。

**当前不建议直接实现并上线有限前瞻。先解决诊断覆盖与候选表达问题；有限前瞻目前只有降低部分预测误差的动机，没有新增平均JCT收益的直接证据。**

1. 原FULL的运行中存在资源互补，已经使用了大量差异化配额；六个结构性状态的已记录固定GPU配额域中，没有发现进一步的无受损方加速方案。
2. 找到了容量允许、执行响应支持、但当前候选域表达不了的CPU剩余配额机会：420秒CIFAR可用5 CPU，比实际4.399/4.546 CPU预测加速7.34%/5.02%，其他任务配额不变。它不是现有策略合法候选，不能算作优化器漏选了合法解；没有新JCT证据。
3. 79轮原FULL预测不可用的主要原因是未训练作业缺当前Goodput模型，57轮未知者全在等待集合。全局完整接续预检阻断了已知运行任务的比较。这是保守设计的覆盖盲区，不能直接删掉等待者而忽略其机会成本。
4. 可预测的41轮中，只比较三类提案；不同静态ETA方案仅3轮，现有共享接续均判断更差。两个已有完整分叉也没有显示更低平均JCT。不能宣称原FULL全局最优，也不能声称已经发现可直接兑现的遗漏JCT收益。

## 2. 版本、样本与完整性

- 冻结源码：`D:\科研启航\Sia\cc-joint-uncertainty-v4`；340个文件与上一轮source_hashes.json逐一匹配。它是可重建源码副本，不冒称独立Git commit。
- 主证据：`D:\科研启航\Sia\research_runs\sia_synergy\cc_uncertainty_20261101_run1`的NEW_FULL/NEW_FO；其真实执行轨迹已证明分别与OLD_FULL/OLD_FO完全相同。这里只分析两个不重复的物理轨迹。
- 12作业COMPACT12、AWS/DGX各2 GPU/6合成CPU/125GB、R0/E-legacy Medium alpha=0.5。全部已完成；本轮没有产生新JCT。
- 旧产物封印116个文件、源码340文件，在诊断前后均检查。原始四格、4800/5820旧分叉、V3/V4报告全部保留。
- 没有适用AGENTS.md。新目录CURRENT_TASK.md记录授权、只读边界、选样规则与恢复入口。

## 3. 预测可用性：究竟缺什么

|指标|原FULL轨迹（NEW_FULL日志）|原FO轨迹（NEW_FO日志）|
|---|---:|---:|
|有活动作业的决策轮|120|116|
|UNKNOWN_MODEL轮|79|84|
|未知者全部在等待集合的轮数|57|62|
|未知作业×轮记录|302|348|
|未知但已选中（启动/未有效训练）|24|24|
|未知作业已经有效训练的记录|0|0|
|首个全体可预测决策|4800秒|5100秒|

FULL未知区间60–4740秒，FO未知区间60–5040秒。302/348条未知作业轮全部used_gpu_seconds=0，且配置来源为unprofiled_sia_fallback；其中278/324条是未选中的等待者。不是“已训练任务模型仍全部未知”，也不是不确定性端点筛查先拒绝了它们。

根因链：[job.py](D:/科研启航/Sia/cc-joint-uncertainty-v4/sia-simulator/job.py:131)的seed_profiles初始化性能参数；[job.py](D:/科研启航/Sia/cc-joint-uncertainty-v4/sia-simulator/job.py:230)在缺gradient/perf时仍没有Goodput模型；[joint_prototype/completion_time.py](D:/科研启航/Sia/cc-joint-uncertainty-v4/sia-simulator/joint_prototype/completion_time.py:31)因此明确返回raw_rate=None，而非把冷启动归一化占位1当执行速率。[joint_prototype/release_evaluation.py](D:/科研启航/Sia/cc-joint-uncertainty-v4/sia-simulator/joint_prototype/release_evaluation.py:104)要求所有当前已到达作业都有可评分列，否则UNKNOWN_MODEL；[joint_prototype/policy_rollout.py](D:/科研启航/Sia/cc-joint-uncertainty-v4/sia-simulator/joint_prototype/policy_rollout.py:120)在共用接续前仍调用这一预检；[joint_prototype/uncertainty.py](D:/科研启航/Sia/cc-joint-uncertainty-v4/sia-simulator/joint_prototype/uncertainty.py:75)再把它归入UNKNOWN_POINT_MODEL。

这属于可用性限制，不足以认定实现错误。若改为只优化已知任务，必须保持未知任务准入/服务承诺、排队次序及未来可用资源不被损害，或对这些外部影响给出显式保守界；不能将其从总完成时间目标中静默删掉。

## 4. 候选机会：已有域、未表达域与真实JCT必须分开

[joint_prototype/release_evaluation.py](D:/科研启航/Sia/cc-joint-uncertainty-v4/sia-simulator/joint_prototype/release_evaluation.py:178)先固定原策略已选服务集合，随后比较LEGACY、MIN_STATIC_SUM_ETA、CURRENT_ELIGIBLE_BUNDLE。[joint_prototype/release_evaluation.py](D:/科研启航/Sia/cc-joint-uncertainty-v4/sia-simulator/joint_prototype/release_evaluation.py:212)给这些作业选中等式，其余作业不能在纠偏提案中替换；FO再冻结GPU。前置FIFO/持续服务见[joint_prototype/clean_ablation.py](D:/科研启航/Sia/cc-joint-uncertainty-v4/sia-simulator/joint_prototype/clean_ablation.py:172)。因此完成评价器不是从所有等待者、所有候选联合寻找全局最优。

|提案在可预测轮中不同于原方案的次数|FULL（41轮）|FO（32轮）|
|---|---:|---:|
|最小静态ETA|3|0|
|仍合法的旧分配|5|1|
|共享接续预测正收益提案|0|0|

### 4.1 当前原FULL轨迹的三次GPU配置不同提案

|时间|原方案静态ΣETA|提案静态ΣETA|原方案接续Σ完成秒|提案接续Σ完成秒|预测净收益（正为改善）|
|---|---:|---:|---:|---:|---:|
|4800|31788.636|21189.415|6093.715|7406.328|-1312.613|
|5400|10314.064|9863.807|3477.982|3531.839|-53.856|
|5820|8281.790|2592.594|2091.830|2592.594|-500.764|

例如5400秒静态ETA减少450.257秒，但计入原FULL后续合法恢复/资源释放后，预测反而增加53.856秒。减少当前配置ETA不等于减少完整完成时间。5820自然轨迹只剩BERT107/159；不能和含4800坏改选、仍有CIFAR5的历史5820分支混用。

### 4.2 六个结构性状态的固定GPU配额枚举

选样规则记录在CURRENT_TASK.md：首个checkpoint与排队并存、首个未知者全等待、首个完成释放与排队并存，以及最早三个静态ETA不同提案。没有按替代配额收益选样。选出的时刻为180/420/960/4800/5400/5820秒。

从shared_eligibility提取已有合法几何，固定任务集合、GPU类型/数量、节点及checkpoint锁，按节点枚举所有这些配额的共驻组合；没有调用MIP或训练。独立从官方矩阵字面量复算双线性插值与R0 rho，选中点全部吻合。

|时刻|用途|节点级可行配额组合数之和|严格Pareto加速组合|说明|
|---|---|---:|---:|---|
|180|first_checkpoint_and_waiting|197|0|只补入实际已审计选中点：cifar10-1|
|420|first_all_unknown_waiting|15202|0|已有过滤域完整记录|
|960|first_completion_and_waiting|6561|0|只补入实际已审计选中点：cifar10-2|
|4800|different_eta_4800|15059|0|已有过滤域完整记录|
|5400|different_eta_5400|5490|0|已有过滤域完整记录|
|5820|different_eta_5820|13|0|只补入实际已审计选中点：bert-107|

组合数是各节点枚举计数之和，不是独立实验数，也不是全GPU配置域大小。严格Pareto加速要求每个服务作业rho都不下降且至少一个严格上升。六状态均未发现；归一化相对速度和的最大值也没有超过当前方案。这里只排除了记录域中的固定GPU配额提升，不排除任务替换、GPU重选或跨时间方案。

180/960的首次启动作业及5820新恢复几何没有完整共享资格列表，仅补入已提交且已审计的现状；没有伪造其他资格。对这些部分域不能称全候选覆盖。已有源码的未知模型bootstrap只允许比例点（completion_time.py112–116），与960的CIFAR初始比例点一致。

### 4.3 发现的物理机会：剩余CPU，但配额在现有网格外

对同一六状态只做确定性的剩余容量边界算术：固定所有peer配额，将某接收方的CPU或RAM提高至节点剩余上限。它只是执行模型下的物理可行性诊断，不是新增合法候选，更没有实际提交。

|状态/任务|GPU|当前CPU→剩余CPU上限|RAM保持|rho当前→假设|预计同配置速度增幅|当前策略可直接执行？|
|---|---|---:|---:|---:|---:|---|
|420 / cifar10-0|dgx-ext:1:1|4.399394→5.0|103.280883|1.628680→1.748290|7.344%|否，5 CPU不在当前网格|
|420 / cifar10-1|aws:1:1|4.545600→5.0|93.468723|1.599056→1.679347|5.021%|否，5 CPU不在当前网格|

420秒AWS当前占用5.545600 CPU，DGX占用5.399394 CPU；BERT各1 CPU保持不动，可分别给CIFAR增加0.454400、0.600606 CPU，节点仍不超过6 CPU。CIFAR的响应分子/分母均在矩阵支持范围，GPU/batch/placement不变，无让出方配额减少。DGX现有下一较大候选CPU=5.461474，与BERT共驻会超过6；AWS下一候选为6，也不能共驻。因此这是离散表达缺口，不是当前优化器在相同域中错过更高分解。

但BERT当前低配额存在E-legacy裁剪，DGX的参考分母也越界；整套共驻证据仍依赖冻结合成边界假设。不能把接收方坐标合法写成整套配置都具有实测支持。仅RAM填剩余空间在这些已知受益CIFAR点不再加速，因其已到该CPU下的RAM平台。960秒新CIFAR的物理空间同时受冷启动比例bootstrap约束。

**对“原FULL还有什么未抓住”的精确回答：发现了网格外、容量内的局部CPU加速空间；没有发现这六状态已有配额域中的无损速度机会；已有分叉没有证实遗漏的净JCT改善。任务集合/GPU全域的全局机会仍未判定。** 有限前瞻不会自动生成缺失的5 CPU配额，不能用它解决这个候选表达问题。本轮未增加此点或改变网格。

## 5. 误差来源：控制器一致不代表训练动力学准确

当前[joint_prototype/policy_rollout.py](D:/科研启航/Sia/cc-joint-uncertainty-v4/sia-simulator/joint_prototype/policy_rollout.py:24)只推进progress、时间、checkpoint与完成事件；不会产生未来epoch/梯度/性能观测。[joint_prototype/policy_rollout.py](D:/科研启航/Sia/cc-joint-uncertainty-v4/sia-simulator/joint_prototype/policy_rollout.py:59)冻结当前模型并调用真实原FO/FULL控制器，解决了锁、候选与恢复预算不一致，但仍沿用当前epoch与学习参数。[joint_prototype/uncertainty.py](D:/科研启航/Sia/cc-joint-uncertainty-v4/sia-simulator/joint_prototype/uncertainty.py:11)校准键含epoch，不能据此声称未来真实epoch都已覆盖。

|已有单次分叉|接收方提前|让出方延后|训练总秒变化|checkpoint变化|总JCT节省|结论|
|---|---|---|---:|---:|---:|---|
|4800 ALT|BERT159：756秒|BERT107：1236；CIFAR5：893秒|+1393|−20|−1373|原FULL更好|
|历史5820 ALT|BERT107：525秒|CIFAR5：546秒|−29|+50|−21|训练侧收益不能支付切换成本|

两次等待均不变。历史5820在4800错误ALT之后；其makespan改善525秒，但平均JCT恶化1.75秒，体现目标权衡。V3修复把4800预测净收益从+4903.900改为−1312.613，已正确排序；历史5820仍预测+100.272而实际−21，来自接收方收益高估92.404及让出方损失低估28.868，总误差121.272秒翻转小净差。

### 5.1 决策前残差的顺序留出检查

每条已有真实训练标签只用其origin之前已产生、observed_end/observed_at均不晚于origin的同层历史构造范围，再检验该条后续标签；不跨FO/FULL共享，不用未来完成时间。属于日志内顺序检验，不是独立负载泛化。

|方法|标签/分层|有历史支持|落在既有经验范围内|raw速率误差中位/P95|
|---|---:|---:|---:|---:|
|NEW_FULL|366/72|265|259|4.432% / 5.148%|
|NEW_FO|375/103|239|239|4.432% / 5.148%|

FULL的6个后续标签超出此前经验范围，故历史min/max不是可靠概率界；重复间隔相关，不把265个支持标签当265次独立实验。未选中过的配置/batch/epoch仍没有校准。此检查不能证明“有支持即可安全改选”。

### 5.2 已保存预测的短窗精度（未生成新预测）

读取此前已经保存的baseline.model_execution，比较相同原策略之后60/120/300秒的积分训练工作。实际量由raw_before_rho×rho×segment_seconds独立累加，去除整数epoch舍入残差；checkpoint不计训练。只评价已有可预测轮，不补未知模型。

|方法|窗口|有效作业×起点样本|积分工作误差中位|P95|最大|
|---|---|---:|---:|---:|---:|
|NEW_FULL|60s|88|3.176%|5.148%|16.120%|
|NEW_FULL|120s|92|2.859%|5.180%|47.426%|
|NEW_FULL|300s|96|2.072%|13.358%|39.215%|
|NEW_FO|60s|90|5.148%|5.148%|8.179%|
|NEW_FO|120s|91|4.536%|5.148%|8.179%|
|NEW_FO|300s|92|4.528%|8.179%|10.824%|

零实际训练单独计数，不除零、不删除后当成成功样本：FULL60/120秒为8/4条、FO为2/1条，均没有“实际零训练但预测正训练”。300秒均有实际训练。全部是相关的重复时点，且只覆盖晚期可预测状态，不能代表早期冷启动阶段。

FULL原方案完整剩余时间预测的96个作业×起点，绝对相对误差中位3.137%、P95 12.836%；FO92条为4.989%、36.558%。这一指标与短窗积分工作误差不同，不能直接相减作归因。短窗也有明显离群值（FULL120秒最大47.426%），原因可能包含完成/重配置边界；本轮不借此选一个更有利窗口。

## 6. 问题优先级与最小方案（仅建议，未实施）

|级别|证据位置|影响|最小建议|
|---|---|---|---|
|High|release_evaluation.py104–120；availability_rounds.json|未训练等待者使全体完成评价不可用，57个FULL轮连已知运行集合都未比较|先给预测失败分类及保存完整评分；研究不损害未知等待者的局部支配/资源释放条件，不能直接略去未知者|
|High|policy_rollout.py24–57；两次历史分叉|冻结训练阶段的速率误差可翻转小净收益|预测与真实选择规则继续共用；先验证未参与设计的阶段/配置预测，未知回退，不补未来梯度|
|Medium|release_evaluation.py178–259；proposal_coverage.json|三提案覆盖有限，无改选不是全域无机会|记录提案是否真正不同、候选过滤原因、固定集合；不为寻找正结果任意增加提案|
|Medium|joint_scheduler.py714；physical_slack_bounds.json|现有网格不能表达剩余5 CPU，有限前瞻也无法使用|若研究者另行批准候选改动，考虑确定性的剩余容量闭包，并做仅改变候选表达的独立消融；本轮不改|
|Medium|uncertainty.py21–26、83；prequential_calibration_rows.json|精细分层覆盖稀疏，经验范围也可能失效|校准共享规则必须用独立留出数据论证，不能针对四个缺失层放宽|
|Medium|rounds中completion_scores仅count/hash|无法无损重建所有GPU候选的当时raw/批量/ETA|未来只增加观测记录，不回填旧日志；本轮不能伪造全GPU反事实|
|Medium|joint_profile.py186、272；六状态坐标|BERT低配额/参考点依赖裁剪，缺少目标硬件实测支持|保留E-legacy结论范围，区分接收方数据内与整套共驻数据内，不改曲线制造支持|

## 7. 有限前瞻是否值得实现，以及下一步门槛

**结论：有研究动机，但尚不具备立即进入新调度版本/完整训练对照的证据。** 短窗通常减少跨训练阶段的误差暴露，却不能消除未知等待者影响、候选网格缺口和资源释放后的任务替换风险。原FULL还没有被证明全局最优；当前也没有能在其既定合法域内、经独立完成验证证明更好的新增方案。

建议下一步先由研究者选择清晰范围，优先级如下：

1. 先保持原FULL，讨论是否允许处理已明确的候选表达缺口。若允许，应采用对所有任务/策略对称的规则，FO也应用；固定响应/负载，单独识别候选覆盖贡献，不能包装成有限前瞻收益。
2. 若坚持候选域完全不变，下一步只应做观察性补充：保存各候选raw、batch、剩余工作、当前服务/锁和模型缺失原因。已有日志只能确认三提案及部分资格域，不能凭空精确枚举所有跨GPU完成反事实。
3. 有限前瞻的实现门槛：在与4800/5820独立的状态中，找到现有合法域内且数据支持充分的改选；让出方/等待者的损失已计入；预测证据足以覆盖真实切换开销；不能仅依赖短窗积分工作增加。若要有限截断，必须给未完成工作的终端价值或保守边界，不能把晚发生的损失截掉。
4. 只有上述门槛成立，再预注册一次同状态单次分叉，后续都执行原FULL；再通过同负载12作业新旧FO/FULL完整比较。保留不改善结果，不启动80作业，不按JCT挑窗口/阈值/分层。

当前原FULL相对FO的既有12作业8.155%优势保留；最新安全筛查没有新增JCT收益。也不应因为纠偏支线暂未增益就否定已有有序启动、持续服务与CPU/RAM差异化主线。

## 8. 验证、产物与复核

- 8项独立诊断检查通过：矩阵节点插值、比例基准/平坦模板、已选rho吻合、unknown分类、容量/完整锁、网格外不能冒充合法点、顺序校准汇总、已有短窗积分。它们是本轮分析检查，不是重新运行385项调度器测试，也不是性能实验通过。
- diagnose.py、slack_bounds.py、horizon_audit.py只导入Python标准库和本目录纯分析函数；没有导入项目调度模块，不调用Job.step/Cluster.step/optimize_policy/MIP。
- availability_summary.json / availability_rounds.json / unknown_job_rounds.json：逐轮模型可用性。
- proposal_coverage.json / baseline_prediction_errors.json：已有三提案和后验预测误差。
- selected_state_receipt.json / fixed_gpu_quota_enumeration.json / physical_slack_bounds.json：选样、域内枚举、域外容量边界，严格分开。
- prequential_calibration_*.json / existing_forecast_horizon_*.json：顺序残差和已有短窗预测核查。
- integrity_before.json / integrity_after.json / evidence_hashes.json / artifact_hashes.json：原始封印与本轮输入/输出hash。
- test_diagnostics.log / diagnose.log / slack_bounds.log / horizon_audit.log：命令输出。

完成后停止。全部结论是冻结trace与合成跨硬件模型下的诊断，不是目标Saturn集群实测，也不是未运行方案的JCT预测成绩。
