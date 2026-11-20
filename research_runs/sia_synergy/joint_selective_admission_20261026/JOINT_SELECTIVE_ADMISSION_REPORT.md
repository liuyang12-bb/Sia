# 联合调度决策改进：选择性准入 JOINT-S

## 结论

完成了一项有证据支持的决策改进，**工程验收通过，JCT效果尚未验证**。不是换数据、调大权重或修改CPU曲线，而是将“先把所有可并发作业都承诺服务，再优化资源”改为“已承诺服务＋FIFO队首必选，其余作业由同一个公平目标选择是否准入”。新策略名称JOINT-S，旧JOINT-G及历史基线均保留。

154项测试通过（141项旧测试＋13项新增）。其中有真实未seed BERT的首次分配→checkpoint→有效训练→下一次调度；20个负效用作业两轮FIFO服务；真实锁、排空、求解失败原子回滚；三作业候选独立穷举；旧G前8轮完整分配和执行轨迹一致。

预注册10个小规模验证格，但第一格启动子进程导入依赖后可用内存低于2GiB，被安全门槛拒绝，尚未构造模拟器。事后只读采样可用内存**1.800GiB**、磁盘余量22.530GiB。**0/10启动训练、无新JCT，不宣称新S优于AUX或Sia。** 未降低门槛、未自动反复重试、未恢复104格正式实验。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。

## 1. 为什么先改决策，而不找更有利数据

前轮20/M：SIA-P均值3477.15s，SIA-AUX3112.70s，旧G3157.15s。AUX已经能利用当前资源响应，G虽比Sia快9.20%，仍比AUX慢1.43%。相对AUX，G少3120s等待、少780s checkpoint，却多4789s有效训练耗时。这支持检查资源配置/任务选择，但不独立证明所有损失均由准入导致。

本轮只读预定义4个结构状态，未按收益替换样本。同一旧日志状态、同一在线画像、同一候选和真实锁、同一原公平目标：

|预定义状态|时间s|旧G选中数/效用|非队首未承诺可选：选中数/效用|
|---|---:|---:|---:|
|首轮调度|60|20 / 3.25|14 / 3.82|
|首次服务到期|420|20 / 9.33|19 / 9.50|
|BERT54首次RTX1已有效训练且解锁|240|20 / 8.64|20 / 8.64|
|首次drain|—|NO_MATCH|不替换样本|

首轮旧G的“最大可行并发”会压缩一些作业的GPU配置；原效用其实更偏好少量、更有效的配置。240秒所有旧服务仍未兑现，不能强行撤销来制造收益，因此同状态结果不变。这是预测决策证据，**不是JCT反事实**；数据在same_state_admission.json。

## 2. 准确算法与活性

完整定义和状态机论证见DESIGN.md。令H为未完成300秒物理有效训练服务的作业，h为FIFO队首，O为其他ready作业。在H与h兼容时：

`max Σ[j,c] u(j,c)x(j,c)`

`Σc x(j,c)=1, j∈H∪{h}`；`0≤Σc x(j,c)≤1, j∈O`；其余为0。

GPU/CPU/RAM节点容量、合法GPU配置、CPU/RAM网格、比例进度下限、完整checkpoint锁全部保留。列效用、原restart factor以及精确同分后的真实启动成本次目标不变；没有加入并发奖励、改变p或lambda_n。

若h单独不可行，明确报错；若h与H不兼容，排空期间只允许H，后到小任务不能绕过。H的承诺不中途撤销，不重置服务时钟/启动预算。仅实际被选中的optional建立新服务。负效用队首仍必选，打破全空；负效用非队首可以等待，并在其成为队首时得到服务。

活性需要单独可行、当前锁合法、真实执行率正、有限队列/前序任务与可决求解器；累计真实启动D≤250、Q=300有效训练、60秒离散轮次意味着每个承诺最迟610秒到期。队首受阻时不新增承诺，H有界退出；未完成者排队尾，不插到队首前。按有限前序位置归纳可得首次与重复服务；正进展下界和有限工作量才可进一步推出最终完成。不对无限过载/零速率/无合法候选作完成保证。

仍有结构局限：若队首不断能与H共存，可能逐轮增加强制服务，重新压到高并发。首轮14个不是“永久并发上限”。本轮预先记录此边界，**不在看到JCT后新增预留、训练租约或第二套规则**。

## 3. 改动范围

隔离树：D:/科研启航/Sia/sia-joint-selective-admission；分支exp/joint-selective-admission；基线HEAD为2d5f9dbfcb7e7bfc2bfd9dc194b1838674a227c2。前轮未提交的实际执行源码完整恢复并逐一验证105个Python文件hash，不把裸HEAD冒充新版本。未提交或推送本轮代码；执行快照与hash为权威版本。

- 新增joint_prototype/selective_admission.py：SelectiveAdmission._admission_sets（第9行）、_decide（第21行）、SelectiveJoint与install（第34行）。默认策略不走新类。
- adaptive_service.py：新增默认保持旧贪心的_admission_sets hook（第59行）；solve增加optional集合的0/1上界；校验必选子集/允许集合；optional同分可确定性不选；只有实际选中者建立承诺。旧调用optional为空，保持原等式语义。
- 新增run_selective_cell.py及3个测试文件。原bundle_adaptation.py、execution_audit.py及所有原GPU/Job执行/响应代码相对前轮内容完全相同。execution_audit相对裸Git基线已有的改动是继承前轮修复，不是本轮再次修改。
- 新增独立诊断、预注册、运行与原始审计脚本，全部在新产物目录。

完整本轮相对父版本diff在selective_diff.patch；Git短diff不会列出全部untracked文件，不能拿它代替快照。

## 4. 红绿与回归证据

|检查|结果|
|---|---|
|旧准入运行新断言（修正fixture后）|10 tests，4 assertion failures，0 errors|
|初始红测|4 failures＋1构造期fixture error，原日志保留|
|新完整测试|154 tests，全部通过，68.24s进程墙钟|
|真实未画像BERT|实际10轮调度、checkpoint及正训练通过，未伪造Goodput|
|旧G有界回归|8轮选中配置/CPU/RAM/rho/progress/restart相关执行字段逐项相同|
|原Sia相关既有测试|实际调用SiaPolicy.optimize，无L2；原始Job.step取整边界等既有测试通过|
|新增活性测试|20个负效用作业各两次FIFO服务；gang排空保护；负效用锁必选|
|最优性小穷举|3作业全部有限候选枚举与新MIP主效用一致|

初始no-candidate测试用超过GPU容量的配置，在构造期被合法拒绝，不能宣称为准入缺陷。修正为GPU合法但CPU最低要求不可行后重跑旧类，4个真实失败仍存在；所有日志保留。第一次prepare另遇Windows路径分隔符表示不一致，在启动前停止，使用Path.as_posix比较修正，未改科学参数。

## 5. 冻结验证计划与实际状态

|数据|响应|方法|计划格数|本轮训练执行|
|---|---|---|---:|---|
|既有8作业|Neutral/Medium|JOINT-S|2|0（首格在2GiB门槛前停止）|
|既有20作业|Neutral/Medium|JOINT-S|2|0|
|独立workload5轮转20作业|Neutral/Medium|SIA-P、SIA-AUX、JOINT-S|6|0|

独立数据在任何新策略结果出现前冻结。按与原负载相同的BERT/CIFAR/DeepSpeech2轮转、类内原行序、不重复、类耗尽跳过；到达时间控制为0，保留original_arrival/source_row_index，原batch/GPU demand/总工作量不变。这是相对当前workload1诊断的独立验证，不声称该trace从未用于任何历史研究。

holdout_workload_20.csv SHA256：C8C2E2180E026880DFBB84B748831420B5C6F1A5C5EB0F51EC60F4417A8E0AE0。其余全部hash见preregistration.json。资源仍AWS4/12CPU/250GB、DGX8/24/500、RTX8/24/500，R0/E-legacy、alpha0/.5、seed_profiles(1,8)、Q300/D250、600轮×60秒均不变。CPU单位与节点规格仍synthetic。

实际退出日志：original8_neutral_joint_s.log，退出码1，3.84秒，MEMORY_GATE_2GiB；执行发生在make_cluster之前，没有策略JSON、训练进度或部分分配。不是600轮删失，不计算任何新JCT或完成率。

## 6. 下一步与复现

当前只能判断：准入层确实排除了某些原目标更优的同状态决策，新模式在正确性测试中能解除这层约束并保留FIFO/锁/预算。**尚不能判断它是否改善平均JCT、是否胜过AUX或是否减少BERT损失。** 冻结的10格完成且原始审计通过后才能回答，不需要现在换数据或再设计一套权重。

资源满足后在新的attempt目录复制本次冻结preregistration.json、run.py和audit.py，再按同一manifest运行；不要在根目录重用同名log，不覆盖安全停止证据。恢复时首先重新验证源码/输入hash，确认没有重复实验进程。原8两格审计通过才20，不通过则停止；不按结果修改参数。这里没有安排自动续跑。

复建：从基线HEAD创建新目录，恢复execution_source.zip；用frozen_prerequisites.zip恢复根目录相对布局与完整输入，依赖按requirements-lock.txt；运行测试，再以全新目录执行冻结run.py。父源码parent_complete_source.zip、parent_tracked.patch、restored_source_hashes.json保留继承前版本。所有产物hash由delivery_manifest.json封存。

本轮保护官方Sia和Synergy；Synergy原有未跟踪论文PDF保留；旧run与暂停记录hash不变。104仍暂停。没有运行160-job、九cohort或正式参数搜索。
