# B3/B3-PROP Stage1.1 工程收尾

## 1. 验收结论与边界

本轮两个指定缺陷均已修复。原有 61 项测试原文未修改，新增 10 项测试；最终 **71/71 通过**。官方 Sia、medium 冻结树、Stage1、Stage1.1 的一作业三轮有界回归状态完全一致，并与 Stage1 已保存回归一致。

**B3-PROP 全空 OPTIMAL 吸收态仍未解决。** 既有测试仍明确复现三轮空分配、零进度；本轮没有增加服务轮转、训练租约、资源预留、allocation bonus 或其他活性规则。

没有运行 w10_late 20-job、6000 轮、九 cohort 或正式实验。p、lambda_n、目标、phi 公式、clipping、响应模板、容量、workload、观察窗和求解参数未修改。工程修复不意味着既有性能结论得到重新验证。

## 2. 隔离与版本

- 起点：`0461dcdf061197fdfffe23b2b3d07e46e0d9efb2`。
- 新工作树：`D:\科研启航\Sia\sia-joint-correctness-stage1p1`。
- 新分支：`exp/b3-correctness-stage1p1`。
- 本地修复 commit：`216573cb26f3136520ce9d4f609623b4c2c75df1`，未推送。
- 输出：`D:\科研启航\Sia\research_runs\sia_synergy\correctness_stage1p1`。

起点实际源码 786 文件已做完整字节快照和 Git bundle；tracked/index patch 均为空，必要 untracked 源码为空，生成缓存未纳入。恢复到新工作树后逐文件 SHA-256 一致。修复后快照包含 787 文件。3393 个受保护文件已重新核验：Stage1、medium 冻结树、official-artifacts、Synergy 官方源码、旧报告、Stage1 产物、2×2 run1/run2 均未改变。

Git 的换行符处理曾使 22 个字节恢复后的文件显示 stat-only 修改；逐文件确认其字节与基线相同、canonical diff 为空后，仅刷新明确列出的索引。最终 commit **只有下列 4 个文件**，工作树干净；未使用 reset、clean、add . 或 add -A。

## 3. 根因一：共享 replica 边界被联合枚举忽略

源证据：

- [SiaPolicy.__init__，sia.py:79](D:/科研启航/Sia/sia-joint-correctness-stage1p1/sia-simulator/sia.py:79)：启用 `project_throughputs` 会自动将 `share_max_replicas` 设为 True，即使调用方传入 False。
- [SiaPolicy.optimize_mip，sia.py:621](D:/科研启航/Sia/sia-joint-correctness-stage1p1/sia-simulator/sia.py:621)：共享模式使用所有类型的最大 max_replicas 和最小 min_replicas。
- Stage1 的 `JointSiaPolicy._raw_candidates` 原第 122–123 行始终使用 cluster-local bounds，未读取这个实际生效的共享标志。
- [修复后的 _raw_candidates，joint_integration.py:121](D:/科研启航/Sia/sia-joint-correctness-stage1p1/sia-simulator/joint_prototype/joint_integration.py:121)：adaptive 候选复用当前 Sia 的共享边界规则；rigid 及 SY 固定 demand 的既有逻辑不改。

### 3.1 完整最小复现

测试输入仍为 `bert-mid / arrival=17 / application=bert / replicas=1 / batch=384`，原测试节点与配置不变。执行 `step(60), step(60), step(60), step(1)`；每次都调用真实 `Cluster.optimize_policy → JointSiaPolicy.optimize → SiaPolicy.optimize`。观测子类只记录输入、输出并调用原实现，没有 mock 下一次调度、seed profile、伪造 Goodput 或关闭 audit。

| 调度时刻 | 实际进度 | 剩余重配置时间 | 分配 | 事件 |
|---:|---:|---:|---|---|
| 60 | 0 | 120 | RTX 1 GPU | 首次分配；部分周期等待 43 秒 |
| 120 | 0 | 60 | RTX 1 GPU | checkpoint 锁保留 |
| 180 | 0 | 0 | RTX 1 GPU | 原延迟结束 |
| 181 | 1.8646307921300191 | 120 | DGX-ext 2 GPU | 180 秒首次有效训练；181 秒真实重调度成功 |

181 秒同一 JobInfo 的 local max 为 `{aws:1, dgx-ext:1, rtx:2}`，只有 RTX 有 direct profile。Sia 的共享上限是 2，而旧联合表在 AWS/DGX-ext 上只枚举 1 GPU。

红测中真实 Sia 对 DGX-ext 查询 1、2 GPU，返回 raw Goodput `141.28050231933594`、`274.9748840332031`；其 intent 为 `dgx-ext:1:2`，旧联合表没有此列。修复后该配置明确标为 `projected_profile`，使用同一个 Sia `_compute_goodputs_v2` 获取值。

因此分类是：**硬件/训练配置合法、目的 GPU 尚无 direct profile、但已有来源 profile 可投影；两端合法边界规则不一致**。不是配置真实非法，也不是两次调用间 JobInfo/profile 被更新。红测保存在 `red_evidence/bert_mid_full_chain.json`，绿测在 `final_tests_evidence/bert_mid_full_chain.json`；`root_cause.json` 提供机器可读对照。

原有独立计时单元测试仍保留，不把其 mock 边界当作完整链路证据；新完整链路测试见 [test_stage1p1_correctness.py:55](D:/科研启航/Sia/sia-joint-correctness-stage1p1/sia-simulator/tests/test_stage1p1_correctness.py:55)。

## 4. 根因二：列标签被误当作候选资格

Stage1 `refresh_column` 原第 243 行附近只更新 query/config/round，保留旧 `candidate_kind`。原 audit 的普通候选检查只对 `proportional`、`joint_aux` 标签生效。解锁后的 `locked`，以及未知标签，均可跳过网格/progress-floor 验证。

修复点：

1. [JointScheduler.__init__:176](D:/科研启航/Sia/sia-joint-correctness-stage1p1/sia-simulator/joint_prototype/joint_scheduler.py:176) 新增当前调用方的 `candidate_policy`；集成入口显式传 `self.mode`，原型入口和 TUNE adapter 显式选择其现有契约。该信息不由列提供。
2. [refresh_column:247](D:/科研启航/Sia/sia-joint-correctness-stage1p1/sia-simulator/joint_prototype/joint_scheduler.py:247) 只复用旧 geometry/quotas，使用当前 JobSpec/config/oracle 重建。仍锁定时构造 lock 资格；解锁后按当前策略重建普通资格。未知旧标签拒绝；重新 audit 不通过则拒绝整个 fallback。
3. [audit:655](D:/科研启航/Sia/sia-joint-correctness-stage1p1/sia-simulator/joint_prototype/joint_scheduler.py:655) 校验当前 JobSpec.configs、placement、query round、完整 oracle query、quota/usage、容量与锁。未知、伪造锁、跨契约标签拒绝；正常配额资格由**当前策略**决定。
4. [fallback:426](D:/科研启航/Sia/sia-joint-correctness-stage1p1/sia-simulator/joint_prototype/joint_integration.py:426) 留存具体 `requalification_errors`，不再只有“上一分配不可验证”的笼统说明。

| 当前状态/策略 | 允许的正常资格 |
|---|---|
| 真实 checkpoint lock | 必须 exact config、节点及 GPU 分布、CPU/RAM 配额；必须被选中；标签必须 locked |
| B1 / B3-PROP 解锁 | 比例配额与既有 minimum floor 的唯一确定点，不允许 joint_aux/TUNE 标签 |
| B3 / B4 解锁 | 当前辅助网格（包含比例点），且满足既有 proportional progress floor；比例标签只能用于比例点 |
| SY / B5 解锁 | 当前 TUNE demand 或 fair-share 点；比例 intent 仍只能是比例点 |

真实锁可以暂时保留网格外旧配额，这是完整 checkpoint 保护，不是标签豁免：一旦解锁，该配额必须重新取得普通候选资格。保留的 `locked` 历史标签不被直接提交；合法旧几何会被重建为普通列。对 raw forged locked 列则直接拒绝。

加强单列校验后补查到 B4 的兼容边界：当比例 intent 自身超 CPU，不能让 refresh 抛错覆盖已有合法求解结果。现 [run_b4:764](D:/科研启航/Sia/sia-joint-correctness-stage1p1/sia-simulator/joint_prototype/joint_scheduler.py:764) 捕获 requalification 失败、拒绝整份 fallback、保留求解器已审计的完整结果，绝不提交成功重建的前缀。`boundary1.log` 保留修复中途这一失败证据。

## 5. 最小 diff

| 文件（均位于新树 sia-simulator） | 改动 |
|---|---|
| joint_prototype/joint_integration.py | 对齐共享 replica 边界；传当前契约；记录 fallback 拒绝细节 |
| joint_prototype/joint_scheduler.py | 当前契约校验、解锁重资格、B4 整体 fallback 拒绝 |
| joint_prototype/synergy_adapter.py | 3 行：在可信入口声明现有 TUNE 候选契约，不改其分配流程 |
| tests/test_stage1p1_correctness.py | 10 项短小测试，含真实训练/调度与故障注入 |

总计 281 插入、10 删除，其中新增测试 211 行；生产代码 70 插入、10 删除。`repair.patch`、`repair_diff_stat.txt` 保存完整差异。`_b3_result`、`_restart_factor`、MILP `solve` 的 AST 未变；sia.py、job.py、cluster.py、goodput.py、simulator_config.py、joint_profile.py、execution_audit.py 的字节哈希未变。

## 6. 红绿与回归证据

| 证据 | 结果 | 含义 |
|---|---|---|
| baseline_existing61.log | 61/61 通过，修改前 | 既有测试确实未覆盖两项缺陷 |
| red.log + red_test_source.py | 7 个测试方法；12 个失败断言/子测试、1 个 error | 未改生产代码前复现缺陷；真实 lock 正例通过 |
| green1.log | 7/7 通过 | 两项主要修复红转绿 |
| green_full1.log | 68/68 通过 | 61 旧测试 + 首批 7 项 |
| boundary1.log | 10 项中 1 error | 新增 B4 超配 fallback 边界捕获中途回归；证据未覆盖 |
| green_full2.log | 71/71 通过 | 补齐 B4 整体拒绝处理 |
| final_tests.log | **71/71，17.661 秒测试时间，22.967 秒墙钟，退出 0** | 对最终本地 commit 验收 |
| legacy_regression/comparison.json | 4 个版本、各 1 作业 × 3 轮完全一致 | 原始 Sia 有界状态回归，不冒充完整 JCT/CLI 实验 |

额外保存的 `baseline61.log` 实际是首次筛选表达式运行的 **44 项子集**，文件名不代表 61 项；随后专用 `existing_suite.py` 明确断言并运行完整 61 项。该日志未删除、未用作 61 项验收证据。

新增 10 项覆盖：完整 unseeded bert 链路；合法 lock→unlock→MILP 无解→fallback；网格外拒绝；progress-floor 拒绝；未知/伪造标签拒绝；B3-PROP 非比例/跨策略拒绝；仍锁定的完整保护；B3 与 B3-PROP 真实 Cluster 解锁故障注入；合法 proportional 解锁及未知 refresh 拒绝；B4 不可行比例 intent 的整体拒绝。

真实 Cluster 解锁测试只注入 MILP `x=None`，实际执行 60/120/180 秒时间推进、锁判断、fallback、独立原子审计和提交，不 mock optimize_policy。既有独立倍率、checkpoint、epoch/完成边界、资源计时测试仍全部通过。PMP 路径仍有既有 autograd log RuntimeWarning，不伪装为本轮新增失败。

## 7. 对历史指标的影响

- 补全共享边界可能改变 GPU 候选数、GPU 决策、重配置、资源小时及 JCT；历史漏枚举结果不能当作修复版结果复用。
- **phi 公式未改，但数值可能间接变化**：当前模型的 per-GPU reference minimum 依赖候选集合。本 fixture 中由 `13.099814414978027` 变为 `12.748114585876465`（新增 AWS 2-GPU 投影点），所以部分既有配置的 demand scale 也会依既有公式重算。没有固定 phi、外推、补 1、改变 clipping；这是仍待响应模型审定的重要路径依赖，不应宣称本轮消除了它。
- 原来靠残留 locked 标签通过的非法 fallback 现在会显式拒绝，必要时安全停止，而非继续生成误导性结果。合法 fallback 保留几何和配额，使用当前 query/rho/版本。
- 本轮没有直接改动 waiting/start/training 计时；调度路径改变仍可能使其历史汇总值变化。不能推断性能更好。

## 8. 可重建交付与哈希

完整包位于 [correctness_stage1p1](D:/科研启航/Sia/research_runs/sia_synergy/correctness_stage1p1)。恢复说明见 [REBUILD.md](D:/科研启航/Sia/research_runs/sia_synergy/correctness_stage1p1/REBUILD.md)。完整每文件索引见 `snapshot_manifest.json`、`repair_manifest.json`、`delivery_manifest.json`，验收见 `validation_summary.json`。

| 产物 | SHA-256 |
|---|---|
| repair.patch | CDC1B16943BD4CE7FE1FB2D4BB43EA3281D0B55F46B95A1BC6F60329D8270BD7 |
| repair_source_snapshot.zip | 0372456516A336208ACE485B2920F04C8A2EAA7E0DBF3D7C9BC39D62A989195F |
| repair.bundle | 7D1BBAC944A68946660186310B2A2EB9C140C108274AE4ABEE6DD225DF37D253 |
| final_tests.log | 4B7B0DF0BF0B3E6173D51A3D671BDEA650B1674FFE6C1D98B453CCE22BB7580F |

依赖：Python 3.10.10、NumPy 1.23.5、SciPy 1.10.1、CVXPY 1.3.2、CyLP 0.91.6、Pandas 1.5.3、autograd 1.9.1。完整 pip 锁保留。CBC/HiGHS 引擎独立版本未另查，不猜测。快照/bundle 都已独立校验，不仅保存哈希；未打包离线 wheel 或操作系统，跨平台数值重现不作保证。

## 9. 停止点

Stage1.1 工程验收通过；**不代表正式实验或活性修复获批**。保留未解决的 B3-PROP 全空吸收态和响应边界/在线 phi 路径依赖。下一阶段仅需研究者审定活性机制、GPU 对照定义和响应边界模型，本轮不实施或自动启动后续实验。
