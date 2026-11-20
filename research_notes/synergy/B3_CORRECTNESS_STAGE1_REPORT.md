# B3/B3-PROP 第一阶段：工程正确性修复与数据预检

## 结论与边界

本轮完成的是工程修复，不是活性算法修复，也不是新的性能实验。最终测试 **61/61 通过**；原始 Sia 单作业三轮确定性状态与冻结源码完全一致。没有运行 w10_late 20-job、6000 轮、九 cohort 或任何正式矩阵。

**B3-PROP 全空吸收态仍存在。** 新增短测保留了连续三轮 `OPTIMAL`、空分配、零进度的证据。不能把测试通过解释成 B3-PROP 已可用于完整 2×2 性能实验。

响应预检见 [B3_RESPONSE_DATA_PREFLIGHT.md](B3_RESPONSE_DATA_PREFLIGHT.md)。本轮没有冻结新的 phi、修改 clipping、调整容量、工作量、模板或响应指数。

## 1. 保护、隔离与可重建版本

| 项目 | 记录 |
|---|---|
| 工作区 | `D:\科研启航\Sia` |
| 受保护源树 | `sia-joint-medium-frozen`，`exp/b3-medium-nonneutral` |
| 原 HEAD | `be03f95c7bc641cbef0e2a4452bc4c7a3b5404e4` |
| 新工作树 | `sia-joint-correctness-stage1` |
| 新分支 | `exp/b3-correctness-stage1` |
| 实际源码封存基线 commit | `1cbc3f5abb4de18ccca3a75749e1fc704fd46027` |
| 修复 commit | `0461dcdf061197fdfffe23b2b3d07e46e0d9efb2` |
| 本轮产物根目录 | `research_runs/sia_synergy/correctness_stage1/` |

原树的 3 个 tracked 修改及 6 个必要 untracked 源码/配置文件全部纳入基线。归档包含 782 个文件，恢复后逐文件 SHA-256 一致；run2 manifest 中 job、集成器、调度器、Oracle、公共运行器的执行源码 hash 均匹配。没有仅从 HEAD 丢掉历史修改后开始修复。

封存内容：

- `source_snapshot.zip`：实际工作区字节快照；`tracked.patch`、`index.patch`；`base.bundle`。
- `snapshot_manifest.json`：全文件 hash、原状态、必要 untracked 清单和执行 hash 核验。
- `repair.bundle`、`repair_source_snapshot.zip`、`repair.patch`、`repair_diff_stat.txt`：修复后的提交、完整源码和差异。
- `requirements.lock.txt`、`dependencies.json`：Python 与安装依赖版本；未臆测独立 CBC/HiGHS engine 版本。
- `repair_manifest.json`：修复 commit、源码 hash、保护核验、未改动目标/响应函数的 AST 核验。
- `REBUILD.md`：在全新位置恢复的方法；`external_response_templates.zip` 提供必要的官方响应矩阵快照。

结束核验：1801 个旧报告/旧 run1/run2/官方文件、782 个原树文件的字节均未变化。官方 Sia 保持干净；Synergy 原有 `osdi22-mohan.pdf` untracked 状态保留。新树 tracked 内容已提交，仅保留测试生成的 untracked `__pycache__`，没有执行清理或推送。恢复后的部分 LF 文件曾显示 Git stat/EOL 待刷新；逐一 `git add -- <明确文件>` 刷新后 staged diff 为空，没有改写源码。

## 2. 已实施的修复

以下路径均相对新工作树的 `sia-simulator/`。

| 修复 | 实现与证据位置 | 可能改变的历史指标 |
|---|---|---|
| 真实到达与稳定次序 | `joint_prototype/joint_integration.py:210` `_job_specs()` 使用 `creation_timestamp`；`JointCluster.get_job_infos():747` 传递 `source_row_index`；`synergy_adapter.py` `run()` 按 `(arrival, source_row, job_id)` 排序 | SY/B5 的分配顺序、等待、JCT、重启与资源小时可能改变 |
| 部分周期等待、首次分配/生产训练 | `JointCluster.step():764`、`optimize_policy():754`；`job.py:419` `step()` | waiting 不再漏掉周期内 arrival 到下次决策的剩余秒；start_time 从空值变成明确的首次分配别名 |
| 互斥时间与周期内完成 | `JointCluster.step()` 记录 observed/wait/checkpoint/training/allocated/post-completion；`run_cpu_bottleneck_smoke.py::summarize()` 使用新字段 | 修正等待口径；明确完成后不再计入作业资源小时，且不提前为别的作业调度 |
| checkpoint 强制完整锁 | `joint_scheduler.py:257` `locked_column()`；`:272` `locked_columns()`；`:558` `solve()` 的锁定行下界=上界=1；`:633` `audit()` | 防止锁定作业丢失、换节点/数量/配额；会改变旧分配轨迹、JCT 和 restart，不是历史结果等价修补 |
| 覆盖所有相关路径 | `_b1_result():279` 保留完整锁，二次 admission 必须选中；B4 精确锁候选；B3/B3-PROP 相同硬锁；B5 `_repack_all_fair()` 先放锁定 peer、不降级、不搬移 | 原先错误中断或降级的轨迹不应继续复现 |
| fallback 重建与提交前复核 | `_b3_failure_fallback():416` → `refresh_column():243`；B4 fallback 同样刷新；`optimize():599` 独立 precommit audit | 当前 Goodput、phi/rho、目标日志与版本更新；旧 rho 不再沿用，可能改变后续进度 |
| 直接 raw rate 与非循环倍率审计 | `job.py:419` 在乘 rho 前记录；`joint_prototype/execution_audit.py:5` 独立检查 | 训练 rate 不再由 `effective/rho` 倒推；原始 epoch 取整残差被显式分离，历史 raw/effective 数值不可直接混表 |

`source_row_index` 是输入 DataFrame 的稳定零基行序；若输入显式提供同名源行字段则保留该字段。它不是从 job 名称解析出来的编号。

### checkpoint 与原子提交的确切语义

锁定作业只构造一个重新查询过当前性能的精确既有分配列，并施加：

`sum(x[j, exact_existing_allocation]) = 1`。

锁包括 GPU type/count、每节点 GPU 数量、CPU/GPU、RAM/GPU；不能只比较节点集合。锁定作业即使效用为负也必须选中。锁定分配不参与自由配额搜索或 peer downgrade。普通无锁作业的空分配规则和目标未变。

缺少完整旧配额证据、旧 geometry 不合法或全部锁的资源总量不可行，均显式 `INFEASIBLE_CHECKPOINT_LOCK` 停止，不放宽容量，不提交空锁。提交前重新查询 Oracle、检查候选类型/网格/当前 progress floor、查询轮次、配置、placement、配额、完整锁及三种资源总量。不能仅凭返回对象带有 `audit.valid=true` 放行。

fallback 只复用 allocation geometry 与 quotas；当前 config 对象、Goodput、phi/rho、Oracle 查询、`query_round` 全部刷新。B3 fallback 的效用用原公式重新计算。若旧配额在当前状态已不满足候选规则，停止而不是强行保留。锁的硬保护优先于自由配额候选过滤，但仍必须满足物理容量和合法配置。

### 时间口径

对调度区间 `[t,t+Δ]`，作业到达 `a`、完成 `c`：

`observed = max(0, min(t+Δ,c) - max(t,a))`，未完成时用 `t+Δ` 代替 `c`。

`observed = waiting + startup/checkpoint + productive_training`，三者互斥。

- 未分配且已到达的重叠秒数全部计入 waiting；例：17 秒到达、60 秒首次分配，等待为 **43 秒**。
- `first_allocation_time` 是首次实际提交非空分配的决策时间；`start_time` 明确作为它的兼容别名。
- `first_productive_training_time` 是第一次出现正训练进度的子段起点；不是首次分配时间，也不是首次有在线 profile 的时间。
- allocated time 只包含 startup/checkpoint 与训练，截断到周期内完成时刻。完成后的剩余区间记录为 post-completion，不计入该作业 GPU/CPU/RAM hours；没有增加周期内重新调度。
- `Job.used_gpu_seconds` **仍保留原 Sia 决策用计数语义**：它可能预加完成所在周期的剩余秒，不能拿来替代新资源积分。没有偷偷改动基于该字段的 restart factor。新 `resource_hours` 与 `exclusive_time_by_job` 才是本轮的物理时间统计口径；该兼容字段在 `summary.jobs` 中尚保留，分析者必须区分。

### Goodput 审计的边界

直接记录 `raw_rate_before_rho`，再记录 `effective_rate`，检查后者等于前者乘当轮 rho。非 epoch 边界还独立检查 `progress_delta = raw × rho × dt`。

原 Sia 的 epoch 边界耗时采用 `round(remaining_progress/goodput)`，因此跨边界时原始 progress 与连续时间积分不一定严格相等。本轮**未改变该语义**：记录 rounding residual，并检查单边界残差绝对值不超过 `0.5 × effective_rate`。完成后截断训练，不把 `rho=k` 的线性关系错误延伸到已经完成的作业。

候选 Goodput 是 Sia-normalized 调度量；执行 raw 是真实 `Job.step()` 的 progress/second。两者单位不同，没有强行要求数值相同。

## 3. 红绿测试与小型回归

全部命令、退出码、墙钟时间位于产物根的同名 `.json/.log`。主命令为：

```powershell
.\.venv-sia-baseline\Scripts\python.exe research_runs\sia_synergy\correctness_stage1\run_checks.py <新标签>
```

它在新 worktree 下运行 `python -m unittest discover -s tests -p test_*.py -v`，拒绝覆盖同名日志。

| 阶段 | 实际结果 | 含义 |
|---|---|---|
| `baseline_tests` | 原 47/47 通过，退出 0 | 旧实现原本就通过，不能当成本轮修复的功劳 |
| `stage1_red` | 5 个新增目标检查全部红：1 failure、4 errors，退出 1 | 旧 fallback 仍返回旧 Goodput；arrival/order/full-lock/raw telemetry API 不存在。日志另外发现了 21 个导入 fixture 的旧测试，不计为新增覆盖 |
| `stage1_green1` | 中间集成失败，日志保留 | 旧锁 fixture 缺完整配额、新 round 字段使旧整字典相等断言失效；另暴露未 seeded 的 B1 candidate/interface 不一致，见风险 |
| `stage1_green2` | 59/59 通过 | 明确升级旧 fixture 契约，不放松完整锁要求 |
| `stage1_green3` | 60/60 通过 | 加入独立分段执行审计和 epoch 取整边界 |
| `stage1_final_tests` | **61/61 通过**，退出 0，墙钟 17.924 s | 47 项既有测试＋14 项新增检查；最终源码测试依据 |
| `legacy_regression` | 1 作业、3 轮，全部状态逐项相等，退出 0 | 原始 Sia 非 joint 路径的有界确定性回归，不是新的完整 CLI 性能实验 |

旧测试 fixture 的修改：锁测试现在提供 GPU 节点计数和 CPU/RAM 旧配额；fallback 测试要求 round 更新，但其余资源内容一致。没有删除旧测试。PMP 的既有 autograd log RuntimeWarning 仍出现，未被隐藏。

新增文件 `tests/test_stage1_correctness.py` 覆盖：

1. 负效用 locked job 强制选中且保留全部配额。
2. FIFO 使用真实 arrival，再以源行顺序打破平局。
3. 中途到达等待 43 秒；首次分配 60、首次生产训练 180 秒分开记录。
4. fallback 将旧 raw Goodput=1 更新为当前 7，并拒绝旧 query。
5. 直接记录 raw rate，而非 effective/rho。
6. B1/B4/B5/B3/B3-PROP 实际集成路径的锁与资源竞争；B1 触发二次 CPU admission。
7. B5 经 Cluster 的实际 FIFO 路径；不依赖 job ID 字典序。
8. peer repack 不能降级或移动 locked peer。
9. 容量故障注入使 lock 不可行时显式停止，资源状态不提交。
10. 三轮 cold-start OPTIMAL 空分配仍存在，明确标为 unresolved。
11. 独立 rho=1/2 配对测试覆盖 checkpoint、跨 epoch、完成截断。
12. 人工固定 raw=1、rho=2、delay=3、总工作量20：13秒完成；只计 13 GPU-s、26 CPU-core-s、650 RAM-GB-s；伪造双倍 effective rate 被拒绝。
13. rho=3 的 epoch 取整残差显式为1，不藏进 raw。
14. 伪造 `OPTIMAL/valid=true` 丢弃 locked job，被独立 precommit 审计拒绝。

逐轮短 fixture JSON 保存在 `stage1_final_tests_evidence/`，包括五种策略的 full-lock 路径、FIFO、partial-arrival、fresh-fallback、completion accounting，以及 `UNRESOLVED_empty_optimal.json`。这些是最多数轮的故障注入/人工单元测试，不是 w10_late 结果。

## 4. 明确保留的语义与风险

- `_raw_candidates()`、`_b3_result()`、`_restart_factor()`、Oracle `query()`、`_bilinear()` 的 AST 与封存基线相同；p、lambda_n、原目标、现有 bonus、phi、clipping、响应公式与求解器参数均未调整。
- 没有增加服务轮转、训练租约、资源预留、相对 proportional admission、bootstrap 或新的 GPU 基线。
- **活性未修复**，四个因子单元仍不是统一 admission/目标的干净 2×2。
- 在线 phi 的路径依赖与响应 clipping 仍存在；本轮只报告覆盖证据。
- 新锁保护会使旧轨迹变化，或把原先静默错误变成显式失败；不能要求修复后复现旧 JCT。
- 小型计时测试曾在“未 seed 的 BERT、先 checkpoint 后仅1秒训练再重优化”触发既有接口错误：原 Sia 选择 `dgx-ext:1:2`，而当轮集成候选缺该配置。证据在 `stage1_green1.log`；本轮未修改合法候选算法。最终该计时测试只隔离生产训练边界，不宣称解决这个额外问题。正式实验前必须重新审查这个未画像候选接口边界。
- 旧 JSON/验证器历史结论没有重写。新分段审计不能反向补足旧日志中不存在的 raw 执行证据；旧数据中 Goodput“只乘一次”的证据强度仍有限。
- 未执行完整正式实验，因此不能声称修复版本已经具备 20-job 或九 cohort 稳定性。

## 5. 下一阶段最少需要决定的三件事

1. **活性机制**：批准什么对 B3/B3-PROP 对称的 admission/liveness 规则；本轮没有选择或实施。
2. **GPU 对照定义**：四单元如何统一目标/admission，并严格限定 GPU 候选因子；B5 是否继续单独作为复合策略，而不混入 2×2。
3. **响应边界模型**：如何用策略无关参考状态生成完整 phi 表，以及越界时采用哪种预注册数据/边界规则。现有观测子集不能替代完整覆盖；禁止默认补 phi=1。

上述决定前不启动新的正式对照。本轮证据属于冻结 trace 和 synthetic cross-hardware transfer 的工程测试，**不是真实 Saturn 集群性能测量**。
