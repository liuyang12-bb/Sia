# Sia + Synergy 集成覆盖扩展报告

> 结论：**INTEGRATION_COVERAGE_PASS（27/27 tests，18/18 persisted audits）**  
> 性能证据范围：**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**

## 1. 本轮范围和保护状态

本轮在新分支 `exp/sia-synergy-response-audit` 上完成。基准 HEAD 为 `aaf7a0de7b6cb5eab06ea020467ede5d2fd18e70`。没有运行 20-job/160-job、参数搜索或真实 GPU profiling。

`official-artifacts/` 最终为 `main...origin/main` 且工作区干净。任务 13 的三个关键历史产物 hash 未变化：

| 文件 | SHA-256 |
|---|---|
| joint_integration_smoke/run_manifest.json | `DB68695977A16813E33D967E0E1272998519CC9EDCE302458033441E1F6071F3` |
| joint_integration_smoke/validation_summary.json | `025D72CD163E9739DB68F4628BAD32151C236A61DCBC3BEF1BF496E1C4E37D3E` |
| joint_integration_smoke/comparison_summary.json | `66B3A615F7BFF1570F269BB53E4FF4A345DA73E6C18D56B62B6BC35ABC6C63DC` |

## 2. 修改文件和行为

| 文件/位置 | 修改 |
|---|---|
| `joint_prototype/joint_profile.py:71-102` | `ProfileQuery` 增加 raw rho、响应指数、请求/clip 后坐标及 clip 标记。 |
| `joint_prototype/joint_profile.py:211-271` | 增加预注册 `response_exponent`，以 `raw_rho ** alpha` 进行对数效应缩放。默认 alpha=1，neutral=0。 |
| `joint_prototype/joint_scheduler.py:68-106` | 将分子、分母、坐标和缩放参数写入每轮结果。 |
| `joint_prototype/joint_integration.py:70-90` | 支持测试所需的逐节点 synthetic capacity override，默认同构容量行为不变。 |
| `joint_prototype/joint_integration.py:99-127` | rigid 候选强制保持固定 GPU demand；未画像 rigid fallback 使用其完整 demand，不再错误退化为 1 GPU。 |
| `joint_prototype/joint_integration.py:327-359` | 每轮资源日志增加完整响应审计字段。 |
| `joint_prototype/joint_integration.py:417-444` | 仅在联合包装层为官方 `Cluster` 跳过的 PMP throughput map 建桥；来源是现有 Sia placement trace 的最小 stage 行。 |
| `tests/test_joint_integration_coverage.py` | 新增 8 个实际集成路径测试。 |
| `audit_joint_response.py` | 读取全部官方矩阵并复算历史最大 rho。 |
| `run_joint_response_sensitivity.py` | 串行运行四档×三策略，冻结 manifest。 |
| `run_joint_coverage.py` | 保存覆盖测试日志和结构化摘要。 |
| `validate_joint_response_audit.py` | 对容量、Goodput、checkpoint、B4、回归和历史 hash 作持久化审计。 |

原始 `SiaPolicy`、`Cluster`、`sia.py` 和 `multi_simulator.py` 未修改。`job.py` 的倍率默认仍为 1，原始 CLI 不进入联合代码。

## 3. 八项新增覆盖

新增测试不是只调用独立 helper；每项至少经过实际 `Cluster`/`JobInfo`/`JointSiaPolicy`/`JointScheduler` 中与目标相关的路径。

| 场景 | 实际路径和断言 | 结果 |
|---|---|---|
| projected profile E2E smoke | `JointCluster.step(121)` 推进真实 CIFAR10 Job，在 aws 形成 direct profile；下一轮候选同时出现 `direct_profile` 与 `projected_profile`，且 progress>0 | PASS |
| rigid 配置与进度 | rigid CIFAR10 经过 4 个 `Cluster.step`；所有联合候选/选中配置严格为 4 GPU，训练进度增加 | PASS |
| PMP/scale-unit | 实际 `gpt_pmp` Job、Sia profile、`get_job_infos`、候选枚举和 `optimize_policy`；dgx-ext/rtx 的所有候选及选择均可整除各自 scale unit | PASS（有非致命 warning） |
| 多节点整节点 | 16-GPU rigid CIFAR10 在 2×8 RTX 节点上选择 `num_nodes=2`、placement `[8,8]` | PASS |
| B4 placement repair | 原 Sia GPU config 固定；节点 0/1 使用显式 synthetic 异构辅助容量，B4 从原 placement 重装箱，审计原因 `REPACK_FIXED_GPU_CONFIG` | PASS |
| B4 fallback/原子回退 | 测试专用 fault injection 让 B4 solver 结果丢一个作业；生产 `run_b4` 路径检测缺失并整体恢复完整 B1 proportional allocation，未部分提交 | PASS |
| B3 solver failure | 首轮成功提交后，测试专用 solver failure 返回 invalid audit；`optimize` 在 commit 前抛错，cluster allocations 和 current_resources 均保持旧值 | PASS |
| checkpoint lock | BERT 首次迁移后仍处于 restore；下一轮配置和节点完全保持，完整 delay 区间 progress_delta=0 | PASS |

测试专用 fault injection 只定义在测试文件的 policy subclass 中，不进入正式联合调度器默认行为。

## 4. 测试中发现并修正的问题

### 4.1 rigid 候选语义

旧 `_raw_candidates` 只使用 `max_replicas=target`，却仍让 `min_replicas=1`，导致 B3 可将 rigid Job 当作 elastic Job 缩小。修订后 rigid 的 min=max=target。随后测试又暴露：未形成 direct profile 时 fallback 仅为 `scale_unit` GPU 赋正 Goodput，固定 4/16 GPU 的 rigid Job会没有候选；现改为 rigid 的完整固定 demand。

### 4.2 PMP throughput map

官方 `Cluster.compute_throughput_ratios()` 和 `compute_throughputs()` 明确跳过 `gpt_pmp`，但后续 `get_sia_multi_job_info()` 会索引这两个字典。因此完整 PMP JobInfo 路径原本会缺 key。联合包装层现仅对缺失应用建立 map，使用已有 Sia placement trace 中 `num_replicas == num_stages` 的行计算原子 throughput 与类型比率，并把桥接原因/来源写入 `joint_profile_bridges`。

该修复允许检查 scale-unit 和联合 placement，但 PMP profile 拟合仍发出：

```text
RuntimeWarning: invalid value encountered in log
```

本轮候选均通过有限性和合法配置过滤，测试通过；但尚未执行 PMP 完整完成/JCT 实验，因此不能声称 PMP 性能语义已经完全验证。

## 5. 完整正确性回归

全套 `unittest discover`：**27/27 PASS**，其中原有 19 项和新增 8 项全部通过。日志：

- `research_runs/sia_synergy/joint_response_audit/all_tests.log`
- `research_runs/sia_synergy/joint_response_audit/all_tests_summary.json`
- `research_runs/sia_synergy/joint_response_audit/coverage_test.log`
- `research_runs/sia_synergy/joint_response_audit/coverage_summary.json`

持久化验证为 **18/18 PASS**：

1. 12 次运行严格串行；
2. 未超过两作业范围；
3. 所有作业完成；
4. JCT 来自完成事件；
5. 每节点 GPU/CPU/RAM audit 全部合法；
6. progress 单调；
7. Goodput 只乘一次；
8. checkpoint/restore 无训练推进；
9. `rho = raw_rho ** alpha`；
10. 坐标审计字段完整；
11. neutral rho 严格为 1；
12. neutral B1/B4 完全一致；
13. B4 GPU config 每轮固定；
14. 新增覆盖 8/8；
15. 全测试 27/27；
16. 原始 Sia CLI 字节一致；
17. 历史产物 hash 未变化；
18. `rho=4.2981` 精确复算。

验证文件：`research_runs/sia_synergy/joint_response_audit/validation_summary.json`。

## 6. 原始 Sia CLI 回归

在官方仓库和联合实验仓库中使用同一个两任务 workload、同一个原始命令：

```powershell
python multi_simulator.py <workload> --policy sia `
  --oracle_num_nodes 1 --oracle_num_replicas 8 --output <new-log>
```

结果：

```text
official     24D8FC9AFF0412A79C222E7090952156E4877FE2E7FEB0DED3F4C41A8905FF35
experimental 24D8FC9AFF0412A79C222E7090952156E4877FE2E7FEB0DED3F4C41A8905FF35
```

两者逐字节一致；原始 JCT 仍为 bert-28=547 s、cifar10-29=410 s。新日志存于 `joint_response_audit/`，没有覆盖任务 13 的回归日志。

## 7. 运行命令

```powershell
# 矩阵与历史 rho 复算
python audit_joint_response.py --models <Synergy/models> `
  --history <joint_integration_smoke/b3.sensitive.json> `
  --output <joint_response_audit>

# 四档×B1/B4/B3，严格串行
python run_joint_response_sensitivity.py `
  --config joint_prototype/integration_smoke_config.json `
  --workload joint_prototype/integration_smoke_workload.csv `
  --synergy-models <Synergy/models> --output <joint_response_audit>

# 新增八项与全套测试
python run_joint_coverage.py --output <joint_response_audit>
python run_joint_coverage.py --suite all --output <joint_response_audit>

# 持久化审计
python validate_joint_response_audit.py <joint_response_audit> `
  --historical <joint_integration_smoke>
```

## 8. 是否允许进入 20-job smoke

从软件正确性门槛看，允许进入下一阶段的 **20-job synthetic sensitivity smoke**：响应计算可复算、日志字段完整、manifest 已冻结、集成覆盖与原始回归均通过。

但需要附带三个硬条件：

1. neutral/low/medium/high 四档全部运行和报告，禁止按结果选择档位；
2. 固定哈希与预注册 category-mix 分成两个独立实验族，不混用、不换 seed；
3. 结论只能是合成机制与敏感性结果，不能称为真实 Saturn 或真实异构 GPU 集群收益。

在解决 `phi` 的真实校准和 PMP 警告之前，不建议进入真实性能主张、参数搜索或更大规模正式实验。
