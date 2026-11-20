# CPU 瓶颈两作业机制验证报告

> **SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**

## 1. 范围与结论

本轮顺序运行了 2 个真实 Sia Saturn workload 行、2 种合成 CPU 容量、2 种预注册响应设置和 5 个策略，共 20 个两作业配置。没有运行 20-job/160-job，也没有参数搜索。

结果确认：硬 CPU 容量参与可行性审计，CPU/RAM 供给通过统一 Oracle 影响实际训练推进，所有策略无节点超配；SY 固定 demand/type、B5 GPU 冻结、B3 多 GPU 候选联合选择均通过日志验证。CPU 受限使敏感任务显著变慢，但 B3 相对 B5 的小幅差异高度依赖合成响应；当前不具备发布或正式 20-job 性能对照的模型可信度。

## 2. 输入与可复现性

- workload：`joint_prototype/cpu_bottleneck_smoke_workload.csv`
- 来源：`workloads/saturn/workload-1.csv` 中 `bert-28`、`cifar10-29` 两行；arrival 与训练工作量字段保持原值。
- workload SHA-256：`7E7BEC84471123AA098212EC5CF8EF622EECAC7A6C23C956A86A2C1E403EE100`
- 配置：`joint_prototype/cpu_bottleneck_smoke_config.json`
- 配置 SHA-256：`0F12094E8569ED88DE2287D3DAF2F246AF4ABB882167E080B7997C771239082F`
- 代码 HEAD：`aaf7a0de7b6cb5eab06ea020467ede5d2fd18e70`
- Python：3.10.10，Windows 10/11 build 26200 环境。
- 运行时长：134.829 秒；全部顺序执行。

固定 category mix：

- `bert-28 → bert_1.py`，标记为 CPU-insensitive borrowed shape；
- `cifar10-29 → alexnet_1.py`，标记为 CPU-sensitive borrowed shape。

模板哈希与完整 manifest 位于 `research_runs/sia_synergy/cpu_bottleneck_smoke/run_manifest.json`。该映射在运行前冻结，没有按结果挑选模板。

## 3. 集群与 CPU 瓶颈

GPU 拓扑在两种容量场景中相同：1×`aws` 节点（4 GPU）、1×`dgx-ext`（8 GPU）、1×`rtx`（8 GPU）。GPU 类型与 Sia profile 来自 Artifact；以下 CPU/RAM 均为合成容量，不是 AWS、DGX 或 RTX 真实规格。

| 场景 | aws CPU/RAM | dgx-ext CPU/RAM | rtx CPU/RAM |
|---|---:|---:|---:|
| CPU 充足 | 48 / 500 GB | 96 / 1000 GB | 96 / 1000 GB |
| CPU 受限 | 4 / 500 GB | 8 / 1000 GB | 6 / 1000 GB |

最小合法画像点为每 GPU 1 CPU、20.83 GB。故 CPU 受限的 RTX 节点最多合法使用 6/8 GPU，完整 8-GPU 配置因 CPU floor 不可行；CPU 充足时三类完整节点都可行。这个约束进入候选生成、TUNE 放置、MIP 与提交前审计，不是仅记录在配置文件中。

响应采用 fixed reference `(3 CPU/GPU, 62.5 GB/GPU)`：neutral 为 `alpha=0`，因此 `rho≡1`；medium 为任务 14 预注册的 `alpha=0.5`。没有事后裁剪倍率。

## 4. 执行命令

```powershell
..\..\.venv-sia-baseline\Scripts\python.exe run_cpu_bottleneck_smoke.py `
  --config joint_prototype\cpu_bottleneck_smoke_config.json `
  --workload joint_prototype\cpu_bottleneck_smoke_workload.csv `
  --synergy-models ..\..\research_sources\Synergy\simulator\models `
  --output ..\..\research_runs\sia_synergy\cpu_bottleneck_smoke

..\..\.venv-sia-baseline\Scripts\python.exe validate_cpu_bottleneck_smoke.py `
  --input ..\..\research_runs\sia_synergy\cpu_bottleneck_smoke

..\..\.venv-sia-baseline\Scripts\python.exe -m unittest discover -s tests -v
```

## 5. 绝对结果

所有配置均完成 2/2 作业，日志中的等待时间均为 0；因此本例显示的是性能降级和配置可行域收缩，而不是实际排队。JCT 单位为秒。

| CPU | 响应 | 策略 | 平均 JCT | bert JCT | cifar JCT | rho 范围 | GPU-h | CPU-core-h | 重启 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 充足 | neutral | B1 | 673.5 | 783 | 564 | 1–1 | 1.5411 | 18.4933 | 4 |
| 充足 | neutral | SY | 2381.5 | 4274 | 489 | 1–1 | 5.1878 | 10.6144 | 0 |
| 充足 | neutral | B4 | 673.5 | 783 | 564 | 1–1 | 1.5411 | 1.5411 | 4 |
| 充足 | neutral | B5 | 673.5 | 783 | 564 | 1–1 | 1.5411 | 11.2944 | 4 |
| 充足 | neutral | B3 | 674.0 | 783 | 565 | 1–1 | 1.5422 | 1.5422 | 4 |
| 充足 | medium | B1 | 506.0 | 694 | 318 | 1–2.6624 | 1.2700 | 15.2400 | 4 |
| 充足 | medium | SY | 2257.5 | 4274 | 241 | 1–2.5819 | 4.9122 | 7.3078 | 0 |
| 充足 | medium | B4 | 506.0 | 694 | 318 | 1–2.6624 | 1.2700 | 5.5233 | 4 |
| 充足 | medium | B5 | 506.0 | 694 | 318 | 1–2.6624 | 1.2700 | 8.0233 | 4 |
| 充足 | medium | B3 | 483.0 | 695 | 271 | 1–3.4155 | 1.0867 | 4.8989 | 3 |
| 受限 | neutral | B1 | 673.5 | 783 | 564 | 1–1 | 1.5411 | 1.5411 | 4 |
| 受限 | neutral | SY | 2381.5 | 4274 | 489 | 1–1 | 5.1878 | 5.1878 | 0 |
| 受限 | neutral | B4 | 673.5 | 783 | 564 | 1–1 | 1.5411 | 1.5411 | 4 |
| 受限 | neutral | B5 | 673.5 | 783 | 564 | 1–1 | 1.5411 | 1.5411 | 4 |
| 受限 | neutral | B3 | 674.0 | 783 | 565 | 1–1 | 1.5422 | 1.5422 | 4 |
| 受限 | medium | B1 | 591.0 | 724 | 458 | 0.5477–1.4142 | 1.4256 | 1.4256 | 4 |
| 受限 | medium | SY | 2324.0 | 4274 | 374 | 0.5477–1.4142 | 5.0600 | 5.0600 | 0 |
| 受限 | medium | B4 | 590.0 | 724 | 456 | 1–1.4142 | 1.4233 | 1.4733 | 4 |
| 受限 | medium | B5 | 591.0 | 724 | 458 | 0.5477–1.4142 | 1.4256 | 1.4256 | 4 |
| 受限 | medium | B3 | 584.5 | 721 | 448 | 1–1.7888 | 1.2244 | 1.3911 | 3 |

注意：CPU-core-hours 反映各策略所选择的合成配额，不能直接解释为真实能耗或成本。

CPU 受限、medium 场景的逐节点峰值如下（格式为 GPU / CPU / RAM-GB）：

| 策略 | aws:0 | dgx-ext:0 | rtx:0 | RAM-GB-h |
|---|---:|---:|---:|---:|
| B1 | 0/0/0 | 8/8/1000 | 2/2/250 | 178.1944 |
| SY | 4/4/83.32 | 4/4/500 | 0/0/0 | 143.4797 |
| B4 | 0/0/0 | 8/8/1000 | 2/5/83.33 | 84.8582 |
| B5 | 0/0/0 | 8/8/1000 | 2/2/145.83 | 172.9859 |
| B3 | 0/0/0 | 8/8/583.32 | 0/0/0 | 60.6915 |

所有 B3/B4 轮次状态为 `OPTIMAL`，B1 为 `ORIGINAL_SIA_OPTIMAL`，SY/B5 为 `SYNERGY_TUNE_ADAPTED`；20 个运行均无 fallback。CPU 受限 medium 下，SY 的 TUNE action 为 72 次 demand、6 次当前任务降到 fair，B5 为 3 次 demand、17 次降到 fair。完整逐轮 action 与 placement 在对应 JSON 中。

## 6. 机制问题回答

### 6.1 CPU 瓶颈是否真正生效？

是。除 RTX 8-GPU 列被硬 floor 排除外，medium 下 CPU 受限使敏感的 `cifar10-29` JCT 明显上升：B1/B5 从 318 到 458 秒，SY 从 241 到 374 秒，B3 从 271 到 448 秒。B1/B5 还出现最低 `rho=0.5477`，说明统一执行模型允许资源不足导致 Goodput 低于固定参考值。所有逐节点容量审计为零违规。

本次只有两个同时可容纳的任务，因此没有观测到 CPU 引起的非零 waiting；不能声称已验证排队效应。GPU 因 CPU 被迫闲置由 RTX 的“最多 6/8 GPU 可用”结构审计验证，而不是由一个实际等待事件验证。

### 6.2 SY 是否保留官方核心？

保留了 TUNE 的 demand→当前任务 fair-share→peer fair-share/repack→最终重试顺序，以及 FIFO 顺序与固定 GPU demand 语义。异构 GPU 类型映射、Sia 整数 placement、最终失败等待和统一执行模型是明确列出的适配规则。没有移植 Synergy-OPT，也没有把 SY 描述为未经修改的官方实现。

### 6.3 B5 相比 B1 的收益来自哪里？

本两作业配置中 **没有 JCT 收益**：四个容量×响应组合里 B5 与 B1 的 JCT 完全相同。TUNE 改变了部分 CPU 配额与 action trace，但没有改变冻结的 GPU 配置，也没有改变完成时间。这是有效的负结果。

### 6.4 B3 是否优于 B5？

- neutral：B3 平均 JCT 比 B5 高 0.5 秒；没有辅助资源收益。
- medium、CPU 充足：B3 低 23 秒，即 4.55%。
- medium、CPU 受限：B3 低 6.5 秒，即 1.10%。

差异来自 B3 可联合更换 GPU 配置、辅助资源点和 placement，并减少一次 restart；但最大 `rho` 同时升到 3.4155/1.7888。由于优势在 CPU 充足时反而更大，结果更像响应模型与配置选择共同作用，而不是“CPU 瓶颈必然带来联合优势”。

### 6.5 CPU 充足时 B3 是否有额外开销？

有求解开销但很小：四个 B3 配置的优化总时间约 0.01495–0.02860 秒；B1 不运行辅助 MIP。neutral 下没有性能收益且平均 JCT多 0.5 秒，说明联合选择不天然免费或天然更好。

### 6.6 去掉限制或设为 insensitive 后收益是否减弱？

neutral 下所有 `rho=1`，B1/B4/B5 完全一致，B3 没有收益。`bert-28` 使用 flat 模板，在 medium 中各策略的 rho 始终为 1；其 JCT 基本接近。观测到的主要变化集中在映射到 `alexnet_1.py` 的敏感任务，支持“收益依赖响应假设”的判断。

SY 的平均 JCT 很高是因为其预注册固定 GPU type/demand 对 `bert-28` 不利；这是固定-demand 对照的行为，不是官方 Synergy 与 B3 的公平系统级优劣结论。

## 7. 正确性结果

`validation_summary.json` 的 13 项检查全部通过：

- 20 个组合齐全，全部 2/2 完成；
- 无 GPU/CPU/RAM 容量违规；
- neutral `rho` 严格为 1；
- B5 从不改变当轮 Sia GPU config；
- SY 使用审计过的 TUNE core，且 type/demand 全程固定；
- B3 每轮审计到多个 GPU 候选；
- 完整 checkpoint delay 区间无训练推进；
- CPU floor 使 RTX 满节点配置不可行，而充足场景可行；
- CPU 压力改变敏感任务实际执行。
- 原始 `--policy=sia` 在官方仓库与实验仓库输出逐字节一致，SHA-256 均为 `F81E0B22C76734E5264742A7231687F366CE77B4D3448155E444A64D42E02DFF`；本次默认 CLI 回归 JCT 为 915/1610 秒。该默认大集群回归只用于一致性检查，不能与三节点联合 smoke 的 JCT 横向比较。

单元与集成测试：30/30 通过。PMP 测试仍有一个既有 autograd `invalid value encountered in log` warning，测试本身通过。

## 8. 产物

目录：`research_runs/sia_synergy/cpu_bottleneck_smoke/`

- `run_manifest.json`：环境、输入/画像 hash、硬 CPU 可行性审计、20 次运行摘要；
- `comparison_summary.json`：绝对数值与 CPU 压力/B3-B5 对比；
- `validation_summary.json`：独立检查结果；
- `code_manifest.json`：代码与配置 hash；
- `all_tests.log`、`all_tests_summary.json`：测试证据；
- `official_sia_regression.log`、`experimental_sia_regression.log`：本轮原始 CLI 字节一致性回归；
- `cpu_{sufficient,constrained}.{neutral,medium}.{b1,sy,b4,b5,b3}.json`：逐轮决策、资源、Goodput、checkpoint 和完成事件。

## 9. 是否进入 20-job 阶段

判定：`MECHANISM_VALID_BUT_NOT_READY_FOR_20_JOB_PERFORMANCE_CLAIMS`。

代码路径与容量约束已经足以做进一步工程测试，但在启动正式 20-job 对照前至少需要：

1. 冻结更广的模板映射或多个预注册 category mixes，避免两任务结论由单个敏感模板支配；
2. 对 `phi_c/phi_m` 和 fixed-reference 分母做外部依据或区间敏感性校准，特别处理高达 3.4155/历史 4.2981 的倍率；
3. 增加能实际产生 CPU 排队、GPU 空闲和多任务 peer downgrade 的 3–5 作业集成场景；
4. 解释或校准候选归一化 Sia Goodput与 `Job.step()` 瞬时执行 Goodput 的口径差；
5. 预注册 20-job 的模板映射、种子、容量和响应档位后再运行，禁止按结果调参。

因此，本轮停止于小规模机制验证，不自动开展正式实验。
