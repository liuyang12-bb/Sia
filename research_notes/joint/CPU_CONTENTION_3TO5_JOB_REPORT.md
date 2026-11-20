# Sia + Synergy 四作业资源竞争机制验证报告

> **SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**

## 1. 范围与结论

本轮使用 4 条真实 Saturn workload 记录，在运行前冻结三种合成容量场景、两档响应参数和任务到 Synergy 模板的映射，顺序运行 B1、SY、B4、B5、B3 共 30 个配置。没有运行 20-job/160-job，没有搜索参数，也没有修改调度算法。

主要结论：

1. 30/30 配置均完成 4/4 作业，无 GPU/CPU/RAM 超配。
2. CPU/RAM 压力产生了真实运行现象，而非只做静态配置检查：压力场景累计等待 28,620 job-seconds；检测到 484 个“CPU 已饱和但节点仍有空闲 GPU”的逐轮节点事件，其中 45 个轮次同时存在未分配作业。
3. Synergy-TUNE 的当前任务 fair-share 降级和最终不可行等待均实际触发；本轮没有触发 `ATOMIC_PEER_DOWNGRADE_TO_FAIR`，应如实视为零次。
4. CPU 充足、neutral 下 B3 与 B1/B4/B5 几乎一致，B3 平均 JCT 多 0.25 秒；没有凭空收益。
5. CPU 压力下 B3 能选择与 Sia 串行意图不同的 GPU/placement，从而避免本场景中 B1/B4/B5 的长等待。该差异在 neutral 下也存在，说明主要来源是逐节点联合可行域，而不只是 `rho` 加速。
6. medium 响应仍出现最大 `rho=3.4155`，且来自低归一化分母与合成 `phi` 映射。当前结果只能证明机制，不足以支持真实集群性能主张。

最终判定：`MULTI_JOB_MECHANISM_VALID_BUT_MODEL_NOT_READY_FOR_20_JOB_CLAIMS`。

## 2. 版本保护与运行环境

- 实验仓库：`sia-joint-experiments/`
- 分支：`exp/synergy-baselines-cpu-bottleneck`
- HEAD：`aaf7a0de7b6cb5eab06ea020467ede5d2fd18e70`
- 任务开始时已有任务 12–15 的未提交原型；本轮只新增独立配置、runner、validator、产物和本报告。
- `official-artifacts/` 最终为干净的 `main...origin/main`。
- 正式运行开始时可用内存 2.263 GiB，高于预注册的 2 GiB 门槛。
- 第一次预检因可用内存仅 1.910 GiB而在模拟前主动退出；内存恢复到 2.501 GiB 后才执行成功预检。未降低门槛。
- 正式 30 次运行顺序执行，用时 653.536 秒。

## 3. 预注册输入

### 3.1 四个 Saturn 作业

`joint_prototype/cpu_contention_4job_workload.csv` 是 `workloads/saturn/workload-1.csv` 的精确行提取：

| 作业 | arrival | application | 原始 GPU demand | batch size | 合成响应模板 | 类别 |
|---|---:|---|---:|---:|---|---|
| bert-28 | 4931 | bert | 12 | 384 | bert_1.py | insensitive/flat |
| cifar10-29 | 4935 | cifar10 | 6 | 2048 | alexnet_1.py | sensitive |
| deepspeech2-30 | 5017 | deepspeech2 | 12 | 640 | deepspeech_1.py | sensitive |
| deepspeech2-31 | 5291 | deepspeech2 | 6 | 320 | transformer_1.py | insensitive/flat |

所有 arrival、application、GPU demand、batch 和训练工作量来自 Saturn；CPU/RAM 模板映射是预注册的跨应用、跨硬件合成迁移，不能称为这些 Saturn 作业的实测画像。

### 3.2 异构集群与三种容量

GPU 拓扑固定为：`aws: 1×4 GPU`、`dgx-ext: 1×8 GPU`、`rtx: 1×8 GPU`。CPU/RAM 容量均为合成值：

| 场景 | aws CPU/RAM | dgx-ext CPU/RAM | rtx CPU/RAM | 目的 |
|---|---:|---:|---:|---|
| A_cpu_sufficient | 48 / 500 GB | 96 / 1000 GB | 96 / 1000 GB | 辅助资源不构成硬瓶颈 |
| B_cpu_constrained | 4 / 500 GB | 8 / 1000 GB | 6 / 1000 GB | RTX 只能支持 6 个最小 CPU share，至少 2 GPU 受 CPU 限制 |
| C_mixed_pressure | 4 / 250 GB | 8 / 500 GB | 4 / 250 GB | 两类敏感任务竞争，仅 16 个最小 CPU share，且 RAM 同时收紧 |

最小合法执行点固定为每 GPU 1 CPU、20.83 GB RAM。场景 B 的 RTX 8-GPU 整节点配置不可行；场景 C 的 RTX 最多合法使用 4/8 GPU。

SY 固定映射为：bert→aws、cifar→dgx-ext、deepspeech2-30→dgx-ext、deepspeech2-31→rtx；固定 demand 使用 workload demand 并裁到不超过它的最大合法配置。该规则不读取 CPU/RAM response。

### 3.3 响应档位

- neutral：`alpha=0`，`rho≡1`。
- medium：`alpha=0.5`，即 `rho=(F_current/F_baseline)^0.5`。
- fixed reference：3 CPU/GPU、62.5 GB/GPU。

模板、容量、档位和 SY 映射均写入配置并在任何正式结果产生前冻结。

## 4. 策略口径

- B1：原 Sia MIP 选择 GPU，proportional CPU/RAM；若该意图违反硬辅助资源约束，仅允许完整作业 admission，不改变 GPU config。
- SY：固定 GPU type/demand + Synergy-TUNE 核心适配；不是未经修改的官方 Synergy。
- B4：原 Sia GPU 决策冻结后，运行本项目的有限列辅助资源优化与 placement 修复。
- B5：原 Sia GPU 决策冻结后，运行 Synergy-TUNE。
- B3：一个 MIP 同时选择 GPU 类型/数量、CPU、RAM 与逐节点 placement。

所有策略共享同一 workload、节点容量、Oracle、响应矩阵、`Job.step()`、checkpoint/restore 和完成判定。

## 5. 完整 JCT 与等待结果

单位均为秒；顺序为 `bert / cifar / ds30 / ds31`。所有行完成率均为 4/4。

| 容量 | 响应 | 策略 | 平均 JCT | 四作业 JCT | 总等待 | rho 范围 |
|---|---|---|---:|---|---:|---|
| A | neutral | B1 | 1700.50 | 783 / 564 / 2723 / 2732 | 0 | 1–1 |
| A | neutral | SY | 2981.75 | 4274 / 489 / 2995 / 4169 | 420 | 1–1 |
| A | neutral | B4 | 1700.50 | 783 / 564 / 2723 / 2732 | 0 | 1–1 |
| A | neutral | B5 | 1700.50 | 783 / 564 / 2723 / 2732 | 0 | 1–1 |
| A | neutral | B3 | 1700.75 | 783 / 565 / 2723 / 2732 | 0 | 1–1 |
| A | medium | B1 | 1420.75 | 694 / 318 / 1962 / 2709 | 0 | 1–2.6624 |
| A | medium | SY | 2737.00 | 4274 / 241 / 2264 / 4169 | 180 | 1–2.5819 |
| A | medium | B4 | 1420.75 | 694 / 318 / 1962 / 2709 | 0 | 1–2.6624 |
| A | medium | B5 | 1420.75 | 694 / 318 / 1962 / 2709 | 0 | 1–2.6624 |
| A | medium | B3 | 1408.50 | 695 / 271 / 1917 / 2751 | 0 | 1–3.4155 |
| B | neutral | B1 | 2152.75 | 783 / 564 / 2723 / 4541 | 1800 | 1–1 |
| B | neutral | SY | 2981.75 | 4274 / 489 / 2995 / 4169 | 420 | 1–1 |
| B | neutral | B4 | 2152.75 | 783 / 564 / 2723 / 4541 | 1800 | 1–1 |
| B | neutral | B5 | 2152.75 | 783 / 564 / 2723 / 4541 | 1800 | 1–1 |
| B | neutral | B3 | 1869.00 | 783 / 565 / 2723 / 3405 | 0 | 1–1 |
| B | medium | B1 | 2490.75 | 724 / 458 / 3460 / 5321 | 2580 | 0.5477–1.4142 |
| B | medium | SY | 3212.50 | 4274 / 374 / 4033 / 4169 | 300 | 0.5477–1.4142 |
| B | medium | B4 | 2470.75 | 724 / 456 / 3442 / 5261 | 2520 | 0.5652–1.4142 |
| B | medium | B5 | 2471.25 | 724 / 458 / 3442 / 5261 | 2520 | 0.5477–1.4142 |
| B | medium | B3 | 2042.75 | 721 / 448 / 3265 / 3737 | 0 | 0.6719–1.8227 |
| C | neutral | B1 | 2152.75 | 783 / 564 / 2723 / 4541 | 1800 | 1–1 |
| C | neutral | SY | 2981.75 | 4274 / 489 / 2995 / 4169 | 420 | 1–1 |
| C | neutral | B4 | 2152.75 | 783 / 564 / 2723 / 4541 | 1800 | 1–1 |
| C | neutral | B5 | 2152.75 | 783 / 564 / 2723 / 4541 | 1800 | 1–1 |
| C | neutral | B3 | 1898.50 | 783 / 565 / 2750 / 3496 | 0 | 1–1 |
| C | medium | B1 | 2801.25 | 783 / 570 / 3991 / 5861 | 3120 | 0.3873–1.0000 |
| C | medium | SY | 3331.00 | 4274 / 492 / 4389 / 4169 | 420 | 0.5477–1.0000 |
| C | medium | B4 | 2562.00 | 783 / 569 / 6155 / 2741 | 2400 | 0.5583–1.4142 |
| C | medium | B5 | 2799.25 | 783 / 570 / 3983 / 5861 | 3120 | 0.3873–1.0000 |
| C | medium | B3 | 2220.00 | 809 / 606 / 3601 / 3864 | 0 | 0.6098–1.7888 |

等待是实际 `Job.step()` 区间中“已到达、未完成、无 placement”的累计时间。主要事件：

- B-neutral：B1/B4/B5 的 `deepspeech2-31` 等待 1800 秒；B3 为 0。
- B-medium：B1 等待 2580 秒；B4/B5 各 2520 秒；均集中于 `deepspeech2-31`。
- C-medium：B1/B5 的 `deepspeech2-31` 等待 3120 秒；B4 的 `deepspeech2-30` 等待 2400 秒；SY 的 `deepspeech2-30` 等待 420 秒；B3 为 0。
- C-medium 中，未分配任务与“CPU 饱和但仍有空闲 GPU”同时出现：B1 7 轮、B4 9 轮、B5 7 轮、SY 2 轮。其余等待被保守标为 placement/config fragmentation，未强行归因给 CPU。

## 6. 资源、开销和求解结果

下表列出 medium 档；neutral 的同字段完整保存在 `comparison_summary.json`。资源单位为 GPU-h、CPU-core-h、RAM-GB-h。

| 场景 | 策略 | GPU-h | CPU-h | RAM-h | restart | checkpoint s | wasted GPU-s | solve total / P99 s | fallback |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | B1 | 10.007 | 120.080 | 1250.8 | 9 | 685 | 3060 | 0 / 0 | 0 |
| A | SY | 14.070 | 32.049 | 562.2 | 0 | 220 | 980 | .0560 / .0040 | 0 |
| A | B4 | 10.007 | 31.288 | 616.7 | 9 | 685 | 3060 | .0654 / .0024 | 0 |
| A | B5 | 10.007 | 30.389 | 468.3 | 9 | 685 | 3060 | .0221 / .0009 | 0 |
| A | B3 | 9.567 | 29.457 | 558.7 | 9 | 660 | 2710 | .1215 / .0031 | 0 |
| B | B1 | 13.496 | 13.496 | 1686.9 | 8 | 660 | 2860 | .0441 / .0014 | 0 |
| B | SY | 17.882 | 19.477 | 1371.1 | 0 | 220 | 980 | .0811 / .0038 | 0 |
| B | B4 | 13.453 | 13.537 | 1039.1 | 8 | 660 | 2860 | .1044 / .0021 | 42 |
| B | B5 | 13.456 | 13.489 | 1674.0 | 8 | 660 | 2860 | .0880 / .0020 | 0 |
| B | B3 | 11.574 | 12.991 | 852.1 | 10 | 685 | 2635 | .1946 / .0041 | 0 |
| C | B1 | 14.664 | 14.664 | 871.7 | 8 | 660 | 2860 | .0366 / .0011 | 0 |
| C | SY | 18.538 | 18.538 | 819.2 | 0 | 220 | 980 | .0628 / .0034 | 0 |
| C | B4 | 14.339 | 14.406 | 641.1 | 9 | 685 | 2960 | .0665 / .0009 | 40 |
| C | B5 | 14.647 | 14.680 | 869.9 | 8 | 660 | 2860 | .0564 / .0013 | 0 |
| C | B3 | 12.213 | 12.813 | 574.1 | 9 | 660 | 2460 | .1560 / .0040 | 0 |

所有 B3/B4 MIP 为 `OPTIMAL`。B1 在压力场景中包含 `ORIGINAL_SIA_OPTIMAL` 后的 whole-job admission `OPTIMAL`。B4 的 fallback 计数表示原 Sia 固定 GPU intent 无法让所有作业共同通过辅助资源约束，代码保留合法完整子集并记录原子回退；不是求解器崩溃。SY/B5 状态均为 `SYNERGY_TUNE_ADAPTED`。

代表性的 C-medium 时间加权节点利用率：

| 策略 | aws GPU/CPU/RAM | dgx-ext GPU/CPU/RAM | rtx GPU/CPU/RAM |
|---|---|---|---|
| B1 | 0.2% / 0.2% / 0.2% | 95.5% / 95.5% / 95.5% | 10.4% / 20.7% / 10.4% |
| SY | 92.9% / 92.9% / 31.0% | 91.7% / 91.7% / 91.7% | 45.3% / 90.6% / 45.1% |
| B4 | 0.2% / 0.2% / 0.1% | 93.7% / 93.7% / 67.5% | 8.8% / 18.6% / 11.7% |
| B5 | 0.2% / 0.2% / 0.1% | 95.3% / 95.3% / 95.3% | 10.4% / 21.2% / 10.4% |
| B3 | 8.5% / 8.5% / 2.8% | 87.5% / 93.5% / 77.3% | 37.6% / 75.9% / 37.1% |

利用率以首个 arrival 到该策略仿真结束作为观察窗口，因此不同策略的窗口长度不同；适合检查容量与闲置，不宜直接视为能源效率结论。

## 7. Synergy-TUNE 行为

TUNE 实际发生的动作包括：

- A-medium SY：177 demand、3 current→fair、3 final-wait；B5：91/4/0。
- B-medium SY：128 demand、81 current→fair、5 final-wait；B5：5/117/42。
- C-medium SY：72 demand、143 current→fair、7 final-wait；B5：5/130/52。

本轮 `ATOMIC_PEER_DOWNGRADE_TO_FAIR=0`。原因是当前四作业映射下，尝试当前任务 fair-share 后要么能够放置，要么固定 GPU 配置/placement 仍不可行；没有出现“必须降低已放置 peer 且全量重装后可行”的运行状态。该路径已有单元测试覆盖，但本轮不能宣称端到端触发。

SY 的较高 JCT主要来自预注册固定 GPU type/demand 与长作业的组合。例如 A-neutral 中 bert JCT 为 4274 秒，而 B1/B3 为 783 秒。它不能被解释为未经修改的官方 Synergy 弱于 B3。

## 8. B3 的成对比较

负数表示 B3 平均 JCT 更低：

| 场景 | B3−B1 | B3−SY | B3−B4 | B3−B5 |
|---|---:|---:|---:|---:|
| A-neutral | +0.25 | -1281.00 | +0.25 | +0.25 |
| A-medium | -12.25 | -1328.50 | -12.25 | -12.25 |
| B-neutral | -283.75 | -1112.75 | -283.75 | -283.75 |
| B-medium | -448.00 | -1169.75 | -428.00 | -428.50 |
| C-neutral | -254.25 | -1083.25 | -254.25 | -254.25 |
| C-medium | -581.25 | -1111.00 | -342.00 | -579.25 |

解释边界：

- A-neutral 是最干净的退化检查：B3 没有收益，且仅因逐节点 MIP tie-break/离散完成秒多 0.25 秒。
- B/C-neutral 中 `rho=1`，B3 仍减少 254–284 秒平均 JCT，来源只能是联合 GPU/placement 可行性选择与避免长等待，不能归因于性能倍率。
- C-medium 中 B3 相对 B1 低 20.75%，相对 B4 低 13.35%，相对 B5 低 20.69%。但这同时叠加了合成 slowdown/gain，不是可外推的真实收益。
- B3 并非逐任务都更好。C-medium 中 bert/cifar 的 JCT 分别为 809/606，高于 B1 的 783/570；改善主要来自两个长 DeepSpeech 作业，尤其避免 `deepspeech2-31` 的 3120 秒等待。

## 9. 响应模型复核

### 9.1 历史 `rho=4.2981`

历史复算文件：`research_runs/sia_synergy/joint_response_audit/rho_4_2981_reproduction.json`。

- `res18_1.py`，配置 `dgx-ext:1:1`，8 CPU/GPU、125 GB/GPU；
- `phi_c=4`、`phi_m=2`；
- 分子源坐标 `(2,62.5)`，矩阵值 0.72；
- baseline 请求 `(0.75,31.25)`，CPU clip 到 1，分母 0.1675161987041037；
- `0.72 / 0.1675161987041037 = 4.298091799896853`。

### 9.2 当前 medium 最大 `rho=3.4155`

当前最大值位于 A-medium/B3、`cifar10-29`、`dgx-ext:1:4`、24 CPU/GPU、125 GB/GPU、`alexnet_1.py`：

- `phi_c=4`、`phi_m=2`；
- 分子源坐标 `(6,62.5)`，矩阵值 1.75；
- baseline 请求 `(0.75,31.25)`，CPU clip 到 1，分母 0.1500119990400768；
- `raw_rho=1.75/0.1500119990400768=11.6657334826`；
- medium 使用平方根衰减：`rho=sqrt(raw_rho)=3.4155136484`。

这条链路数值上可复算，但物理上未校准。官方数据仅包括 Synergy 矩阵单元；任务映射、固定参考点、GPU→`phi` 缩放、插值/clip 和与 Sia Goodput 相乘均为本项目合成假设。没有为了本轮结果截断 3.4155。

### 9.3 候选与实际推进口径

B3 决策列使用 Sia 归一化 scheduling Goodput；`Job.step()` 日志使用实际 progress/second。二者量纲/归一化不同，不能逐值相等。例如最大 rho 轮的 predicted normalized effective Goodput 为 13.9678，而实际 raw/effective progress rate 为 23.5727/80.5128。

可验证的是倍率只施加一次：全 30 次运行中 `|effective - raw × rho|` 最大残差为 `1.7764e-15`。checkpoint/restore 倒计时没有乘 `rho`，完整 delay 区间无训练进度。

## 10. 正确性、测试和回归

`validation_summary.json` 的 16 项检查全部通过，包括：

- 30 个组合齐全且全部 4/4 完成；
- 容量与时间加权利用率均不超过 1；
- neutral `rho=1`、倍率只乘一次；
- B5 保持 Sia GPU config，SY 使用 TUNE core，B3 枚举多个 GPU 候选；
- checkpoint delay 无进度；
- TUNE 降级、实际等待、CPU 饱和伴随 GPU 闲置均出现；
- medium 改变敏感任务实际执行；
- 原始 Sia 官方/实验日志 SHA-256 均为 `F81E0B22C76734E5264742A7231687F366CE77B4D3448155E444A64D42E02DFF`。

既有测试：30/30 通过，`Ran 30 tests in 8.042s`。PMP 覆盖测试仍有既有 autograd `invalid value encountered in log` warning，测试通过。

## 11. 哈希与运行产物

- 配置 SHA-256：`2F4D6C1AE765DD68DB1CAFAF0223CFFAD852F5094B9EBE30054A3325DE0D0CE9`
- workload SHA-256：`43EE4622CF80096847BDF4FAF9A032EE9BB56A03FA070214473E176C4017A07B`
- 源 workload SHA-256、四个画像 hash、代码 hash：见 manifests。

正式目录：`research_runs/sia_synergy/cpu_contention_4job/`

- `run_manifest.json`：输入、环境、画像哈希、硬容量审计、30 次摘要；
- `comparison_summary.json`：完整 JCT、等待、资源、重启、checkpoint、TUNE、求解及成对差值；
- `validation_summary.json`：16 项独立验收；
- `code_manifest.json`：核心代码和新文件哈希；
- `all_tests.log`、`all_tests_summary.json`：30 项测试证据；
- `A/B/C.{neutral,medium}.{b1,sy,b4,b5,b3}.json`：逐轮资源、Goodput、训练进度、完成事件和 runtime contention evidence。

预检目录 `research_runs/sia_synergy/cpu_contention_4job_preflight/` 独立保留，没有覆盖正式结果。

## 12. 是否具备开展 20-job 对照的条件

### 工程条件

基本具备：五个策略可以在统一时间推进中完成多作业运行；CPU/RAM 硬约束、真实等待、GPU 闲置、TUNE 降级、checkpoint 和原子提交都有日志证据；测试与原始 Sia 回归通过。

### 结论可信度条件

尚不具备直接开展“可用于性能结论”的 20-job 对照。会直接影响结论的问题包括：

1. `phi_c/phi_m` 与 fixed-reference 分母未由真实跨 GPU CPU/RAM 测量校准，最大 medium `rho=3.4155` 仍偏高。
2. B1 在辅助资源不可行时采用 whole-job admission；B3 从一开始就在逐节点联合可行域求解。neutral 下的大差异证明这个基线语义本身会显著影响结果，20-job 前必须将其作为预注册实验定义，而不能笼统称为“原 Sia 性能”。
3. 候选 normalized Goodput 与 `Job.step()` 实际 progress/sec 的数值口径不同；需要校准或至少报告预测误差，避免将 MIP 目标值当作实际吞吐。
4. 当前只有一个固定 50/50 category mix。20-job 前应冻结多个不依赖结果的模板映射/种子，或给出唯一确定映射规则。
5. 本轮未端到端触发 peer downgrade；若 20-job 研究问题依赖该机制，应先增加一个预注册、非调参式 3–5 作业场景覆盖它。
6. CPU/RAM 节点规格仍为 synthetic，不能外推为真实 AWS/DGX/RTX 集群。

建议下一步先冻结 20-job workload、模板映射、容量、响应档位和 B1 admission 语义，并进行一次仅用于正确性/稳定性的 20-job smoke；在完成 `rho` 校准前，不应把它升级为正式性能对照或论文结论。
