# Sia + Synergy 联合调度最小原型报告

- 日期：2026-10-24
- 分支：`sia-joint-experiments/` → `exp/sia-synergy-joint-prototype`
- 基线：Sia Artifact `aaf7a0de7b6cb5eab06ea020467ede5d2fd18e70`
- Python：3.10.10；NumPy 1.23.5；SciPy 1.10.1；MIP 为 `scipy.optimize.milp`/HiGHS
- 判定：**MINIMAL_PROTOTYPE_VALID；尚未具备直接开展 20-job 端到端对照的条件**
- 声明：**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**

## 1. 完成内容

本轮在干净官方 commit 上新建独立仓库和分支，没有修改：`official-artifacts/`、`sia-synergy-experiments/`、`research_sources/Synergy/` 或历史运行产物。

实现包括：

- 直接解析 Synergy 官方 9×5 CPU/RAM 矩阵的 `JointProfileOracle`；
- fixed-reference absolute 与 capacity-relative 两种归一化；
- B1 GPU-only 决策但统一 actual execution；
- B4 固定 B1 GPU config 的 auxiliary 优化与多资源装箱；
- B3 GPU+CPU+RAM+placement 有限列 MIP；
- 逐节点三资源审计、checkpoint lock、B4 proportional fallback；
- 独立穷举器验证每个小 MIP 的最优目标；
- 3 个固定 2-job 场景、14 项单元测试和 6 次运行（3 场景×2 归一化）。

## 2. 文件清单

### 2.1 代码与输入

| 文件 | 内容 |
|---|---|
| `sia-joint-experiments/sia-simulator/joint_prototype/joint_profile.py` | 官方矩阵 loader、插值、统一 Oracle、两类模板映射。 |
| `joint_prototype/joint_scheduler.py` | placement 枚举、联合列、MIP、穷举、审计、B1/B4/B3。 |
| `joint_prototype/scenario_io.py` | synthetic schema 与输入 hash。 |
| `run_joint_prototype.py` | 串行执行三场景、两归一化并输出 JSON。 |
| `joint_prototype/scenarios/scenario_a_*.json` | 快 GPU/CPU 不足。 |
| `joint_prototype/scenarios/scenario_b_*.json` | 弹性 GPU/CPU 限制。 |
| `joint_prototype/scenarios/scenario_c_*.json` | CPU/RAM 争用和 placement。 |
| `tests/test_joint_prototype.py` | 14 项测试。 |

### 2.2 运行产物

目录：`research_runs/sia_synergy/joint_prototype/`

- `run_manifest.json`：命令、环境、输入 hash、退出状态、运行时间和全局验证。
- 每场景分别有 `.fixed_reference.json` 与 `.capacity_relative.json`。
- 每个结果保存 B1/B4/B3 的完整 config、GPU 数、CPU/RAM、placement、raw Goodput、rho、effective Goodput、官方模板文件/hash、节点使用、MIP/穷举结果和修复审计。

## 3. 统一性能查询实装

`JointProfileOracle` 位于 `joint_profile.py:187`，查询入口 `:218`。输出 `ProfileQuery` 包含：

```text
raw_sia_goodput
template_id / source_path / source_sha256
cpu_per_gpu / ram_gb_per_gpu
profile baseline 与映射后的 source coordinates
synthetic CPU/RAM demand
rho / effective_goodput
normalization mode / evidence label
```

loader 只接受官方 `*_1.py` 文件名，用 AST 解析 literal/arithmetic 表达式，不执行 Synergy 模块；矩阵必须为 9×5、有限且正。结果中使用的 `res18_1.py`、`m5_1.py`、`transformer_1.py` 均来自 `research_sources/Synergy/simulator/models/`，并逐查询保存 SHA-256。

无未来信息检查：`JobSpec` 只有 arrival 已发生、observed progress、当前 config/node、checkpoint lock；不存在 completion time、future arrival 或 true remaining runtime 字段。

## 4. B1/B4/B3 的实际代码行为

### B1

`build_b1_columns()`（`joint_scheduler.py:264`）以 raw Sia Goodput 选择 GPU config/node pattern，CPU/RAM 使用所在节点的 proportional share。MIP 决策不利用 response，但结果的实际进度由同一个 Oracle 查询，因此 absolute shortage 会降低 B1 effective Goodput。

### B4

`run_b4()`（`joint_scheduler.py:409`）从 B1 结果构造 `allowed_configs[j]={B1_config}`；CPU/RAM 与 placement 可以改变，GPU config 不可改变。结果记录保持/重装箱原因。B1 proportional map 是显式构造的全量 fallback；若 B4 MIP遗漏 B1 job，则整体回退并重新审计，绝不提交半解。

### B3

`run_b3()`（`joint_scheduler.py:452`）把每个合法 GPU config、auxiliary 点和 placement pattern 组成列，一次 MIP 决定。它没有调用 B1/Sia 先选唯一 config，因此不是 S2/B4 改名。

## 5. 测试

命令：

```powershell
cd D:\科研启航\Sia\sia-joint-experiments\sia-simulator
..\..\.venv-sia-baseline\Scripts\python.exe -m unittest tests.test_joint_prototype -v
```

最终结果：退出码 0，14/14 通过；最终测试本体 0.225 秒。覆盖：

1. 官方 matrix 形状、数值和 SHA provenance。
2. Synergy 文件中的乘除/int arithmetic 安全解析。
3. fixed reference 点 `rho=1` 且不足时可 `<1`。
4. capacity-relative proportional 点严格为 1。
5. 非法 GPU config 拒绝查询。
6. fixed-hash/category-mix 确定性映射。
7. 多节点整节点与 per-GPU 比例一致。
8. B1/B4/B3 的 MIP 目标全部匹配独立穷举。
9. B4 从不改变 B1 GPU config。
10. 场景 A 改 GPU 类型、场景 B 改 GPU 数。
11. 场景 C B4 修复 placement 且与 B3 相同。
12. Job 输入无未来信息。
13. checkpoint lock 固定 config 和 node pattern。
14. 所有结果通过逐节点容量审计。

## 6. 运行命令与总状态

```powershell
cd D:\科研启航\Sia\sia-joint-experiments\sia-simulator
..\..\.venv-sia-baseline\Scripts\python.exe run_joint_prototype.py `
  --synergy-models ..\..\research_sources\Synergy\simulator\models `
  --output ..\..\research_runs\sia_synergy\joint_prototype `
  --normalization all
```

最终结果：退出码 0，3 个输入、6 个运行全部成功；所有逐节点容量审计为真，18/18 policy-run 的 MIP 目标与独立穷举目标一致。最终整套运行 0.163 秒，没有并行执行。

## 7. fixed-reference 主结果

本节数值是单轮总 effective Goodput，不是 JCT；所有场景均为 2 jobs。

| 场景 | B1 | B4 | B3 | B4 vs B1 | B3 vs B4 | 是否改变 GPU config |
|---|---:|---:|---:|---:|---:|---|
| A 快 GPU/CPU 瓶颈 | 1.857798 | 1.857798 | 1.900000 | 0% | +2.27% | 是：fast→slow |
| B 弹性 GPU/CPU 限制 | 1.857798 | 2.178899 | 2.795000 | +17.28% | +28.28% | 是：2 GPU→1 GPU |
| C CPU/RAM placement | 2.000000 | 3.416667 | 3.416667 | +70.83% | 0% | 否；B4 已充分 |

### 7.1 场景 A：快 GPU 不一定有更高 actual Goodput

| 策略/Job | GPU config | GPU | CPU/GPU | RAM/GPU | placement | raw | rho | effective |
|---|---|---:|---:|---:|---|---:|---:|---:|
| B1 target | fast-1g | 1 | 2 | 62.5 | fast-0 | 2.0 | 0.678899 | 1.357798 |
| B4 target | fast-1g | 1 | 1 | 62.5 | fast-0 | 2.0 | 0.678899 | 1.357798 |
| B3 target | slow-1g | 1 | 3 | 62.5 | slow-0 | 1.4 | 1.000000 | 1.400000 |
| all / neutral | background-1g | 1 | 3(B1)/1(B4,B3) | 62.5/20.83 | background-0 | 0.5 | 1 | 0.5 |

ResNet18 source curve在 fast config 的低 CPU 区间已被 clip 到同一最低矩阵点，所以 B4 把 2 降至 1 core/GPU 不再进一步降低性能，但也无法通过固定 fast config 超过 1.357798。B3 能比较另一个 GPU 类型，选择 nominal raw 较低但 actual 更高的 slow config。收益很小但方向可解释。

### 7.2 场景 B：更多 GPU 的 nominal 收益被 CPU 限制

| 策略/Job | GPU config | GPU | CPU/GPU | RAM/GPU | placement | raw | rho | effective |
|---|---|---:|---:|---:|---|---:|---:|---:|
| B1 elastic | elastic-2g | 2 | 1.5 | 62.5 | elastic-0 | 2.0 | 0.678899 | 1.357798 |
| B4 elastic | elastic-2g | 2 | 2.5 | 62.5 | elastic-0 | 2.0 | 0.839450 | 1.678899 |
| B3 elastic | elastic-1g | 1 | 5.0 | 62.5 | elastic-0 | 1.5 | 1.530000 | 2.295000 |
| all / neutral peer | peer-1g | 1 | 1(B4,B3) | 20.83 | elastic-0 | 0.5 | 1 | 0.5 |

B4 从不敏感 peer 回收资源后能改善固定 2-GPU config，但每份 CPU 被两个 GPU 分摊。B3 选择 1 GPU，把 5 cores 集中给 elastic job，实际总 Goodput 最高。该结果验证新增能力来自 GPU 数与辅助资源的共同选择。

### 7.3 场景 C：串行修复已经足够

| 策略/Job | GPU | CPU/GPU | RAM/GPU | placement | rho | effective |
|---|---:|---:|---:|---|---:|---:|
| B1 ResNet18 | 1 | 3 | 62.5 | cpu-rich | 1 | 1.0 |
| B1 M5 | 1 | 3 | 62.5 | cpu-rich | 1 | 1.0 |
| B4/B3 ResNet18 | 1 | 6 | 62.5 | cpu-rich | 1.75 | 1.75 |
| B4/B3 M5 | 1 | 2 | 187.5 | memory-rich | 1.666667 | 1.666667 |

两作业各只有一个 GPU config，因此 B3 没有额外 GPU 决策空间。B4 重新装箱后已经达到与 B3 相同的最优解。这是重要的反例：联合 GPU 选择并非所有多资源场景都必要；若固定 GPU config 足够且 placement 可修复，B4 可以完全解释收益。

## 8. capacity-relative 诊断结果

| 场景 | B1 | B4 | B3 | 解释 |
|---|---:|---:|---:|---|
| A | 2.500000 | 2.500000 | 2.500000 | proportional 重归一把 fast 节点 CPU不足隐藏为 baseline；三者无差异。 |
| B | 2.500000 | 3.655963 | 4.100000 | 仍能看到 baseline 以上资源收益和弹性差异。 |
| C | 2.000000 | 3.083333 | 3.083333 | placement 收益存在，B4仍等于B3。 |

场景 A 直接证明两类模型回答不同问题。不能只报告 relative 结果并据此说 GPU-only 不受辅助资源短缺影响。

## 9. 正确性与审计

- 所有 selected columns 在提交前逐节点汇总 GPU/CPU/RAM；零超配。
- 每 job 最多一个 column；无法分配可以显式留空。
- B4 的 `fixed_gpu_configs` 与 B1 完全一致；场景 C 记录 `REPACK_FIXED_GPU_CONFIG`。
- 三个场景 fresh start，不触发 restart factor；checkpoint lock另有单测。
- MIP 不是唯一正确性依据：每个 policy-run 对所有 job 的“None 或一列”做笛卡尔穷举，最优 objective 与 MIP 相等。
- 完整结果保存 Oracle template/source/hash 和 normalization；没有手工 1.30/1.25 曲线。
- 结果提交是返回完整 `PolicyResult` 后一次完成；审计失败抛错，不返回部分解。

## 10. 与真实集群的差距

1. 三个场景的 raw Sia Goodput 是 synthetic Sia-like 输入，不是从当前 epoch 的真实 `SpeedupFunction` 提取。
2. Synergy 矩阵来自 V100/其数据栈的 `_1` profile；与 Saturn 应用/GPU 配置没有 A 级对应。
3. `cpu_demand_scale` 是场景假设；没有真实 GPU 类型×配置的 CPU/RAM saturation 测量。
4. 只验证单轮 Goodput，没有推进 epoch、JCT、makespan、resource-hours 或后续调度反馈。
5. 没有接入 Sia `p=-0.5`、`lambda_n`、CBC、projection、PMP 或真实 restart delay。
6. auxiliary quota/cache change 和 node remap 开销仍未建模。
7. prototype placement 枚举只覆盖 Sia 式单节点 partial/多节点 full-node；尚未做大规模有界 pattern generation。

## 11. 是否具备 20-job 条件

**尚未具备直接运行有意义的 20-job 端到端对照条件。** 已通过的是算法核心门槛：同一 Oracle、公平 B1/B4/B3、联合列 MIP、容量/锁/回退和独立最优性核验。

进入 20-job 前仍必须完成：

1. 把 `g_sia(j,k)` 接到 `SiaPolicy._compute_goodputs_v2()`/当前 Job state，而不是 JSON 常量。
2. 在 `Cluster.step()` 中以完整联合解原子提交，并让所有策略用 Oracle 的 actual Goodput 推进训练。
3. 恢复 Artifact 的归一化、`p`、`lambda_n`、restart/checkpoint和 rigid/PMP语义。
4. 将 fixed-hash 与 category-mix 的 20-job manifest、capacity 档和 sensitivity range 预注册并冻结。
5. 增加 neutral objective-equivalence、Sia placement 可行性和逐轮状态回归；明确何时只要求目标相同，何时能要求日志相同。
6. 为 20-job runner 增加内存门槛、每轮 timeout、solver gap、fallback 和完整日志。

建议下一步是“完整模拟器集成 smoke”而不是参数搜索：先接入 2-job 场景的 `Cluster.step()`，确认 progress/restart/JCT 语义，再批准固定 20-job 对照。
