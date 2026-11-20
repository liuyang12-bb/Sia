# Sia + Synergy 联合数据集设计

- 日期：2026-10-23
- 阶段：数据方案与生成器规范；本任务**不生成正式数据、不运行实验**
- 当前证据等级：Sia GPU profile 为 Artifact 实测/拟合输入；Synergy CPU×RAM 矩阵为其平台画像；二者的组合是 **synthetic cross-hardware transfer**，不是同平台实测。
- 强制结果标签：`SYNTHETIC JOINT-RESOURCE EVALUATION — NOT REAL SATURN PERFORMANCE`

## 1. 数据目标与禁止事项

联合查询必须回答：对同一作业 `j`、合法 Sia GPU 配置 `k` 和 CPU/RAM 点 `q`，给出原始 Sia Goodput、相对辅助资源倍率、逐节点资源需求和完整来源。

禁止：

- 用 Synergy V100 的绝对 iteration time 替换 Sia 训练时长；
- 把 ResNet18/ImageNet profile 标成 CIFAR10/ResNet18 实测；
- 把 Synergy `_1` profile 说成多 GPU 实测；
- 用策略运行结果反向调参或读取未来任务信息；
- 让不同策略使用不同 response function、硬件容量、arrival 或训练工作量；
- 将 synthetic CPU unit 写成真实 AWS vCPU、DGX core 或 RTX host core。

## 2. 原始数据源

### 2.1 Sia GPU 性能与训练工作量

来源 commit：`official-artifacts` 的 `aaf7a0de7b6cb5eab06ea020467ede5d2fd18e70`。

| 内容 | 路径/函数 | 用法 |
|---|---|---|
| GPU profile loader | `sia-simulator/applications.py:25-54` | 读取 validation、placement、scalability CSV。 |
| GPU 类型绑定 | `applications.py:342-375` | `aws/dgx-ext/rtx` 分别加载对应 suffix。 |
| 原始 trace | `sia-simulator/traces/<app>/placements-*.csv`, `scalability-*.csv`, `validation-*.csv` | 保留 hash；不直接把单行吞吐当最终 Goodput。 |
| Goodput | `goodput.py:55-113`, `job.py:240-252` | 使用原 Artifact performance + efficiency 查询。 |
| 异构/投影 | `sia.py:423-507` | 记录 direct 还是 projected；两类结果不能混为同证据。 |
| 合法 GPU 配置 | `sia.py:100-142,652-669` | 生成 `K_j`。 |
| 工作负载 | `sia-simulator/workloads/saturn/workload-{1,5,10}.csv` | 保持 name、application、submission_time、replicas、batch 和工作量。 |

生成器应实例化 Artifact 的 `Application/GoodputFunction/SpeedupFunction`，对每个调度状态查询 `g_gpu(j,k)`；不能只把 CSV 的 step time 做静态拼接，因为 batch 和统计效率会随作业状态变化。为可复核性，静态数据保存 profile key 和初始候选；运行时仍通过同一只读接口按当前 epoch/参数查询。

### 2.2 Synergy CPU×RAM 响应形状

来源 commit：`research_sources/Synergy` 的 `1bb2bed6edc9c5f953d0b7f0a40e4fdad053489c`。

- profile classes：`simulator/models/*_1.py`；活跃十模型列表在 `jobs/model_zoo.py:178-214`。
- 坐标：`jobs/job.py:96-97` 的 9 CPU 点和 5 RAM 点。
- loader：`jobs/model.py:169-183`；明确记录其强制 `_1` 行为。
- source proportional point：默认 8 GPU、24 CPU、500 GB RAM，所以每 GPU 为 `(3 CPU, 62.5 GB)`，该坐标在矩阵中存在。

对每个 source template `s`，先构造无量纲曲面：

```text
F_s(c,m) = tput_s(c,m) / tput_s(3,62.5)
```

只借用矩阵形状；`iter_time`、`speedup`、`placement_penalty` 不进入 Sia 工作量。若分母非正、矩阵缺值或 profile 未处于激活清单，模板校验失败。

### 2.3 模板分配不冒充应用映射

真实 A 级覆盖为 0/480，因此 synthetic 主数据不按应用名声称对应。为保持可复现，使用固定 manifest 中的 `template_assignment_seed`，按 `SHA256(workload_hash || job_name || seed)` 将 job 分层均匀分到通过校验的模板；字段写：

```text
template_assignment: deterministic_hash
semantic_model_mapping: none
profile_evidence: borrowed_synthetic_response
```

同一 job 在所有策略、压力档和重复运行中保持相同模板。可另建 `neutral` 模板 `F=1` 做退化测试，但不能为“让 B3 更好”挑选模板。

## 3. 合成跨 GPU 迁移模型

### 3.1 配置条件化需求

Synergy 原矩阵没有 GPU 类型维。为明确表达“更快 GPU 可能需要更强数据供给”，定义参考 GPU 配置 `k_ref(j)`：该 job 最小合法 GPU 数下，原始 Sia per-GPU Goodput 最低的 direct-profile 配置；若只有 projected 配置，则记录 `reference_is_projected=true`。

```text
s(j,k) = [g_gpu(j,k) / replicas(k)] /
         [g_gpu(j,k_ref) / replicas(k_ref)]

b(j,k) = local_batch(j,k) / local_batch(j,k_ref)

phi_cpu(j,k) = clip(s(j,k)^eta_cpu, 0.5, 2.0)
phi_mem(j,k) = clip(s(j,k)^eta_mem * b(j,k)^eta_batch, 0.75, 1.5)
```

预注册合成指数固定为：

```text
eta_cpu = 0.50
eta_mem = 0.25
eta_batch = 0.50
```

这些指数是机制假设，不是拟合结果；不在主实验后搜索。附加 sanity dataset 可把三者全设为 0，用于验证“响应与 GPU 配置无耦合”时算法是否退化，但不得混入主结果。

### 3.2 联合响应查询

对某压力档节点的 proportional per-GPU 点 `(c0_t,m0_t)` 和实际点 `(c,m)`，令资源倍率 `u=c/c0_t, v=m/m0_t`。将其映射回 Synergy source 坐标：

```text
c_src = 3.0  * u / phi_cpu(j,k)
m_src = 62.5 * v / phi_mem(j,k)
raw(j,k,c,m) = Interpolate(F_s, c_src, m_src)

rho(j,k,c,m) = raw(j,k,c,m) / raw(j,k,c0_t,m0_t)
```

由构造 `rho(j,k,c0_t,m0_t)=1`。`Interpolate` 使用矩形网格双线性插值；超出坐标范围时 clip 到边界，不外推。原矩阵若局部非单调，不擅自做 isotonic 修复；低于 baseline 的点可以存在，但联合调度候选只保留 `rho >= 1-1e-9`。资源超过矩阵边界后倍率保持饱和值。

这一定义有三个作用：保留 Synergy 曲线形状；让更快 GPU/更大 batch 的饱和需求可上移；通过同一配置 baseline 比值避免在 Sia Goodput 上重复计算绝对收益。

### 3.3 离散 auxiliary 候选

每个 `(j,k)` 使用 CPU/RAM multiplier 的笛卡尔积：

```text
L_cpu = L_mem = {0.50, 0.75, 1.00, 1.25, 1.50, 2.00}
c = quantize(c0_t * L_cpu, 0.25 synthetic-core)
m = quantize(m0_t * L_mem, 0.25 GB)
```

保留满足以下条件的点：单 job/node 不超过容量；多节点 per-GPU 比可一致；`rho>=1`；且不被另一个“CPU/RAM 都不多、rho 不低”的点支配。`(1,1)` baseline 永不被剪枝。允许低于 proportional 的资源点仅因为某些矩阵在该区域仍 `rho>=1`，这与 Synergy 公式 (5) 的性能保证一致，而不是资源保证。

## 4. 三档预注册资源压力

三档保持完全相同的 GPU 节点数/类型、workload arrival、训练工作量、GPU Goodput 和 response template；只改变 **synthetic CPU/RAM capacity**。CPU 单位定义为 `synthetic physical-core-equivalent`，不与真实 vCPU 混用。

基础比例来自 Synergy 官方模拟器 8 GPU : 24 CPU : 500 GB，仅作为合成锚点。节点 GPU 仍采用 Sia：6×4 `aws`、2×8 `dgx-ext`、3×8 `rtx`。

| 场景 | capacity factor | 每 GPU CPU/RAM | 4-GPU 节点 CPU/RAM | 8-GPU 节点 CPU/RAM | 解释 |
|---|---:|---:|---:|---:|---|
| low | 1.50 | 4.5 / 93.75 GB | 18 / 375 GB | 36 / 750 GB | 辅助资源相对宽松。 |
| medium | 1.00 | 3 / 62.5 GB | 12 / 250 GB | 24 / 500 GB | 复用 Synergy 比例。 |
| high | 0.75 | 2.25 / 46.875 GB | 9 / 187.5 GB | 18 / 375 GB | 辅助资源紧张。 |

每个场景的 proportional baseline 定义为该场景的 `C_v/G_v,M_v/G_v`，所以 B0/B1 始终有可解释 baseline。所有数值字段必须携带：

```yaml
resource_model_type: synthetic
capacity_source: synergy_ratio_scaled
is_real_host_spec: false
cpu_unit: synthetic_physical_core_equivalent
```

不通过改变 arrival rate 制造压力，避免与任务完成反馈混淆。若后续需要 CPU-only/RAM-only stress，应另行预注册，不能从主三档中挑有利结果。

## 5. 统一查询接口

建议接口：

```text
JointProfileOracle.query(job_state, gpu_config, aux_point, pressure_id)
  -> {
       raw_sia_goodput,
       reference_goodput,
       source_template_id,
       source_matrix_value,
       baseline_matrix_value,
       rho,
       effective_goodput,
       cpu_per_gpu,
       ram_gb_per_gpu,
       saturation_flags,
       gpu_profile_mode: direct|projected,
       provenance_id
     }
```

强制断言：

```text
abs(query(q0).rho - 1) <= 1e-9
effective_goodput == raw_sia_goodput * rho
同一个 simulation step 的 progress 只读取一次 effective_goodput
输入不含 future arrival/completion/true remaining runtime
```

## 6. 文件格式与目录设计

后续实现建议生成到独立目录，绝不写进官方仓库：

```text
research_data/joint/v1/
├── manifest.json
├── cluster/
│   ├── low.json
│   ├── medium.json
│   └── high.json
├── workloads/
│   ├── workload-1.csv
│   ├── workload-5.csv
│   └── workload-10.csv
├── gpu_profiles/
│   ├── sia_profile_index.json
│   └── initial_gpu_candidates.parquet
├── auxiliary_profiles/
│   ├── synergy_source_matrices.npz
│   ├── source_profile_index.json
│   └── job_template_assignments.csv
└── joint/
    ├── response_spec.json
    └── precomputed_response_points.parquet
```

### 6.1 `manifest.json`

```json
{
  "schema_version": "joint-profile-v1",
  "dataset_class": "synthetic_cross_hardware_transfer",
  "claim_scope": "mechanism_only",
  "sia_commit": "aaf7a0de7b6cb5eab06ea020467ede5d2fd18e70",
  "synergy_commit": "1bb2bed6edc9c5f953d0b7f0a40e4fdad053489c",
  "template_assignment_seed": 20261023,
  "response_exponents": {"cpu": 0.5, "memory_speed": 0.25, "memory_batch": 0.5},
  "input_sha256": {},
  "generator_commit": null
}
```

`generator_commit` 在未来实验分支提交后填入；未提交代码不得伪造 hash。

### 6.2 响应点字段

`precomputed_response_points.parquet` 每行至少包含：

```text
job_profile_id, app_label_for_trace_only, template_id
gpu_type, num_nodes, num_replicas, scale_unit, local_batch
pressure_id, cpu_per_gpu, ram_gb_per_gpu
sia_goodput, sia_goodput_source, g_ref
phi_cpu, phi_mem, source_cpu_coord, source_mem_coord
source_raw, source_baseline, rho, effective_goodput
is_baseline, is_saturated_cpu, is_saturated_mem
profile_evidence, semantic_model_mapping, provenance_id
```

运行时状态变化导致的 Goodput 不能全部静态预计算；Parquet 用于初始状态/一致性核验，oracle 仍按当前状态查询并把输入/输出写到每轮日志。

## 7. 生成脚本设计

本轮不创建脚本；后续按以下单向流水线实现：

1. `extract_sia_profiles.py`
   - 校验官方 commit 与工作区干净状态；
   - hash `applications.py`、trace CSV 和 workload；
   - 枚举合法 GPU configs，标 direct/projected；
   - 不写官方目录。
2. `extract_synergy_profiles.py`
   - 只读导入激活 `_1` classes；
   - 导出坐标、矩阵、source baseline 与原文件 hash；
   - 拒绝全空/非正 baseline；不导出 absolute iter time 为联合训练时长。
3. `build_joint_dataset.py`
   - 固定 seed 分配匿名模板；
   - 应用 `phi` 和 baseline normalization；
   - 生成三压力档候选与 provenance。
4. `validate_joint_dataset.py`
   - schema、hash、baseline、范围、Pareto、容量和跨策略一致性；
   - 输出 machine-readable validation report，任一错误返回非零。

命令必须把所有参数写入 manifest，不依赖未记录环境变量。生成过程确定性运行两次后，除 timestamp 外 hash 应一致。

## 8. 数据正确性与泄漏测试

至少覆盖：

1. 三条 workload 的行数、到达、工作量与官方输入 hash 一致。
2. Sia 每个候选符合 type/node/replica/scale-unit 规则。
3. direct 与 projected Goodput 可区分，缺 profile 不静默变 direct。
4. 每个 source matrix 坐标/形状为 9×5，source baseline 可精确定位。
5. 每个 `(j,k,pressure)` 的 q0 恰好存在一次且 `rho=1`。
6. `g_eff=g_gpu*rho`，模拟器不再次乘 `speedup`。
7. 插值只在矩形内；边界外 clip 且 saturation flag 为真。
8. 被保留候选 `rho>=1`；无 profile 时只生成 neutral baseline。
9. 多节点 job 每 GPU CPU/RAM 完全一致，量化后总量可复算。
10. 三压力档只有 CPU/RAM 容量不同，GPU/arrival/workload/Goodput/template assignment 相同。
11. 所有策略加载相同 manifest hash；运行时拒绝混合 dataset version。
12. query 输入序列化后不含未来 arrival、completion time 或真实剩余时长。
13. 同 seed 重建完全一致；改变 seed 只改变匿名模板分配。
14. neutral 数据中所有 q 的 `rho=1`，联合无辅助收益。
15. synthetic/borrowed/real 的 provenance 枚举严格校验，禁止空标签。

## 9. 对照策略的数据口径

- B0：读固定 GPU mapping 和 q0；不读取 q>q0 的收益。
- B1：读动态 Sia GPU Goodput和 q0；结果应等于原 Sia进度。
- B2：读固定 GPU mapping、同一 `Q/rho` 和节点容量；TUNE 只改变 auxiliary/placement。
- B3/消融：读完全相同的 `K/Q/rho/capacity`；消融通过屏蔽决策维度实现，不能换数据。

固定 GPU mapping 文件在 dataset 构建时生成，按 `JOINT_SCHEDULER_DESIGN.md` 的规则计算并 hash。所有策略保持每个 job 的训练总 progress target 不变。

## 10. 实验输出与可复核性

每次运行目录应包含：

```text
run_config.json             # policy/pressure/trace/solver/time limit
input_manifest.sha256
events.jsonl                # 每轮候选摘要、选择、placement、状态推进
node_usage.jsonl            # 每节点 GPU/CPU/RAM
solver.jsonl                # status, objective, bound, gap, time, fallback
jobs.csv                    # completion/JCT/resource-hours/restart/realloc
summary.json
stdout.log / stderr.log
```

`events.jsonl` 对每个运行 job 记录 `raw_sia_goodput,rho,effective_goodput,profile_id,baseline_id`，从而能证明收益来自何种资源重新分配。B3 还需记录“相对 B1 改变 GPU 类型/数量/placement”的决策原因摘要，以及 capacity slack。

## 11. 从 synthetic 到真实 profile 的升级路径

真正的性能论文结论至少需要对 Sia 五类应用，在三类真实节点上按候选 GPU config 采集：GPU 数/节点数、global/local batch、CPU 配额单位、RAM、cache/storage、throughput、方差、warmup 和 auxiliary reallocation overhead。最小优先级是 workload 覆盖最大的 CIFAR10/ResNet18，其次 DeepSpeech2、BERT、ImageNet、YOLOv3；但必须使用 Sia 相同数据集/loader，而非名称近似。

实测数据进入同一 schema 时：

- `dataset_class` 改为 `same_platform_measured`；
- `phi` transfer 全部关闭；
- `R(j,k,q)` 直接来自该 `(app,gpu_type,config,batch)` 的测量；
- baseline 指向产生 Sia GPU profile 的实际 CPU/RAM/cache 条件；
- synthetic 与 measured 结果分目录、分表报告，不能拼成一个平均数。

## 12. 数据方案判定

本方案可执行且足以研究“联合选择机制”是否在受控条件下产生与 S2 不同的决策；它不能证明真实 Saturn 性能。三档压力、模板分配、迁移公式和候选网格均在实现前固定，避免事后为 B3 调参。若主结果对匿名模板 seed 或合成指数高度敏感，应报告敏感性/不稳健，而不是选择最有利配置。
