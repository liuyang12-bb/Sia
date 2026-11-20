# Sia + Synergy 联合响应模型审计

> 结论标签：**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**  
> 审计日期：2026-10-24  
> 实验分支：`exp/sia-synergy-response-audit`  
> 基准 HEAD：`aaf7a0de7b6cb5eab06ea020467ede5d2fd18e70`

## 1. 结论摘要

历史结果中的最大 `rho=4.298091799896853` 已从官方 Synergy 源矩阵逐项复算，数值完全一致。它对应：

- 作业：`cifar10-29`；
- Sia 候选：`dgx-ext:1:1`（后续轮次也出现 `dgx-ext:1:2`，两者每 GPU 条件相同）；
- 分配：每 GPU 8 synthetic CPU cores、125 GB RAM；
- 模板：`res18_1.py`，SHA-256 `4F6421A904476472DB0A2D62F9130B16CA776F3A8D677283C40C3DE457C7AC42`；
- synthetic demand scale：`phi_c=4`、`phi_m=2`；
- 分子 `F(2,62.5)=0.72`；
- 分母 `F(clip(0.75),31.25)=F(1,31.25)=0.1675161987041037`；
- `rho=0.72/0.1675161987041037=4.298091799896853`。

因此该值不是重复乘法或日志错误，但也不是官方测得的“DGX 上 CIFAR10 获得 4.3 倍加速”。官方数据只支持 `res18_1` 的离散 CPU×RAM 响应矩阵；把该矩阵映射到 Saturn 作业、异构 GPU 和 Sia Goodput，并用 GPU 每卡速率推导 `phi`，均属于本项目的合成假设。

## 2. 数据和实现路径

响应查询由 `joint_prototype/joint_profile.py:197` 的 `JointProfileOracle` 完成。官方 `*_1.py` 只通过 AST 读取常量字段，不执行模型文件。网格为：

- CPU：`[1,2,3,4,5,6,9,12,24]`；
- RAM（GB）：`[20.83,62.5,125,187.5,250]`；
- 源参考点：`(3 CPU, 62.5 GB)`。

`joint_prototype/joint_integration.py:99` 从 Sia 的真实 `JobInfo` 枚举合法 GPU 配置并调用 `_compute_goodputs_v2`。`joint_prototype/joint_integration.py:149-151` 再根据候选的每 GPU Goodput 生成 synthetic demand scale。查询结果现在同时记录：原矩阵分子、分母、原始比率、敏感性指数、请求坐标、clip 后坐标和 clip 标记（`joint_profile.py:71-102`）。

完整机器可读数据：

- `research_runs/sia_synergy/joint_response_audit/source_matrix_audit.json`
- `research_runs/sia_synergy/joint_response_audit/rho_4_2981_reproduction.json`
- `research_runs/sia_synergy/joint_response_audit/sensitivity_manifest.json`

## 3. 官方源矩阵范围

下表的 min/max 和位置均直接从 `research_sources/Synergy/simulator/models/*_1.py` 读取。`F(3,62.5)` 是统一源参考点；`F(9,62.5)` 只是便于横向检查的关键点，不代表所有模板的声明需求。

| 模板 | min（坐标） | max（坐标） | max/min | F(3,62.5) | F(9,62.5) |
|---|---:|---:|---:|---:|---:|
| alexnet_1 | 0.10 @ (1,20.83) | 2.94 @ (12,125) | 29.4 | 1.00 | 2.00 |
| bert_1 | 1.00 | 1.00 | 1.0 | 1.00 | 1.00 |
| deepspeech_1 | 0.10 @ (1,20.83) | 1.00 @ (3,62.5) | 10.0 | 1.00 | 1.00 |
| gnmt_1 | 1.00 | 1.00 | 1.0 | 1.00 | 1.00 |
| lstm_1 | 1.00 | 1.00 | 1.0 | 1.00 | 1.00 |
| m5_1 | 0.10 @ (1,20.83) | 2.50 @ (6,187.5) | 25.0 | 1.00 | 1.50 |
| mobilenet_1 | 0.10 @ (1,20.83) | 1.55 @ (6,62.5) | 15.5 | 1.00 | 1.55 |
| res18_1 | 0.10 @ (1,20.83) | 2.40 @ (9,62.5) | 24.0 | 1.00 | 2.40 |
| res50_1 | 0.10 @ (1,20.83) | 1.15 @ (4,62.5) | 11.5 | 1.00 | 1.15 |
| shufflenet_1 | 0.10 @ (1,20.83) | 2.89 @ (12,125) | 28.9 | 1.00 | 2.00 |
| transformer_1 | 1.00 | 1.00 | 1.0 | 1.00 | 1.00 |
| vae_1 | 1.00 | 1.00 | 1.0 | 1.00 | 1.00 |

全部 12 个矩阵的全局范围为 `[0.1,2.94]`，全局最大单矩阵比率为 29.4。`res18_1` 自身允许的极端矩阵比率为 24。因此 4.2981 没有超出源矩阵的纯数值比率范围；问题在于该比率的分母和 GPU 需求映射是否具有真实物理含义，而不是算术越界。

## 4. fixed-reference 模型逐步计算

设固定参考点 `(c0,m0)=(3,62.5)`，候选实际每 GPU 资源为 `(c,m)`，synthetic demand scale 为 `(phi_c,phi_m)`：

```text
x_c = c0 * (c/c0) / phi_c
x_m = m0 * (m/m0) / phi_m
b_c = c0 / phi_c
b_m = m0 / phi_m
raw_rho = F(clip(x_c), clip(x_m)) / F(clip(b_c), clip(b_m))
effective_goodput = normalized_sia_goodput * rho
```

最大值样例中：

```text
c=8, m=125, phi_c=4, phi_m=2
x_c = 3*(8/3)/4 = 2
x_m = 62.5*(125/62.5)/2 = 62.5
b_c = 3/4 = 0.75 -> clip 到 1
b_m = 62.5/2 = 31.25
F(2,62.5) = 0.72
F(1,31.25) = 0.1 + (31.25-20.83)/(62.5-20.83)*(0.37-0.1)
             = 0.1675161987041037
rho = 0.72 / 0.1675161987041037 = 4.298091799896853
effective_goodput = 2.896887026113962 * rho = 12.451086372168001
```

分子坐标没有 clip；分母 CPU 请求坐标 0.75 被 clip 到 1。clip 不会制造超出矩阵 min/max 的响应值，但会把网格外区域变成边界常数，改变网格外的延伸形状。这里的低分母主要来自：`phi_c=4` 将固定参考点推到 CPU 网格下界之外，同时 RAM=31.25 落在第一行的 0.10 与 0.37 之间。

## 5. phi_c / phi_m 的来源和边界

`phi` 不是 Synergy 官方画像字段。当前规则是：

```text
r = (candidate_raw_goodput / candidate_gpus)
    / min_all_candidates(raw_goodput / gpus)
phi_c = clip(r, 0.5, 4.0)
phi_m = clip(sqrt(r), 0.5, 2.0)
```

其意图是表达“每 GPU 更快的配置可能需要更多 feeder CPU 和内存带宽”，但上下界 `[0.5,4]`、`[0.5,2]` 不是官方实测校准。由于 `r` 以当轮候选集合中的最慢每卡速率为分母，它还会随候选集合、direct/projected 可用性和 GPU 类型变化。现阶段没有数据可以唯一确定更合理的 `phi` 范围。

适用边界：

1. 只可作为 synthetic mechanism validation；
2. 不可把 `phi` 解释为特定 AWS/DGX/RTX 主机的 CPU 或内存需求；
3. 网格外不外推，只 clip；若基准坐标被 clip，必须在日志中显示；
4. flat 模板的 `rho` 恒为 1；responsive 模板最多只能声称“继承了官方矩阵形状”，不能声称继承了跨 GPU 的绝对速度。

## 6. 官方证据与合成假设边界

| 项目 | 证据性质 |
|---|---|
| `*_1.py` 的 9×5 数值、声明 CPU/RAM/batch、文件 hash | Synergy 官方 artifact 数据 |
| Sia 候选 GPU 类型/数量、direct/projected Goodput | Sia 实际模拟器路径 |
| Saturn 作业映射到 `res18_1`/`m5_1` | 固定哈希 synthetic mapping |
| `phi_c/phi_m` 规则 | 本项目 synthetic assumption |
| fixed-reference `(3,62.5)` 跨模板、跨 GPU 使用 | 本项目 synthetic calibration |
| 双线性插值和边界 clip | 本项目查询实现 |
| `effective_goodput = Sia Goodput × rho` | 本项目联合执行模型 |
| 两任务 JCT 差异 | 该合成模型内的模拟结果，不是真实集群证据 |

## 7. 预注册响应强度

没有对 `rho` 做有利的硬截断。令源矩阵与当前映射给出的完整比率为 `rho_raw`，使用对数效应缩放：

```text
rho_alpha = exp(alpha * ln(rho_raw)) = rho_raw ** alpha
```

该定义同时压缩增益和减速，保持 `rho=1` 不动并保持排序。四档在查看本轮 JCT 前固定：

| 档位 | alpha | 含义 |
|---|---:|---|
| neutral | 0 | rho 恒为 1 |
| low | 0.25 | 完整 log-effect 的 25% |
| medium | 0.50 | 完整 log-effect 的 50% |
| high | 1.00 | 不衰减的原 synthetic transfer |

这些档位是模型不确定性分析，不是由真实集群置信区间校准出来的“低/中/高硬件收益”。

## 8. 两任务合法性运行

所有 12 个配置严格串行，使用相同 workload、容量、Sia Goodput、固定哈希和训练状态。

| 档位 | 策略 | 平均 JCT (s) | 执行 rho 范围 | rounds |
|---|---|---:|---:|---:|
| neutral | B1 | 673.5 | 1–1 | 96 |
| neutral | B4 | 673.5 | 1–1 | 96 |
| neutral | B3 | 674.0 | 1–1 | 96 |
| low | B1 | 707.5 | 0.9212–0.9332 | 96 |
| low | B4 | 706.0 | 0.9274–1.2156 | 96 |
| low | B3 | 755.5 | 0.9332–1.4399 | 97 |
| medium | B1 | 726.0 | 0.8485–0.8708 | 97 |
| medium | B4 | 724.0 | 0.8601–1.4776 | 97 |
| medium | B3 | 654.5 | 0.8708–2.0732 | 97 |
| high | B1 | 792.5 | 0.7200–0.7583 | 98 |
| high | B4 | 787.5 | 0.7397–2.1833 | 98 |
| high | B3 | 524.0 | 2.1833–4.2981 | 94 |

结果明显非单调：low 档 B3 比 B1 差 6.78%，medium 档才改善 9.85%，high 档改善 33.88%。原因包括 MIP 配置切换、离散完成时间和 restart 语义，不应从四档中事后选择 high 作为主结果。

## 9. 映射与 manifest

模式 A 沿用固定哈希：seed `1301`，候选模板 `res18_1.py,m5_1.py`，两个作业仍都映射到 `res18_1.py`。历史映射没有覆盖。

模式 B 已预注册但未用于这 12 次运行：50% responsive、50% flat；responsive 候选为 alexnet/m5/res18/shufflenet，flat 候选为 bert/gnmt/lstm/transformer/vae。它只是 synthetic category mix，不能按 JCT 临时换模板或比例。

`sensitivity_manifest.json` 冻结了 workload/profile hash、mapping seed、映射结果、phi 公式与范围、节点容量、reference point、normalization、响应档位、Git HEAD、dirty status、代码 hash 和完整命令。

## 10. 结论与未解决问题

计算路径现已达到“可审计”：最大 rho 可从原矩阵复算，坐标 clip 可见，四档参数已冻结，所有策略共享同一 Oracle。它仍未达到“可物理校准”：

1. `phi` 没有真实跨 GPU profile 支撑；
2. fixed reference 不等于 Sia 三类节点的已知硬件配置；
3. 官方矩阵主要呈 CPU 方向变化，不能替代真实内存带宽画像；
4. 两任务结果对档位和离散调度变化敏感；
5. PMP profile 拟合仍产生非致命 `autograd invalid value in log` 警告，需要在正式 PMP 性能实验前单独处理。

在“仅做 synthetic sensitivity、不宣称真实性能、四档全部报告”的前提下，可以进入 20-job smoke；不建议把 high 档单独作为性能主结论，也不建议在获得真实 profile 前开展参数搜索。
