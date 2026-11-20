# Sia–Synergy 画像与资源覆盖审计

- 日期：2026-10-23
- 判定：**不通过正式跨论文性能实验门槛**
- 直接原因：三条预注册 Saturn traces 的 480 个 jobs 中，没有任何 job 同时具备“同应用/数据流水线、同 batch 语义、同 GPU 类型与 GPU 配置、同缓存/存储假设”的 Synergy CPU×RAM 响应曲线。不能用现有数据可信地构造 S2/S3 的训练进度倍率。

## 1. Saturn 工作负载覆盖

三条固定 trace 均存在且未修改：`workload-1.csv`、`workload-5.csv`、`workload-10.csv`，每条 160 jobs。

| 应用 | workload-1 | workload-5 | workload-10 | 合计 | Saturn batch size | trace 中 replicas |
|---|---:|---:|---:|---:|---|---|
| CIFAR10 | 134 | 137 | 131 | 402 | 2048, 4096 | 4, 6, 8, 10, 12, 16 |
| DeepSpeech2 | 10 | 8 | 11 | 29 | 320, 640 | 6, 8, 12 |
| BERT | 6 | 4 | 7 | 17 | 384 | 6, 8, 12, 16 |
| ImageNet | 3 | 4 | 6 | 13 | 1600, 3200 | 10, 12, 14 |
| YOLOv3 | 7 | 7 | 5 | 19 | 64 | 4 |

这里的 Saturn batch size 是 Sia workload 的全局 batch 配置；Synergy profile class 中的 `batch` 字段不能未经验证地当成同一含义或直接对齐。

## 2. Sia 应用的实际语义

| Sia 应用 | 仓库证据 | 实际工作负载 |
|---|---|---|
| CIFAR10 | `adaptdl/benchmark/models/cifar10/adaptdljob.yaml:24` | ResNet18，CIFAR-10；不是 ImageNet ResNet18。 |
| DeepSpeech2 | `.../deepspeech2/adaptdljob.yaml:25-28` | DeepSpeech2，batch 20 的启动配置，训练清单明确含 `voxforge_cmu_us`。 |
| BERT | `.../bert/adaptdljob.yaml:25-31` | `bert-base-uncased`，SQuAD 1.1，per-GPU initial batch 12。 |
| ImageNet | `.../imagenet/adaptdljob.yaml:24-29` | ResNet50，ImageNet，initial batch 200，随后允许自适应 batch。 |
| YOLOv3 | `.../yolov3/adaptdljob.yaml:24` 与 README:4 | YOLOv3，PASCAL VOC。 |

Sia 的应用 profile 由 `sia-simulator/applications.py:25-66,342-375` 载入不同 GPU cluster 的 placement/scalability CSV；Goodput 根据 batch、节点数、replicas、统计效率及吞吐模型计算，而不是一个独立于 GPU 的固定迭代时间。

## 3. Synergy 画像库存

论文 Table 4（PDF 第 10 页，论文页 587）包含：

- ImageNet：ShuffleNetv2、AlexNet、ResNet18、MobileNetv2、ResNet50；
- WMT16：GNMT；
- Wikitext-2：LSTM；
- Wikitext-103：Transformer-XL；
- Free Music：M5；
- LibriSpeech：DeepSpeech。

当前源码的激活 ModelZoo 与此十模型集合一致，位置为 `simulator/jobs/model_zoo.py:178-214`。`bert_1.py` 虽存在，但 BERT 在 `model_zoo.py:183` 只出现在注释候选中，且其 9×5 matrix 全为 1；论文 Table 4 和默认实验均没有 BERT。

硬编码 `_1` profiles 的主要字段如下：

| Profile | CPU/GPU | RAM/GPU (GB) | storage/GPU (MB/s) | profile batch | baseline iter (s) | matrix 最大值 |
|---|---:|---:|---:|---:|---:|---:|
| AlexNet | 12 | 125 | 125 | 512 | 0.600 | 2.94 |
| ResNet18 | 9 | 62.5 | 62.5 | 512 | 0.640 | 2.40 |
| ResNet50 | 4 | 62.5 | 62.5 | 512 | 0.780 | 1.15 |
| MobileNet | 6 | 62.5 | 62.5 | 512 | 0.710 | 1.55 |
| ShuffleNet | 12 | 125 | 125 | 512 | 0.615 | 2.89 |
| GNMT | 1 | 20 | 20 | 128 | 0.120 | 1.00 |
| Transformer-XL | 1 | 20 | 20 | 128 | 0.670 | 1.00 |
| LSTM | 1 | 20 | 20 | 20 | 0.011 | 1.00 |
| M5 | 6 | 187.5 | 187.5 | 512 | 0.600 | 2.50 |
| DeepSpeech | 3 | 62.5 | 62.5 | 20 | 0.757 | 1.00 |

每个 profile 的 throughput matrix 是 9×5，相应坐标在 `jobs/job.py:96-97`。matrix 是相对于资源配置的 normalized throughput/progress，不含明确的 GPU-type 字段；V100 和 24 CPU/500 GB/8 GPU 的实验平台来自论文 §5.1，而不是 profile 文件自身的可机读元数据。

仓库虽含部分 `_2.py`、`_4.py` profile，运行路径 `jobs/model.py:169-183` 却明确使用 `model_name + "_1"`；`gpu_demand` 选择代码被注释。因此当前官方 simulator 的多 GPU job 并没有可靠地查询相应 GPU 数画像。

## 4. 应用映射表

证据等级定义：

- A：模型、数据集、pipeline、batch、GPU 类型与配置均一致；
- B：模型/数据集基本相同，但硬件或 pipeline/batch 不一致，只能作为重新画像的候选；
- C：只有模型家族或任务类别相近；
- D：没有可用对应画像。

| Sia 应用 | Synergy 候选 | CPU Profile | 内存 Profile | 同 GPU 类型 | 证据等级 | 结论 |
|---|---|---|---|---|---|---|
| CIFAR10 / ResNet18 | ResNet18 / ImageNet | 有 9×5 | 有 9×5，依赖 ImageNet 缓存 | 否；V100 vs T4/A100/2080Ti | C | 同网络名不足以迁移。数据集大小、解码/增强和缓存敏感性完全不同；402/480 jobs 不能套用。 |
| DeepSpeech2 / VoxForge+CMU | DeepSpeech / LibriSpeech | 有 9×5 | 有 9×5 | 否 | C | 实现家族接近且 profile batch 同为 20，但数据清单不同，输入长度/缓存/I/O 分布不等价。 |
| BERT-base / SQuAD | `bert_1.py` | 矩阵全 1 | 矩阵全 1 | 否 | D | 未进入论文 Table 4 或激活 ModelZoo；不能视为经评测的 BERT profile。 |
| ImageNet / ResNet50 | ResNet50 / ImageNet | 有 9×5 | 有 9×5 | 否 | B | 模型与数据集匹配，但 Sia 使用 T4/A100/2080Ti、不同 batch/loader；仅是重新画像候选，不能迁移数值。 |
| YOLOv3 / VOC | 无 | 无 | 无 | 否 | D | 官方 Synergy 模型集合无 YOLOv3。 |

严格可迁移的 A 级覆盖为 **0/480（0%）**。最接近的 B 级只有 ImageNet/ResNet50 **13/480（2.71%）**，仍不能用于性能倍率。C/D 级不能补造曲线。

## 5. GPU 类型与集群资源清单

### 5.1 Sia 仓库明确给出的内容

`sia-simulator/simulator_config.py:3-15` 明确：

| Sia cluster | 节点数 | GPU/节点 | GPU/系统说明 | 仓库中 CPU/RAM 容量 |
|---|---:|---:|---|---|
| `aws` | 6 | 4 | AWS `g4dn.12xlarge`，NVIDIA T4 | 未进入 simulator capacity model |
| `dgx-ext` | 2 | 8 | DGX-A100，A100 40 GB | 未进入 simulator capacity model |
| `rtx` | 3 | 8 | 8× RTX 2080 Ti on-prem | 未给出 host SKU、CPU 或 RAM |

总计 64 GPUs。Sia 的 `NodeInfo` 可容纳通用 resource dict，但当前构造只写 `nvidia.com/gpu`（`utils.py:77-84`，`cluster.py:505`）。策略容量约束也只聚合各 GPU 类型数量（`sia.py:983-1008`）。它已有 job 到 GPU 类型、节点 ID 的映射，但没有 CPU/RAM request、占用和跨节点辅助资源分布状态。

### 5.2 可靠外部规格能补充、但不能自动成为模拟真值的内容

- AWS 官方规格给 `g4dn.12xlarge`：4×T4、48 vCPU/24 physical cores、192 GiB RAM。
- NVIDIA 官方 DGX A100 规格给 8×A100、128 CPU cores；系统内存存在 1 TB base / 2 TB 配置差异。Sia 仓库没有说明实际机器采用哪一内存配置。
- `rtx` 只描述 8×RTX 2080 Ti on-prem，没有足以唯一定位 host CPU/RAM 的型号。

因此即使采用外部资料，三类节点的完整、同口径资源容量仍不闭合。尤其 CPU 的“physical core”与 cloud vCPU 不能静默混用。

### 5.3 必须标作 synthetic 的内容

下列参数若用于实验都只能是显式 synthetic assumption：

- RTX 节点 CPU/RAM；
- DGX-A100 究竟 1 TB 还是 2 TB；
- 各 GPU 类型的 CPU/RAM allocation granularity；
- storage bandwidth、MinIO cache 或等价缓存；
- Sia 每个应用、每种 GPU、每个合法 GPU 配置下的 CPU×RAM response surface。

训练电脑的 8 physical/16 logical CPUs、13.69 GiB RAM 仅用于判断本机能否运行 simulator；没有被当作模拟集群配置。

## 6. Goodput 重复计算风险

Sia `_compute_goodputs_v2()`（`sia.py:423-516`）使用按 GPU 类型拟合/投影的 performance model 和统计效率得到 Goodput。它的原始 placement/scalability 采样是在某个实际 CPU、内存和数据 pipeline 条件下测得的，因此已隐含该基准机器的辅助资源性能。

只有满足以下定义时，才可额外加入 Synergy 风格倍率：

```text
effective_goodput = sia_goodput(config) *
                    R(app, gpu_type, config, cpu, memory) /
                    R(app, gpu_type, config, baseline_cpu, baseline_memory)
```

其中分母必须对应产生该条 Sia throughput profile 的同一硬件/软件基准。若直接乘 Synergy V100 的 `matrix[c,m]` 或其 `speedup` 字段：

- 会把 Synergy 的绝对/归一化基准误当成 Sia 基准；
- 会跨 GPU、跨数据集、跨 batch 和跨 loader；
- “充分资源”时可能在 Sia 已测吞吐上再次施加最高 speedup，构成重复收益。

现有数据无法确定上述分母，因此性能缩放一致性测试 1–6 无法成立。

## 7. 停止条件与结论

已触发任务定义的停止条件：

1. **CPU/内存性能模型缺少可追溯来源**：Sia 五应用 × 三 GPU 类型 × 合法配置没有对应曲线。
2. **模型映射不清/不完整**：YOLOv3 缺失；BERT 未评测；CIFAR10 与 DeepSpeech 数据流水线不匹配。
3. **CPU/内存容量不能完整成立**：RTX host 未知，DGX RAM 配置不确定。
4. **Goodput 存在重复计数风险**：缺少同基准归一化点。

所以不得实现 S2/S3，也不得为了完成 smoke 随意补造 synthetic curve。S1 虽可形式化为 GPU-proportional accounting，但在缺少完整节点容量且没有可信 auxiliary-resource progress model 时，只会成为任意合成环境；按照“若 S1 尚未验证不得运行 S2/S3”以及正式实验停止条件，本轮不修改 Sia 调度器。
