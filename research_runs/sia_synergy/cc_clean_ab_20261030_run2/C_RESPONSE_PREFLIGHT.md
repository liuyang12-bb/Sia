# C组：冻结响应的离线数据预检

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE

本预检不运行训练。枚举三应用、AWS/DGX各1/2GPU、两种既有容量、P/CPU/MEM/FULL配额域，共96行、3286个物理可行查询点。实际A使用CIFAR/DeepSpeech、B使用CIFAR/BERT；其余是预检超集，不伪称执行负载。所有registry源行能在冻结GPU画像中按内容哈希解析；旧绝对路径失效不等于源内容缺失。

## 不变的执行定义

rho=[M(cpu/phi_c,ram/phi_m)/M(3/phi_c,62.5/phi_m)]^alpha。M为官方矩阵双线性插值，越界按既有E-legacy端点裁剪；alpha A=0、B=0.5。R0是固定离线吞吐参考产生的合成跨硬件需求缩放，不给在线Goodput注入未来信息。缺registry或源内容拒绝，不补phi=1。

离线supported采用1e-12端点分析容差以单列浮点端点误差，不改变执行clipping规则。分子与分母、CPU与RAM分别计数，不能加总重叠比例。Neutral在数学上恒rho=1，因此其坐标越界不构成性能加速，但仍保留记录。

## 实际应用配置域的比例参考覆盖

|组|应用|GPU配置|比例CPU/RAM每GPU|分子CPU越界|分子RAM越界|分母CPU越界|分母RAM越界|域内|
|---|---|---|---|---|---|---|---|---|
|A|cifar10|aws:1:1|3/93.75|0|0|0|0|1|
|A|cifar10|aws:1:2|3/93.75|0|0|0|0|1|
|A|cifar10|dgx-ext:1:1|3/93.75|0|0|0|0|1|
|A|cifar10|dgx-ext:1:2|3/93.75|1|0|1|0|0|
|A|deepspeech2|aws:1:1|3/93.75|0|0|0|0|1|
|A|deepspeech2|aws:1:2|3/93.75|0|0|0|0|1|
|A|deepspeech2|dgx-ext:1:1|3/93.75|0|0|0|0|1|
|A|deepspeech2|dgx-ext:1:2|3/93.75|0|0|0|0|1|
|B|cifar10|aws:1:1|3/62.5|0|0|0|0|1|
|B|cifar10|aws:1:2|3/62.5|0|0|0|0|1|
|B|cifar10|dgx-ext:1:1|3/62.5|0|0|0|0|1|
|B|cifar10|dgx-ext:1:2|3/62.5|1|0|1|0|0|
|B|bert|aws:1:1|3/62.5|0|0|0|0|1|
|B|bert|aws:1:2|3/62.5|0|0|0|0|1|
|B|bert|dgx-ext:1:1|3/62.5|1|0|1|0|0|
|B|bert|dgx-ext:1:2|3/62.5|1|0|1|0|0|

A相关8种app/config中7种比例点域内；B相关8种中5种域内。特别是CIFAR DGX双GPU及BERT DGX配置的既有参考缺口保留。严格禁止越界将使对应配置的所有查询失去分母支持，不是只删掉一个资源网格点。不能把它们静默排除后仍声称完整异构配置域。

## 官方矩阵与合成假设

|应用映射|矩阵min/max|平坦|说明|
|---|---|---|---|
|cifar10 → alexnet_1|0.1/2.94|False|矩阵来自官方Synergy；应用关联和phi为合成迁移|
|deepspeech2 → deepspeech_1|0.1/1.0|False|矩阵来自官方Synergy；应用关联和phi为合成迁移|
|bert → bert_1|1.0/1.0|True|矩阵来自官方Synergy；应用关联和phi为合成迁移|

三矩阵在网格上未发现CPU/RAM增加引起响应下降。但单调不等于边际收益全程递减；本次不更改矩阵以强制凹性。BERT完全平坦，是理想化donor；因此需要用非平坦donor检验稳健性，不能把借用标签称为真实BERT资源不敏感测量。

## 同节点静态互补证据

以下是两任务各1GPU、同一容量下的归一化响应净变化；不是总工作吞吐之和，也不是JCT节省。每项包含receiver提升与donor损失，完整点见C_pair_witnesses.json。

|容量|GPU|receiver/donor|全域内|receiver CPU/RAM|donor CPU/RAM|归一化响应净变化|donor损失|
|---|---|---|---|---|---|---|---|
|A|aws|cifar10/deepspeech2|True|3.0912/93.4687|2.9088/75.3741|0.0014|0.0158|
|A|aws|cifar10/deepspeech2|False|3.0912/93.4687|2.9088/75.3741|0.0014|0.0158|
|A|dgx-ext|cifar10/deepspeech2|True|3/103.281|3/79.0719|0.0524|0.0000|
|A|dgx-ext|cifar10/deepspeech2|False|3/103.281|3/79.0719|0.0524|0.0000|
|A|aws|deepspeech2/cifar10|False|4.3632/75.3741|1/93.4687|0.0538|0.1577|
|A|dgx-ext|deepspeech2/cifar10|True|3.20121/79.0719|2.79879/103.281|0.0461|0.0000|
|A|dgx-ext|deepspeech2/cifar10|False|4.80182/79.0719|1/103.281|0.2706|0.0048|
|B|aws|cifar10/bert|True|4.32359/93.4687|1.67641/26.9699|0.5567|0.0000|
|B|aws|cifar10/bert|False|4.5456/93.4687|1/20.83|0.5991|0.0000|
|B|dgx-ext|cifar10/bert|False|4.39939/103.281|1/20.83|0.6287|0.0000|

## 判断与研究者后续数据决策

允许本次固定E-legacy的干净算法消融，结论严格限于该合成模型。预检未校准目标硬件，也没有修复三个参考缺口。下一阶段若要提升外部可信度，应选择：补实际联合profile；或预注册明确的保守边界/覆盖范围，并在完整配对版本中验证。不得根据JCT挑选phi或边界，不默认phi1，不以静默删除合法配置代替数据闭合。

本次C组交付的是覆盖与边界预检，不是新响应模型实验。源矩阵、registry、查询器和候选生成与原版哈希一致，见C_reuse_verification.json。