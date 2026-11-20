# 完整30格的重建与复核

本轮仅恢复原冻结矩阵的未启动后缀；未修改调度算法、候选网格、响应模型、任务或求解参数。
`../source/` 是完整执行源码与输入，`../preregistration.json` 固定282个文件的SHA-256和依赖版本。
`preflight.json` 固定恢复时原9格、部分报告与暂停证据的哈希，原文件仍保留。

## 只读复核

1. 检查预注册的源码、输入、GPU画像、Synergy模板和R0 registry哈希。
2. 各格的 `output_hashes.json` 核对 `start.json`、`rounds.jsonl.gz` 和 `summary.json`。
3. 新的最终审计在 `../completed_analysis_20261028_run1/`；上级旧 `REPORT.md` 仅代表原9格部分结果。
4. 独立审计从原始完成事件、训练片段和分配重算指标，不只读取summary。
5. 本次恢复不是对原9格重跑，执行顺序仍是预注册30格的原顺序。

## 在新的空目录重建（需要研究者另行决定何时执行）

以下说明不在当前工作区自动启动新的实验，也不覆盖当前目录。

选择一个全新的Windows工作目录 `<replica>`，还原如下结构：

```text
<replica>/cc-joint-resource-study/                  完整复制本轮source的内容
<replica>/maintenance/run_cc_resource_factorial_20261027.py
<replica>/maintenance/audit_cc_resource_factorial_20261027.py
<replica>/research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1/
    preregistration.json
    source/                                      再保存一份相同完整源码
```

不把旧 `cells/`、旧日志、execution manifest、部分/最终分析目录复制到新执行输出目录；它们作为历史证据另存。
原运行器默认拒绝已有格或日志，因此不能直接在当前历史目录运行 `batch`。
保持运行器文件名和字节不变，其ROOT通过脚本所在路径确定；无需修改算法或参数。

使用独立Python环境恢复 `source/dependencies.lock.txt`，还应核对 `preregistration.json` 包版本、
原 `environment`/测试日志及Python版本；不升级全局环境。所有画像与矩阵已在source内，无需下载新数据。

在新目录按原预注册启动命令：

```powershell
<独立环境Python> -X utf8 -B <replica>/maintenance/run_cc_resource_factorial_20261027.py batch
```

该命令保持原顺序、独立进程串行、60秒/6000轮以及2GiB内存/5GiB磁盘门槛。
`prepare` 不需要再次执行：完整源码和预注册清单已经还原；不能用重新prepare替代原冻结参数。
结果停止或删失均保留，不根据JCT选择更好结果。资源不足时先停止，不能减少作业数或降低门槛。

在这个全新重建目录可运行原独立审计器：

```powershell
<独立环境Python> -X utf8 -B <replica>/maintenance/audit_cc_resource_factorial_20261027.py
```

原审计器使用独占创建输出；已有报告时应使用单独输出版本，不能覆盖。
本次专用的 `resume_cc_resource_factorial_20261028.py` 只处理原9格后的单次续跑，
不应用于任意新前缀或含失败部分格的目录。

## 证据适用范围

同一20作业负载的CPU/RAM容量消融，不是多个独立workload。
官方矩阵数据之外的应用映射、R0缩放、插值/裁剪和与Sia执行率的结合为合成迁移假设。

**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**
