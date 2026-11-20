# CC-Joint 按需候选与局部联合搜索

独立开发副本以 `cc-joint-recovery-equivalent-v1` 为基础。研究输入、容量、CPU/内存响应、完成时间目标，以及原有准入、持续服务、checkpoint 和恢复资格继续沿用。历史结果与其冻结源码另存。`source_manifest.json` 属于原版本。用本文的验证脚本检查新增策略。

## 策略入口

用 `core80.build()` 选择方法。

- `FO` 使用原串行决策。第一阶段在比例候选选出作业和 GPU 配置，第二阶段调 CPU/内存与放置。
- `LAZY_FO` 将候选过滤前移。第一阶段只构造比例及 checkpoint 护持候选，并保留原方案比较所需的当前配置。第二阶段只展开第一阶段选中的作业与 GPU 配置。恢复预判用原响应缓存给备选放置计算乐观 ETA，再与当前几何的一条普通合法候选比较。多作业路径还可按其他作业当前放置排除 GPU 余量不足的备选。已证明原规则不会发放恢复许可时使用按需候选，其余轮次走原完整枚举。`recovery_precheck=False` 可复现上一版按需 FO。
- `LOCAL` 使用上一版按需 FO 作保底，然后尝试小范围改选。最多扩展三个关联作业、每作业两种 GPU 配置，最多生成 2400 个局部候选，经筛选后最多比较 512 个组合。沿用 FO 本轮已接纳集合。完整审计通过、原目标不降低且局部 ETA 之和下降时，原子提交替代方案。
- `FULL` 仍走原全量联合路径。

`LAZY_FO` 与 `LOCAL` 保留同一候选域的 ETA 归一化。计时从配置候选准备前开始。`LOCAL` 只使用 FO 得到后剩余的时间，默认目标预算 5 秒，可用 `search_budget_seconds` 设置。FO 仍沿用精确求解与原失败处理，若 FO 本身超时，只记录超过预算并跳过局部搜索。这个版本提供局部搜索预算与候选组合数上限。在线硬实时截止保证和模拟决策延迟计入 JCT 仍需独立实现和验证。

## 本地复查

用已有 Sia Python 环境运行小样本验证。程序默认取输入前六个作业，限制最大轮数，全程逐轮核对 FO 与 LAZY_FO 的分配、训练和状态。

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python -X utf8 -B verify_search.py --jobs 6 --rounds 150 --until-complete --include-old-lazy
python -X utf8 -B bench_search.py --jobs 6 --rounds 150 --until-complete --repeats 3 --warmup-decisions 5 --output recovery_timing.json
$env:PYTHONPATH=(Join-Path (Get-Location) 'sia-simulator')
python -X utf8 -B -m unittest -v test_local_search test_recovery_precheck
```

`verify_search.py` 支持 `--include-full` 纳入全量联合对照，`--search-budget-seconds` 验证预算不足时的回退，以及 `--output <路径>` 将本次结果写入新 JSON 文件。该脚本的性能数据是小样本诊断。它使用原仿真时钟，调度计算墙钟与 JCT 分开计。

`bench_search.py` 在独立进程里顺序运行原 FO、上一版按需 FO、带预判的按需 FO，轮转方法顺序，并报告 P95/P99 决策墙钟和阶段耗时。预判统计单独记录潜在触发、经过证明可跳过、实际进入全量枚举、许可发放和许可提交。计时覆盖共同入口的完整决策过程。

## 文件定位

- `core80.py` 装配策略、初始化决策计时。
- `factor_cache.py` 复用原候选评分和共同归一化。
- `lazy_search.py` 提供比例候选、按配置展开、资格过滤和预算统计。
- `recovery_precheck.py` 从当前轮资源和 ETA 证据判断是否可以跳过恢复候选搜索。
- `local_search.py` 选择关联作业，限定候选和组合后验收改选。
- `sia-simulator/joint_prototype/clean_ablation.py` 将上述行为接入实际准入、选择、审计和提交路径。
- `verify_search.py` 对照执行，输出候选、决策时间、回退和 JCT。
- `bench_search.py` 在独立进程重复执行对照并保留逐轮时序证据。
