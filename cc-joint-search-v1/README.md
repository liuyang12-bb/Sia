# CC-Joint 最新搜索优化版

本目录是 2026-11-20 同步的最新开发版本，以恢复等效版为基础，包含按需候选生成、保守恢复预判和可选局部联合搜索。策略构造入口为 core80.build()。

## 代码入口

- core80.py 装配 FO、LAZY_FO、LOCAL、FULL 及 Sia 对照。
- README_SEARCH.md 解释候选生成、预算和复现命令。
- sia-simulator/joint_prototype/clean_ablation.py 实现准入、选择、审计和提交。
- lazy_search.py 实现两阶段按需展开。
- recovery_precheck.py 实现恢复许可的保守排除判断。
- local_search.py 提供受限局部 GPU 与辅助资源联合搜索。

## 验证和实验

项目根目录可使用已有 Python 环境。

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
& ./.venv-sia-baseline/Scripts/python.exe -X utf8 -B ./cc-joint-search-v1/verify_search.py --jobs 6 --rounds 150 --until-complete --include-old-lazy
```

重复计时使用 bench_search.py。固定 20 作业 CPU 互补实验使用 run_cpu20.py，其显式节点容量为每节点 2 GPU、12 CPU 单位、250 GB RAM，所有方法使用同一容量。输出路径需为全新路径。

12 作业三次重复和 20 作业四单元的完整报告、代码及输入指纹、原始记录位于
../research_runs/sia_synergy/cc_search_validation_20261120/。

12 作业验证了等效提速。20 作业中按需 FO 相对 Sia 的平均 JCT 降低 46.6%，同时存在长任务退化。LOCAL 本轮接受改选次数为 0。响应为合成跨硬件映射，决策墙钟尚未计入 JCT。

## 版本溯源

source_manifest.json 来自继承的冻结基线。早期入口 run_cc_joint.py、cc_factory.py 和旧 manifest 驱动脚本继续用于历史复现，当前策略通过上述新入口构造。

同步校验记录在 ../maintenance/search_sync_20261120/。代码和两项实验的证据随本次 Git 提交上传。项目首页和导航列出各历史版本对应的实验。
