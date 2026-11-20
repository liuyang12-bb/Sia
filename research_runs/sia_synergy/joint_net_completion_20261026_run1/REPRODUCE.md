# 实际源码与证据重建

本轮只修正固定参考服务集合内的配置目标，不调整数据、响应、容量或候选网格。
裸历史Git HEAD不是本轮版本。实际源码406个文件的逐文件SHA-256与组合fingerprint
在preregistration.json/source_changes.json；source.patch列出相对上一执行版本的最小差异。
运行时读取本目录source/，不从Git HEAD猜测或恢复缺失的未跟踪实现。

将complete_rebuildable_evidence.zip解压到全新的项目根，不覆盖任何旧研究目录。
新工作源码位于sia-joint-net-completion/sia-simulator；冻结执行副本位于本产物的source/。
inputs/包含与上一轮相同的六任务、合成容量、R0 registry、Synergy矩阵和相关画像。
包内保留上一版实际源码及历史测试fixture、官方只读源码副本、对照原始日志，避免只保存hash。
delivery_manifest.json记录压缩包和所有成员的hash，打包后逐成员解压读取校验。

dependencies.lock.txt及environment.json记录解释器、库和平台。测试在新源码目录运行：

```powershell
python -X utf8 -B -m unittest discover -s tests -p test_*.py -v
```

部分历史fixture包含原项目绝对路径。若恢复到另一个根，仅在**新恢复副本**重定位
`D:/科研启航/Sia`及反斜杠形式，不修改原档或科学参数。不宣称已在第二台机器复现。

本轮交付后不自动再跑。若另行授权独立复现，把experiment.py、audit_results.py、inputs/、
source/和preregistration.json复制到新空的同级产物目录，不复制cells、日志或execution_manifest。
冻结输入和脚本哈希应完整验证，然后运行：

```powershell
python -X utf8 -B experiment.py validate
python -X utf8 -B experiment.py batch
python -X utf8 -B audit_results.py
```

六策略按冻结顺序独立进程串行执行；逐格内存至少2GiB、磁盘至少5GiB。
每格最多6000轮、每轮60秒；不能为某策略单独延长或插补未完成JCT。
原始四个对照的逐轮配置/配额/进度/完成/restart回归参照
joint_completion_continuity_20261026_run2，完整原始日志保留在包中。
所有写出使用独占创建，拒绝覆盖。不要通过删除保护或清理历史产物重复执行。

红测是测试适配器调用旧reciprocal-ETA控制器的历史反例，不是改变执行响应。
新增净代价目标不构成全局JCT最优；同分解、模型误差及后续资源释放仍须如实报告。
实际辅助资源换配额沿用无额外CPU/RAM加载开销假设，原Synergy矩阵不等于目标硬件测量。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。
