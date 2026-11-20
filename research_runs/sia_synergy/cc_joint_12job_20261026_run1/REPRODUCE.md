# 重建与审查

CC-Joint沿用joint_completion_continuity_20261026_run2的404文件实际版本，
指纹21CA3574775F3A10030473BD22183802CC5DED06D2ED57EDE35F3D1A332F5A57。
裸历史Git HEAD不能代表实际执行源码。新算法树为sia-joint-cc-12job/sia-simulator，
source/是逐字节相同冻结副本。原始Sia、Synergy、旧策略与旧结果没有覆盖。

输入取官方workload-1的CF0–5及DS8/9/12/16/21/30，批量到达0秒为合成压力场景；
训练参数和工作量沿用原记录。全部八格、顺序、6000轮上限和响应/容量在运行前冻结。
完整源trace、抽取CSV、原六任务、R0 registry、模板、容量配置在inputs/。

将complete_rebuildable_evidence.zip解压到新项目根，不能覆盖已有研究。
delivery_manifest.json记录包hash和逐成员hash，打包后独立读回校验。
包内包含历史测试fixture、只读官方源、父版本实际源、全部本轮原始日志。
dependencies.lock.txt/environment.json记录包版本/解释器；部分旧fixture有原根绝对路径，
只能在新恢复副本重定位D:/科研启航/Sia前缀，保留参数和旧原档。

测试须在项目根下新树sia-joint-cc-12job/sia-simulator执行，不能从更深的产物source目录
直接执行依赖parents[3]的历史测试。第一次路径失败保存在tests.log/tests.json，
布局恢复后的通过证据为tests_relocated.log/tests_relocated.json，算法源和测试均未改。

本目录已有独占输出，不能重复运行覆盖。若另行授权复现，复制脚本、inputs、source、
prepared.json及preregistration.json到全新同级输出目录，不复制cells或execution_manifest。
保持同目录层次及LIVE指向的新树，逐文件验证source/input/runner哈希，运行：

```powershell
python -X utf8 -B experiment.py validate
python -X utf8 -B experiment.py batch
python -X utf8 -B audit.py
```

每格独立Python进程，严格串行，启动内存>=2GiB、磁盘>=5GiB；不延长观察窗、
不插补删失JCT。audit从完成事件、预rho训练率与执行片段积分复算，
report给出所有作业与退化，不选择最好一次。不同平台的求解器同分解应如实记录。

本轮完成后停止，不自动扩大任务、调参或恢复104/160批次。
SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。
