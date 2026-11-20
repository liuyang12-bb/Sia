# 重建、审查与边界

本轮最终尝试位于joint_completion_continuity_20261026_run2，失败尝试在同级run1。
preregistration.json冻结执行源码、输入、包版本、七单元顺序及所有科学参数；source_changes.json
描述相对上一轮实际源码的变更，source.patch可审查。裸Git HEAD不是实际执行版本。

重建时将complete_rebuildable_evidence.zip解压到一个全新的项目根目录，不能覆盖已有研究。
源码在sia-joint-completion-continuity/sia-simulator，冻结运行副本在本产物目录source/。
inputs/包括相同六任务、容量配置、R0 registry和Synergy模型。dependencies.lock.txt记录Python包，
environment.json记录解释器与运行平台。CPU单位为合成单位，不能作为真实机器核性能外推。

正确性测试：在新树sia-simulator目录用Python3.10兼容环境执行
`python -X utf8 -B -m unittest discover -s tests -p test_*.py -v`。
部分历史测试fixture的preregistration.json包含原项目根绝对路径；仅在新恢复副本把前缀
`D:/科研启航/Sia`或对应反斜杠路径重定位到新根，保留所有参数与源文件内容，不修改原档。
包内包含所需历史fixture和官方只读源码副本。未声称在第二台硬件复现性能。

本轮不自动再启动性能实验。经另行授权复跑时，将experiment.py、audit_results.py、inputs/、
source/及冻结preregistration.json复制到**新空的同级产物目录**。不要复制cells/或execution_manifest。
`python -X utf8 -B experiment.py validate`仅构造验收，之后
`python -X utf8 -B experiment.py batch`按冻结7单元、独立进程串行运行。
`python -X utf8 -B audit_results.py`复算原始完成/训练片段。
当前目录已非空，重复执行cell/batch/audit的独占创建保护会拒绝覆盖；不得移除保护。

三个历史基线轨迹回归读取joint_ect_six_job_20261026_run1的原始轮日志；包中保留这些参照日志。
相同版本/参数在不同求解器平台可能选择并列解，应保留版本信息并报告差异，不选最好一次。

NUMERICAL_REPAIR.md解释run1停止及run2的近整数规范化，仅在完整原约束复核合法时接受，
不改变评分/网格/资源/响应/求解器参数。run1的失败快照、原冻结源码及部分日志全部保留。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。
