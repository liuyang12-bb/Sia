# 独立只读设计复核

本复核只读当前新实现与其父版本；未运行训练仿真、未修改算法。证据基于 `sia-joint-completion-time/sia-simulator/joint_prototype/completion_time.py` 及同树依赖。测试与诊断数值由主任务独立记录，本复核不把这些数值重新认定为完整JCT结果。

## 判定

在本轮已限定的 DP、小规模、固定epoch公开预算范围内，未发现需要通过修改算法或参数才能继续的明确实现错误。预测误差大于所声称的局部净收益、GPU耦合门槛未通过，都是应停止完整性能实验的证据，不能通过重复调整目标或模型来消除不利结果。

## 单位与信息合法性

- `goodput.py:64–86` 的模型Goodput为 `init_batch_size × gain / iteration_time`；`job.py:489–497` 的执行进度率为 `calibration × gain / iteration_time`。新代码 `predict_job():54–56` 的除初始batch并乘calibration正确，避免拿每作业归一化speedup除剩余progress。
- `predict_job():46–55` 浅复制Job后调用原 `update_local_bsz`，并使用当前已观测梯度/拟合吞吐模型评价该batch；没有调用真实未来梯度、未来吞吐或 `Job.step`。原 `update_local_bsz` 在当前路径仅写复制对象的batch/accum属性，不修改live Job训练状态。
- `declare_work():65–74` 必须按设计将 `get_progress(max_epochs)` 在提交时登记为公开的固定epoch训练预算。它来自Artifact工作量定义，不能被泛化为未知收敛时间估计。`target_metric != None` 已显式拒绝。
- `score_columns():92` 使用已声明预算减当前progress；不访问未来完成时间或未来到达事件。`completion_jobs` 仅包含当前到达未完成作业（132–133）。

## 目标与资源契约

列代价为 `d + remaining_work / predicted_execution_rate`，等待参考为 `300 + min(proportional resume ETA)`（77–127）。当服务集合固定时，共同等待常数抵消，因此资源交换确实比较receiver节省的完成秒数与donor损失的完成秒数，不只是加速百分比。

JO/JP在配额消融前使用共同的比例候选参考（85–101），当前checkpoint列走原完整保护（112–117及继承的候选/提交审计）；JP仍通过原 `b3_prop` 候选契约限制CPU/RAM比例点。SIA-ECT-AUX真实调用 `SiaPolicy.optimize` 并两次核验完整GPU意图未改变（173–185），没有加L2。

以下是模型近似，而非应“修到通过”的实现错误：

1. 等待参考使用的300秒是现有服务机会长度，不是该作业实际排队时间或等待上界；独立计算的最佳比例恢复配置并不保证届时资源可用。
2. 静态ETA假设当前梯度、吞吐和batch速度持续，忽略epoch改变及任务完成后的资源释放。其目标不是全局平均JCT最优。
3. 目前真实GPU变更资格与累计启动预算保留，可限制重启振荡；不构成统计置信区间或无误差承诺。
4. `predict_job()` 对 projected-only 或PMP配置返回未知ETA；GPU候选没有被删除，但这些配置只按比例bootstrap资格参与，不能宣称已实现其完成时间精确排名。PMP仍需独立单位/执行验证。
5. 若unknown配置与known配置并存，unknown效用为0；这是一项公开的保守未知处理规则，并不表示其预计完成时间为0。

## 门槛判断

主任务记录的当前模型与独立首段raw执行率，分别对同一个CPU交换给出相反的总ETA方向。两者都不是完整JCT，也不能用其中有利的一项取代另一项。当前证据只支持：CPU差异化确实影响真实训练进度，新的目标能显式计入donor完成时间损失；尚不支持该交换降低完整平均JCT，也尚未证明在选定状态下联合GPU选择优于固定GPU辅助优化。

因此建议按预注册机会门槛停止，不进入小型完整性能比较；交付已实现的可审查机制、测试和失败门槛证据。后续需先取得误差能支撑净收益方向的同状态当前速率证据，以及整个合法资源预算内的GPU—辅助资源耦合证据，不能根据JCT选择参数。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。
