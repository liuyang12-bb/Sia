# 审计边界修复：不是修改训练模型

首次JOINT-G 8任务Medium在t720停止，cifar10-1的checkpoint在t710结束，原Sia剩余最后epoch所需约0.295秒，被`round()`取为0。最终进度增加15.977，raw速率54.089，处于0.5×raw=27.045的原始epoch取整误差内；完成时间710，观测存活时间50秒、checkpoint50秒、有效训练秒0。

旧审计把完成截断后的“观测50秒”误当成整个请求60秒，判断全周期都在checkpoint，产生假阳性。实际`Job.step(60)`在checkpoint之后还剩10秒允许执行；0秒残差在解锁时结算，并未在锁内训练。

用真实应用画像、真实batch/grad/吞吐计算近完成状态，不mock Goodput，直接加载未改动official-artifacts/sia-simulator/job.py：官方Job和实验Job的进度、epoch、completion、execution_time精确相同，均在710完成且execution_time不增加。红测试在旧审计上1失败/3通过。

最小修复仅在execution_audit.py：全checkpoint条件比较请求可执行时间而不是完成截断观测；新加segment不得早于checkpoint结束、不得越过请求终点、无剩余执行窗口不得结算进度。原epoch取整残差≤半秒速率、raw×rho单次、互斥时间/完成后不训练检查全部保留。四个测试包括真实官方复现、锁内伪造、全请求checkpoint伪造、过大残差拒绝；完整139项通过。

没有修改job.py、epoch、训练时长、rho、checkpoint时间、资源容量、目标或JOINT-G设计；`prepare_repair.py`逐文件确认除审计及新增测试外冻结源码不变，Job.py SHA保持原值。首次失败日志和v1完整源码快照保持；新目录attempt2重跑8任务Neutral/Medium完整配对，再按同一gate运行20任务两格。正式结论只使用attempt2版本。

此修复继承官方整数秒仿真的量化，而不是声称零物理时间能完成真实训练。资源小时/JCT均保持官方取整时间口径；残差事件单独计数。这也是模拟与真实训练之间的已知边界。
