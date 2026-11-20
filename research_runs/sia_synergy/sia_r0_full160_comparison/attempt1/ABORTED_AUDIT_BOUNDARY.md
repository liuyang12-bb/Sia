# 实现审计中止（不作为性能结果）

2026-10-25：冻结 ef7266f，正式启动第一格 main_w1_neutral_sia_p 后，在执行器复核中发现 FullMetrics 的 checkpoint 检查读取了“新调度后的 rescale_time”，可能把本轮旧checkpoint已结束后新配置的延迟当作旧锁。它不改变训练，但可能错误拒绝合法轮次。

主动中断该批次执行会话，保留本目录所有文件。中断前尚未生成 rounds.jsonl.gz 或 summary，处于完整160作业初始化阶段。没有可用性能结果，不选择性保留JCT。

修复方案：新适配Cluster只增加调度前 checkpoint 剩余时间观测字段，审计以这个状态判断真锁；不改变 Job.step、Sia、L2、响应或任何参数。增加边界回归后在 attempt2 从固定矩阵第一格重启，四方法配对不混合版本。
