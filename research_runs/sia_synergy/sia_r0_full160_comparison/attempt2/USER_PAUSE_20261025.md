# 用户要求暂停：现场保留记录

- 时间：2026-10-25 21:02–21:04，Asia/Shanghai。
- 原因：协调对话 `01a1719a-a726-7100-a163-0c806c8a4e60` 转达用户明确要求“先把实验停掉”。不是算法失败或按性能选择停止。
- 动作：向现有执行会话42607发送Ctrl-C，返回exit_code=1。未等待当前格完成，未启动替代实验。
- 核验：中断后以系统只读 `Get-CimInstance Win32_Process` 查询名称包含python/cbc/highs的进程，返回空；未发现批次或实验子进程残留。
- 自动续跑：`sia-104` 的配置已为PAUSED（由协调侧暂停，本侧只核验，未重复更新）。
- 恢复：需要用户新的明确授权。不得覆盖中断格或自行重跑；冻结的104格范围不构成当前继续授权。

## 保留状态

已完成6/104格：workload-1 Neutral下四策略，以及Medium SIA-P和SIA-TUNE，均160/160。第7格 `main_w1_medium_joint_p` 已中断，原始流可读78个完整轮次，最后模拟时间4680秒、完成15/160。该格没有summary，不视为6000轮删失，也不估计总体JCT。其余97格未启动。

中断的 `rounds.jsonl.gz` 缺少gzip结束标记（读取完整前缀后出现EOFError），原文件不修补、不覆盖。stdout压缩文件为空，仍保留。完整结果、原manifest、历史产物及源码未修改；只新增本记录并在CURRENT_TASK顶部标记暂停。

## 停止时SHA-256

| 对象 | SHA-256 |
|---|---|
| execution_manifest.jsonl | F5B8DEAC6BA612D44E0422CE18FFF2598DCD6EECE7DA870A525800377F557D77 |
| preregistration.json | 101D54B80B5C1D55CAA059F28D596481B62016DC4E6D920AB9F91208065D713E |
| main_w1_medium_joint_p/rounds.jsonl.gz | B4A19C80BB17647FCB683EDCFCBE125F8182878408232B0D13ED1720FFBC1CAB |
| main_w1_medium_joint_p/start.json | 4ACD9463A0083C79D00A34D907874473E09FD4870B3AF2D90895884527233856 |
| main_w1_medium_joint_p.stdout.log.gz | E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855 |

冻结源码commit仍为 `2d5f9dbfcb7e7bfc2bfd9dc194b1838674a227c2`。本次仅停止与现场记录，没有算法修改、调参或进一步训练仿真。
