# 当前授权与恢复入口

## 当前最高优先级：用户要求暂停（2026-10-25 21:02–21:04，Asia/Shanghai）

协调对话转达用户明确要求“先把实验停掉”。已向执行会话42607发送中断，返回退出码1；只读系统进程核验未发现Python/CBC/HiGHS遗留进程。`sia-104` 已核验为 `PAUSED`。此前自动跑完104格的授权现在暂停，**不得因任何旧续接提示恢复、启动下一格或重跑，必须先取得用户新的明确授权**。

attempt2已完成6/104格（均160/160）；第7格 `main_w1_medium_joint_p` 被用户中断，未形成完成summary。其压缩逐轮流保留78条完整记录（模拟时间4680秒、已完成15/160），因中断缺少gzip结束标记，原字节保留，不修补、不覆盖。其余97格未启动。已完成数据、部分日志、冻结源码与历史产物均保留。停止证据见 `attempt2/USER_PAUSE_20261025.md`。下文“运行中”“继续运行”等表述仅为暂停前历史记录，不是当前执行指令。

本任务：从已验收 Stage2C commit `141138d95fc51755e2f71c15f01964b4c20757da` 隔离实施完整 Sia 对照，自动验收并运行 **104 次完整160作业实验**。旧任务的8次/禁止160限制不适用于此任务。

新树 `D:/科研启航/Sia/sia-joint-full160`；分支 `exp/sia-r0-full160-comparison`。本目录保存所有新产物，不覆盖历史。官方 Sia/Synergy 和旧工作树只读。

四方法 SIA-P（真实官方Sia决策+比例）、SIA-TUNE（相同GPU意图+TUNE适配）、JOINT-P、JOINT-O（原JP/JO及L2）。不偷偷给Sia加L2，不调目标/响应/网格。固定R0/E-legacy、Neutral/Medium、seed_profiles(1,8)，60s，6000轮。完整workload1–10各160、首到达平移0。主容量GPU4/8/8、CPU12/24/24、RAM250/500/500；半/双CPU只用于trace1/5/10 Medium。80+24=104，独立进程串行，2GiB门槛，日志gzip。不得按JCT调参或选择最好重跑。

恢复顺序：核对本目录 setup/snapshot、源码和 tests 状态；通过94旧测试及新增验收、官方逐轮回归、160队列/服务/锁/容量/覆盖后冻结commit和104矩阵；运行并逐项记录，原始事件独立复算；中文报告。技术失败保留并对受影响完整配对重跑，不混版本；删失属于结果并继续其余格。

只在缺材料/权限或必须改变科学定义时请求用户。无此阻塞则继续执行，无需阶段批准。禁止推送/清理旧文件。最后停，不自动下一轮。

标注：SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。

当前：已创建新树并保存794文件字节快照；新增5文件，独立本地commit `ef7266f`。原94+新8=102测试通过（full2）。官方两作业14轮全部完成且逐轮状态精确相同（official_regression2）。三作业30轮×3容量×4方法的12个验收均通过（mixed2）。最初测试name写为cifar导致KeyError已修正为cifar10，失败日志保留；红测red2在旧树缺少新接口，不能称为算法反例。red1仅内存门槛未启动。

attempt1在首格160作业初始化时主动中断（未生成逐轮结果）：新审计器误用“决策后新checkpoint”判断旧锁，需改成直接观察“决策前剩余延迟”。证据见attempt1/ABORTED_AUDIT_BOUNDARY.md，原源码及外部工具精确hash快照均保留。没有性能选择或参数修改。

修复后 full3 共103项通过，官方回归复验为official_regression3。下一步新commit后 `freeze_run.py freeze attempt2`、`freeze_run.py run attempt2`，从第一格开始相同104格。运行器每格独立进程，流式gzip，支持仅验证完成前缀后恢复；部分失败目录绝不覆盖。恢复时先读 attempt2/execution_manifest.jsonl 与 failure.json，勿覆盖已完成格或退回旧8格任务。

运行中：attempt2 已冻结并启动，commit `2d5f9dbfcb7e7bfc2bfd9dc194b1838674a227c2`。当前工具执行会话ID=42607（仅本会话恢复提示，不是可复现参数）。第一格从2026-10-25 18:23:56启动，约18:28开始产生rounds日志；初始化逐个调用seed_profiles，无缓存/未来轨迹灌入。尚未产生任何正式汇总。不要因为初始化慢就重复启动另一个进程。

已创建当前对话的15分钟续接 `sia-104`（heartbeat，ACTIVE），仅用于完成这个104格任务、错误复核和最终交付。无重要变化保持安静，完成或遇到需改变科学定义的阻塞后停止此续接。不是新研究任务，不自动扩展范围。保持现有进程串行运行；不要因为一次检查读不到沙箱外进程就宣称进程已死或重复启动。必要时以执行会话、最新日志和只读进程检查交叉确认。

全部完成后：`audit_results.py attempt2` 从原始事件独立复算，生成raw_review；再中文报告和交付快照/hash。若遇到TUNE full-intent+checkpoint无法同时满足的语义冲突，保留证据，不自行改变核心定义。
