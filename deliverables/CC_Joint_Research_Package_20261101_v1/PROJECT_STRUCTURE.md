# CC-Joint 核心代码与结构说明

本包冻结的是论文主版本v8的完整329文件，不是从裸HEAD重建，也没有删除看似旧名的真实继承依赖。主版本80作业报告、后期恢复等效12作业数字不能混用。

## 目录

```text
workspace/
  cc-joint-core80-full-lex-repair-v8/  329个原始源码/输入/测试文件
    core80.py                       构造七方法及共同执行模型
    run_core80.py                   单格正式运行器（不会自动运行）
    engineering_repair.py           v8正式求解上下文
    numeric_repair.py 等            等价求解与数值证明辅助
    inputs/                        冻结80作业、R0、矩阵和容量配置
    sia-simulator/
      sia.py, job.py, goodput.py, applications.py
      joint_prototype/             核心策略、响应、服务、锁、原子审计
      tests/                       原样保留的测试
provenance/                        原预注册、源码hash、输入hash和环境
fixtures/                          原external_test_dependencies.zip
tools/                             原验收脚本ZIP与离线hash核对脚本
STRUCTURE.md, REPRODUCTION.md       本包新增说明
```

## 调用关系

`run_core80.main → core80.build → make_cluster → install_completion / install_tail / install80 → Cluster.step → optimize_policy → 候选与服务约束 → 联合MIP → 原子提交 → Job.step → 完成事件与指标`。

正式v8会进入engineering_repair.repair_context，不能只调用旧run_cc_joint.py后声称复现了80作业结果。原源码README源自早期统一入口，保留原字节；运行语义以本说明、core80.build和原预注册为准。

## 模块

|模块|职责|
|---|---|
|completion_time|当前模型预测raw rate、声明剩余工作量和ETA|
|completion_continuity|完成导向目标、真实有效服务与连续性|
|clean_ablation|共享候选、队首与held强制集合、确定性择解|
|joint_integration / joint_scheduler|真实Sia候选、联合CPU/RAM/placement、容量与原子提交|
|joint_profile / r0_liveness|固定R0与Synergy矩阵查询|
|tail_recovery|单任务有界收尾恢复|
|job / applications / goodput / sia|实际训练和Sia异构/弹性/batch模型|
|engineering_repair及求解辅助|主值/启动/词典序的等价数值验收|

所有源码文件与历史source_hashes逐项一致。参考算法报告和evidence/algorithm_formalization.md可定位具体函数及行号。
