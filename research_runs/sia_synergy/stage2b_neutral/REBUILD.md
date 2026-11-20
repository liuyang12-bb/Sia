# Stage2B 重建与证据

基线216573cb26f3136520ce9d4f609623b4c2c75df1；新commit bd6105a37c6662336236a135eb7ad610ca6e16de。

所有恢复必须使用新目录；不得覆盖Stage1.1、旧run1/run2或当前证据。

1. 在独立位置从stage2b.bundle克隆并检出上述新commit；bundle含可达完整历史。
2. 核验stage2b_source_snapshot.zip的hash，并把实际字节文件恢复到该新checkout，按照repair_manifest.json逐文件核验791个文件。Windows CRLF可能导致stat-only差异；勿用reset清除快照字节。
3. 官方Synergy模型需在与checkout平级的research_sources/Synergy/simulator/models目录。synergy_matrix_snapshot.zip保存所需*_1.py原始文件，按g0_verified/template_summary.json核验；不需要部署集群。
4. 用独立Python3.10环境按requirements.lock.txt准备依赖。未保存离线wheel或操作系统；不能保证任意平台二进制求解行为相同。
5. 在sia-simulator目录执行独立日志记录的`python -B -m unittest discover -s tests -p test_*.py -v`。88项套件只含既有有界测试及新增≤3作业机制fixture；不得调用cohort运行器。测试证据环境变量STAGE1_EVIDENCE_DIR、STAGE2B_EVIDENCE_DIR必须指向新的空目录。
6. 离线复核：`python -B stage2b_preflight.py --output <新的目录> --models <模型目录>`。它不构造训练Job/Cluster，不将registry接入执行。

参数：D_bar250、Q300、Δ60、rho1；L2四单元为u_fp/u_fo/u_jp/u_jo。普通JointSiaPolicy保持历史行为；新模式通过独立NeutralL2Policy/NeutralL2Cluster使用。不能只用旧JointCluster而漏掉执行观测hook。Stage2B未提供正式实验CLI。

base_snapshot.zip/base.bundle为原始Stage1.1；source与protected完整清单在snapshot_manifest.json。红测、green1、boundary_red、green_full中的失败均保留；final_tests为通过的开发验收，acceptance_tests为提交后的验收并附执行源码hash。完整状态机推导在L2_STATE_MACHINE_PROOF.md。

数据：g0_verified是完整离线预检；初始g0_initial/g0_final亦保留。E-strict丢失全部可执行比例点的结论不允许用删除作业或默认phi=1绕过。E-linear/legacy仅离线分析。
