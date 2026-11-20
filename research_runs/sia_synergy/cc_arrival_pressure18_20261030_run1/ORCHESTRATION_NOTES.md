# 编排与审计补充

初次启动前可用内存1.670GiB，未构造任何模拟器。短时复查恢复到2.058GiB后开始首个Sia有界gate并完成。离线分析曾在启动器中驻留NumPy/应用画像，下一方法前降至1.958GiB而停止；随后改为独立分析进程。另一次CC-CPU子进程在构造模拟器前测得1.999920GiB而退出，未创建case目录，日志保留。

最终启动器在既有2GiB强制门槛之上等待0.125GiB启动余量，以避免父/子进程启动瞬间的边界抖动；子进程仍独立检查原2GiB/5GiB门槛。完整Sia gate经output_hashes核验后复用，没有重复执行或重写完成日志。所有策略串行、独立Python进程，无用户程序被关闭、无内存门槛降低、无虚拟CPU/RAM/模型/算法参数改变。

正式18格启动时最小可用内存2.529GiB。暂停/等待只增加真实墙钟时间，不推进模拟时间。全部正式case完整，失败启动只发生于gate前资源检查，未补写中断JSONL。

gate_eta_boundary_recheck.json单独记录零时长epoch边界的ETA诊断复核；FULL正式门槛判定完全不变，主JCT数据不涉及此诊断公式。

初始delivery_hashes.json由审计子进程生成，其中gate_resume3_console.log在子进程返回后被父启动器正常追加ALL18_AUDITED_STOPPED，造成唯一活动控制台hash变化。所有源码/输入/原始结果hash一致。所有进程退出后再生成delivery_hashes_final.json，作为最终完整封存；不重写初始manifest。
