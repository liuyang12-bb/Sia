# JOINT-R 规模与竞争度验证

最终状态COMPLETE：125 tests通过，8/20各6格均全部完成，validation_8/20及extra_audit均通过；所有子进程已退出。主报告SCALE_COMPETITION_REPORT.md，收尾CLOSEOUT.md，权威最终hash为delivery_manifest.json。不要依照下面的历史进度重新启动。Medium8：R对P改善10.19%、对AUX改善1.04%；20：对P改善5.75%、对AUX退化5.28%。无调参，104保持暂停，完成后停止。

用户授权：先8作业，正确性通过后20作业；SIA-P/SIA-AUX/JOINT-R，Neutral/Medium。104格继续暂停。
新树sia-joint-scale-check，分支exp/joint-r-scale-check，基线2d5f9db加原adaptive-service未提交8文件逐字节复制。仅新增显式20任务上限适配，不改变历史3任务入口。

预注册选择：workload-1，依次BERT/CIFAR10/DeepSpeech2，每类按原始行序循环取下一个，8为20的前缀。真实记录/工作量保留，arrival统一0作为受控竞争场景，原arrival/source index保存；这不是自然到达trace实验。种子seed_profiles(1,8)、main容量、R0/E-legacy、alpha0/.5、Q300/D250均不变。每格固定600轮×60秒，完成即停，不延长、不按结果调参。8任务六格均完成且审计通过才运行20任务六格。独立进程串行，内存2GiB/磁盘5GiB门槛。失败产物不覆盖。

流程：规模状态测试＋全部122旧测试→冻结输入/源码→8任务6格→原始事件审计→通过才20任务6格→审计/报告/停止。没有恢复104或160任务授权。

进展：125项全测通过。数据预检BERT仅6条，因此耗尽类跳过、不复制（规则在任何仿真前冻结）；preregistration_v2为唯一实际执行版本，v1保留了psutil Windows Path启动前修正。
8作业六格全部8/8，validation_8.json valid/all_complete均true；Medium SIA-P1626.625、AUX1476.25、R1460.875。20首个child在内存门槛处退出，目录尚未创建、未构造cluster；原日志n20_neutral_sia_p.log保留。只加载同一环境后可用2.1889GiB，确认无重叠进程并核验完成前缀hash，按retry20_plan.json进行唯一一次安全重试。当前session24558串行运行retry20.py，严禁重复启动。再遇门槛/错误则不第三次重试。20全部6格结束后analyze.py 20，report.py，最后补充结论并封存。104仍暂停。
