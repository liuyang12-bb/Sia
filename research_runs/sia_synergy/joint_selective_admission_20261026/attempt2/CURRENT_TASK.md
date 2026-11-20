# JOINT-S 冻结验证恢复

最终状态COMPLETE：10/10格均完成全部作业，原始审计＋额外计数/取整向量复核通过。18,996执行记录、3,003节点容量检查、105锁跨轮，0已提交违规/真实fallback，2次已拦截并同模型合法重求。原20/M新S3191.65比旧G3157.15慢1.09%；独立20/M新S3500.3比Sia3730.5快6.17%、比AUX3872.1快9.60%，但独立负载未跑G不能归因于新准入。154测试复用hash完全一致版本。两组Medium初始14/13作业分别t240/t300回到20，记录结构性局限，未追加调参。报告JOINT_SELECTIVE_ADMISSION_RESULTS.md＋FINAL_REVIEW.md，source_version/delivery为封存入口。会话9282及67938均已完成，不需恢复任何进程。正式104仍暂停；完成后停止。

用户确认内存已调整并要求继续。仅恢复上一轮已冻结10格小规模验证，正式104继续暂停。父目录工程版本、旧内存阻塞报告/manifest/log全部保护；当前实验结果唯一输出为本attempt2，preregistration与父文件逐字节相同。

启动前核验961源码/输入、112历史文件、原delivery全部hash；无重复实验进程，内存3.093GiB。会话9282执行run.py，顺序独立进程，禁止重复启动/修改冻结源码。8两格原始审计通过后才20，所有600轮，2GiB/5GiB门槛不变。不按结果调参或追加第二种策略。

恢复入口：resume_manifest.json、preregistration.json、execution_manifest.jsonl、各格summary/validation；观察当前进程，完成后从原始片段复核、生成独立结果报告及新delivery，不覆盖父报告。实施/审计错误则保留现场停止，删失不插补；不恢复104或扩大到160。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。
