# 项目整理：2026-11-01

用户授权整理项目、保留有用内容并清理无用内容。本轮不更改算法、不启动实验、不提交/推送。

保留六个源码入口：cc-joint（旧统一入口，2026-10-27）、cc-joint-core80-full-lex-repair-v8（论文主80作业版）、cc-joint-clean-ab-r3（干净消融）、cc-joint-resource-study（早期资源开关）、cc-joint-breakpoint-precision（响应候选/到达压力基础）、cc-joint-recovery-equivalent-v1（最新恢复等效版）。不能把这些版本的结果混同。

其余根目录cc-joint-*中间源码在确认不是Git工作树/链接、无运行进程、与保留目录无显式源码引用后，完整压缩到_archive/cc_intermediate_sources_20261101_v1.zip；逐文件SHA-256复核完成后才删除原目录，按原路径可恢复。历史实验research_runs、论文和报告research_notes、官方源、旧_archive、两个独立环境全部保留。只清除两个虚拟环境的可再生成缓存。

恢复入口：inventory_and_archive.py、archive_manifest.json、restore_intermediate_sources.ps1、ROOT_README_before.md。安全要求：所有删除目标必须是显式根目录的直接子目录、resolve仍在根目录内；归档含全部文件，未知链接/Git目录或内容变化即停止。不移动历史结果，不改实验脚本中的路径。

完成条件：归档全文件校验、保留源码和官方数据内容hash一致、历史结果文件大小/mtime清单不变、论文交付验证通过、新导航与工作区可用，输出整理报告与释放空间。

已完成：28套中间源码8797文件归档并校验后移除根目录副本；六套源码保留。10636个可再生字节码删除，净释放566.83 MiB。3975个保护文件内容hash不变；19158个历史结果路径/大小/mtime不变；论文123文件hash不变。恢复dry run、12个工作区入口及两个环境导入检查通过。见CLEANUP_REPORT.md、final_validation.json、environment_check.json。任务结束，不恢复归档、不启动实验。
