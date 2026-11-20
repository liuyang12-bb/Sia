# 复现说明

本交付只打包，不运行仿真。源码zip包含工作量、GPU画像、R0、矩阵和原依赖锁；不包含500MB虚拟环境或16.4GiB原始实验日志。环境为Python3.10.10、NumPy1.23.5、SciPy1.10.1、CVXPY1.3.2、CyLP0.91.6等，以provenance/environment.json和源码dependencies.lock.txt为准。CBC本地库还受平台安装约束，不能保证pip单条命令可重建二进制环境。

1. 解压到新目录；不要覆盖当前项目。先运行`python -B tools/verify_package.py`，只读校验所有329个源码文件。
2. 原测试有外部fixture依赖。解压fixtures/external_test_dependencies.zip到workspace/，保留其内部路径；不能把外部数据缺失当成算法测试失败。
3. 原验收记录和小型回归见provenance/software_acceptance.json、provenance/REBUILD.md、tools/maintenance_scripts.zip；正式主版本已有335项测试证据。本次没有重新运行它们。
4. 原preregistration.json保存的是历史绝对路径和已关闭输出目录，不能直接启动。需另行预注册新的源码位置、输入位置和全新输出目录，仅迁移路径、不改内容hash或算法参数。校验run_core80.py的--manifest/--cell接口后，才可运行单格；不要直接运行watch/launch脚本。

历史命令接口（仅示意，不在本交付中执行）：

`python -B workspace/cc-joint-core80-full-lex-repair-v8/run_core80.py --manifest NEW_FROZEN_MANIFEST.json --cell dual_medium_FULL`

内存至少2GiB、磁盘至少5GiB；60秒周期、6000轮上限；不减少训练工作量、不改响应、不覆盖旧输出。原始source_hashes.json中的key相对源码根；input_hashes.json中的原绝对路径需按新布局对应核对。某些原测试和历史脚本还使用项目布局，跨机器完整测试需逐项恢复fixture，不能把包描述成无需环境准备的一键部署。

原始源码快照SHA-256：B0DCAEB4A5A205B6CA27893F3ACFE1F43C3F850CA8AF6E5BC79B15029518D6EB。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。
