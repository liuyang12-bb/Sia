# Stage1.1 重建说明

本目录仅包含工程正确性测试和可重建交付，不是 cohort 实验。

## 版本

- 基线：`0461dcdf061197fdfffe23b2b3d07e46e0d9efb2`
- 修复：`216573cb26f3136520ce9d4f609623b4c2c75df1`
- 分支：`exp/b3-correctness-stage1p1`
- 本地工作树：`D:\科研启航\Sia\sia-joint-correctness-stage1p1`

`base.bundle` 和 `repair.bundle` 保存完整 Git 历史；`source_snapshot.zip` 和 `repair_source_snapshot.zip` 保存实际执行字节（含历史 tracked 源码和当前已提交测试）。基线 tracked/index patch 均为空；基线 untracked 只有被排除的生成缓存。没有遗漏的必要 untracked 源码。

## 在全新目录重建

1. 从 `repair.bundle` 本地 clone 到不存在的新目录，检出上述修复 commit。不得操作旧工作树或 run1/run2。
2. 对照 `repair_manifest.json` 检查 commit 与完整源文件名单。Git 会处理换行符；如需逐字节复现，将 `repair_source_snapshot.zip` 的文件恢复到这个**新目录**，然后按 manifest 的每个 SHA-256 核对。
3. 外部 Synergy 模板在 `external_response_templates.zip`；应放在新项目根目录的 `research_sources/Synergy/simulator/models`。Sia applications/profile/workload 数据包含在完整源快照中。禁止用新下载模板替换本包版本。
4. 使用独立 Python 3.10.10 环境，并参考 `requirements.lock.txt`、`dependencies.json` 重建依赖；不得修改系统环境。包未包含离线 wheel、Python 安装包或操作系统，跨平台求解器一致性仍需验证。
5. 在新 `sia-simulator` 目录执行 `python -B -m unittest discover -s tests -p test_*.py -v`，应为 71 项。可设置 `STAGE1_EVIDENCE_DIR` 为一个全新的路径以留存小测试日志。
6. 不运行任何 cohort 脚本。本包不批准活性算法或响应边界模型。

## 证据

- `red_test_source.py` 是生产源码修改前的新增 7 项测试原文；可在基线副本中作为 tests 下新文件运行，复现红测。不要覆盖 Stage1 的既有 tests。
- `red.log/json` 是基线红测；`boundary1.log/json` 是收紧校验后发现的 B4 fallback 边界错误；所有证据原样保留。
- `final_tests.log/json` 是修复 commit 的 71 项测试。
- `legacy_regression/` 是官方 Sia、原冻结树、Stage1、Stage1.1 各自独立进程的一作业三轮回归。
- `delivery_manifest.json` 索引交付文件和报告的哈希，不包含自身哈希；`validation_summary.json` 给出源快照、bundle、受保护文件和测试验收结果。

字节快照用于精确复核 Windows 换行符；Git commit 用于语义 diff。二者同时保留，不能只保留哈希。
