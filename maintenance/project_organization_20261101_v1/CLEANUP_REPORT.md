# 项目整理完成报告

## 结果

- 状态：PASS。
- 保留六套有明确用途的源码入口；28套中间版本共8797文件完成逐文件归档校验后，从根目录移除。
- 归档前原内容612.98 MiB，压缩包177.34 MiB；归档内含逐文件清单与hash。
- 清除两个独立环境中有对应源码的10636个字节码缓存，释放131.19 MiB。
- 合计净释放约566.83 MiB（文件内容字节口径，不是文件系统块计量）。
- 没有永久删除历史源码；没有更改算法、依赖版本或实验输入；未启动仿真。

## 保留与校验

- 六套保留源码、官方源、research_sources与research_notes共3975文件SHA-256前后相同。
- research_runs共19158文件的完整路径、大小、纳秒mtime前后相同；不声称全量重算16.4 GiB日志hash。
- 论文交付的123个文件全部与原delivery_manifest的hash相同。
- 项目根和CC快照目录不是Git工作树；官方仓库原先的未跟踪缓存/PDF保留，没有reset/clean/提交/推送。
- 六套保留源码的Python文件未发现对本次28个归档目录名的显式引用；历史maintenance脚本仍可能要求恢复旧目录，未改写它们。
- 无实验原生Python进程检查在整理前完成；本轮没有停止、启动或恢复任何实验。

## 新入口

- 根README.md及PROJECT_INDEX.md说明论文版本、修正版、消融版、数据和环境用途。
- CC_Joint_Research.code-workspace提供12个具名目录入口，隐藏缓存并避免监听/搜索大体积原始日志。
- 原根README与VSCode设置备份保存在本目录。
- cc-joint/README.md为2026-10-27历史版本，保持字节原样；根README已解释其“唯一入口”不代表后续版本。

## 恢复

归档：`_archive/cc_intermediate_sources_20261101_v1.zip`

SHA-256：`1A694484E00A56FC1B4AFD9DFE254A0FC0B67C2D34E6E65ACE9E936C962D2955`

`restore_intermediate_sources.ps1`默认dry run，添加`-Execute`恢复所有中间目录到原路径，拒绝覆盖并逐文件验证hash。归档不是可随意删除的缓存。

## 有意未删除

research_runs原始结果、失败现场、旧报告、既有_archive、tmp早期试验数据/截图和maintenance历史脚本均可能承担溯源/复现作用，保留原路径。两个Python环境只清缓存，安装包未删。

## 整理工具兼容性

归档脚本首次在Python 3.10因Path.is_junction不可用停止，发生在任何归档/删除前；改为检查Windows reparse属性后成功。该修复只影响本轮整理脚本。恢复脚本另外以dry run核验，不实际恢复重复副本。

缓存清理后，以`-B`方式实际加载两个环境的核心依赖均成功：baseline为Python 3.10.10 / NumPy 1.23.5 / SciPy 1.10.1 / CVXPY 1.3.2；synergy为Python 3.10.10 / NumPy 2.2.6 / pandas 2.3.3。未安装、卸载或升级依赖。未运行完整算法测试套件或训练，源码逐文件一致性作为本轮整理的回归依据。

具体机器验收见final_validation.json、cleanup_receipt.json、archive_manifest.json与cleanup_journal.jsonl。
