# 重建与复核

本轮算法仍为原CC-Joint，404源码文件指纹：
21CA3574775F3A10030473BD22183802CC5DED06D2ED57EDE35F3D1A332F5A57。
实际执行树sia-joint-cc-sustained20/sia-simulator，source/逐字节相同。
裸HEAD不代表实际历史实现；不依赖不完整HEAD恢复。

只读官方workload-1，取前18CF/前2DS，稳定交错初始四任务，后续每300s两CF。
original_arrival/original_source_row_index保留原始出处，source_row_index是受控顺序。
原batch、GPU demand、应用工作量、R0/模板/phi/grid/容量不改。
inputs/包括原始完整trace、新20任务CSV、旧6/12任务输入、配置、registry和响应模板。

参数和八格顺序冻结于preregistration.json。preflight与offline_reference_detail
记录数据机会，非在线额外画像。完整训练独立进程串行，memory>=2GiB/disk>=5GiB。
6000轮统一删失，不扩窗/删任务/插补JCT。所有正式输出独占创建，不允许重复覆盖。

将complete_rebuildable_evidence.zip解压到新的空项目根；禁止覆盖当前工程。
delivery_manifest.json有包SHA与每个成员SHA，逐成员读回验证。
归档继承旧12任务已经校验的完整重建包，因此包含官方只读源码、历史测试fixtures
和所有所需旧实际版本；另含本轮source、执行树、冻结输入、原始结果、测试和分析脚本。
依赖锁dependencies.lock.txt，Python/platform在environment.json。
旧fixtures包含D:/科研启航/Sia绝对路径，只在新恢复副本重定位前缀，不修改参数或原档。

复核（不重新训练）：在本轮目录使用同环境执行load/verify，核对原始文件SHA；
如需重新运行auditor，先复制到全新同级目录，不复制生成汇总，以免exclusive写入冲突。
audit.py从raw rounds重算完成、JCT、互斥时间、资源小时、锁、rho单次、真实进度和目标；
mechanism_analysis.py/outcome_analysis.py/exchange_composition.py负责资源交换与时间分解。
report.py只读取独立审计值。解析配对机会只是平均速率近似，不伪称完整训练测量。

测试必须从项目根下sia-joint-cc-sustained20/sia-simulator启动，不能从更深的source目录
启动依赖parents[3]的旧测试。212测试原样；原始Sia bounded独立3轮逐状态回归。

只有另行授权复现实验，才在全新独立目录重建相同层次/配置，使用experiment.py的
prepare/relocate/preflight/test/regression/freeze/validate/batch入口；每格独立进程。
本轮不自动开启下一场景、调参、104格、160任务或新算法。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE
