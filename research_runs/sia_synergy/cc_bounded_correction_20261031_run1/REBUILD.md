# 重建与判定

新树：`D:/科研启航/Sia/cc-joint-bounded-correction-v1`。这是从329项已封存实际执行源码逐文件核验复制的完整源码版本，不是只从旧Git HEAD重建的版本。

验收通过后，pipeline.py保存完整source_snapshot.zip、source_hashes.json、source_changes.patch、输入哈希、依赖环境及预注册。解压至同级新目录后按清单逐项核验。不得在已运行的树内恢复或覆盖。

环境：项目`.venv-sia-baseline/Scripts/python.exe -X utf8 -B`。依赖锁在源码与preregistration.environment中；外部测试依赖使用项目历史测试资源，不下载、不安装、不升级依赖。

红测：初始bounded_correction.py仅包含空的BoundedCorrectionMixin类，其他逻辑完全继承旧控制器；test_bounded_correction.py前8项测试产生red.log的2项预期失败。通过旧控制器模式仍保留该反例。扩展测试的锁定peer场景曾把零进度保持到360秒，超过原310秒停滞门槛；这是测试输入错误，修正为180秒。full_suite.log和green_expanded_attempt1.log保留该失败，不充作验收通过。

同状态same_state_validation.json使用标注的toy原始率，检验真实候选、资源审计、原子MIP及新旧规则；不将其ETA预测当成真实训练JCT。实际Sia模型、unseeded BERT、checkpoint/epoch/完成/倍率的回归由完整套件和原始Sia探针验证。

正式补跑仅新FO和新FULL，Medium、原80作业、相同GPU/CPU/RAM、60秒与6000轮。原两格均取自cc_core80_full_lex_repair_20261031_run5的封存结果，从原始事件复算数据比较；不替换、不覆盖旧数据。

新规则相同，并不要求两方法选择相同配置。固定其他当前分配的余量见证只证明恢复点可以不驱逐peer落地；普通优化仍可能改变peer。FO仍先比例提案再固定GPU做辅助优化；FULL仍联合选择；两者目标和所有原候选网格不变。一次许可和原单作业尾部恢复共享，不能重复领取。

预先解释规则：平均JCT降低为主要目标；P95/P99、makespan、DeepSpeech训练、等待、checkpoint、重启及资源小时同时报告。若新规则改善两方法但FULL对FO差距未扩大，只认定共同控制器改善。若退化或未触发，则如实报告，不调参数。全部80作业保留；未完成时不插补JCT，不做完整均值配对。

pipeline.py为一次性入口，不得重复启动；先核对pipeline_start.json、原生进程及execution_manifest.jsonl。任何失败目录不复用。回归或资源门槛失败时不启动正式格；正式工程失败保留现场并停止后续格。资源安全门槛为内存2GiB、磁盘5GiB。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE
