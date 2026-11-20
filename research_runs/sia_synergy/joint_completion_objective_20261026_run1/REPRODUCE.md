# 重建和复核说明

本轮是门槛停止的最小实现，不包含完整JCT实验入口或启动授权。

`baseline_actual_source.zip`保留旧实际源码；`rebuildable_source_inputs.zip`保存最终实际源码、新输入、诊断脚本、全部本轮证据及静态分析引用的必要旧日志。文件按项目根目录相对路径保存。`delivery_manifest.json`在压缩包外给出最终哈希，避免归档包含自身的循环哈希。

1. 将最终压缩包解压到一个**新的空根目录**，不要覆盖`D:/科研启航/Sia`。源码在`sia-joint-completion-time/sia-simulator`，本轮记录在`research_runs/sia_synergy/joint_completion_objective_20261026_run1`。
2. 使用Python 3.10兼容环境及`dependencies.lock.txt`。记录中使用的是`D:/科研启航/Sia/.venv-sia-baseline/Scripts/python.exe`；清单同时保存实际Python/平台信息，不要求在其他机器沿用该绝对路径。
3. 在新源码目录运行`python -X utf8 -B -m unittest discover -s tests -p test_*.py -v`，或只运行`test_completion_time.py`。这是有界正确性测试，不是正式性能矩阵。
4. 如需重新生成离线/控制诊断，先在同一`research_runs/sia_synergy/`下新建空兄弟目录，将所需`.py`和`inputs/`复制进去。原脚本根据当前位置定位项目根目录；在新目录依次执行`offline_gate.py`、`offline_d3_first_warm.py`，或执行`controlled_validation.py`后执行`independent_controlled_audit.py`。脚本使用独占创建，直接在已有证据目录重跑会拒绝覆盖。控制诊断仅两任务，共同预热240秒、观察600秒，不完成整批训练。
5. `verify_offline_provenance.py`中的registry source路径是历史来源绝对路径，作为溯源证据保持不变。跨机器重算时，仅将路径前缀映射到解压根，按相对路径读取归档中的源CSV并校验原SHA-256；不要修改phi、源行或内容。此路径映射属于重建工具，不是新数据校准。
6. 官方Sia有界逐轮回归入口为`check_stage1_legacy_regression.py --probe <simulator目录>`。压缩包不冒充另一个官方仓库；原官方路径及hash和本轮两份原始回归日志已保留，需官方源码时使用原官方仓库并校验。

新策略是显式opt-in模块：`completion_time.install(cluster)`为JOINT-ECT，`proportional=True`为JOINT-ECT-P；`install_aux(cluster)`为固定完整Sia意图的SIA-ECT-AUX。不能在已经运行的服务状态中重新安装策略。旧CLI和旧策略不自动进入新代码。

本轮完整性能门槛未通过，禁止据此自动运行2/6/20/160任务完整对照、104格矩阵、容量搜索或响应调参。需要新的研究授权及预注册后再继续。
