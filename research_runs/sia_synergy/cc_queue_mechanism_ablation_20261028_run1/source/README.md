# CC-Joint：当前唯一开发入口

本目录保留已验证的CC-Joint、比例消融以及必要的Sia对照。整理不改变调度算法、
目标、准入、CPU/RAM响应、容量或训练语义；只筛选依赖、集中输入并增加统一入口。

## 入口与四种方法

统一使用`run_cc_joint.py`，不再从历史版本目录启动实验：

- `JOINT_CC`：CC-Joint完整联合方案。
- `JOINT_CC_P`：同版比例配额消融。
- `SIA_P`：官方Sia GPU决策＋比例辅助资源执行适配。
- `SIA_ECT_AUX`：固定当轮Sia意图的辅助资源优化对照。

默认输入是已冻结的20任务持续CPU互补场景。数据均在`inputs/`：
18个真实CIFAR记录、2个真实DeepSpeech记录，受控到达，原batch/GPU demand/工作量保留。
AWS和DGX-ext各2GPU/12合成CPU单位/250GB RAM，R0＋E-legacy，alpha0或0.5，
seed_profiles(1,8)、batch自适应、60秒周期与6000轮上限不变。

只核对构造、不推进训练（项目根运行）：

```powershell
.venv-sia-baseline/Scripts/python.exe -X utf8 -B cc-joint/run_cc_joint.py --check
```

只有另行批准实验后才运行；`--output`必须是全新目录，不会自动启动矩阵：

```powershell
.venv-sia-baseline/Scripts/python.exe -X utf8 -B cc-joint/run_cc_joint.py --method JOINT_CC --response medium --output research_runs/sia_synergy/NEW_APPROVED_RUN
```

每次真实运行检查内存至少2GiB、磁盘至少5GiB。不存在覆盖、自动重试、调参或自动延长观察窗。
运行记录逐轮原始压缩JSON、完成/删失、时间、资源、checkpoint、rho和失败快照。

## 代码地图

- `cc_factory.py`：集中四策略构造；与最近冻结实验同构。
- `cc_metrics.py`：原有指标与独立执行审计的同义提取。
- `sia-simulator/joint_prototype/completion_continuity.py`：完成优先评分、持续服务、事件门控与数值保护。
- `completion_time.py`：声明工作量、当前Sia模型的剩余完成时间估计、串行辅助对照。
- `joint_integration.py`、`joint_scheduler.py`：真实Sia候选、联合列、原子提交、锁与回退。
- `r0_liveness.py`、`joint_profile.py`：固定R0/phi和统一CPU/RAM响应。
- `job.py`、`applications.py`、`goodput.py`、`sia.py`：保留Sia建模、弹性配置和训练推进。
- `tests/`：保留212项既有回归；旧文件名不代表启用旧实验。

`adaptive_service`、`bundle_adaptation`、`protected_admission`等仍在真实继承链中，
不能按旧名字删除。`gavel/policies`由继承的Cluster直接导入，留作兼容依赖，未启用Gavel实验。
少量旧运行器仍被回归测试或指标类导入，不作为本项目入口。无关分析/矩阵启动脚本已排除。
生产配置唯一入口是`inputs/config_symmetric.json`；`joint_prototype/`中的旧JSON供回归测试，
不是本轮运行配置。`service_horizon.py`和`version_bridge.py`等测试依赖不在四策略入口的调度链上。

## 版本与测试

原404文件冻结源码指纹：
`21CA3574775F3A10030473BD22183802CC5DED06D2ED57EDE35F3D1A332F5A57`。
`source_manifest.json`记录所有保留引擎、测试、输入的原SHA；引擎字节不变。
`dependencies.lock.txt`是实际运行依赖锁；不安装或升级全局环境。

从`cc-joint/sia-simulator`运行既有完整测试：

```powershell
../../.venv-sia-baseline/Scripts/python.exe -X utf8 -B -m unittest discover -s tests -p test_*.py -v
```

测试仍读取`research_runs/`中的历史只读fixture以及官方Sia/Synergy资料，这些不是无关数据。
清理记录、前后测试、源码等价性与恢复清单见`maintenance/cc_joint_cleanup_20261027/`。
旧树归档到`_archive/`，官方源、报告和历史产物不删除；冻结报告仍指向其原始版本，不冒称新目录产生了旧结果。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。
CPU额外收益仅已在受控场景观察，内存额外收益未证实；平均JCT改善伴随部分长任务退化。
