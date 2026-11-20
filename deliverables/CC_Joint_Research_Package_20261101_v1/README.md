# CC-Joint 三项交付

1. `01_ALGORITHM_DESIGN_REPORT.pdf` / `.md`：完整算法、流程图、10个公式、伪代码、输入输出、约束、失败与活性边界。
2. `02_EXPERIMENT_REPORT.pdf` / `.md`：Baseline、四组42格完整结果、资源与尾部、曲线、逐作业数据及负结果。这里“完整”指所列四套证据，不代表历史所有探索和失败只有42格。
3. `03_CC_JOINT_CODE_AND_STRUCTURE.zip`：原样329文件v8源码、输入/测试/依赖、外部fixture包、预注册和结构/复现说明。不是仅几段核心函数。

PROJECT_STRUCTURE.md可在不解压ZIP时阅读；evidence/*.csv可用表格软件打开；figures/提供PNG、SVG、PDF原图。v8原源码README沿用早期版本说明，本包STRUCTURE/REPRODUCTION明确实际入口与版本，不改原源码字节。

本次没有运行仿真或改算法。主80作业、干净消融、arrival18和后期12作业分别冻结，不混算版本。原始历史文件与17GB级日志保持原位置。

`CC_Joint_Three_Deliverables.zip`将上述三项、可编辑报告、曲线、CSV、结构说明及交付hash一起打包，排除分页检查PNG等中间文件。两个PDF均已逐页检查，核心ZIP329文件已核对。CSV中原记录没有completed字段的216条，通过已记录的completion事件判定完成，completed_evidence列明确标注；未补造缺失等待细项或性能数据。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。
