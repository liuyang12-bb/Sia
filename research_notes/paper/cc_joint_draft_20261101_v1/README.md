# CC-Joint 文献与论文初稿交付

本目录为2026-11-01新增写作成果，不包含新仿真，不修改调度器，也不覆盖历史报告。论文是可审查初稿，尚未投稿；实验结果的适用范围和待补证据已写明。

## 建议阅读顺序

1. [中文论文初稿 PDF](CC_JOINT_PAPER_DRAFT.pdf)：含摘要、问题定位、数学模型、算法、伪代码、实验表图、局限和参考文献。
2. [可编辑 Markdown 初稿](CC_JOINT_PAPER_DRAFT.md)：后续修订的唯一文本源。
3. [文献调研与写作思路](literature/LITERATURE_REVIEW.md)：13篇精选，CPU/主机RAM、异构弹性、排队/公平三条线；标明原论文图表形式如何借鉴。
4. [完整参考文献](literature/REFERENCES.md)、[BibTeX](literature/references.bib)、[公开论文PDF目录](literature/papers)：11篇PDF，另RAP/Nexus仅有核验链接与元数据，未声称已读其全文。Sia PDF为官方Artifact的14页副本，作者链接是另一16页版本。
5. [证据与版本说明](evidence/evidence_inventory.md)、[数学机制到代码映射](evidence/algorithm_formalization.md)。

## 图表与数据

论文图号按首次出现顺序。文件名保留生成脚本的原始稳定编号，二者映射如下。

|论文图号|图内容|文件名|
|---|---|---|
|图1|源CPU/RAM响应形状|figures/fig06_response_templates.*|
|图2|CC-Joint系统架构|figures/fig01_architecture.*|
|图3|有序/持续服务消融及时间分解|figures/fig02_service_ablation.*|
|图4|CPU/RAM开关及联合与串行对照|figures/fig03_resource_ablation.*|
|图5|80作业均值、CDF与尾部|figures/fig04_core80.*|
|图6|逐作业得失与平均时间分解|figures/fig05_tradeoffs.*|

全部图有300dpi PNG、SVG和矢量PDF；SVG可继续编辑。图由我们数据重新绘制，未复制原论文图像。源矩阵图只显示原始模板支持域内的形状，不将其当成R0迁移后真实目标硬件曲线。

机器数据：[paper_results.json](evidence/paper_results.json)，42格抽取记录来自四套独立历史证据（不是一轮新42格实验）。所有均值、分位数和makespan已从逐作业完成时间重新计算；互斥时间总和逐作业匹配。跨版本不混合计算统一效应。

## 复现图文

在项目根目录依次运行，仅读取既有证据并写本目录，不导入/启动调度器：

```powershell
.\.venv-sia-baseline\Scripts\python.exe -X utf8 -B research_notes\paper\cc_joint_draft_20261101_v1\scripts\plot_results.py
.\.venv-sia-baseline\Scripts\python.exe -X utf8 -B research_notes\paper\cc_joint_draft_20261101_v1\scripts\build_diagrams.py
.\.venv-sia-baseline\Scripts\python.exe -X utf8 -B research_notes\paper\cc_joint_draft_20261101_v1\scripts\render_equations.py
& 'C:\Users\88534\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -X utf8 research_notes\paper\cc_joint_draft_20261101_v1\scripts\build_draft_pdf.py
.\.venv-sia-baseline\Scripts\python.exe -X utf8 -B research_notes\paper\cc_joint_draft_20261101_v1\scripts\pdf_qa.py
```

下载与书目脚本只在需要重新建立文献库时使用。运行`prepare_bibliography.py`可重建BibTeX/元数据；它只在初稿含明确插入标记时写入正文参考文献，不覆盖后续手工修订。

## 当前必须保留的结论边界

- 主80作业FULL对Sia平均JCT下降29.62%，但P99/makespan退化18.29%/14.50%，对FO只有1.09%均值增量。
- 有序/持续的干净A组使用公平评分，其35.56%不是完成优先评分的独立效果。
- 辅助优化相对比例有正效果，但主要依赖固定E-legacy和理想平坦BERT模板；不能说目标硬件响应已校准。
- 后期12作业反例仅0.336%均值增量，保留所有DeepSpeech损失。
- FULL求解P99超过60秒，仿真没有把计算墙钟加入JCT；尚不能宣称实时可部署。
- Nexus是直接近邻工作，全文待补；不宣称首次组合异构GPU和CPU/RAM。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。
