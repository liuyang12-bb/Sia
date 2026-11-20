# 保留的记录工具错误

1. 正式冻结前 freeze.py 首次读取包含中文路径的 input_seed.json 时，Windows默认GBK解码失败；改为显式UTF-8。尚未启动仿真。
2. 正式冻结前 freeze.py 的ROOT误用OUT.parents[3]（解析为D:/科研启航），git找不到sia-joint-version-bridge；改为OUT.parents[2]。尚未启动仿真。
3. 8格完成后第一次 analyze.py，在第一个旧对照的全部原始记录计算完成、尚未保存validation前，audit.py未创建reference_audits父目录，出现 `FileNotFoundError [WinError 3] .../reference_audits/original8_neutral_sia_p`。仅把 `dest.mkdir(exist_ok=True)` 改为 `dest.mkdir(parents=True,exist_ok=True)`。训练源码/preregistration/8格结果/历史文件均未改变，没有重跑或择优选择策略运行。

这些是记录/汇总工具错误，不是求解失败或训练失败。最终源码与工具分别由execution_source.zip和audit_tools_snapshot.zip封存。
