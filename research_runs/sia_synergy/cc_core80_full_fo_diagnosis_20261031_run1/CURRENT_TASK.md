# FULL 与 FO 的只读机制诊断

范围：读取已封存 V8 七方法80作业 Medium 结果，定位 FULL 相对 FO 平均JCT仅改善1.09%的来源。只新增本目录的离线分析脚本、证据与报告。不修改源码、参数或历史数据，不启动任何训练仿真、求解批次或参数搜索。

证据源：cc_core80_full_lex_repair_20261031_run5 的原始逐轮 gzip、audit/per_job.csv、preregistration.json 和最终完整性清单；实现 cc-joint-core80-full-lex-repair-v8。历史12作业干净消融仅作不同场景的参考。

先固定分析方法：逐作业分解等待/检查点/训练；FO 同轮 proposal→selected 属于真实同状态串行辅助优化，不冒充 FULL 的反事实；跨策略同时间轨迹仅作描述。各方法选取最早满足三类结构条件的状态（至多各3个）：全到达后存在排队且辅助资源闲置；DeepSpeech与其他应用共驻；GPU已满且CPU或RAM接近容量。固定GPU下离线响应检查只评价局部速度，不能推断全程JCT。未记录的候选 ETA 和真实后续信息不重建或捏造。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE

诊断结束：80份逐作业账目、FO/FULL逐轮轨迹、4个预先结构条件状态与2个事后晚期预算状态已核对；validation.json=PASS。生成独立REPORT.md与diagnostic_hashes.json后停止。无训练仿真、无算法修改、无正式实验恢复。
