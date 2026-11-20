# 12作业混合场景：真实冷启动机会与FO/FULL配对

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE

算法/响应/容量/工作量保持冻结；6CIFAR+4DeepSpeech2+2BERT，两波0/600秒，AWS4GPU/18CPU/125GB，RTX4GPU/12CPU/375GB。Medium alpha=.5，R0/E-legacy，seed_profiles(1,8)，60秒/6000轮。

真实事件机会门槛：未通过。40轮冷启动前缀，事件时同状态比较，再要求实际两路径GPU差异持续≥6周期且FULL至少2周期推进；没有额外注入gradient。

```json
[]
```

按冻结门槛停止，未启动完整FO/FULL。不能报告完整平均JCT。单次静态机会不等于在线可达、持续或JCT收益。事件明细见gate_result.json；所有完整前缀保留。

## 可复现性与边界

来源与输入hash见preregistration.json，执行脚本hash见runner_receipt.json，344文件源码和旧完整snapshot保持不变。旧结果未覆盖；未修改目标、候选、phi、裁剪或配额。该场景及R0迁移为合成机制验证，不能写成真实Saturn硬件收益。所有最终hash见artifact_hashes.json。
