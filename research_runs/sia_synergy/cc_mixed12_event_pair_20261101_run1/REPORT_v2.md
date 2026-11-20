# 12作业混合场景：真实冷启动机会与FO/FULL配对

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE

算法/响应/容量/工作量保持冻结；6CIFAR+4DeepSpeech2+2BERT，两波0/600秒，AWS4GPU/18CPU/125GB，RTX4GPU/12CPU/375GB。Medium alpha=.5，R0/E-legacy，seed_profiles(1,8)，60秒/6000轮。

真实事件机会门槛：通过。40轮冷启动前缀，事件时同状态比较，再要求实际两路径GPU差异持续≥6周期且FULL至少2周期推进；没有额外注入gradient。首次验收实现额外要求后续服务集合不变，与冻结文字不符，纠正后复核；原判定及报告保留，详见gate_correction_receipt.json。

```json
[
  {
    "owner": "FO",
    "time": 360,
    "same_set": true,
    "changed": [
      "bert-28",
      "cifar10-1"
    ],
    "fo_objective": 1.5569524480473322,
    "full_objective": 1.6263692312025966,
    "structural": true,
    "geometry_events": [
      "FIRST_VALID_CURRENT_MODEL"
    ],
    "key": "bert-28",
    "fo_config": "aws:1:2",
    "full_config": "rtx:1:1",
    "intervals": 34,
    "productive_intervals": 32,
    "end": 2400,
    "key_progress": {
      "FO": 3500.0169560276463,
      "FULL": 3580.0911208896396
    },
    "clipping": {
      "FO": {
        "source_coordinates": true,
        "baseline_source_coordinates": false
      },
      "FULL": {
        "source_coordinates": true,
        "baseline_source_coordinates": true
      }
    }
  },
  {
    "owner": "FO",
    "time": 660,
    "same_set": true,
    "changed": [
      "cifar10-3",
      "deepspeech2-8"
    ],
    "fo_objective": 1.4588136477087674,
    "full_objective": 1.4985495478346216,
    "structural": true,
    "geometry_events": [
      "FIRST_VALID_CURRENT_MODEL"
    ],
    "key": "cifar10-3",
    "fo_config": "rtx:1:1",
    "full_config": "aws:1:1",
    "intervals": 18,
    "productive_intervals": 18,
    "end": 1740,
    "key_progress": {
      "FO": 9900.788896467808,
      "FULL": 10945.810807107931
    },
    "clipping": {
      "FO": {
        "source_coordinates": true,
        "baseline_source_coordinates": false
      },
      "FULL": {
        "source_coordinates": false,
        "baseline_source_coordinates": false
      }
    }
  },
  {
    "owner": "FULL",
    "time": 360,
    "same_set": true,
    "changed": [
      "bert-28",
      "cifar10-0",
      "cifar10-1",
      "deepspeech2-9"
    ],
    "fo_objective": 1.7060221540276195,
    "full_objective": 1.8988633217979236,
    "structural": true,
    "geometry_events": [
      "FIRST_VALID_CURRENT_MODEL"
    ],
    "key": "bert-28",
    "fo_config": "aws:1:2",
    "full_config": "rtx:1:1",
    "intervals": 34,
    "productive_intervals": 32,
    "end": 2400,
    "key_progress": {
      "FO": 3500.0169560276463,
      "FULL": 3580.0911208896396
    },
    "clipping": {
      "FO": {
        "source_coordinates": true,
        "baseline_source_coordinates": false
      },
      "FULL": {
        "source_coordinates": true,
        "baseline_source_coordinates": true
      }
    }
  }
]
```

## 完整结果（从原始完成事件独立复算）

|方法|完成|平均JCT|Median|P95|P99|Makespan|
|---|---:|---:|---:|---:|---:|---:|
|FO|12|6970.0|7166.0|12786.999999999998|13781.400000000001|14630.0|
|FULL|12|6946.583333333333|5875.5|13543.349999999999|14547.87|15399.0|

```json
{
  "complete": true,
  "mean_improvement_pct": 0.3359636537541899,
  "per_job_saved": {
    "cifar10-0": -27.0,
    "deepspeech2-8": -452.0,
    "bert-28": 1466.0,
    "cifar10-1": 155.0,
    "deepspeech2-9": -828.0,
    "cifar10-2": -104.0,
    "cifar10-3": 19.0,
    "deepspeech2-12": -746.0,
    "bert-37": 134.0,
    "cifar10-4": 318.0,
    "deepspeech2-16": -769.0,
    "cifar10-5": 1115.0
  },
  "receiver_jct_saved": 3207.0,
  "donor_jct_loss": 2926.0,
  "net_jct_saved": 281.0,
  "deepspeech_net_saved": -2795.0,
  "improves": 6,
  "worsens": 6,
  "ties": 0
}
```

|作业|FO JCT|FULL JCT|节省s|FO等待|FULL等待|FO训练|FULL训练|FO checkpoint|FULL checkpoint|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|cifar10-0|1693.0|1720.0|-27.0|60.0|60.0|1433.0|1460.0|200.0|200.0|
|deepspeech2-8|8953.0|9405.0|-452.0|120.0|120.0|8633.0|9110.0|200.0|175.0|
|bert-28|8442.0|6976.0|1466.0|180.0|180.0|8022.0|6436.0|240.0|360.0|
|cifar10-1|2638.0|2483.0|155.0|240.0|240.0|2298.0|2143.0|100.0|100.0|
|deepspeech2-9|10619.0|11447.0|-828.0|300.0|300.0|10169.0|11022.0|150.0|125.0|
|cifar10-2|3036.0|3140.0|-104.0|360.0|360.0|2576.0|2580.0|100.0|200.0|
|cifar10-3|2840.0|2821.0|19.0|0.0|0.0|2740.0|2721.0|100.0|100.0|
|deepspeech2-12|11770.0|12516.0|-746.0|1140.0|60.0|10505.0|12356.0|125.0|100.0|
|bert-37|9478.0|9344.0|134.0|2040.0|1140.0|7078.0|7844.0|360.0|360.0|
|cifar10-4|4251.0|3933.0|318.0|2460.0|1920.0|1691.0|1913.0|100.0|100.0|
|deepspeech2-16|14030.0|14799.0|-769.0|2880.0|2580.0|11050.0|12144.0|100.0|75.0|
|cifar10-5|5890.0|4775.0|1115.0|4260.0|2880.0|1530.0|1795.0|100.0|100.0|

## 资源、成本与审计

|方法|GPU h|CPU h|RAM GB h|等待s|训练s|checkpoint s|重启|wasted GPU s|裁剪选中列比例|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|FO|29.41388888888886|78.1412726947946|1574.952473820704|14040.0|67725.0|1875.0|30|2590|0.5793991416309013|
|FULL|28.762499999999957|90.17108508871644|1486.2788049880976|9840.0|71524.0|1995.0|29|2685|0.5694557270511779|

逐节点容量、checkpoint完整锁、独立rho复算、raw×rho执行率、互斥时间与资源小时均从原始日志复算，错误数0。统计详见raw_audit.json。P95/P99仅12作业经验分位。完整均值仅在12/12配对时计算。

资源释放的事件顺序及所有存活任务配置变化保存在resource_release_events.json；两路径历史已经分叉，不能将全部收益归因于某一次释放。接收方收益、DeepSpeech2损失与总和必须共同解释。

## 可复现性与边界

来源与输入hash见preregistration.json，执行脚本hash见runner_receipt.json，344文件源码和旧完整snapshot保持不变。旧结果未覆盖；未修改目标、候选、phi、裁剪或配额。该场景及R0迁移为合成机制验证，不能写成真实Saturn硬件收益。所有最终hash见artifact_hashes.json。
