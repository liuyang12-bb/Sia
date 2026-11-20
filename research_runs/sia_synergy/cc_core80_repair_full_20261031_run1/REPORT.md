# 冻结修复版：11格80作业补跑（10/11格完整）

**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**

本轮固定矩阵完成 **10/11格**，独立原始事件审计违规 **0项**。下表保留全部完成、删失、失败或计算超时结果；不为未完成作业插补JCT，也不以已完成子集均值代替总体均值。

### 直接结论

- 相同官方Sia接口后的辅助资源优化：SIA_AUX相对SIA_P平均JCT改善20.13%，P99改善4.83%，Makespan改善8.81%；负值为退化。
- CC比例配额整套策略，含目标/准入/配置变化：PROP相对SIA_P平均JCT改善8.94%，P99改善-10.43%，Makespan改善-5.38%；负值为退化。
- 同一CC策略下仅开放CPU差异化配额：CPU相对PROP平均JCT改善12.44%，P99改善0.81%，Makespan改善0.01%；负值为退化。
- 同一CC策略下仅开放内存差异化配额：MEM相对PROP平均JCT改善10.21%，P99改善-1.20%，Makespan改善-1.09%；负值为退化。
- 完整双资源联合格FULL未完整完成，不能声称它相对Sia、串行FO或单资源优化更好，也不能计算完整CPU×内存交互效应。工程求解失败不等于训练性能退化。
- 仅将完整且原始审计有效的配对用于JCT比较；等待/训练/检查点分解属于会计归因，不能据此单独证明某机制的因果贡献。

## 1. 授权、冻结与公平性

研究者明确解除本轮11格80作业暂停后，使用全新目录重跑；旧阻止记录、失败尝试、源码及历史结果未覆盖。源树`cc-joint-core80-repair-v3`保持315文件冻结，源码zip SHA-256 `7894EA3C16587113301D4568FAA09C2116E7EB88A171864C5B3D6AFBAD584DCC`；38科学输入hash、依赖及302/302测试验收在启动前核验。原Sia有界回归一致，FULL12/A1020/PROP12故障状态回归通过。

没有修改目标、p、lambda、allocation bonus、CPU/RAM网格、R0/E-legacy、phi、alpha、容量、workload、训练工作量或服务语义。仅使用已验收的规范化整数解验证、精确变量消去/对偶证书/可行上界及独立CBC安全重试。HiGHS30秒/zero-gap、原无presolve30秒重试，CBC30秒/zero-gap不变；只有最优且原始行通过的解可提交。数值后端重试与调度fallback分开计数。

固定80声明为48CIFAR10/16DeepSpeech2/16BERT，真实Saturn记录的batch/GPU需求/训练工作量保留，40@0、20@600、20@1200秒受控到达。不是自然完整trace重放。AWS4GPU/12合成CPU/250GB，DGX-ext8/24/500、RTX8/24/500；所有策略相同。60秒、6000模拟轮、seed_profiles(1,8)。每格独立进程串行，启动门槛2GiB/5GiB；程序计算上限6h/格、24h/批次不等于训练时长或JCT。

## 2. 方法与可识别的因果因素

|组|方法|定义|
|---|---|---|
|资源 Medium|SIA_P|实际官方Sia GPU/placement决策＋比例CPU/RAM执行适配，无L2|
|资源 Medium|SIA_AUX|实际Sia当轮意图固定后的自有辅助资源优化；不是官方Synergy-TUNE|
|资源 Medium|PROP/CPU/MEM/FULL|共享完成优先目标、候选、FIFO/持续服务及切换成本；分别比例、仅CPU自由、仅内存自由、双资源自由|
|资源 Medium|FO|同一CC目标/准入先以比例域选GPU及服务集合，再冻结GPU配置优化辅助资源；不是原Sia|
|服务 Neutral|A00/A01/A10/A11|共同公平效用与历史/锁约束；第一位关闭/开启FIFO队首准入，第二位300秒有效训练量子/持续到完成|

A组和资源组目标不同，不能把两组百分比相加。JOINT-P/PROP相对Sia还同时改变目标、准入及配置选择，不能将其全部收益归因CPU感知。A00仍保留共同的服务状态、候选与历史约束，不等于官方Sia。A11 Neutral与SIA_P Medium的跨组比较仅作描述：两者比例点均需经独立执行审计确认rho=1，不能替代同目标同响应的因子消融。JCT的等待/训练/检查点分解是会计分解，真正资源因子效果应看同组受控消融。

## 3. 11格绝对结果

|方法|状态|完成|均值JCT(s)|Median|P95|P99|Makespan|轮数|计算墙钟(s)|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SIA_P | ALL_COMPLETE | 80/80 | 9290.88 | 10475.50 | 15778.00 | 17149.94 | 18519.00 | 309 | 300.36 |
| FULL | FAILED | 4/80 | — | — | — | — | — | 18 | 1071.30 |
| SIA_AUX | ALL_COMPLETE | 80/80 | 7420.41 | 6904.00 | 14505.65 | 16320.76 | 16888.00 | 282 | 1116.28 |
| PROP | ALL_COMPLETE | 80/80 | 8460.56 | 7426.00 | 17639.15 | 18937.84 | 19516.00 | 326 | 420.52 |
| CPU | ALL_COMPLETE | 80/80 | 7408.18 | 6144.00 | 16385.85 | 18783.86 | 19515.00 | 326 | 976.18 |
| MEM | ALL_COMPLETE | 80/80 | 7596.74 | 6369.00 | 17509.80 | 19164.15 | 19729.00 | 329 | 725.81 |
| FO | ALL_COMPLETE | 80/80 | 6610.69 | 5536.50 | 15320.50 | 17021.07 | 18171.00 | 303 | 7578.70 |
| A00 | ALL_COMPLETE | 80/80 | 11798.89 | 12700.00 | 19347.00 | 23449.07 | 23643.00 | 395 | 520.68 |
| A01 | ALL_COMPLETE | 80/80 | 11798.89 | 12700.00 | 19347.00 | 23449.07 | 23643.00 | 395 | 505.62 |
| A10 | ALL_COMPLETE | 80/80 | 7564.20 | 6287.00 | 17523.75 | 18353.16 | 19108.00 | 319 | 450.63 |
| A11 | ALL_COMPLETE | 80/80 | 7684.35 | 6920.00 | 15485.40 | 18422.94 | 19034.00 | 318 | 385.45 |

P95/P99为单一80作业场景的经验分位数，不能当高精度总体尾延迟估计；没有把80作业视作80次独立实验。

### 未完成/失败的解释

- FULL：FAILED，已记录18个完整轮次，失败时间1140，最后完整时间1080；未记录的失败间隔60秒不补写、不插补。错误`RuntimeError('REPAIRED_SOLVER_NO_VERIFIED_OPTIMUM')`。
  求解器首次/重试status=1/1；CBC={'status': 1, 'success': False, 'message': 'EXISTING_CBC_optimal_inaccurate', 'options': {'maximumSeconds': 30.0, 'allowableFractionGap': 0.0}}；当前成本=5，下界=-0E-76；本阶段非零成本任务=['deepspeech2-10003']。没有证明可用最优解不等于资源违规、训练死锁或JCT性能退化；主结果不计算失败格总体均值。
## 4. 完整配对、排队与辅助资源贡献

|候选 vs 对照|均值JCT改善%|平均节省秒|等待/检查点/训练节省秒|改善/退化/持平|P99改善%|Makespan改善%|
|---|---:|---:|---|---|---:|---:|
| SIA_AUX vs SIA_P | 20.13 | 1870.46 | 1591.50/42.38/236.59 | 56/24/0 | 4.83 | 8.81 |
| PROP vs SIA_P | 8.94 | 830.31 | 1747.50/81.25/-998.44 | 44/36/0 | -10.43 | -5.38 |
| CPU vs SIA_P | 20.26 | 1882.70 | 2476.50/90.62/-684.42 | 46/34/0 | -9.53 | -5.38 |
| MEM vs SIA_P | 18.23 | 1694.14 | 2391.00/92.50/-789.36 | 47/33/0 | -11.74 | -6.53 |
| FO vs SIA_P | 28.85 | 2680.19 | 2917.50/94.69/-332.00 | 58/22/0 | 0.75 | 1.88 |
| CPU vs PROP | 12.44 | 1052.39 | 729.00/9.38/314.01 | 73/6/1 | 0.81 | 0.01 |
| MEM vs PROP | 10.21 | 863.82 | 643.50/11.25/209.07 | 72/8/0 | -1.20 | -1.09 |
| FO vs PROP | 21.86 | 1849.88 | 1170.00/13.44/666.44 | 73/7/0 | 10.12 | 6.89 |
| A10 vs A00 | 35.89 | 4234.69 | 4691.25/-55.06/-401.50 | 52/26/2 | 21.73 | 19.18 |
| A01 vs A00 | 0.00 | 0.00 | 0.00/0.00/0.00 | 0/0/80 | 0.00 | 0.00 |
| A11 vs A10 | -1.59 | -120.15 | -188.25/59.06/9.04 | 48/30/2 | -0.38 | 0.39 |
| A11 vs A01 | 34.87 | 4114.54 | 4503.00/4.00/-392.46 | 52/16/12 | 21.43 | 19.49 |
| A11 vs SIA_P | 17.29 | 1606.52 | 2394.75/124.38/-912.60 | 50/29/1 | -7.42 | -2.78 |

正值为降低JCT/延迟，负值为退化；未完成配对不给出改善率，见paired_comparisons.json的complete_pair。

资源2×2（仅四格完整才计算）：
```json
{}
```

有序准入×持续服务2×2：
```json
{
  "fifo_saved_mean_seconds": 4174.612500000001,
  "continuity_saved_mean_seconds": -60.07500000000027,
  "jct_interaction": 120.15000000000146,
  "sign": "Negative interaction means an additional JCT decrease beyond additive component changes. Do not add these effects to resource effects measured under a different objective."
}
```

主效应正数表示平均节省JCT秒；交互项Y11−Y10−Y01+Y00为负表示超加性改善。A00/A01无差异只说明本场景下量子重入队未改变最终服务轨迹，不证明持续服务普遍无效。

## 5. 资源、执行、求解与边界审计

|方法|GPU-h|CPU单位-h|RAM GB-h|等待(s)|检查点(s)|训练(s)|restart|wasted GPU-s|违规|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SIA_P | 99.73 | 299.18 | 6232.93 | 515460.00 | 23315.00 | 204495.00 | 377.00 | 30475.00 | 0 |
| FULL | 3.47 | 14.66 | 248.46 | 43620.00 | 2925.00 | 4893.00 | 47.00 | 3985.00 | 0 |
| SIA_AUX | 90.51 | 266.15 | 5447.46 | 388140.00 | 19925.00 | 185568.00 | 332.00 | 26185.00 | 0 |
| PROP | 102.51 | 307.54 | 6407.01 | 375660.00 | 16815.00 | 284370.00 | 287.00 | 22670.00 | 0 |
| CPU | 95.29 | 290.23 | 5955.33 | 317340.00 | 16065.00 | 259249.00 | 267.00 | 20835.00 | 0 |
| MEM | 95.49 | 286.47 | 6057.93 | 324180.00 | 15915.00 | 267644.00 | 268.00 | 20045.00 | 0 |
| FO | 83.82 | 258.02 | 5265.19 | 282060.00 | 15740.00 | 231055.00 | 266.00 | 20700.00 | 0 |
| A00 | 109.21 | 327.63 | 6825.69 | 684120.00 | 13685.00 | 246106.00 | 195.00 | 18015.00 | 0 |
| A01 | 109.21 | 327.63 | 6825.69 | 684120.00 | 13685.00 | 246106.00 | 195.00 | 18015.00 | 0 |
| A10 | 98.94 | 296.82 | 6183.65 | 308820.00 | 18090.00 | 278226.00 | 313.00 | 19640.00 | 0 |
| A11 | 93.08 | 279.25 | 5817.64 | 323880.00 | 13365.00 | 277503.00 | 215.00 | 15090.00 | 0 |

从原始完成事件、训练片段和allocated_seconds独立积分，作业周期中完成后不继续计资源。检查未选中锁作业、完整锁几何/配额、节点容量、单次rho、互斥时间、进度及原Sia/FO意图。完整输出summary只用于交叉核对，不作为唯一指标来源。

- SIA_P：执行检查24720，锁检查77，等待轮252，CPU饱和/GPU空闲节点轮0；fallback=0；数值事件`{}`；实际整数后端/LP/证明统计`{'native_calls': 0, 'native_seconds': 0, 'bound_certificates': 0, 'lp_calls': 0, 'lp_seconds': 0, 'peak_variables_removed': 0, 'peak_zero_optional_removed': 0, 'peak_lp_variables_removed': 0}`；clipping`{'num_cpu': 482, 'num_ram': 0, 'den_cpu': 482, 'den_ram': 0, 'num_any': 482, 'den_any': 482, 'any_point': 482, 'selected': 3826}`；裁剪率%`{'num_cpu': 12.598013591217983, 'num_ram': 0.0, 'den_cpu': 12.598013591217983, 'den_ram': 0.0, 'num_any': 12.598013591217983, 'den_any': 12.598013591217983, 'any_point': 12.598013591217983}`；训练暴露`{'fully_supported': 179630.0, 'clipped': 24865.0}`。
- FULL：执行检查1440，锁检查5，等待轮18，CPU饱和/GPU空闲节点轮11；fallback=0；数值事件`{'CANONICAL_BINARY_BEFORE_RETRY': 16, 'FIXED_EXISTING_NO_PRESOLVE_RETRY': 4, 'EXISTING_CBC': 1}`；实际整数后端/LP/证明统计`{'native_calls': 315, 'native_seconds': 455.15602540023974, 'bound_certificates': 54, 'lp_calls': 137, 'lp_seconds': 175.73162499986938, 'peak_variables_removed': 125760, 'peak_zero_optional_removed': 153, 'peak_lp_variables_removed': 144244}`；clipping`{'num_cpu': 67, 'num_ram': 29, 'den_cpu': 42, 'den_ram': 0, 'num_any': 67, 'den_any': 42, 'any_point': 80, 'selected': 146}`；裁剪率%`{'num_cpu': 45.89041095890411, 'num_ram': 19.863013698630137, 'den_cpu': 28.767123287671232, 'den_ram': 0.0, 'num_any': 45.89041095890411, 'den_any': 28.767123287671232, 'any_point': 54.794520547945204}`；训练暴露`{'fully_supported': 2025.0, 'clipped': 2868.0}`。
- SIA_AUX：执行检查22560，锁检查65，等待轮225，CPU饱和/GPU空闲节点轮0；fallback=0；数值事件`{}`；实际整数后端/LP/证明统计`{'native_calls': 0, 'native_seconds': 0, 'bound_certificates': 0, 'lp_calls': 0, 'lp_seconds': 0, 'peak_variables_removed': 0, 'peak_zero_optional_removed': 0, 'peak_lp_variables_removed': 0}`；clipping`{'num_cpu': 1360, 'num_ram': 776, 'den_cpu': 456, 'den_ram': 0, 'num_any': 1360, 'den_any': 456, 'any_point': 1360, 'selected': 3461}`；裁剪率%`{'num_cpu': 39.29500144466917, 'num_ram': 22.421265530193587, 'den_cpu': 13.175382837330252, 'den_ram': 0.0, 'num_any': 39.29500144466917, 'den_any': 13.175382837330252, 'any_point': 39.29500144466917}`；训练暴露`{'fully_supported': 111110.0, 'clipped': 74458.0}`。
- PROP：执行检查26080，锁检查32，等待轮210，CPU饱和/GPU空闲节点轮0；fallback=0；数值事件`{'CANONICAL_BINARY_BEFORE_RETRY': 22}`；实际整数后端/LP/证明统计`{'native_calls': 1170, 'native_seconds': 0.7558752997720148, 'bound_certificates': 5418, 'lp_calls': 1112, 'lp_seconds': 2.929531400150154, 'peak_variables_removed': 195, 'peak_zero_optional_removed': 195, 'peak_lp_variables_removed': 281}`；clipping`{'num_cpu': 690, 'num_ram': 0, 'den_cpu': 690, 'den_ram': 0, 'num_any': 690, 'den_any': 690, 'any_point': 690, 'selected': 5052}`；裁剪率%`{'num_cpu': 13.657957244655583, 'num_ram': 0.0, 'den_cpu': 13.657957244655583, 'den_ram': 0.0, 'num_any': 13.657957244655583, 'den_any': 13.657957244655583, 'any_point': 13.657957244655583}`；训练暴露`{'fully_supported': 246445.0, 'clipped': 37925.0}`。
- CPU：执行检查26080，锁检查32，等待轮179，CPU饱和/GPU空闲节点轮13；fallback=0；数值事件`{'CANONICAL_BINARY_BEFORE_RETRY': 300, 'FIXED_EXISTING_NO_PRESOLVE_RETRY': 36}`；实际整数后端/LP/证明统计`{'native_calls': 4461, 'native_seconds': 286.9518950008787, 'bound_certificates': 3354, 'lp_calls': 2863, 'lp_seconds': 88.88021410169313, 'peak_variables_removed': 4421, 'peak_zero_optional_removed': 192, 'peak_lp_variables_removed': 7956}`；clipping`{'num_cpu': 2665, 'num_ram': 0, 'den_cpu': 451, 'den_ram': 0, 'num_any': 2665, 'den_any': 451, 'any_point': 2677, 'selected': 4624}`；裁剪率%`{'num_cpu': 57.6340830449827, 'num_ram': 0.0, 'den_cpu': 9.753460207612457, 'den_ram': 0.0, 'num_any': 57.6340830449827, 'den_any': 9.753460207612457, 'any_point': 57.89359861591696}`；训练暴露`{'fully_supported': 107594.0, 'clipped': 151655.0}`。
- MEM：执行检查26320，锁检查32，等待轮191，CPU饱和/GPU空闲节点轮0；fallback=0；数值事件`{'CANONICAL_BINARY_BEFORE_RETRY': 506}`；实际整数后端/LP/证明统计`{'native_calls': 5444, 'native_seconds': 110.56991899941931, 'bound_certificates': 3015, 'lp_calls': 4013, 'lp_seconds': 64.07123649996356, 'peak_variables_removed': 3491, 'peak_zero_optional_removed': 192, 'peak_lp_variables_removed': 4411}`；clipping`{'num_cpu': 621, 'num_ram': 1196, 'den_cpu': 621, 'den_ram': 0, 'num_any': 1284, 'den_any': 621, 'any_point': 1284, 'selected': 4760}`；裁剪率%`{'num_cpu': 13.046218487394958, 'num_ram': 25.126050420168067, 'den_cpu': 13.046218487394958, 'den_ram': 0.0, 'num_any': 26.974789915966387, 'den_any': 13.046218487394958, 'any_point': 26.974789915966387}`；训练暴露`{'fully_supported': 197362.0, 'clipped': 70282.0}`。
- FO：执行检查24240，锁检查32，等待轮160，CPU饱和/GPU空闲节点轮33；fallback=0；数值事件`{'CANONICAL_BINARY_BEFORE_RETRY': 799, 'FIXED_EXISTING_NO_PRESOLVE_RETRY': 12}`；实际整数后端/LP/证明统计`{'native_calls': 6550, 'native_seconds': 544.355565899692, 'bound_certificates': 6239, 'lp_calls': 3737, 'lp_seconds': 885.9841058998136, 'peak_variables_removed': 45107, 'peak_zero_optional_removed': 192, 'peak_lp_variables_removed': 45364}`；clipping`{'num_cpu': 2183, 'num_ram': 1173, 'den_cpu': 566, 'den_ram': 0, 'num_any': 2183, 'den_any': 566, 'any_point': 2206, 'selected': 4148}`；裁剪率%`{'num_cpu': 52.627772420443584, 'num_ram': 28.278688524590162, 'den_cpu': 13.645130183220829, 'den_ram': 0.0, 'num_any': 52.627772420443584, 'den_any': 13.645130183220829, 'any_point': 53.18225650916104}`；训练暴露`{'fully_supported': 107419.0, 'clipped': 123636.0}`。
- A00：执行检查31600，锁检查33，等待轮276，CPU饱和/GPU空闲节点轮0；fallback=0；数值事件`{'CANONICAL_BINARY_BEFORE_RETRY': 3}`；实际整数后端/LP/证明统计`{'native_calls': 1129, 'native_seconds': 0.948307999788085, 'bound_certificates': 16864, 'lp_calls': 12458, 'lp_seconds': 26.829603103746194, 'peak_variables_removed': 0, 'peak_zero_optional_removed': 0, 'peak_lp_variables_removed': 365}`；clipping`{'num_cpu': 427, 'num_ram': 0, 'den_cpu': 427, 'den_ram': 0, 'num_any': 427, 'den_any': 427, 'any_point': 427, 'selected': 4362}`；裁剪率%`{'num_cpu': 9.789087574507107, 'num_ram': 0.0, 'den_cpu': 9.789087574507107, 'den_ram': 0.0, 'num_any': 9.789087574507107, 'den_any': 9.789087574507107, 'any_point': 9.789087574507107}`；训练暴露`{'fully_supported': 223347.0, 'clipped': 22759.0}`。
- A01：执行检查31600，锁检查33，等待轮276，CPU饱和/GPU空闲节点轮0；fallback=0；数值事件`{'CANONICAL_BINARY_BEFORE_RETRY': 3}`；实际整数后端/LP/证明统计`{'native_calls': 1120, 'native_seconds': 0.8468437999254093, 'bound_certificates': 16285, 'lp_calls': 12451, 'lp_seconds': 27.067893700877903, 'peak_variables_removed': 0, 'peak_zero_optional_removed': 0, 'peak_lp_variables_removed': 365}`；clipping`{'num_cpu': 427, 'num_ram': 0, 'den_cpu': 427, 'den_ram': 0, 'num_any': 427, 'den_any': 427, 'any_point': 427, 'selected': 4362}`；裁剪率%`{'num_cpu': 9.789087574507107, 'num_ram': 0.0, 'den_cpu': 9.789087574507107, 'den_ram': 0.0, 'num_any': 9.789087574507107, 'den_any': 9.789087574507107, 'any_point': 9.789087574507107}`；训练暴露`{'fully_supported': 223347.0, 'clipped': 22759.0}`。
- A10：执行检查25520，锁检查32，等待轮163，CPU饱和/GPU空闲节点轮0；fallback=0；数值事件`{'CANONICAL_BINARY_BEFORE_RETRY': 2}`；实际整数后端/LP/证明统计`{'native_calls': 1371, 'native_seconds': 0.8849818007729482, 'bound_certificates': 9419, 'lp_calls': 5044, 'lp_seconds': 19.9378384994925, 'peak_variables_removed': 0, 'peak_zero_optional_removed': 0, 'peak_lp_variables_removed': 443}`；clipping`{'num_cpu': 451, 'num_ram': 0, 'den_cpu': 451, 'den_ram': 0, 'num_any': 451, 'den_any': 451, 'any_point': 451, 'selected': 4970}`；裁剪率%`{'num_cpu': 9.074446680080483, 'num_ram': 0.0, 'den_cpu': 9.074446680080483, 'den_ram': 0.0, 'num_any': 9.074446680080483, 'den_any': 9.074446680080483, 'any_point': 9.074446680080483}`；训练暴露`{'fully_supported': 253280.0, 'clipped': 24946.0}`。
- A11：执行检查25440，锁检查32，等待轮202，CPU饱和/GPU空闲节点轮0；fallback=0；数值事件`{'CANONICAL_BINARY_BEFORE_RETRY': 18}`；实际整数后端/LP/证明统计`{'native_calls': 1212, 'native_seconds': 0.9635598001768813, 'bound_certificates': 6980, 'lp_calls': 3264, 'lp_seconds': 10.804483501560753, 'peak_variables_removed': 0, 'peak_zero_optional_removed': 0, 'peak_lp_variables_removed': 318}`；clipping`{'num_cpu': 448, 'num_ram': 0, 'den_cpu': 448, 'den_ram': 0, 'num_any': 448, 'den_any': 448, 'any_point': 448, 'selected': 4879}`；裁剪率%`{'num_cpu': 9.182209469153515, 'num_ram': 0.0, 'den_cpu': 9.182209469153515, 'den_ram': 0.0, 'num_any': 9.182209469153515, 'den_any': 9.182209469153515, 'any_point': 9.182209469153515}`；训练暴露`{'fully_supported': 252591.0, 'clipped': 24912.0}`。

整数后端总次数包含支撑集上界子问题、重试和CBC；外层solver_trace不是实际后端次数。后端计时含其调用封装，选择/优化计时还含候选构建和审计。仿真不把求解墙钟加入作业训练时钟，开销应单列，不应当作真实性能。

## 6. 长尾与局部训练收益

- dual_medium_SIA_AUX vs dual_medium_SIA_P：最大受益cifar10-10066（原cifar10-57），节省13061.00秒；占净节省8.73%；剔除此单作业的**诊断**改善18.75%，不替代80作业主结果。应用净贡献`{"cifar10": {"count": 48, "sum_saved_seconds": 98725.0, "improved": 33, "worse": 15}, "deepspeech2": {"count": 16, "sum_saved_seconds": -3048.0, "improved": 8, "worse": 8}, "bert": {"count": 16, "sum_saved_seconds": 53960.0, "improved": 15, "worse": 1}}`。
- dual_medium_PROP vs dual_medium_SIA_P：最大受益bert-10009（原bert-37），节省11075.00秒；占净节省16.67%；剔除此单作业的**诊断**改善7.57%，不替代80作业主结果。应用净贡献`{"cifar10": {"count": 48, "sum_saved_seconds": 41606.0, "improved": 25, "worse": 23}, "deepspeech2": {"count": 16, "sum_saved_seconds": -72648.0, "improved": 4, "worse": 12}, "bert": {"count": 16, "sum_saved_seconds": 97467.0, "improved": 15, "worse": 1}}`。
- dual_medium_CPU vs dual_medium_SIA_P：最大受益bert-10009（原bert-37），节省11134.00秒；占净节省7.39%；剔除此单作业的**诊断**改善19.08%，不替代80作业主结果。应用净贡献`{"cifar10": {"count": 48, "sum_saved_seconds": 83350.0, "improved": 26, "worse": 22}, "deepspeech2": {"count": 16, "sum_saved_seconds": -56546.0, "improved": 4, "worse": 12}, "bert": {"count": 16, "sum_saved_seconds": 123812.0, "improved": 16, "worse": 0}}`。
- dual_medium_MEM vs dual_medium_SIA_P：最大受益bert-10014（原bert-54），节省11100.00秒；占净节省8.19%；剔除此单作业的**诊断**改善17.06%，不替代80作业主结果。应用净贡献`{"cifar10": {"count": 48, "sum_saved_seconds": 85081.0, "improved": 27, "worse": 21}, "deepspeech2": {"count": 16, "sum_saved_seconds": -64558.0, "improved": 5, "worse": 11}, "bert": {"count": 16, "sum_saved_seconds": 115008.0, "improved": 15, "worse": 1}}`。
- dual_medium_FO vs dual_medium_SIA_P：最大受益bert-10014（原bert-54），节省11849.00秒；占净节省5.53%；剔除此单作业的**诊断**改善27.77%，不替代80作业主结果。应用净贡献`{"cifar10": {"count": 48, "sum_saved_seconds": 123628.0, "improved": 35, "worse": 13}, "deepspeech2": {"count": 16, "sum_saved_seconds": -32405.0, "improved": 7, "worse": 9}, "bert": {"count": 16, "sum_saved_seconds": 123192.0, "improved": 16, "worse": 0}}`。
- dual_medium_CPU vs dual_medium_PROP：最大受益bert-10039（原bert-15），节省6758.00秒；占净节省8.03%；剔除此单作业的**诊断**改善11.66%，不替代80作业主结果。应用净贡献`{"cifar10": {"count": 48, "sum_saved_seconds": 41744.0, "improved": 44, "worse": 3}, "deepspeech2": {"count": 16, "sum_saved_seconds": 16102.0, "improved": 13, "worse": 3}, "bert": {"count": 16, "sum_saved_seconds": 26345.0, "improved": 16, "worse": 0}}`。
- dual_medium_MEM vs dual_medium_PROP：最大受益bert-10039（原bert-15），节省7058.00秒；占净节省10.21%；剔除此单作业的**诊断**改善9.34%，不替代80作业主结果。应用净贡献`{"cifar10": {"count": 48, "sum_saved_seconds": 43475.0, "improved": 46, "worse": 2}, "deepspeech2": {"count": 16, "sum_saved_seconds": 8090.0, "improved": 12, "worse": 4}, "bert": {"count": 16, "sum_saved_seconds": 17541.0, "improved": 14, "worse": 2}}`。
- dual_medium_FO vs dual_medium_PROP：最大受益bert-10039（原bert-15），节省7118.00秒；占净节省4.81%；剔除此单作业的**诊断**改善21.21%，不替代80作业主结果。应用净贡献`{"cifar10": {"count": 48, "sum_saved_seconds": 82022.0, "improved": 46, "worse": 2}, "deepspeech2": {"count": 16, "sum_saved_seconds": 40243.0, "improved": 14, "worse": 2}, "bert": {"count": 16, "sum_saved_seconds": 25725.0, "improved": 13, "worse": 3}}`。
- dual_neutral_A10 vs dual_neutral_A00：最大受益bert-10034（原bert-0），节省19845.00秒；占净节省5.86%；剔除此单作业的**诊断**改善34.65%，不替代80作业主结果。应用净贡献`{"cifar10": {"count": 48, "sum_saved_seconds": 217687.0, "improved": 36, "worse": 10}, "deepspeech2": {"count": 16, "sum_saved_seconds": -113104.0, "improved": 0, "worse": 16}, "bert": {"count": 16, "sum_saved_seconds": 234192.0, "improved": 16, "worse": 0}}`。
- dual_neutral_A01 vs dual_neutral_A00：最大受益cifar10-10000（原cifar10-0），节省0.00秒；占净节省—%；剔除此单作业的**诊断**改善0.00%，不替代80作业主结果。应用净贡献`{"cifar10": {"count": 48, "sum_saved_seconds": 0.0, "improved": 0, "worse": 0}, "deepspeech2": {"count": 16, "sum_saved_seconds": 0.0, "improved": 0, "worse": 0}, "bert": {"count": 16, "sum_saved_seconds": 0.0, "improved": 0, "worse": 0}}`。
- dual_neutral_A11 vs dual_neutral_A10：最大受益deepspeech2-10013（原deepspeech2-12），节省5303.00秒；占净节省—%；剔除此单作业的**诊断**改善-2.54%，不替代80作业主结果。应用净贡献`{"cifar10": {"count": 48, "sum_saved_seconds": 5153.0, "improved": 35, "worse": 11}, "deepspeech2": {"count": 16, "sum_saved_seconds": 30201.0, "improved": 13, "worse": 3}, "bert": {"count": 16, "sum_saved_seconds": -44966.0, "improved": 0, "worse": 16}}`。
- dual_neutral_A11 vs dual_neutral_A01：最大受益bert-10034（原bert-0），节省18105.00秒；占净节省5.50%；剔除此单作业的**诊断**改善33.80%，不替代80作业主结果。应用净贡献`{"cifar10": {"count": 48, "sum_saved_seconds": 222840.0, "improved": 36, "worse": 0}, "deepspeech2": {"count": 16, "sum_saved_seconds": -82903.0, "improved": 0, "worse": 16}, "bert": {"count": 16, "sum_saved_seconds": 189226.0, "improved": 16, "worse": 0}}`。
- dual_neutral_A11 vs dual_medium_SIA_P：最大受益cifar10-10022（原cifar10-19），节省10498.00秒；占净节省8.17%；剔除此单作业的**诊断**改善16.18%，不替代80作业主结果。应用净贡献`{"cifar10": {"count": 48, "sum_saved_seconds": 61442.0, "improved": 29, "worse": 18}, "deepspeech2": {"count": 16, "sum_saved_seconds": -46508.0, "improved": 6, "worse": 10}, "bert": {"count": 16, "sum_saved_seconds": 113588.0, "improved": 15, "worse": 1}}`。
- PROP同状态/同raw/config/batch的比例参考等效训练秒`{'receiver_gain': 0.0, 'donor_loss': 0.0, 'net': 0.0, 'interpretation': 'Within-logged-segment fixed raw/config/batch reference; not a replay or causal JCT attribution.'}`，其中数据内/裁剪支持分层`{'fully_supported': {'receiver_gain': 0.0, 'donor_loss': 0.0, 'net': 0.0}, 'clipped': {'receiver_gain': 0.0, 'donor_loss': 0.0, 'net': 0.0}}`；这不是完整反事实重放或JCT因果贡献。
- CPU同状态/同raw/config/batch的比例参考等效训练秒`{'receiver_gain': 37180.88015584466, 'donor_loss': 19051.647695990734, 'net': 18129.232459853924, 'interpretation': 'Within-logged-segment fixed raw/config/batch reference; not a replay or causal JCT attribution.'}`，其中数据内/裁剪支持分层`{'fully_supported': {'receiver_gain': 36904.43663494297, 'donor_loss': 73.92820616546791, 'net': 36830.5084287775}, 'clipped': {'receiver_gain': 276.4435209016789, 'donor_loss': 18977.719489825264, 'net': -18701.275968923586}}`；这不是完整反事实重放或JCT因果贡献。
- MEM同状态/同raw/config/batch的比例参考等效训练秒`{'receiver_gain': 29365.04703574602, 'donor_loss': 15833.648953560061, 'net': 13531.39808218596, 'interpretation': 'Within-logged-segment fixed raw/config/batch reference; not a replay or causal JCT attribution.'}`，其中数据内/裁剪支持分层`{'fully_supported': {'receiver_gain': 28299.283050436603, 'donor_loss': 1281.662610871961, 'net': 27017.620439564642}, 'clipped': {'receiver_gain': 1065.763985309547, 'donor_loss': 14551.986342688117, 'net': -13486.22235737857}}`；这不是完整反事实重放或JCT因果贡献。
- FULL同状态/同raw/config/batch的比例参考等效训练秒`{'receiver_gain': 2393.6575993081415, 'donor_loss': 219.65880173871685, 'net': 2173.998797569425, 'interpretation': 'Within-logged-segment fixed raw/config/batch reference; not a replay or causal JCT attribution.'}`，其中数据内/裁剪支持分层`{'fully_supported': {'receiver_gain': 1265.8521853400312, 'donor_loss': 0.0, 'net': 1265.8521853400312}, 'clipped': {'receiver_gain': 1127.8054139681085, 'donor_loss': 219.65880173871685, 'net': 908.1466122293916}}`；这不是完整反事实重放或JCT因果贡献。
- FO同状态/同raw/config/batch的比例参考等效训练秒`{'receiver_gain': 74295.89197317259, 'donor_loss': 21852.51686623065, 'net': 52443.37510694194, 'interpretation': 'Within-logged-segment fixed raw/config/batch reference; not a replay or causal JCT attribution.'}`，其中数据内/裁剪支持分层`{'fully_supported': {'receiver_gain': 69253.14741940098, 'donor_loss': 232.25252333540885, 'net': 69020.89489606557}, 'clipped': {'receiver_gain': 5042.744553771985, 'donor_loss': 21620.264342895236, 'net': -16577.519789123253}}`；这不是完整反事实重放或JCT因果贡献。
- SIA_AUX同状态/同raw/config/batch的比例参考等效训练秒`{'receiver_gain': 33659.47505615129, 'donor_loss': 13412.782447409658, 'net': 20246.692608741636, 'interpretation': 'Within-logged-segment fixed raw/config/batch reference; not a replay or causal JCT attribution.'}`，其中数据内/裁剪支持分层`{'fully_supported': {'receiver_gain': 32837.11010210546, 'donor_loss': 2486.153944887616, 'net': 30350.95615721784}, 'clipped': {'receiver_gain': 822.3649540457064, 'donor_loss': 10926.628502522084, 'net': -10104.263548476378}}`；这不是完整反事实重放或JCT因果贡献。

最差8作业的逐项JCT、首轮/后续等待、训练、检查点和重启见audit/tail_diagnostics.json；主分析不删长任务。

## 7. 历史回归及可复现性

- dual_medium_SIA_P：新状态ALL_COMPLETE；与旧完整结果逐作业完成/JCT/互斥时间/restart等精确一致=True。
- dual_medium_SIA_AUX：新状态ALL_COMPLETE；与旧完整结果逐作业完成/JCT/互斥时间/restart等精确一致=True。
- dual_neutral_A00：新状态ALL_COMPLETE；与旧完整结果逐作业完成/JCT/互斥时间/restart等精确一致=True。
- dual_neutral_A01：新状态ALL_COMPLETE；与旧完整结果逐作业完成/JCT/互斥时间/restart等精确一致=True。
- dual_neutral_A11：新状态ALL_COMPLETE；与旧完整结果逐作业完成/JCT/互斥时间/restart等精确一致=True。

若有差异，old_complete_regressions.json逐项保留，不因均值接近就认定相同，不挑选较好版本。源码及依赖快照见工程冻结目录；本目录保留新授权、预检、固定命令、逐轮压缩JSON、每格manifest/hash、逐作业CSV和独立审计。科学条件一致不等于预测模型已被真实硬件校准。

本轮批次实测墙钟 **3.903小时**。全部11格来自同一315文件修复版，没有与旧5个完成格混成新矩阵。

## 8. 结论边界

这是一组受控80作业、三类GPU、合成CPU/RAM迁移的机制对照，不能证明Sia普遍存在缺陷，也不是同平台真实Saturn测量。E-legacy分子/分母裁剪仍存在，报告必须与clipping暴露一同解释。更大规模/跨负载结论不由这一个场景保证。对于未完成或失败格，保持证据、不给完整JCT结论；不通过删任务、改响应、改预算内配额或改目标使方法获胜。本轮结束后停止，不自动追加可选4格、104/160作业或调参。
