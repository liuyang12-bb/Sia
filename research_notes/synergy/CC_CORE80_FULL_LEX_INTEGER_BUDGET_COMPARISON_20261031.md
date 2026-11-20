# FULL词典序修复及80作业完整资源对照

**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**

正式矩阵完整 **7/7格**，独立原始事件审计违规 **0项**，补跑墙钟 **6.552小时**。不插补删失JCT，不将不同源码版本混成配对。

从本轮最初修复准备记录起至报告生成的总历时约12.938小时，包含失败复现、开发、测试、验收、等待和正式补跑；这不是模拟时间。各次失败的源码与日志均保留。

## 1. 直接回答

- FULL vs SIA_P：平均JCT改善29.620%，P95/P99/Makespan改善0.272/-18.289/-14.499%，逐作业改善/退化/持平=55/25/0。
- FULL vs SIA_AUX：平均JCT改善11.879%，P95/P99/Makespan改善-8.476/-24.298/-25.557%，逐作业改善/退化/持平=42/38/0。
- FULL vs PROP：平均JCT改善22.713%，P95/P99/Makespan改善10.794/-7.121/-8.649%，逐作业改善/退化/持平=74/6/0。
- FULL vs CPU：平均JCT改善11.734%，P95/P99/Makespan改善3.971/-7.999/-8.655%，逐作业改善/退化/持平=71/9/0。
- FULL vs MEM：平均JCT改善13.925%，P95/P99/Makespan改善10.135/-5.856/-7.476%，逐作业改善/退化/持平=72/7/1。
- FULL vs FO：平均JCT改善1.086%，P95/P99/Makespan改善-2.706/-19.184/-16.691%，逐作业改善/退化/持平=61/18/1。
正值为改善、负值为退化。单轮目标更优不保证平均JCT更低；FULL不必优于FO或每个单资源版本。

## 2. 根因、证据与最小修复

V7保存的真实第42轮故障已从启动阶段转移到后续规范排名：deepspeech2-10008，整数排名可行值8186、LP下界0，已有4组约束（物理/作业、主目标、启动225、此前作业掩码）。V8将同一预算证明扩展到一般非负整数词典序目标，全部保留前级启动及掩码；不混淆启动成本与排名编号。对预算8185，零gap主目标全局上界2.569973689748低于原下界2.570187416674，证明8186是最优排名。新增红测直接捕获旧入口返回None；没有丢弃前级约束或接受旧incumbent代替证明。

本次最终修复链：旧第19轮per-job排名、V5第37轮启动次目标、V6第42轮启动次目标均保留原始失败。第42轮168184列，已验证启动可行值225，LP下界86.4117；不能把可行值当最优。V7将更优启动预算限制为220，在等价GPU分组和合法容量cover下最大化原主目标，零gap全局上界约2.5684724低于原主目标下界2.5701874，因而证明225全局最小。一般实现仅使用当前轮合法incumbent及整数gcd，预算由V-g推导，不写死225/220；超时或无证明会回到原后端/显式失败，不静默提交。

旧冻结修复版在第19次决策失败：131244列，前级主目标区间约[1.3341274568444956,1.3341274588444958]，启动代价275±1e-9，当前排名对象deepspeech2-10003；合法整数可行解排名成本5，旧LP排名下界0，两次HiGHS超时及CBC optimal_inaccurate不能证明成本5最优。失败不是容量不可行，也不是训练死锁。旧现场和封存hash未改。

旧版只用当前排名目标的下界和逐配置支配关系消元，遗漏已固定主目标对其他变量的约束；在窄浮点带、大量近似/重复列下搜索困难。单独把整数启动行等价改写成=275仍超时，probe_lattice.json保留此证据。

实现还将可行域消元前移到当前排名LP及其支持集证明之前；这些证明仅在已认证的等价域上执行，再把支持集、约化成本及整数格点索引还原到原变量编号。没有调整LP/MIP时间限或公平/完成目标。V4部分回归日志与源码见相邻run1/DEVELOPMENT_SUPERSEDED.json；中断日志未修补、没有启动正式实验。

第二个封存故障：V5在第37次决策的第二级启动目标超时，200462列，旧可行启动成本250，旧LP下界102.6124不足以证明最优，36完整轮/9完成。V6不再要求启动最优先固定才启用主目标域证明；仅主目标带已固定即可。离线模型排除171141列，保留29321列，仍以原约束和原30秒零gap求解，证明启动最优值250。没有更换数据、增大时间限或用未证明点继续。

最小工程修复：①整数行gcd格点向内取界，只有原数学域和原audit容差域都相同时才改写；该启动行化为精确275。②使用既有主目标带进行可证明安全的条件弱对偶消元。设c=−w，外包约束Ax≤b、单位盒；任意y≤0给出B=bᵀy+Σmin(0,cᵢ−Aᵢᵀy)。若xᵢ=1，目标下界至少B+max(0,cᵢ−Aᵢᵀy)。仅当它严格超过−L及原容差/浮点误差的保守上界时删除变量。这里L是原主目标下界，未增加新优化目标。

使用80位有向Decimal区间；包含原验证容差与保守点积误差。缓存绑定当前列对象及前三组原始约束的字节hash，后续加约束可复用，前缀改变则失效；LP失败不删变量；丢失现有合法incumbent则明确停止。全部返回解还须全局最优且逐行通过原始约束。候选网格、Goodput/评分、目标带、CPU/RAM容量、R0/phi/clipping和任何训练/锁语义均未修改。

同一捕获模型红/绿：旧版115.331秒后再次拒绝；修复重放14.558秒取得最优排名5.0，主目标条件证书排除113532列。只是求解实现改进，不是训练速度或JCT收益。

改动文件：`engineering_repair.py`, `numeric_repair.py`, `supervise80.py`；新增`capacity_cover.py`, `epsilon_lex_repair.py`, `gpu_group_formulation.py`, `incumbent_complement.py`, `integer_row_lattice.py`, `primary_band_presolve.py`, `sia-simulator/tests/fixtures/core80_full_rank_step19.json.gz`, `sia-simulator/tests/fixtures/core80_full_rank_step42.json.gz`, `sia-simulator/tests/fixtures/core80_full_startup_step37.json.gz`, `sia-simulator/tests/fixtures/core80_full_startup_step42.json.gz`, `sia-simulator/tests/test_capacity_cover_repair.py`, `sia-simulator/tests/test_epsilon_lex_repair.py`, `sia-simulator/tests/test_integer_lex_epsilon_repair.py`, `sia-simulator/tests/test_rank_lattice_repair.py`。supervise80只新增默认关闭的工程失败停止选项；本次开启以在新工程失败时停止批次，非调度规则。完整diff在source_changes.patch。

- 代码依据：[sia-simulator/joint_prototype/clean_ablation.py::_decide](D:/科研启航/Sia/cc-joint-core80-full-lex-repair-v8/sia-simulator/joint_prototype/clean_ablation.py:13)。
- 代码依据：[numeric_repair.py::canonical_bundle](D:/科研启航/Sia/cc-joint-core80-full-lex-repair-v8/numeric_repair.py:13)。
- 代码依据：[integer_row_lattice.py::tighten_integer_rows](D:/科研启航/Sia/cc-joint-core80-full-lex-repair-v8/integer_row_lattice.py:16)。
- 代码依据：[primary_band_presolve.py::primary_certificate](D:/科研启航/Sia/cc-joint-core80-full-lex-repair-v8/primary_band_presolve.py:50)。
- 代码依据：[primary_band_presolve.py::certificate_on_domain](D:/科研启航/Sia/cc-joint-core80-full-lex-repair-v8/primary_band_presolve.py:100)。
- 代码依据：[capacity_cover.py::strengthen](D:/科研启航/Sia/cc-joint-core80-full-lex-repair-v8/capacity_cover.py:16)。
- 代码依据：[gpu_group_formulation.py::prepare](D:/科研启航/Sia/cc-joint-core80-full-lex-repair-v8/gpu_group_formulation.py:20)。
- 代码依据：[epsilon_lex_repair.py::solve](D:/科研启航/Sia/cc-joint-core80-full-lex-repair-v8/epsilon_lex_repair.py:42)。
- 代码依据：[epsilon_lex_repair.py::outer_rows](D:/科研启航/Sia/cc-joint-core80-full-lex-repair-v8/epsilon_lex_repair.py:24)。
## 3. 验收、冻结及公平条件

```json
{
  "status": "PASS",
  "existing_tests": 302,
  "new_tests": 33,
  "total_tests": 335,
  "full_suite": "tests/full_suite_final.log",
  "full_real_prefix": 45,
  "old_failure_decision": 19,
  "original_sia_exact_rounds": 3,
  "bridges": [
    {
      "method": "FULL",
      "compared_rounds": 41,
      "exact_match": true,
      "different_rounds": [],
      "ignored": "Relocated template_source only; byte hashes preserved"
    },
    {
      "method": "A10",
      "compared_rounds": 20,
      "exact_match": true,
      "different_rounds": [],
      "ignored": "Relocated template_source only; byte hashes preserved"
    },
    {
      "method": "PROP",
      "compared_rounds": 12,
      "exact_match": true,
      "different_rounds": [],
      "ignored": "Relocated template_source only; byte hashes preserved"
    }
  ],
  "execution_checks": 6160,
  "locks": 25,
  "violations": 0
}
```

红测red_backend_contract2.log捕获原后端未使用等价整数行；red_literal_replay.log复现真实131244列失败。red_backend_contract.log仅是首次测试cwd导入错误，不能算缺陷红测。probe_primary.log的离线工具稠密/稀疏兼容错误保留，修正后probe_primary2.log完成；未修改原矩阵。

来源：实际315文件旧源码完整复制，保持旧树及官方源；最终329文件，source_snapshot.zip SHA256 `B0DCAEB4A5A205B6CA27893F3ACFE1F43C3F850CA8AF6E5BC79B15029518D6EB`，输入38项逐一核验。没有可假定代表实际源码的裸Git HEAD；完整快照、patch和测试fixture共同重建，不推送。

335项测试含原302项与累计新增33项；真实FULL45轮跨越已保存的第19、37、42轮失败，A1020/PROP12短回归，原Sia3轮精确回归。V4开发版315/316项通过，V5为318项通过但正式第37轮失败，V6为320项通过但真实45轮验收在第42轮失败；这些版本均单独保留，不补写中断日志、不填补总体JCT。V7新增等价分组、容量cover、预算判定与错误输出保护回归。这里是有界正确性验收，不是80作业完成证据；下表来自全程正式补跑。

80作业、48CIFAR10/16DeepSpeech2/16BERT；40@0、20@600、20@1200的既有受控到达。原始训练模型、work、batch不变。AWS4GPU/12合成CPU/250GB、DGX-ext8/24/500、RTX8/24/500。R0+E-legacy、Medium alpha=.5、seed_profiles(1,8)、60秒/6000轮，7格独立进程串行。门槛2GiB/5GiB，每格计算watchdog6h、批次24h，不调整求解器30秒/零gap或原重试设置。

FULL/CPU/MEM/PROP共享完成导向目标、GPU域、服务/准入、切换成本，分别开放双资源、仅CPU、仅RAM、比例配额。FO共享CC目标/准入，先在比例域确定服务集合与GPU，再冻结GPU配置做辅助优化。FULL vs FO检验联合与串行决策，不单独识别服务集合和GPU选择贡献。SIA_P实际调用官方Sia GPU/placement后比例执行，不加CC/L2；SIA_AUX固定该意图后用自有辅助优化，不是官方Synergy-TUNE。

## 4. 七格绝对数值

|方法|状态|完成|Mean JCT|Median|P95|P99|Makespan|轮数|墙钟(s)|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FULL | ALL_COMPLETE | 80/80 | 6538.913 | 5068.000 | 15735.100 | 20286.420 | 21204.000 | 354 | 12724.756 |
| SIA_P | ALL_COMPLETE | 80/80 | 9290.875 | 10475.500 | 15778.000 | 17149.940 | 18519.000 | 309 | 305.447 |
| SIA_AUX | ALL_COMPLETE | 80/80 | 7420.413 | 6904.000 | 14505.650 | 16320.760 | 16888.000 | 282 | 1131.252 |
| PROP | ALL_COMPLETE | 80/80 | 8460.562 | 7426.000 | 17639.150 | 18937.840 | 19516.000 | 326 | 430.531 |
| CPU | ALL_COMPLETE | 80/80 | 7408.175 | 6144.000 | 16385.850 | 18783.860 | 19515.000 | 326 | 795.959 |
| MEM | ALL_COMPLETE | 80/80 | 7596.738 | 6369.000 | 17509.800 | 19164.150 | 19729.000 | 329 | 685.939 |
| FO | ALL_COMPLETE | 80/80 | 6610.688 | 5536.500 | 15320.500 | 17021.070 | 18171.000 | 303 | 7513.762 |

P95/P99是单一80作业场景经验分位数，不把80个作业当80次独立实验。失败/删失格均值留空。

## 5. 配对收益、互斥时间和双资源交互

|候选 vs 对照|平均JCT改善%|节省秒/作业|等待/检查点/训练节省秒|改善/退化/平|
|---|---:|---:|---|---|
|FULL vs SIA_P|29.620|2751.962|3183.000/86.562/-517.600|55/25/0|
|FULL vs SIA_AUX|11.879|881.500|1591.500/44.188/-754.188|42/38/0|
|FULL vs PROP|22.713|1921.650|1435.500/5.312/480.837|74/6/0|
|FULL vs CPU|11.734|869.262|706.500/-4.062/166.825|71/9/0|
|FULL vs MEM|13.925|1057.825|792.000/-5.938/271.762|72/7/1|
|FULL vs FO|1.086|71.775|265.500/-8.125/-185.600|61/18/1|
|SIA_AUX vs SIA_P|20.132|1870.462|1591.500/42.375/236.588|56/24/0|
|PROP vs SIA_P|8.937|830.312|1747.500/81.250/-998.438|44/36/0|
|CPU vs SIA_P|20.264|1882.700|2476.500/90.625/-684.425|46/34/0|
|MEM vs SIA_P|18.234|1694.137|2391.000/92.500/-789.362|47/33/0|
|FO vs SIA_P|28.848|2680.188|2917.500/94.688/-332.000|58/22/0|
|CPU vs PROP|12.439|1052.387|729.000/9.375/314.012|73/6/1|
|MEM vs PROP|10.210|863.825|643.500/11.250/209.075|72/8/0|
|FO vs PROP|21.865|1849.875|1170.000/13.438/666.438|73/7/0|
|FO vs SIA_AUX|10.912|809.725|1326.000/52.312/-568.587|41/39/0|

CPU×RAM四格（只有PROP/CPU/MEM/FULL全部完成才计算）：
```json
{
  "cpu_saved_mean_seconds": 1055.1062499999998,
  "memory_saved_mean_seconds": 866.5437499999998,
  "jct_interaction": -5.4375,
  "sign": "Positive main effects reduce JCT; negative interaction is additional reduction. Same completion objective/FIFO/continuity, variable auxiliary factors only."
}
```

主效应为开启该因子的平均JCT节省秒；交互YFULL−YCPU−YMEM+YPROP，负值代表超加性改善。局部实际训练加速可能通过释放资源间接降低排队，时间分解只是会计恒等式，不能将全部等待减少独占归因某模块。

## 6. 资源、重启与约束审计

|方法|GPU-h|CPU单位-h|RAM GB-h|等待(s)|检查点(s)|训练(s)|restart|wasted GPU-s|违规|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FULL | 79.981 | 266.585 | 5331.912 | 260820.000 | 16390.000 | 245903.000 | 280.000 | 18775.000 | 0 |
| SIA_P | 99.727 | 299.181 | 6232.934 | 515460.000 | 23315.000 | 204495.000 | 377.000 | 30475.000 | 0 |
| SIA_AUX | 90.507 | 266.145 | 5447.462 | 388140.000 | 19925.000 | 185568.000 | 332.000 | 26185.000 | 0 |
| PROP | 102.512 | 307.537 | 6407.014 | 375660.000 | 16815.000 | 284370.000 | 287.000 | 22670.000 | 0 |
| CPU | 95.285 | 290.234 | 5955.330 | 317340.000 | 16065.000 | 259249.000 | 267.000 | 20835.000 | 0 |
| MEM | 95.490 | 286.471 | 6057.931 | 324180.000 | 15915.000 | 267644.000 | 268.000 | 20045.000 | 0 |
| FO | 83.822 | 258.022 | 5265.187 | 282060.000 | 15740.000 | 231055.000 | 266.000 | 20700.000 | 0 |
- FULL：执行审计28320、完整锁检查32、fallback0，求解状态`{'OPTIMAL_CLEAN_AB': 276, 'DRAIN_FOR_HEAD': 77}`；工程统计`{'native_calls': 3843, 'native_seconds': 617.0743132014759, 'bound_certificates': 2256, 'lp_calls': 3758, 'lp_seconds': 803.3712825002731, 'primary_band_lp_calls': 463, 'primary_band_lp_seconds': 1288.4174702999298, 'integer_rows_tightened': 2064, 'epsilon_queries': 2509, 'epsilon_seconds': 541.1394913005352, 'epsilon_global_certificates': 2463, 'epsilon_unknown': 45, 'peak_variables_removed': 218302, 'peak_zero_optional_removed': 183, 'peak_lp_variables_removed': 210796, 'peak_primary_band_variables_removed': 227807}`；选择总/P99秒`{'total': 7289.254030399461, 'p99': 90.76997965200344, 'definition': 'Logged PolicyResult.solve_seconds; includes multi-MIP selection/control work, not an isolated single-backend-call timer.'}`。
- SIA_P：执行审计24720、完整锁检查77、fallback0，求解状态`{'ORIGINAL_SIA_OPTIMAL': 308}`；工程统计`{'native_calls': 0, 'native_seconds': 0, 'bound_certificates': 0, 'lp_calls': 0, 'lp_seconds': 0, 'primary_band_lp_calls': 0, 'primary_band_lp_seconds': 0, 'integer_rows_tightened': 0, 'peak_variables_removed': 0, 'peak_zero_optional_removed': 0, 'peak_lp_variables_removed': 0, 'peak_primary_band_variables_removed': 0}`；选择总/P99秒`{'total': 15.407287399837514, 'p99': 0.090790894991369, 'definition': 'Logged PolicyResult.solve_seconds; includes multi-MIP selection/control work, not an isolated single-backend-call timer.'}`。
- SIA_AUX：执行审计22560、完整锁检查65、fallback0，求解状态`{'OPTIMAL_FIXED_OFFICIAL_INTENT': 281}`；工程统计`{'native_calls': 0, 'native_seconds': 0, 'bound_certificates': 0, 'lp_calls': 0, 'lp_seconds': 0, 'primary_band_lp_calls': 0, 'primary_band_lp_seconds': 0, 'integer_rows_tightened': 0, 'peak_variables_removed': 0, 'peak_zero_optional_removed': 0, 'peak_lp_variables_removed': 0, 'peak_primary_band_variables_removed': 0}`；选择总/P99秒`{'total': 391.6685528998496, 'p99': 19.15745876001891, 'definition': 'Logged PolicyResult.solve_seconds; includes multi-MIP selection/control work, not an isolated single-backend-call timer.'}`。
- PROP：执行审计26080、完整锁检查32、fallback0，求解状态`{'OPTIMAL_CLEAN_AB': 194, 'DRAIN_FOR_HEAD': 131}`；工程统计`{'native_calls': 1129, 'native_seconds': 0.7485628002905287, 'bound_certificates': 5414, 'lp_calls': 1112, 'lp_seconds': 1.5483980004792102, 'primary_band_lp_calls': 248, 'primary_band_lp_seconds': 0.4007220997882541, 'integer_rows_tightened': 891, 'epsilon_queries': 55, 'epsilon_seconds': 0.08031370007665828, 'epsilon_global_certificates': 40, 'epsilon_unknown': 15, 'peak_variables_removed': 195, 'peak_zero_optional_removed': 195, 'peak_lp_variables_removed': 281, 'peak_primary_band_variables_removed': 273}`；选择总/P99秒`{'total': 28.88584210019326, 'p99': 0.2802554400032386, 'definition': 'Logged PolicyResult.solve_seconds; includes multi-MIP selection/control work, not an isolated single-backend-call timer.'}`。
- CPU：执行审计26080、完整锁检查32、fallback0，求解状态`{'OPTIMAL_CLEAN_AB': 225, 'DRAIN_FOR_HEAD': 100}`；工程统计`{'native_calls': 2576, 'native_seconds': 60.71219770019525, 'bound_certificates': 3709, 'lp_calls': 2862, 'lp_seconds': 15.04885839993949, 'primary_band_lp_calls': 435, 'primary_band_lp_seconds': 7.990446499723475, 'integer_rows_tightened': 1807, 'epsilon_queries': 1243, 'epsilon_seconds': 33.53299609993701, 'epsilon_global_certificates': 1142, 'epsilon_unknown': 101, 'peak_variables_removed': 4421, 'peak_zero_optional_removed': 192, 'peak_lp_variables_removed': 6760, 'peak_primary_band_variables_removed': 7389}`；选择总/P99秒`{'total': 257.96967009990476, 'p99': 3.633436620008433, 'definition': 'Logged PolicyResult.solve_seconds; includes multi-MIP selection/control work, not an isolated single-backend-call timer.'}`。
- MEM：执行审计26320、完整锁检查32、fallback0，求解状态`{'OPTIMAL_CLEAN_AB': 216, 'DRAIN_FOR_HEAD': 112}`；工程统计`{'native_calls': 3144, 'native_seconds': 22.02158739810693, 'bound_certificates': 3270, 'lp_calls': 4012, 'lp_seconds': 11.864928698982112, 'primary_band_lp_calls': 443, 'primary_band_lp_seconds': 4.606686400220497, 'integer_rows_tightened': 2677, 'epsilon_queries': 1821, 'epsilon_seconds': 23.884492000652244, 'epsilon_global_certificates': 1801, 'epsilon_unknown': 19, 'peak_variables_removed': 3491, 'peak_zero_optional_removed': 192, 'peak_lp_variables_removed': 4286, 'peak_primary_band_variables_removed': 4390}`；选择总/P99秒`{'total': 189.91978449991439, 'p99': 2.121815506012356, 'definition': 'Logged PolicyResult.solve_seconds; includes multi-MIP selection/control work, not an isolated single-backend-call timer.'}`。
- FO：执行审计24240、完整锁检查32、fallback0，求解状态`{'OPTIMAL_CLEAN_AB': 221, 'DRAIN_FOR_HEAD': 81}`；工程统计`{'native_calls': 3729, 'native_seconds': 134.2949346996611, 'bound_certificates': 6400, 'lp_calls': 3737, 'lp_seconds': 113.85884460105444, 'primary_band_lp_calls': 535, 'primary_band_lp_seconds': 93.70812779993867, 'integer_rows_tightened': 1891, 'epsilon_queries': 2500, 'epsilon_seconds': 296.6457174999814, 'epsilon_global_certificates': 2481, 'epsilon_unknown': 19, 'peak_variables_removed': 45107, 'peak_zero_optional_removed': 192, 'peak_lp_variables_removed': 9, 'peak_primary_band_variables_removed': 43950}`；选择总/P99秒`{'total': 2398.777006000164, 'p99': 20.711214042003856, 'definition': 'Logged PolicyResult.solve_seconds; includes multi-MIP selection/control work, not an isolated single-backend-call timer.'}`。
资源小时从执行片段allocated_seconds积分，周期中完成后不继续占用计时。checkpoint与训练互斥、rho只乘原始执行率一次；节点容量和锁检查不只看已选中的任务。仿真未将求解墙钟加入训练时间，计算开销须与JCT分开解读。

## 7. 长尾、资源响应和局限

- FULL vs dual_medium_SIA_P：最大受益cifar10-10022（原cifar10-19），节省11118.000秒，占正净收益5.050%；去除此单作业诊断改善28.660%，不替代包含全部80作业的主结果。应用净贡献`{"cifar10": {"count": 48, "sum_saved_seconds": 134332.0, "improved": 33, "worse": 15}, "deepspeech2": {"count": 16, "sum_saved_seconds": -43599.0, "improved": 6, "worse": 10}, "bert": {"count": 16, "sum_saved_seconds": 129424.0, "improved": 16, "worse": 0}}`。
- FULL vs dual_medium_SIA_AUX：最大受益bert-10014（原bert-54），节省10200.000秒，占正净收益14.464%；去除此单作业诊断改善10.388%，不替代包含全部80作业的主结果。应用净贡献`{"cifar10": {"count": 48, "sum_saved_seconds": 35607.0, "improved": 25, "worse": 23}, "deepspeech2": {"count": 16, "sum_saved_seconds": -40551.0, "improved": 5, "worse": 11}, "bert": {"count": 16, "sum_saved_seconds": 75464.0, "improved": 12, "worse": 4}}`。
- FULL vs dual_medium_PROP：最大受益bert-10039（原bert-15），节省7718.000秒，占正净收益5.020%；去除此单作业诊断改善21.985%，不替代包含全部80作业的主结果。应用净贡献`{"cifar10": {"count": 48, "sum_saved_seconds": 92726.0, "improved": 47, "worse": 1}, "deepspeech2": {"count": 16, "sum_saved_seconds": 29049.0, "improved": 12, "worse": 4}, "bert": {"count": 16, "sum_saved_seconds": 31957.0, "improved": 15, "worse": 1}}`。
- FULL vs dual_medium_CPU：最大受益deepspeech2-10018（原deepspeech2-16），节省4220.000秒，占正净收益6.068%；去除此单作业诊断改善11.311%，不替代包含全部80作业的主结果。应用净贡献`{"cifar10": {"count": 48, "sum_saved_seconds": 50982.0, "improved": 46, "worse": 2}, "deepspeech2": {"count": 16, "sum_saved_seconds": 12947.0, "improved": 10, "worse": 6}, "bert": {"count": 16, "sum_saved_seconds": 5612.0, "improved": 15, "worse": 1}}`。
- FULL vs dual_medium_MEM：最大受益deepspeech2-10018（原deepspeech2-16），节省8786.000秒，占正净收益10.382%；去除此单作业诊断改善12.898%，不替代包含全部80作业的主结果。应用净贡献`{"cifar10": {"count": 48, "sum_saved_seconds": 49251.0, "improved": 47, "worse": 1}, "deepspeech2": {"count": 16, "sum_saved_seconds": 20959.0, "improved": 11, "worse": 5}, "bert": {"count": 16, "sum_saved_seconds": 14416.0, "improved": 14, "worse": 1}}`。
- FULL vs dual_medium_FO：最大受益bert-10019（原bert-99），节省6818.000秒，占正净收益118.739%；去除此单作业诊断改善-0.207%，不替代包含全部80作业的主结果。应用净贡献`{"cifar10": {"count": 48, "sum_saved_seconds": 10704.0, "improved": 43, "worse": 5}, "deepspeech2": {"count": 16, "sum_saved_seconds": -11194.0, "improved": 5, "worse": 11}, "bert": {"count": 16, "sum_saved_seconds": 6232.0, "improved": 13, "worse": 2}}`。
- FULL：分子/分母分CPU/RAM的裁剪率% `{'num_cpu': 47.39110707803993, 'num_ram': 26.134301270417424, 'den_cpu': 13.634301270417422, 'den_ram': 0.0, 'num_any': 47.39110707803993, 'den_any': 13.634301270417422, 'any_point': 48.00362976406534}`；数据内/裁剪训练秒`{'fully_supported': 128494.0, 'clipped': 117409.0}`；同节点真实差异化配额且receiver加速的重叠训练对秒312205.0，其中全坐标数据内3357.0。同raw/config/batch比例参考的局部净等效训练秒`{'receiver_gain': 82747.82713520426, 'donor_loss': 19615.65837268348, 'net': 63132.168762520785, 'interpretation': 'Within-logged-segment fixed raw/config/batch reference; not a replay or causal JCT attribution.'}`；这不是完整反事实JCT。
- SIA_P：分子/分母分CPU/RAM的裁剪率% `{'num_cpu': 12.598013591217983, 'num_ram': 0.0, 'den_cpu': 12.598013591217983, 'den_ram': 0.0, 'num_any': 12.598013591217983, 'den_any': 12.598013591217983, 'any_point': 12.598013591217983}`；数据内/裁剪训练秒`{'fully_supported': 179630.0, 'clipped': 24865.0}`；同节点真实差异化配额且receiver加速的重叠训练对秒0.0，其中全坐标数据内0.0。同raw/config/batch比例参考的局部净等效训练秒`{'receiver_gain': 0.0, 'donor_loss': 0.0, 'net': 0.0, 'interpretation': 'Within-logged-segment fixed raw/config/batch reference; not a replay or causal JCT attribution.'}`；这不是完整反事实JCT。
- SIA_AUX：分子/分母分CPU/RAM的裁剪率% `{'num_cpu': 39.29500144466917, 'num_ram': 22.421265530193587, 'den_cpu': 13.175382837330252, 'den_ram': 0.0, 'num_any': 39.29500144466917, 'den_any': 13.175382837330252, 'any_point': 39.29500144466917}`；数据内/裁剪训练秒`{'fully_supported': 111110.0, 'clipped': 74458.0}`；同节点真实差异化配额且receiver加速的重叠训练对秒235721.0，其中全坐标数据内32718.0。同raw/config/batch比例参考的局部净等效训练秒`{'receiver_gain': 33659.47505615129, 'donor_loss': 13412.782447409658, 'net': 20246.692608741636, 'interpretation': 'Within-logged-segment fixed raw/config/batch reference; not a replay or causal JCT attribution.'}`；这不是完整反事实JCT。
- PROP：分子/分母分CPU/RAM的裁剪率% `{'num_cpu': 13.657957244655583, 'num_ram': 0.0, 'den_cpu': 13.657957244655583, 'den_ram': 0.0, 'num_any': 13.657957244655583, 'den_any': 13.657957244655583, 'any_point': 13.657957244655583}`；数据内/裁剪训练秒`{'fully_supported': 246445.0, 'clipped': 37925.0}`；同节点真实差异化配额且receiver加速的重叠训练对秒0.0，其中全坐标数据内0.0。同raw/config/batch比例参考的局部净等效训练秒`{'receiver_gain': 0.0, 'donor_loss': 0.0, 'net': 0.0, 'interpretation': 'Within-logged-segment fixed raw/config/batch reference; not a replay or causal JCT attribution.'}`；这不是完整反事实JCT。
- CPU：分子/分母分CPU/RAM的裁剪率% `{'num_cpu': 57.6340830449827, 'num_ram': 0.0, 'den_cpu': 9.753460207612457, 'den_ram': 0.0, 'num_any': 57.6340830449827, 'den_any': 9.753460207612457, 'any_point': 57.89359861591696}`；数据内/裁剪训练秒`{'fully_supported': 107594.0, 'clipped': 151655.0}`；同节点真实差异化配额且receiver加速的重叠训练对秒286735.0，其中全坐标数据内1003.0。同raw/config/batch比例参考的局部净等效训练秒`{'receiver_gain': 37180.88015584466, 'donor_loss': 19051.647695990734, 'net': 18129.232459853924, 'interpretation': 'Within-logged-segment fixed raw/config/batch reference; not a replay or causal JCT attribution.'}`；这不是完整反事实JCT。
- MEM：分子/分母分CPU/RAM的裁剪率% `{'num_cpu': 13.046218487394958, 'num_ram': 25.126050420168067, 'den_cpu': 13.046218487394958, 'den_ram': 0.0, 'num_any': 26.974789915966387, 'den_any': 13.046218487394958, 'any_point': 26.974789915966387}`；数据内/裁剪训练秒`{'fully_supported': 197362.0, 'clipped': 70282.0}`；同节点真实差异化配额且receiver加速的重叠训练对秒285918.0，其中全坐标数据内24485.0。同raw/config/batch比例参考的局部净等效训练秒`{'receiver_gain': 29365.04703574602, 'donor_loss': 15833.648953560061, 'net': 13531.39808218596, 'interpretation': 'Within-logged-segment fixed raw/config/batch reference; not a replay or causal JCT attribution.'}`；这不是完整反事实JCT。
- FO：分子/分母分CPU/RAM的裁剪率% `{'num_cpu': 52.627772420443584, 'num_ram': 28.278688524590162, 'den_cpu': 13.645130183220829, 'den_ram': 0.0, 'num_any': 52.627772420443584, 'den_any': 13.645130183220829, 'any_point': 53.18225650916104}`；数据内/裁剪训练秒`{'fully_supported': 107419.0, 'clipped': 123636.0}`；同节点真实差异化配额且receiver加速的重叠训练对秒328598.0，其中全坐标数据内4951.0。同raw/config/batch比例参考的局部净等效训练秒`{'receiver_gain': 74295.89197317259, 'donor_loss': 21852.51686623065, 'net': 52443.37510694194, 'interpretation': 'Within-logged-segment fixed raw/config/batch reference; not a replay or causal JCT attribution.'}`；这不是完整反事实JCT。
所有策略共享R0/E-legacy合成跨硬件执行函数；比例点rho=1是参考归一化，不是无CPU瓶颈证明。裁剪允许且未改，尤其不能把边界外推假设下的收益称为目标硬件实测。真实共驻片段、局部预测/执行收益和全程JCT分别报告，禁止互相替代。最差8作业详见audit/tail_diagnostics.json；主结果不删长尾。

## 8. 同版本回归与交付

- SIA_P：ALL_COMPLETE；逐作业完成事件/JCT/等待/训练/检查点/资源小时/重启精确复现旧完整格=True。
- SIA_AUX：ALL_COMPLETE；逐作业完成事件/JCT/等待/训练/检查点/资源小时/重启精确复现旧完整格=True。
- PROP：ALL_COMPLETE；逐作业完成事件/JCT/等待/训练/检查点/资源小时/重启精确复现旧完整格=True。
- CPU：ALL_COMPLETE；逐作业完成事件/JCT/等待/训练/检查点/资源小时/重启精确复现旧完整格=True。
- MEM：ALL_COMPLETE；逐作业完成事件/JCT/等待/训练/检查点/资源小时/重启精确复现旧完整格=True。
- FO：ALL_COMPLETE；逐作业完成事件/JCT/等待/训练/检查点/资源小时/重启精确复现旧完整格=True。
若有差异，control_regressions.json保留逐项证据，不因均值接近就认定相同，不选择较好一次。完整正式结果均来自同一实际源码快照；旧结果只用于回归核验，不填入新矩阵。legacy native计数不含独立列出的epsilon_queries/epsilon_seconds；总选择墙钟包含全部数值证明，求解墙钟不计入模拟JCT。

原始rounds.jsonl.gz、solver诊断、输入/源码hash、source_snapshot.zip、source_changes.patch、335项测试和短回归均保留。本轮停止，不追加容量/响应档、九cohort、104/160或参数搜索。
