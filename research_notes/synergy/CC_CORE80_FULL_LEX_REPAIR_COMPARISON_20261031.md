# FULL词典序修复及80作业完整资源对照

**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**

正式矩阵完整 **0/7格**，独立原始事件审计违规 **0项**，补跑墙钟 **0.520小时**。不插补删失JCT，不将不同源码版本混成配对。

## 1. 直接回答

- FULL vs SIA_P：不完整配对，禁止给出总体JCT改善率。
- FULL vs SIA_AUX：不完整配对，禁止给出总体JCT改善率。
- FULL vs PROP：不完整配对，禁止给出总体JCT改善率。
- FULL vs CPU：不完整配对，禁止给出总体JCT改善率。
- FULL vs MEM：不完整配对，禁止给出总体JCT改善率。
- FULL vs FO：不完整配对，禁止给出总体JCT改善率。
正值为改善、负值为退化。单轮目标更优不保证平均JCT更低；FULL不必优于FO或每个单资源版本。

## 2. 根因、证据与最小修复

旧冻结修复版在第19次决策失败：131244列，前级主目标区间约[1.3341274568444956,1.3341274588444958]，启动代价275±1e-9，当前排名对象deepspeech2-10003；合法整数可行解排名成本5，旧LP排名下界0，两次HiGHS超时及CBC optimal_inaccurate不能证明成本5最优。失败不是容量不可行，也不是训练死锁。旧现场和封存hash未改。

旧版只用当前排名目标的下界和逐配置支配关系消元，遗漏已固定主目标对其他变量的约束；在窄浮点带、大量近似/重复列下搜索困难。单独把整数启动行等价改写成=275仍超时，probe_lattice.json保留此证据。

实现还将可行域消元前移到当前排名LP及其支持集证明之前；这些证明仅在已认证的等价域上执行，再把支持集、约化成本及整数格点索引还原到原变量编号。没有调整LP/MIP时间限或公平/完成目标。V4部分回归日志与源码见相邻run1/DEVELOPMENT_SUPERSEDED.json；中断日志未修补、没有启动正式实验。

最小工程修复：①整数行gcd格点向内取界，只有原数学域和原audit容差域都相同时才改写；该启动行化为精确275。②使用既有主目标带进行可证明安全的条件弱对偶消元。设c=−w，外包约束Ax≤b、单位盒；任意y≤0给出B=bᵀy+Σmin(0,cᵢ−Aᵢᵀy)。若xᵢ=1，目标下界至少B+max(0,cᵢ−Aᵢᵀy)。仅当它严格超过−L及原容差/浮点误差的保守上界时删除变量。这里L是原主目标下界，未增加新优化目标。

使用80位有向Decimal区间；包含原验证容差与保守点积误差。缓存绑定当前列对象及前三组原始约束的字节hash，后续加约束可复用，前缀改变则失效；LP失败不删变量；丢失现有合法incumbent则明确停止。全部返回解还须全局最优且逐行通过原始约束。候选网格、Goodput/评分、目标带、CPU/RAM容量、R0/phi/clipping和任何训练/锁语义均未修改。

同一捕获模型红/绿：旧版115.331秒后再次拒绝；修复重放14.558秒取得最优排名5.0，主目标条件证书排除113532列。只是求解实现改进，不是训练速度或JCT收益。

改动文件：`engineering_repair.py`, `numeric_repair.py`, `supervise80.py`；新增`integer_row_lattice.py`, `primary_band_presolve.py`, `sia-simulator/tests/fixtures/core80_full_rank_step19.json.gz`, `sia-simulator/tests/test_rank_lattice_repair.py`。supervise80只新增默认关闭的工程失败停止选项；本次开启以在新工程失败时停止批次，非调度规则。完整diff在source_changes.patch。

- 代码依据：[sia-simulator/joint_prototype/clean_ablation.py::_decide](D:/科研启航/Sia/cc-joint-core80-full-lex-repair-v5/sia-simulator/joint_prototype/clean_ablation.py:13)。
- 代码依据：[numeric_repair.py::canonical_bundle](D:/科研启航/Sia/cc-joint-core80-full-lex-repair-v5/numeric_repair.py:13)。
- 代码依据：[integer_row_lattice.py::tighten_integer_rows](D:/科研启航/Sia/cc-joint-core80-full-lex-repair-v5/integer_row_lattice.py:16)。
- 代码依据：[primary_band_presolve.py::primary_certificate](D:/科研启航/Sia/cc-joint-core80-full-lex-repair-v5/primary_band_presolve.py:50)。
- 代码依据：[primary_band_presolve.py::certificate_on_domain](D:/科研启航/Sia/cc-joint-core80-full-lex-repair-v5/primary_band_presolve.py:98)。
## 3. 验收、冻结及公平条件

```json
{
  "status": "PASS",
  "existing_tests": 302,
  "new_tests": 16,
  "total_tests": 318,
  "full_suite": "tests/full_suite_final.log",
  "full_real_prefix": 30,
  "old_failure_decision": 19,
  "original_sia_exact_rounds": 3,
  "bridges": [
    {
      "method": "FULL",
      "compared_rounds": 18,
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
  "execution_checks": 4960,
  "locks": 22,
  "violations": 0
}
```

红测red_backend_contract2.log捕获原后端未使用等价整数行；red_literal_replay.log复现真实131244列失败。red_backend_contract.log仅是首次测试cwd导入错误，不能算缺陷红测。probe_primary.log的离线工具稠密/稀疏兼容错误保留，修正后probe_primary2.log完成；未修改原矩阵。

来源：实际315文件旧源码完整复制，保持旧树及官方源；最终319文件，source_snapshot.zip SHA256 `37D0D0A205761D7C8B2614FC273C0A0945A5454812BE15811676D6786B2B2FAA`，输入38项逐一核验。没有可假定代表实际源码的裸Git HEAD；完整快照、patch和测试fixture共同重建，不推送。

318项测试含原302项与新增16项；真实FULL30轮跨越原第19轮及第三到达波，A1020/PROP12短回归，原Sia3轮精确回归。V4开发版315/316项测试通过，但其证明顺序仍在大矩阵上执行冗余LP；正式运行前已停止并封存V4，不作性能结果。V5新增证明域顺序和非连续索引映射回归，最终318项再通过。这里是有界正确性验收，不是80作业完成证据；下表来自全程正式补跑。

80作业、48CIFAR10/16DeepSpeech2/16BERT；40@0、20@600、20@1200的既有受控到达。原始训练模型、work、batch不变。AWS4GPU/12合成CPU/250GB、DGX-ext8/24/500、RTX8/24/500。R0+E-legacy、Medium alpha=.5、seed_profiles(1,8)、60秒/6000轮，7格独立进程串行。门槛2GiB/5GiB，每格计算watchdog6h、批次24h，不调整求解器30秒/零gap或原重试设置。

FULL/CPU/MEM/PROP共享完成导向目标、GPU域、服务/准入、切换成本，分别开放双资源、仅CPU、仅RAM、比例配额。FO共享CC目标/准入，先在比例域确定服务集合与GPU，再冻结GPU配置做辅助优化。FULL vs FO检验联合与串行决策，不单独识别服务集合和GPU选择贡献。SIA_P实际调用官方Sia GPU/placement后比例执行，不加CC/L2；SIA_AUX固定该意图后用自有辅助优化，不是官方Synergy-TUNE。

## 4. 七格绝对数值

|方法|状态|完成|Mean JCT|Median|P95|P99|Makespan|轮数|墙钟(s)|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FULL | FAILED | 9/80 | — | — | — | — | — | 36 | 1872.066 |
| SIA_P | NOT_RUN | None/80 | — | — | — | — | — | None | — |
| SIA_AUX | NOT_RUN | None/80 | — | — | — | — | — | None | — |
| PROP | NOT_RUN | None/80 | — | — | — | — | — | None | — |
| CPU | NOT_RUN | None/80 | — | — | — | — | — | None | — |
| MEM | NOT_RUN | None/80 | — | — | — | — | — | None | — |
| FO | NOT_RUN | None/80 | — | — | — | — | — | None | — |

P95/P99是单一80作业场景经验分位数，不把80个作业当80次独立实验。失败/删失格均值留空。

- 未完成格FULL：FAILED，已记录完整轮次36，最后记录时间2160，失败时间2220；错误`RuntimeError('REPAIRED_SOLVER_NO_VERIFIED_OPTIMUM')`。失败区间不补写；NOT_RUN不等于算法失败。
- 未完成格SIA_P：NOT_RUN，已记录完整轮次None，最后记录时间None，失败时间None；错误`None`。失败区间不补写；NOT_RUN不等于算法失败。
- 未完成格SIA_AUX：NOT_RUN，已记录完整轮次None，最后记录时间None，失败时间None；错误`None`。失败区间不补写；NOT_RUN不等于算法失败。
- 未完成格PROP：NOT_RUN，已记录完整轮次None，最后记录时间None，失败时间None；错误`None`。失败区间不补写；NOT_RUN不等于算法失败。
- 未完成格CPU：NOT_RUN，已记录完整轮次None，最后记录时间None，失败时间None；错误`None`。失败区间不补写；NOT_RUN不等于算法失败。
- 未完成格MEM：NOT_RUN，已记录完整轮次None，最后记录时间None，失败时间None；错误`None`。失败区间不补写；NOT_RUN不等于算法失败。
- 未完成格FO：NOT_RUN，已记录完整轮次None，最后记录时间None，失败时间None；错误`None`。失败区间不补写；NOT_RUN不等于算法失败。
## 5. 配对收益、互斥时间和双资源交互

|候选 vs 对照|平均JCT改善%|节省秒/作业|等待/检查点/训练节省秒|改善/退化/平|
|---|---:|---:|---|---|

CPU×RAM四格（只有PROP/CPU/MEM/FULL全部完成才计算）：
```json
{}
```

主效应为开启该因子的平均JCT节省秒；交互YFULL−YCPU−YMEM+YPROP，负值代表超加性改善。局部实际训练加速可能通过释放资源间接降低排队，时间分解只是会计恒等式，不能将全部等待减少独占归因某模块。

## 6. 资源、重启与约束审计

|方法|GPU-h|CPU单位-h|RAM GB-h|等待(s)|检查点(s)|训练(s)|restart|wasted GPU-s|违规|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FULL | 9.208 | 31.645 | 605.997 | 100920.000 | 5350.000 | 21185.000 | 82.000 | 6660.000 | 0 |
| SIA_P | — | — | — | — | — | — | — | — | None |
| SIA_AUX | — | — | — | — | — | — | — | — | None |
| PROP | — | — | — | — | — | — | — | — | None |
| CPU | — | — | — | — | — | — | — | — | None |
| MEM | — | — | — | — | — | — | — | — | None |
| FO | — | — | — | — | — | — | — | — | None |
- FULL：执行审计2880、完整锁检查10、fallback0，求解状态`{'OPTIMAL_CLEAN_AB': 28, 'DRAIN_FOR_HEAD': 8}`；工程统计`{'native_calls': 713, 'native_seconds': 515.7550194003561, 'bound_certificates': 234, 'lp_calls': 377, 'lp_seconds': 171.6668775002181, 'primary_band_lp_calls': 35, 'primary_band_lp_seconds': 98.74293319991557, 'integer_rows_tightened': 322, 'peak_variables_removed': 191086, 'peak_zero_optional_removed': 183, 'peak_lp_variables_removed': 203599, 'peak_primary_band_variables_removed': 217348}`；选择总/P99秒`{'total': 1108.2066179001122, 'p99': 124.22735526501631, 'definition': 'Logged PolicyResult.solve_seconds; includes multi-MIP selection/control work, not an isolated single-backend-call timer.'}`。
- SIA_P：执行审计None、完整锁检查None、fallbackNone，求解状态`None`；工程统计`None`；选择总/P99秒`None`。
- SIA_AUX：执行审计None、完整锁检查None、fallbackNone，求解状态`None`；工程统计`None`；选择总/P99秒`None`。
- PROP：执行审计None、完整锁检查None、fallbackNone，求解状态`None`；工程统计`None`；选择总/P99秒`None`。
- CPU：执行审计None、完整锁检查None、fallbackNone，求解状态`None`；工程统计`None`；选择总/P99秒`None`。
- MEM：执行审计None、完整锁检查None、fallbackNone，求解状态`None`；工程统计`None`；选择总/P99秒`None`。
- FO：执行审计None、完整锁检查None、fallbackNone，求解状态`None`；工程统计`None`；选择总/P99秒`None`。
资源小时从执行片段allocated_seconds积分，周期中完成后不继续占用计时。checkpoint与训练互斥、rho只乘原始执行率一次；节点容量和锁检查不只看已选中的任务。仿真未将求解墙钟加入训练时间，计算开销须与JCT分开解读。

## 7. 长尾、资源响应和局限

- FULL：分子/分母分CPU/RAM的裁剪率% `{'num_cpu': 62.44635193133047, 'num_ram': 33.47639484978541, 'den_cpu': 24.892703862660944, 'den_ram': 0.0, 'num_any': 62.44635193133047, 'den_any': 24.892703862660944, 'any_point': 66.30901287553648}`；数据内/裁剪训练秒`{'fully_supported': 6588.0, 'clipped': 14597.0}`；同节点真实差异化配额且receiver加速的重叠训练对秒32662.0，其中全坐标数据内0.0。同raw/config/batch比例参考的局部净等效训练秒`{'receiver_gain': 7298.806154353416, 'donor_loss': 1883.6861698147452, 'net': 5415.119984538671, 'interpretation': 'Within-logged-segment fixed raw/config/batch reference; not a replay or causal JCT attribution.'}`；这不是完整反事实JCT。
- SIA_P：分子/分母分CPU/RAM的裁剪率% `None`；数据内/裁剪训练秒`None`；同节点真实差异化配额且receiver加速的重叠训练对秒None，其中全坐标数据内None。同raw/config/batch比例参考的局部净等效训练秒`None`；这不是完整反事实JCT。
- SIA_AUX：分子/分母分CPU/RAM的裁剪率% `None`；数据内/裁剪训练秒`None`；同节点真实差异化配额且receiver加速的重叠训练对秒None，其中全坐标数据内None。同raw/config/batch比例参考的局部净等效训练秒`None`；这不是完整反事实JCT。
- PROP：分子/分母分CPU/RAM的裁剪率% `None`；数据内/裁剪训练秒`None`；同节点真实差异化配额且receiver加速的重叠训练对秒None，其中全坐标数据内None。同raw/config/batch比例参考的局部净等效训练秒`None`；这不是完整反事实JCT。
- CPU：分子/分母分CPU/RAM的裁剪率% `None`；数据内/裁剪训练秒`None`；同节点真实差异化配额且receiver加速的重叠训练对秒None，其中全坐标数据内None。同raw/config/batch比例参考的局部净等效训练秒`None`；这不是完整反事实JCT。
- MEM：分子/分母分CPU/RAM的裁剪率% `None`；数据内/裁剪训练秒`None`；同节点真实差异化配额且receiver加速的重叠训练对秒None，其中全坐标数据内None。同raw/config/batch比例参考的局部净等效训练秒`None`；这不是完整反事实JCT。
- FO：分子/分母分CPU/RAM的裁剪率% `None`；数据内/裁剪训练秒`None`；同节点真实差异化配额且receiver加速的重叠训练对秒None，其中全坐标数据内None。同raw/config/batch比例参考的局部净等效训练秒`None`；这不是完整反事实JCT。
所有策略共享R0/E-legacy合成跨硬件执行函数；比例点rho=1是参考归一化，不是无CPU瓶颈证明。裁剪允许且未改，尤其不能把边界外推假设下的收益称为目标硬件实测。真实共驻片段、局部预测/执行收益和全程JCT分别报告，禁止互相替代。最差8作业详见audit/tail_diagnostics.json；主结果不删长尾。

## 8. 同版本回归与交付

若有差异，control_regressions.json保留逐项证据，不因均值接近就认定相同，不选择较好一次。完整正式结果均来自新319文件版本；旧结果只用于回归核验，不填入新矩阵。

原始rounds.jsonl.gz、solver诊断、输入/源码hash、source_snapshot.zip、source_changes.patch、318项测试和短回归均保留。本轮停止，不追加容量/响应档、九cohort、104/160或参数搜索。
