# JOINT-S：选择性准入验证结果

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。

## 结论与边界

本轮仅改变新服务准入：未兑现承诺和FIFO队首必选，其他作业由相同原公平目标决定是否加入。旧G整组交换、真实checkpoint、300秒有效服务、250秒启动预算、R0/E-legacy响应和全部工作负载/容量/参数不变。SIA-AUX是固定官方Sia GPU意图后的本项目辅助资源MIP，不是官方Synergy-TUNE。

新矩阵10/10格全部完成；154项既有/新增测试的冻结源码与本轮一致。训练前内存阻塞证据保留在父目录，当前attempt2未覆盖它。104正式批次未恢复。

**效果验收结论：不建议把新S直接作为旧G的升级替代。** 原8作业两格与G相同，原20作业Neutral/Medium分别比G慢0.09%/1.09%。独立20作业Medium中S比Sia快6.17%、比AUX快9.60%，但未运行同负载G，不能把这部分整体方法收益归因于新增准入。两组20作业Medium均优于Sia，只在一组优于AUX，说明优势依赖负载；这些不是目标硬件实测证据。关键机制解释见FINAL_REVIEW.md。

## 1. 原负载结果

|规模/响应|SIA-P|SIA-AUX|旧R|旧G|新S|S对P改善|S对AUX改善|S对G改善|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|8/neutral|1626.62|1626.62|1639.88|1606.88|1606.88|+1.21%|+1.21%|+0.00%|
|8/medium|1626.62|1476.25|1460.88|1385.50|1385.50|+14.82%|+6.15%|+0.00%|
|20/neutral|3477.15|3477.15|3394.80|3385.40|3388.45|+2.55%|+2.55%|-0.09%|
|20/medium|3477.15|3112.70|3277.15|3157.15|3191.65|+8.21%|-2.54%|-1.09%|

## 2. 预先冻结的独立负载

在任何新结果出现前按相同轮转规则提取workload-5的20条真实记录，控制到达均为0；保留原batch、GPU demand、工作量与source row。它独立于当前设计所诊断的workload-1，不声称历史项目从未使用过该trace。不能将这一组确定性负载当作多次独立重复。

|响应|SIA-P|SIA-AUX|新S|S对P改善|S对AUX改善|
|---|---:|---:|---:|---:|---:|
|neutral|3730.50|3730.50|3667.60|+1.69%|+1.69%|
|medium|3730.50|3872.10|3500.30|+6.17%|+9.60%|

### 响应开关的完整运行差分

|负载|方法|Neutral均值−Medium均值(s)|
|---|---|---:|
|original8|sia_p|+0.00|
|original8|sia_aux|+150.38|
|original8|joint_s|+221.38|
|original20|sia_p|+0.00|
|original20|sia_aux|+364.45|
|original20|joint_s|+196.80|
|holdout20|sia_p|+0.00|
|holdout20|sia_aux|-141.60|
|holdout20|joint_s|+167.30|

正值表示Medium在该完整策略路径上更快，负值表示更慢；包含作业完成顺序、GPU决策、在线画像和重启的路径变化，不等于固定状态的纯CPU边际。即使每个合法辅助点不低于该GPU配置的比例基准，也不保证整条在线调度的JCT单调改善。

## 3. 尾部、资源与开销

|数据/响应/方法|完成|Median|P95|P99|Makespan|GPU-h|CPU-h|RAM-GB-h|等待s|checkpoint s|训练s|restart|wasted GPU-s|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|original8/neutral/joint_s|8/8|1341.50|3035.90|3160.78|3192.00|13.874|13.874|288.993|480|1660|10715|18|3420|
|original8/medium/joint_s|8/8|1108.00|2729.65|2825.13|2849.00|11.289|35.572|704.080|480|1465|9139|15|3010|
|original20/neutral/joint_s|20/20|2208.00|6446.95|6461.39|6465.00|34.842|34.842|725.752|1860|3500|62409|64|5100|
|original20/medium/joint_s|20/20|2218.50|6162.25|6166.05|6167.00|31.565|98.010|2033.812|2280|3650|57903|53|5185|
|holdout20/neutral/sia_p|20/20|2181.00|6900.50|6999.30|7024.00|37.204|111.612|2325.243|5700|3920|64990|70|5990|
|holdout20/neutral/sia_aux|20/20|2181.00|6900.50|6999.30|7024.00|37.204|37.204|774.957|5700|3920|64990|70|5990|
|holdout20/neutral/joint_s|20/20|2190.00|6766.30|6816.46|6829.00|36.838|36.838|767.342|2160|3310|67882|66|4810|
|holdout20/medium/sia_p|20/20|2181.00|6900.50|6999.30|7024.00|37.204|111.612|2325.243|5700|3920|64990|70|5990|
|holdout20/medium/sia_aux|20/20|3357.00|6793.25|6842.65|6855.00|36.820|108.332|2201.970|10500|3925|63017|63|7050|
|holdout20/medium/joint_s|20/20|2055.00|6575.70|6799.14|6855.00|33.057|108.858|2241.138|3000|3180|63826|53|4475|

P95/P99只基于8或20作业，不能解释为总体尾分位的高精度估计。等待、checkpoint与有效训练从原始执行片段互斥积分；资源小时按实际完成截断的allocated_seconds计算，rho不加速checkpoint/有效服务秒数。

## 4. 逐作业与净差分

### original8/neutral/S_vs_sia_p

均值变化-19.75s，改善率+1.21%；3改善/3退化/2相同。总JCT变化-158s。最有利差值deepspeech2-9=-480s，最不利差值deepspeech2-8=+446s。
总差分=等待+0+checkpoint+55+有效训练-213s。
仅诊断去掉最大正贡献作业后，对照/新策略均值分别1398.14/1444.14s；主结果保留全部作业。

|Job|新S−对照JCT(s)|
|---|---:|
|bert-28|+192|
|cifar10-0|+64|
|deepspeech2-8|+446|
|bert-37|-6|
|cifar10-1|+0|
|deepspeech2-9|-480|
|bert-54|-374|
|cifar10-2|+0|

### original8/neutral/S_vs_sia_aux

均值变化-19.75s，改善率+1.21%；3改善/3退化/2相同。总JCT变化-158s。最有利差值deepspeech2-9=-480s，最不利差值deepspeech2-8=+446s。
总差分=等待+0+checkpoint+55+有效训练-213s。
仅诊断去掉最大正贡献作业后，对照/新策略均值分别1398.14/1444.14s；主结果保留全部作业。

|Job|新S−对照JCT(s)|
|---|---:|
|bert-28|+192|
|cifar10-0|+64|
|deepspeech2-8|+446|
|bert-37|-6|
|cifar10-1|+0|
|deepspeech2-9|-480|
|bert-54|-374|
|cifar10-2|+0|

### original8/neutral/S_vs_joint_r

均值变化-33.00s，改善率+2.01%；4改善/2退化/2相同。总JCT变化-264s。最有利差值deepspeech2-9=-300s，最不利差值cifar10-0=+411s。
总差分=等待+0+checkpoint+80+有效训练-344s。
仅诊断去掉最大正贡献作业后，对照/新策略均值分别1439.00/1444.14s；主结果保留全部作业。

|Job|新S−对照JCT(s)|
|---|---:|
|bert-28|-136|
|cifar10-0|+411|
|deepspeech2-8|+109|
|bert-37|-136|
|cifar10-1|+0|
|deepspeech2-9|-300|
|bert-54|-212|
|cifar10-2|+0|

### original8/neutral/S_vs_joint_g

均值变化+0.00s，改善率+0.00%；0改善/0退化/8相同。总JCT变化+0s。最有利差值bert-28=+0s，最不利差值bert-28=+0s。
总差分=等待+0+checkpoint+0+有效训练+0s。
仅诊断去掉最大正贡献作业后，对照/新策略均值分别1625.14/1625.14s；主结果保留全部作业。

|Job|新S−对照JCT(s)|
|---|---:|
|bert-28|+0|
|cifar10-0|+0|
|deepspeech2-8|+0|
|bert-37|+0|
|cifar10-1|+0|
|deepspeech2-9|+0|
|bert-54|+0|
|cifar10-2|+0|

### original8/medium/S_vs_sia_p

均值变化-241.12s，改善率+14.82%；6改善/2退化/0相同。总JCT变化-1929s。最有利差值deepspeech2-9=-718s，最不利差值deepspeech2-8=+103s。
总差分=等待+0+checkpoint-140+有效训练-1789s。
仅诊断去掉最大正贡献作业后，对照/新策略均值分别1398.14/1225.14s；主结果保留全部作业。

|Job|新S−对照JCT(s)|
|---|---:|
|bert-28|-179|
|cifar10-0|-419|
|deepspeech2-8|+103|
|bert-37|-377|
|cifar10-1|-112|
|deepspeech2-9|-718|
|bert-54|-330|
|cifar10-2|+103|

### original8/medium/S_vs_sia_aux

均值变化-90.75s，改善率+6.15%；5改善/3退化/0相同。总JCT变化-726s。最有利差值deepspeech2-9=-651s，最不利差值cifar10-2=+317s。
总差分=等待+0+checkpoint-140+有效训练-586s。
仅诊断去掉最大正贡献作业后，对照/新策略均值分别1235.86/1225.14s；主结果保留全部作业。

|Job|新S−对照JCT(s)|
|---|---:|
|bert-28|-31|
|cifar10-0|-101|
|deepspeech2-8|+143|
|bert-37|-242|
|cifar10-1|+34|
|deepspeech2-9|-651|
|bert-54|-195|
|cifar10-2|+317|

### original8/medium/S_vs_joint_r

均值变化-75.38s，改善率+5.16%；6改善/2退化/0相同。总JCT变化-603s。最有利差值bert-54=-316s，最不利差值cifar10-2=+370s。
总差分=等待+0+checkpoint-290+有效训练-313s。
仅诊断去掉最大正贡献作业后，对照/新策略均值分别1459.43/1418.43s；主结果保留全部作业。

|Job|新S−对照JCT(s)|
|---|---:|
|bert-28|-217|
|cifar10-0|-106|
|deepspeech2-8|-32|
|bert-37|-128|
|cifar10-1|+120|
|deepspeech2-9|-294|
|bert-54|-316|
|cifar10-2|+370|

### original8/medium/S_vs_joint_g

均值变化+0.00s，改善率+0.00%；0改善/0退化/8相同。总JCT变化+0s。最有利差值bert-28=+0s，最不利差值bert-28=+0s。
总差分=等待+0+checkpoint+0+有效训练+0s。
仅诊断去掉最大正贡献作业后，对照/新策略均值分别1425.14/1425.14s；主结果保留全部作业。

|Job|新S−对照JCT(s)|
|---|---:|
|bert-28|+0|
|cifar10-0|+0|
|deepspeech2-8|+0|
|bert-37|+0|
|cifar10-1|+0|
|deepspeech2-9|+0|
|bert-54|+0|
|cifar10-2|+0|

### original20/neutral/S_vs_sia_p

均值变化-88.70s，改善率+2.55%；9改善/9退化/2相同。总JCT变化-1774s。最有利差值bert-28=-1860s，最不利差值deepspeech2-12=+1227s。
总差分=等待-3900+checkpoint-300+有效训练+2426s。
仅诊断去掉最大正贡献作业后，对照/新策略均值分别3461.37/3465.89s；主结果保留全部作业。

|Job|新S−对照JCT(s)|
|---|---:|
|bert-28|-1860|
|cifar10-0|-57|
|deepspeech2-8|-252|
|bert-37|-1860|
|cifar10-1|-80|
|deepspeech2-9|+119|
|bert-54|-146|
|cifar10-2|+47|
|deepspeech2-12|+1227|
|bert-99|+0|
|cifar10-3|+687|
|deepspeech2-16|+654|
|bert-107|+0|
|cifar10-4|+168|
|deepspeech2-21|-787|
|bert-159|+6|
|cifar10-5|+746|
|deepspeech2-30|-566|
|cifar10-6|+181|
|deepspeech2-31|-1|

### original20/neutral/S_vs_sia_aux

均值变化-88.70s，改善率+2.55%；9改善/9退化/2相同。总JCT变化-1774s。最有利差值bert-28=-1860s，最不利差值deepspeech2-12=+1227s。
总差分=等待-3900+checkpoint-300+有效训练+2426s。
仅诊断去掉最大正贡献作业后，对照/新策略均值分别3461.37/3465.89s；主结果保留全部作业。

|Job|新S−对照JCT(s)|
|---|---:|
|bert-28|-1860|
|cifar10-0|-57|
|deepspeech2-8|-252|
|bert-37|-1860|
|cifar10-1|-80|
|deepspeech2-9|+119|
|bert-54|-146|
|cifar10-2|+47|
|deepspeech2-12|+1227|
|bert-99|+0|
|cifar10-3|+687|
|deepspeech2-16|+654|
|bert-107|+0|
|cifar10-4|+168|
|deepspeech2-21|-787|
|bert-159|+6|
|cifar10-5|+746|
|deepspeech2-30|-566|
|cifar10-6|+181|
|deepspeech2-31|-1|

### original20/neutral/S_vs_joint_r

均值变化-6.35s，改善率+0.19%；6改善/10退化/4相同。总JCT变化-127s。最有利差值bert-28=-1084s，最不利差值cifar10-4=+973s。
总差分=等待+660+checkpoint+1100+有效训练-1887s。
仅诊断去掉最大正贡献作业后，对照/新策略均值分别3415.53/3465.89s；主结果保留全部作业。

|Job|新S−对照JCT(s)|
|---|---:|
|bert-28|-1084|
|cifar10-0|+0|
|deepspeech2-8|+677|
|bert-37|-1084|
|cifar10-1|+0|
|deepspeech2-9|+669|
|bert-54|+0|
|cifar10-2|+767|
|deepspeech2-12|+298|
|bert-99|+0|
|cifar10-3|+767|
|deepspeech2-16|+285|
|bert-107|-1084|
|cifar10-4|+973|
|deepspeech2-21|-622|
|bert-159|-1084|
|cifar10-5|+803|
|deepspeech2-30|-762|
|cifar10-6|+339|
|deepspeech2-31|+15|

### original20/neutral/S_vs_joint_g

均值变化+3.05s，改善率-0.09%；4改善/3退化/13相同。总JCT变化+61s。最有利差值deepspeech2-30=-933s，最不利差值deepspeech2-16=+1049s。
总差分=等待+660+checkpoint+350+有效训练-949s。
仅诊断去掉最大正贡献作业后，对照/新策略均值分别3232.11/3284.42s；主结果保留全部作业。

|Job|新S−对照JCT(s)|
|---|---:|
|bert-28|+0|
|cifar10-0|+0|
|deepspeech2-8|-53|
|bert-37|+0|
|cifar10-1|+0|
|deepspeech2-9|-448|
|bert-54|+0|
|cifar10-2|+0|
|deepspeech2-12|+992|
|bert-99|+0|
|cifar10-3|+0|
|deepspeech2-16|+1049|
|bert-107|+0|
|cifar10-4|+0|
|deepspeech2-21|-871|
|bert-159|+0|
|cifar10-5|+0|
|deepspeech2-30|-933|
|cifar10-6|+0|
|deepspeech2-31|+325|

### original20/medium/S_vs_sia_p

均值变化-285.50s，改善率+8.21%；13改善/6退化/1相同。总JCT变化-5710s。最有利差值bert-28=-1866s，最不利差值bert-99=+1132s。
总差分=等待-3480+checkpoint-150+有效训练-2080s。
仅诊断去掉最大正贡献作业后，对照/新策略均值分别3461.37/3259.05s；主结果保留全部作业。

|Job|新S−对照JCT(s)|
|---|---:|
|bert-28|-1866|
|cifar10-0|-240|
|deepspeech2-8|-519|
|bert-37|-1098|
|cifar10-1|-231|
|deepspeech2-9|-191|
|bert-54|+543|
|cifar10-2|-969|
|deepspeech2-12|+308|
|bert-99|+1132|
|cifar10-3|-29|
|deepspeech2-16|+356|
|bert-107|+60|
|cifar10-4|-1197|
|deepspeech2-21|-858|
|bert-159|+0|
|cifar10-5|-71|
|deepspeech2-30|+231|
|cifar10-6|-746|
|deepspeech2-31|-325|

### original20/medium/S_vs_sia_aux

均值变化+78.95s，改善率-2.54%；6改善/13退化/1相同。总JCT变化+1579s。最有利差值bert-28=-760s，最不利差值bert-99=+1132s。
总差分=等待-2040+checkpoint-270+有效训练+3889s。
仅诊断去掉最大正贡献作业后，对照/新策略均值分别3135.95/3259.05s；主结果保留全部作业。

|Job|新S−对照JCT(s)|
|---|---:|
|bert-28|-760|
|cifar10-0|+142|
|deepspeech2-8|+176|
|bert-37|-112|
|cifar10-1|+153|
|deepspeech2-9|+382|
|bert-54|+543|
|cifar10-2|-365|
|deepspeech2-12|-755|
|bert-99|+1132|
|cifar10-3|+428|
|deepspeech2-16|+417|
|bert-107|+60|
|cifar10-4|-673|
|deepspeech2-21|+342|
|bert-159|+0|
|cifar10-5|+384|
|deepspeech2-30|+164|
|cifar10-6|-111|
|deepspeech2-31|+32|

### original20/medium/S_vs_joint_r

均值变化-85.50s，改善率+2.61%；12改善/7退化/1相同。总JCT变化-1710s。最有利差值deepspeech2-12=-610s，最不利差值deepspeech2-30=+520s。
总差分=等待+1080+checkpoint+340+有效训练-3130s。
仅诊断去掉最大正贡献作业后，对照/新策略均值分别3128.63/3070.74s；主结果保留全部作业。

|Job|新S−对照JCT(s)|
|---|---:|
|bert-28|-275|
|cifar10-0|-48|
|deepspeech2-8|-212|
|bert-37|+0|
|cifar10-1|-255|
|deepspeech2-9|+121|
|bert-54|-364|
|cifar10-2|-512|
|deepspeech2-12|-610|
|bert-99|+79|
|cifar10-3|+331|
|deepspeech2-16|-229|
|bert-107|+206|
|cifar10-4|-150|
|deepspeech2-21|-417|
|bert-159|-129|
|cifar10-5|+355|
|deepspeech2-30|+520|
|cifar10-6|+43|
|deepspeech2-31|-164|

### original20/medium/S_vs_joint_g

均值变化+34.50s，改善率-1.09%；9改善/11退化/0相同。总JCT变化+690s。最有利差值deepspeech2-12=-605s，最不利差值deepspeech2-31=+787s。
总差分=等待+1080+checkpoint+510+有效训练-900s。
仅诊断去掉最大正贡献作业后，对照/新策略均值分别3002.58/3070.74s；主结果保留全部作业。

|Job|新S−对照JCT(s)|
|---|---:|
|bert-28|-127|
|cifar10-0|-215|
|deepspeech2-8|-277|
|bert-37|-48|
|cifar10-1|+70|
|deepspeech2-9|+84|
|bert-54|-267|
|cifar10-2|-84|
|deepspeech2-12|-605|
|bert-99|+412|
|cifar10-3|+384|
|deepspeech2-16|+184|
|bert-107|+206|
|cifar10-4|-364|
|deepspeech2-21|+39|
|bert-159|+140|
|cifar10-5|-118|
|deepspeech2-30|+276|
|cifar10-6|+213|
|deepspeech2-31|+787|

### holdout20/neutral/S_vs_sia_p

均值变化-62.90s，改善率+1.69%；10改善/9退化/1相同。总JCT变化-1258s。最有利差值bert-15=-1221s，最不利差值cifar10-6=+710s。
总差分=等待-3540+checkpoint-610+有效训练+2892s。
仅诊断去掉最大正贡献作业后，对照/新策略均值分别3761.68/3759.74s；主结果保留全部作业。

|Job|新S−对照JCT(s)|
|---|---:|
|bert-15|-1221|
|cifar10-0|-202|
|deepspeech2-21|-756|
|bert-22|-180|
|cifar10-1|-80|
|deepspeech2-35|-65|
|bert-73|+0|
|cifar10-2|-842|
|deepspeech2-37|+224|
|bert-118|-180|
|cifar10-3|-80|
|deepspeech2-44|+270|
|cifar10-4|+233|
|deepspeech2-52|+57|
|cifar10-5|+397|
|deepspeech2-70|+604|
|cifar10-6|+710|
|deepspeech2-115|+17|
|cifar10-7|+97|
|deepspeech2-123|-261|

### holdout20/neutral/S_vs_sia_aux

均值变化-62.90s，改善率+1.69%；10改善/9退化/1相同。总JCT变化-1258s。最有利差值bert-15=-1221s，最不利差值cifar10-6=+710s。
总差分=等待-3540+checkpoint-610+有效训练+2892s。
仅诊断去掉最大正贡献作业后，对照/新策略均值分别3761.68/3759.74s；主结果保留全部作业。

|Job|新S−对照JCT(s)|
|---|---:|
|bert-15|-1221|
|cifar10-0|-202|
|deepspeech2-21|-756|
|bert-22|-180|
|cifar10-1|-80|
|deepspeech2-35|-65|
|bert-73|+0|
|cifar10-2|-842|
|deepspeech2-37|+224|
|bert-118|-180|
|cifar10-3|-80|
|deepspeech2-44|+270|
|cifar10-4|+233|
|deepspeech2-52|+57|
|cifar10-5|+397|
|deepspeech2-70|+604|
|cifar10-6|+710|
|deepspeech2-115|+17|
|cifar10-7|+97|
|deepspeech2-123|-261|

### holdout20/medium/S_vs_sia_p

均值变化-230.20s，改善率+6.17%；12改善/8退化/0相同。总JCT变化-4604s。最有利差值deepspeech2-35=-1545s，最不利差值deepspeech2-70=+484s。
总差分=等待-2700+checkpoint-740+有效训练-1164s。
仅诊断去掉最大正贡献作业后，对照/新策略均值分别3564.00/3403.00s；主结果保留全部作业。

|Job|新S−对照JCT(s)|
|---|---:|
|bert-15|-1083|
|cifar10-0|-296|
|deepspeech2-21|+373|
|bert-22|-42|
|cifar10-1|-291|
|deepspeech2-35|-1545|
|bert-73|+138|
|cifar10-2|-1163|
|deepspeech2-37|+156|
|bert-118|+150|
|cifar10-3|-490|
|deepspeech2-44|-155|
|cifar10-4|-244|
|deepspeech2-52|+110|
|cifar10-5|-292|
|deepspeech2-70|+484|
|cifar10-6|+230|
|deepspeech2-115|+118|
|cifar10-7|-299|
|deepspeech2-123|-463|

### holdout20/medium/S_vs_sia_aux

均值变化-371.80s，改善率+9.60%；11改善/9退化/0相同。总JCT变化-7436s。最有利差值bert-118=-4608s，最不利差值deepspeech2-115=+420s。
总差分=等待-7500+checkpoint-745+有效训练+809s。
仅诊断去掉最大正贡献作业后，对照/新策略均值分别3715.11/3566.26s；主结果保留全部作业。

|Job|新S−对照JCT(s)|
|---|---:|
|bert-15|-1302|
|cifar10-0|-88|
|deepspeech2-21|-51|
|bert-22|-1302|
|cifar10-1|-241|
|deepspeech2-35|-330|
|bert-73|+138|
|cifar10-2|-693|
|deepspeech2-37|+334|
|bert-118|-4608|
|cifar10-3|-448|
|deepspeech2-44|+218|
|cifar10-4|+332|
|deepspeech2-52|+160|
|cifar10-5|+15|
|deepspeech2-70|-396|
|cifar10-6|+247|
|deepspeech2-115|+420|
|cifar10-7|+388|
|deepspeech2-123|-229|

## 5. 准入是否真正改变、保障与审计

|格|首轮选中|最大同时选中|optional未选总数|head保障轮|drain轮|最大累计启动|数值重求|weighted rho|分子CPU/RAM clipping|分母CPU/RAM clipping|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|original8_neutral_joint_s|8|8|0|22|0|240|0|1.0000|178/178|84/0|
|original8_medium_joint_s|8|8|0|17|0|240|0|1.3172|56/55|59/0|
|original20_neutral_joint_s|20|20|11|88|0|240|0|1.0000|1100/1100|181/0|
|original20_medium_joint_s|14|20|18|85|0|240|1|1.1536|247/226|219/0|
|holdout20_neutral_sia_p|20|20|0|0|0|0|0|1.0000|108/0|108/0|
|holdout20_neutral_sia_aux|20|20|0|0|0|0|0|1.0000|1148/1148|108/0|
|holdout20_neutral_joint_s|20|20|16|75|0|240|0|1.0000|1189/1189|112/0|
|holdout20_medium_sia_p|20|20|0|0|0|0|0|1.0000|108/0|108/0|
|holdout20_medium_sia_aux|20|20|0|0|0|0|0|1.0164|119/117|101/0|
|holdout20_medium_joint_s|13|20|30|85|0|240|1|1.1837|157/135|132/0|

clipping单位为每轮选中列，不是作业比例；它是已冻结合成模型的重要边界，不因无容量违规而变为真实画像。raw训练率直接记录、独立矩阵重算rho、checkpoint完整几何、进度单调、原始完成事件JCT、三资源容量、承诺/队首与可选集合、服务时钟不重置和实际启动预算均复核；全部见validation_summary和逐格validation。

## 6. 解释与下一阶段限制

待结合下方结论审查文件 FINAL_REVIEW.md 阅读。只能把新S与旧G的变化归因于本轮准入规则（含optional精确同分的确定性处理），不能把S相对Sia所有差异都归因于CPU优化：两者仍同时存在准入、GPU选择和放置差异。Neutral→Medium差分也包含完整路径变化，不是固定GPU的纯CPU边际。

保留FIFO队首与服务承诺避免全空和饥饿；但随着下一轮队首必选，仍可能重新增加同时运行数。若出现退化，如实记录，不在本轮按JCT再加入预留、调整Q/D、修改公平权重或响应指数。辅助资源改变无额外加载成本仍是合成执行假设，一节点每类型无法证明复杂跨节点放置收益。

## 7. 复现入口

preregistration.json与父冻结manifest逐字节一致；resume_manifest记录961源码/输入及112历史文件核验、启动前资源与无重复进程。每格独立Python进程串行，60秒/600轮/seed_profiles(1,8)，内存2GiB、磁盘5GiB。完整命令/退出码/墙钟和原始输出hash见execution_manifest.jsonl。源码在父execution_source.zip，前置材料在父frozen_prerequisites.zip，依赖锁与工程diff均已保留。旧安全停止报告不覆盖。

正式104格继续暂停，没有新增160作业、九cohort、Low/High或参数搜索。
