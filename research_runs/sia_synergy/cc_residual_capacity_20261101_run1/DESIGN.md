# 运行前方案

唯一研究变量：候选点表达补齐。旧点保持；不接入V3/V4完成预测评价器。FO/FULL使用相同点生成器与当前轮独立审计。

对CPU和RAM分别构造：本候选在每节点使用g_n个GPU，可共驻GPU数k从0到min(G_n-g_n)；点q_k=min_n((C_n-k*f)/g_n)。f来自既有执行下限；不读取任务ID、JCT、未来到达、未观测吞吐，不递归生成。新点与原breakpoint/peer-complement网格取并集。最后仍用完整逐节点容量约束和已有资格/锁/冷启动过滤。

当前只有单节点GPU候选，故补入6-1*1=5 CPU、125-1*20.83=104.17GB等由硬件预算直接确定的点，而非两个任务专用值。更一般共驻的任意配额残差未全部闭合，不宣称连续最优。

假设：如果原域遗漏了正收益余量，新域可能提高训练速率并影响JCT；也可能改变全局排序，造成某些任务或平均JCT退化。两种结果都保留。FO同样获益时，共享部分不能归于联合GPU选择。

验收：旧provider红测；新点构建/审计/响应不变和禁用轴测试；四个同状态；完整旧385项及新增测试；Sia/FO/FULL各3轮精确回归。全部通过后冻结源码zip+hash、依赖、输入和四格顺序。

四格：OLD_FO → OLD_FULL → NEW_FO → NEW_FULL，各独立进程/目录。COMPACT12、原工作量和到达、R0/E-legacy Medium=.5、seed_profiles(1,8)、AWS/DGX各2GPU/6CPU/125GB、60秒、6000轮不变。启动前2GiB内存/5GiB磁盘。失败保留、停止流水线；不因不利结果调参或重跑。12/12才计算完整平均JCT。

原始完成事件独立重算JCT、互斥时间、容量、完整checkpoint锁、FO冻结GPU、raw×rho、资源小时、restart、裁剪；报告逐任务损益、平均/尾部/墙钟。只完成本12作业对照后停止，不80作业。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE
