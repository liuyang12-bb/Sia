用户批准统一确定性剩余容量候选补齐，FO/FULL共用。基于完整340文件源码副本cc-joint-uncertainty-v4，复制cc-joint-residual-capacity-v1。只启用新候选provider，不启用V4预测纠偏。保护旧源/产物。先红绿、同状态、385旧测试和原Sia回归，再四格OLD_FO/OLD_FULL/NEW_FO/NEW_FULL，相同COMPACT12、Medium、60秒6000轮。2GiB/5GiB门槛；独立进程串行。不调参、不80作业。入口本目录。
完成并停止：393测试通过；同状态及Sia/FO/FULL回归通过；四格12/12，raw审计0；新FO−0.687%、新FULL−1.033%平均JCT。源码342文件冻结，旧340文件及116旧产物未变。完整结果REPORT.md。未80作业、未调参。
