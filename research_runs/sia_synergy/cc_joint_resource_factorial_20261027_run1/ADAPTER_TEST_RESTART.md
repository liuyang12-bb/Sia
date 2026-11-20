# 适配验收夹具调整

adapter_green1.log保留：六方法各5轮真实执行和四CC Neutral轨迹一致性已经通过；后续容量组合构造反复初始化完整20作业，测试耗时过长，主动中断。未运行正式格。
改为显式3作业测试夹具，记录仍为CF0、DS8、CF1的原始GPU demand与batch，受控arrival0；只减少测试构造工作，不改任何正式输入。
build新增可选workload_path转发接口，默认仍为原20作业，正式30格不传该参数。重新验收日志为adapter_green2.log。
224项既有测试不调用新适配器，内核279个父文件保持字节不变，其通过证据existing_tests.log仍有效。
