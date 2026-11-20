# 测试工作目录恢复

首次212项测试从产物目录source/tests执行，有176个FileNotFoundError。
历史测试以Path(__file__).parents[3]寻找项目根，产物目录的嵌套深度不符合
原有工作树布局，导致重复research_runs/sia_synergy前缀。tests.log/tests.json保留。

处理：把相同404文件逐字节复制到新隔离树sia-joint-cc-12job/sia-simulator，
恢复原有目录深度，从该处测试和运行；产物source/继续保存相同冻结副本。
两份源码均逐文件验证父版本hash，不改测试或算法去绕过失败。
新的测试证据为tests_relocated.log/json。原始Sia3轮回归已独立通过。
本修正仅恢复历史fixture定位，未修改任务、响应或策略参数；正式运行尚未开始。

冻结前安全门槛首次调用另发现本机psutil.disk_usage不接受WindowsPath，
TypeError: argument 1 must be str, not WindowsPath。改为显式str(OUT)。
该次在写preregistration和启动任何训练前失败；这是运行器接口修正。
