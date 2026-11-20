# 预检验证器错误与更正

attempt1在训练/离线资源求解前的R0来源验证处退出。新验证器错误地假定phi_c等于未截断吞吐比、phi_m等于sqrt(phi_c)。真实冻结定义是stage2b_preflight.py:102的phi_c=clip(r0,.5,4)、phi_m=clip(sqrt(r0),.5,2)。BERT在DGX四个配置的原始r0>4，原registry按既有规则为4/2，数据本身未改。

保留attempt1/preregistration.json、preflight_source.py和attempt1_console.log。attempt2只更正离线验证器以核验已有公式；未改变registry、phi、CPU响应clipping、参数、负载或任何算法。R0的phi饱和与响应坐标clipping在报告中分别说明。
