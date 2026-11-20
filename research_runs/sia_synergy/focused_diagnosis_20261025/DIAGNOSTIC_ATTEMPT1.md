# 离线诊断首次尝试

六个完整结果的独立原始审计均完成并保存在本目录。随后静态比较错误地把Neutral历史服务承诺点放进Medium响应和progress-floor过滤，导致候选为空，SciPy报错：`ValueError: c must be a one-dimensional array of finite numbers with at least one element.`

这是新诊断脚本的反事实定义错误，没有运行训练仿真、修改实验代码或历史结果。修正为每个静态状态沿用其实际响应档，并保护已锁定/承诺的完整配额。新计算写入review2，六份原始审计直接复用。没有因预测结果调整参数。

补充：model_quality.py首次启动时ROOT父层级多取一级，导入joint_prototype失败，未产生结果；修正为与其他诊断脚本相同的项目根路径后执行。该导入错误也未涉及训练或实验代码。
