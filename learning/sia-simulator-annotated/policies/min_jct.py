# 导读 以作业完成时间为目标的对照入口，调用对应的时长优化逻辑。
# 导读 来源 policies/min_jct.py。以下带「导读」的行是阅读注释。
from audioop import avg
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import cvxpy as cp
import numpy as np

from policy import Policy, PolicyWithPacking

class MinJCTPerf(Policy):

    # 导读 MinJCTPerf.__init__（原源码第 12 行）
    # 导读 构造 MinJCTPerf 时执行。为实例建立 self._name 等状态。
    # 导读 参数 solver 表示选用的数值求解器。
    # 导读 函数体直接赋值的实例字段包括 self._name。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, solver):
        Policy.__init__(self, solver)
        self._name = 'min_jct_perf'

    # 导读 MinJCTPerf.get_allocation（原源码第 16 行）
    # 导读 本策略的分配入口。接收上层传入的状态，并使用 cp.Minimize 等调用生成分配。
    # 导读 参数 unflattened_throughputs 表示求解前以作业或组合、worker 类型为键组织的吞吐字典，scale_factors 表示按作业索引的资源规模需求，cluster_spec 表示每类 worker 的可用数量。
    # 导读 参数 num_steps_remaining 表示每个作业剩余的训练迭代数。
    # 导读 其余显式参数为 age，默认值见函数签名。
    # 导读 函数体中的相关调用包括 cp.Minimize、self.scale_factors_array、self.get_base_constraints、cp.Problem、cvxprob.solve。
    # 导读 从返回表达式可以继续追踪 super().unflatten(x.value.round(decimals=5).clip(min=0.0).clip(max=1.0), index)。
    def get_allocation(self, unflattened_throughputs, scale_factors, cluster_spec, num_steps_remaining, age):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (m, n) = throughputs.shape
        (job_ids, worker_types) = index


        num_steps = []
        time_elapsed = []

        for job_id in job_ids:
            num_steps.append(num_steps_remaining[job_id])
            time_elapsed.append(age[job_id])

        num_steps = np.asarray(num_steps).reshape(-1, 1)
        time_elapsed = np.asarray(time_elapsed).reshape(-1, 1)



        # 导读 CVXPY 在这里创建求解变量。其形状和整数或布尔类型由调用参数给出，取值约束在后面构造。
        x = cp.Variable(throughputs.shape)
        avg_throughput = cp.reshape(cp.sum(cp.multiply(throughputs, x), axis = 1), (m, 1))
        jct = time_elapsed + cp.multiply(num_steps, cp.inv_pos(avg_throughput))
        objective = cp.Minimize(cp.sum(jct, axis=0))

        # 导读 scale_factors_array 是按矩阵顺序排列的作业资源需求。
        scale_factors_array = self.scale_factors_array(scale_factors, job_ids, m, n)
        constraints = self.get_base_constraints(x, scale_factors_array)

        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver=self._solver)

        # print("\t--- passed jobs:", job_ids)
        # print("\t--- avg: throughput:", avg_throughput.value)
        # for i in range(len(job_ids)):
        #     print(f"\t---{job_ids[i]}, x: {x.value.round(decimals=5)[i, :]}, avg_throughput: {avg_throughput.value[i, :]}, jct:{jct.value[i, :]}")

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')
            exit()

        return super().unflatten(x.value.round(decimals=5).clip(min=0.0).clip(max=1.0), index)