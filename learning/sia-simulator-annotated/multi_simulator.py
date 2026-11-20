# 导读 官方异构模拟入口。命令行与 CSV 提供作业和运行设置，simulator_config 提供机器配置。
# 导读 来源 multi_simulator.py。以下带「导读」的行是阅读注释。
import argparse
import copy
import glob
import json
import multiprocessing
import os
import pandas
from rich.console import Console
from rich.table import Table

from cluster import Cluster
from gavel import GavelPolicy
from sia import SiaPolicy
from sia_fix import SiaFixPolicy
import simulator_config as sim_config


# 导读 multi_simulate（原源码第 18 行）
# 导读 从 args.workload 读入作业表，准备策略和集群，循环推进，返回日志列表及按作业名索引的 JCT。
# 导读 参数 args 表示命令行解析结果。
# 导读 函数体中的相关调用包括 pandas.read_csv、job.seed_profiles、cluster.step。
# 导读 从返回表达式可以继续追踪 (cluster.logs, cluster.get_jcts())。
def multi_simulate(args):
    # 导读 args.workload 是命令传入的 CSV 路径。workload 是读取后的 DataFrame，每行描述一个作业。
    workload = pandas.read_csv(args.workload)
    # 导读 cluster_scale 乘的是各类型的节点台数。这里直接修改模块配置字典，同一进程多次调用时会沿用已改的值。
    if args.cluster_scale is not None:
        for k in sim_config.cluster_nnodes.keys():
            sim_config.cluster_nnodes[k] = sim_config.cluster_nnodes[k] * args.cluster_scale
    cluster_populate_nnodes = sim_config.cluster_nnodes

    # filter out GPU types not in simulated cluster
    # 导读 只保留本次实际出现的机器类型。值是每台机器上的 GPU 张数。
    sim_ngpus_per_node = {k: sim_config.cluster_ngpus_per_node[k] for k in sim_config.cluster_nnodes.keys()}

    # 导读 policy 的三个分支分别建立 Gavel、SiaFix 或 Sia 对象。后面 Cluster 通过统一的 optimize 接口调用它。
    if args.policy == "gavel":
        policy = GavelPolicy(args.interval, policy=sim_config.gavel_default_policy)
    elif args.policy == "sia_fix":
        policy = policy = SiaFixPolicy(p_fairness=args.policy_p_val,
                                               restart_penalty=30,
                                               lambda_a=args.mip_lambda_a,
                                               lambda_n=args.mip_lambda_n,
                                               project_throughputs=args.project_throughputs,
                                               share_max_replicas=args.share_max_replicas)
    else:
        policy = SiaPolicy(p_fairness=args.policy_p_val,
                                   restart_penalty=30,
                                   lambda_a=args.mip_lambda_a,
                                   lambda_n=args.mip_lambda_n,
                                   project_throughputs=args.project_throughputs,
                                   share_max_replicas=args.share_max_replicas,
                                   enable_bsz_tuning=not args.disable_bsz_tuning)
    # 导读 先初始化策略能选择的配置。候选配置描述 GPU 类型、节点数量和 GPU 数量。
    policy.populate_valid_configs(
        cluster_populate_nnodes, sim_ngpus_per_node, sim_config.cluster_max_physical_nnodes)
    # 导读 创建 Cluster 时交入作业表、策略、节点台数、单节点 GPU 数、物理节点上限。
    cluster = Cluster(workload, policy, sim_config.cluster_nnodes, sim_ngpus_per_node,
                      max_physical_nodes=sim_config.cluster_max_physical_nnodes)
    if args.disable_bsz_tuning and args.policy == "sia":#关闭job的BatchSize的自适应
        print(f"Diabled bsz tuning for {args.policy} policy")
        cluster.disable_bsz_tuning()
    if args.policy == "gavel":
        cluster.disable_bsz_tuning()
    # 导读 普通 Sia 路径为 rigid 作业预装性能画像。Cluster 已经在构造阶段把表格行转成 Job。
    if args.policy == "sia":
        # seed profiles for all rigid jobs
        rigid_jobs = [job for job in cluster.jobs if job.category == "rigid"]
        for job in rigid_jobs:
            job.seed_profiles_rigid(sim_ngpus_per_node)

    # 导读 oracle 参数允许用历史画像预先拟合部分配置，作为已有性能信息进入模拟。
#     允许任务在真正运行之前，就拥有指定范围内的参考性能观测。
#     第一行：判断是否开启预先画像。
#     第二行：遍历 Cluster 中所有 Job。
#     第三行：对每个 Job 预先填充性能记录
#     比较两种情况：
# 普通模式： Job 需要随着模拟执行逐渐获取性能观测，建立预测模型。
# Oracle 画像模式： 在执行前就给 Job 更多已有性能数据，减少冷启动阶段的信息不足。
# 不过它不等于直接告诉调度器未来完整的调度结果，只是提前提供指定范围的参考性能观测。
# GitHub
# seed_profiles() 是预先填充性能画像。
# Job.update_params() 是在执行过程中，根据新配置观测更新画像
    if args.oracle_num_nodes != 0:
        for job in cluster.jobs:
            job.seed_profiles(args.oracle_num_nodes, args.oracle_num_replicas)

    current_time = 0
    # 导读 外层循环在每轮开始前检查作业的 completion_time。只要有一个 None 就继续。
    while not cluster.all_complete():
        ####### STEP #######
        # 导读 interval 是模拟秒。一个 step 先执行旧分配，再决定下一轮分配，最后记录状态。
        cluster.step(args.interval)
        table = Table(title=f"SIMULATOR TIME:{cluster.current_time}")
        table.add_column("Jobname", justify="left", style="cyan", no_wrap=True)
        table.add_column("Epoch", justify="left", style="white", no_wrap=True)
        table.add_column("Restarts", justify="left", style="white", no_wrap=True)
        table.add_column("Batch size", justify="left", style="white", no_wrap=True)
        table.add_column("Placement", justify="left", style="white", no_wrap=True)
        table.add_column("Contention", justify="left", style="white", no_wrap=True)
        table.add_column("Cluster", justify="left", style="white", no_wrap=True)
        table.add_column("Category", justify="left", style="white", no_wrap=True)

        # 导读 读取最后一轮日志来显示已提交且未完成的作业。表格只是把状态变成人可读的输出。
        for val in cluster.logs[-1]["submitted_jobs"]:
            if val["submission_time"] > cluster.current_time or val["completion_time"] is not None:
                continue
            table.add_row(val["name"], str(val["epoch"]), str(val["num_restarts"]),
                          str(val["batch_size"]), str(val["placement"]), str(val["n_avg"]),
                          str(val["cluster"]), str(val["category"]))
        console = Console()
        console.print(table)
        '''
        print("Active jobs:")
        print("  Jobname\t\tEpoch\tRestarts\tBatch size\tPlacement\t\tContention\tCluster \tCategory".expandtabs(8))
        print("  =======\t\t=====\t========\t==========\t=========\t\t==========\t========\t========".expandtabs(8))
        for val in cluster.logs[-1]["submitted_jobs"]:
            if val["submission_time"] <= cluster.current_time and val["completion_time"] is None:
                print(f"  {str(val['name']):<20}\t{str(val['epoch']):<4}\t{str(val['num_restarts']):<8}\t{str(val['batch_size']):<8}\t{str(val['placement']):<8}\t\t{str(val['n_avg'])[:5]:<8}\t{str(val['cluster']):<8}\t{str(val['category']):<8}".expandtabs(8))
        '''
        # 导读 get_used_gpus 汇总卡数。终端标签 GPU utilization 显示分配数量，百分比需要再与总容量比较。
        used_gpus = get_used_gpus(cluster.logs[-1], cluster.current_time)
        print("GPU utilization: {}".format(used_gpus))
        print("Completed jobs:")
        # 导读 jct_dict 是作业标识到提交至完成耗时的映射。
        jct_dict = cluster.get_jcts()
        print(jct_dict)
        print("Average JCT:", sum(jct_dict.values()) /
              len(jct_dict) if jct_dict else 0)

        # 导读 调试开关让每一轮暂停。Enter 继续一轮，c 取消后续暂停，x 直接退出。
        if args.debug:
            inp = input(f'\n\nPress any key to simlate next round.... (x to quit)')
            if inp == "x":
                print(f"Quitting...")
                exit(0)
            elif inp == "c":
                print(f"Continuing...")
                args.debug = False

    # 导读 指定 output 时按路径写逐轮记录。返回给调用者的是日志列表和作业 JCT 字典。
    if args.output:
        cluster.output_logs(args.output)
    return cluster.logs, cluster.get_jcts()


# 导读 get_used_gpus（原源码第 116 行）
# 导读 读一轮的 submitted_jobs，按集群类型汇总未完成且有分配的作业的 GPU 数。返回集群类型到张数的字典。
# 导读 参数 log_entry 表示一轮的日志记录字典，current_time 表示当前模拟时刻。
# 导读 从返回表达式可以继续追踪 used_gpus。
def get_used_gpus(log_entry, current_time):
    # 导读 used_gpus 是按类型或当前节点汇总的 GPU 分配数量。
    used_gpus = dict()
    for val in log_entry["submitted_jobs"]:
        if val["submission_time"] <= current_time and val["completion_time"] is None:
            cluster = val["cluster"]
            if cluster is None:
                continue
            else:
                if cluster not in used_gpus:
                    used_gpus[cluster] = 0
                used_gpus[cluster] += sum(val["placement"])
    return used_gpus

# 导读 直接运行本文件时执行这里的命令行入口。上方定义的函数此时已经可调用。
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 导读 workload 是必填位置参数，命令中直接填写路径。interval 与 policy 等带双横线的参数有默认值。
    parser.add_argument("workload", type=str, help="path to workload csv")
    parser.add_argument("--policy", type=str, default="sia",
                        choices=["sia", "gavel", "sia_fix"])
    parser.add_argument("--policy_p_val", type=float,
                        default=0.5, help="value of p for policy=sia")
    parser.add_argument("--mip_lambda_n", type=float, default=None,
                        help="sia regularization: no-alloc")
    parser.add_argument("--mip_lambda_a", type=float, default=None,
                        help="sia regularization: change of alloc")
    parser.add_argument("--cluster_scale", type=int, default=None,
                        help="scale of cluster relative to hardcoded values")
    # 导读 store_true 是布尔开关，命令中出现选项名就得到 True。
    parser.add_argument("--project_throughputs", action='store_true', default=False,
                        help="projects throughput functions from one cluster to another by multiplying by a constant = ratio of mean of throughputs for num_replicas=1")
    parser.add_argument("--disable_bsz_tuning", action='store_true', default=False,
                        help="use supplied bsz from workload, scale only num GPUs (sia only)")
    parser.add_argument("--share_max_replicas", action='store_true',
                        default=False, help="share the maximum number of profiled replicas across clusters")
    parser.add_argument("--oracle_num_nodes", type=int, default=0,
                        help="max-num-nodes to seed profiles for, in each cluster")
    parser.add_argument("--oracle_num_replicas", type=int,
                        default=0, help="number of replicas to seed profiles for")
    parser.add_argument("--interval", type=int, default=60,
                        help="scheduling interval in seconds")
    parser.add_argument("--output", type=str, help="path to output logs")
    parser.add_argument("--debug", action='store_true', default=False,
                        help="DEBUG MODE: simulate a scheduling round, pause and wait for user input to simulate next round")

    # 导读 parse_args 将终端里的字符串按照 type 与 choices 规则转换，结果可用 args.workload 等属性读取。
    args = parser.parse_args()
    if sim_config.num_parallel_jobs > 1:
        assert sim_config.num_parallel_traces == 1, f"num_parallel_jobs={sim_config.num_parallel_jobs} so, num_parallel_traces must be 1 (check simulator_config.py)"
    if sim_config.num_parallel_traces > 1:
        assert sim_config.num_parallel_jobs == 1, f"num_parallel_traces={sim_config.num_parallel_traces} so, num_parallel_jobs must be 1 (check simulator_config.py)"
    # 导读 路径如果是目录，每个 CSV 复制一份参数，逐文件运行并分开存日志。
    if os.path.isdir(args.workload):
        assert args.output is not None and os.path.isdir(args.output)
        args_list = []
        for workload in glob.glob(args.workload + "/*.csv"):
            name = os.path.basename(workload)[:-4]
            args_list.append(copy.deepcopy(args))
            args_list[-1].workload = workload
            args_list[-1].output = args.output + "/" + name + ".log"
        if sim_config.num_parallel_traces > 1:
            with multiprocessing.Pool(processes=sim_config.num_parallel_traces) as pool:
                ret_list = pool.map(multi_simulate, args_list)
        else:
            ret_list = []
            for args_item in args_list:
                ret_list.append(multi_simulate(args_item))
        # 导读 jcts 保存每份输入的逐作业结果，avgs 保存其均值，mean 再对这些均值求平均。
        summary = {"jcts": {}, "avgs": {}}
        for args_item, (_, jct_dict) in zip(args_list, ret_list):
            name = os.path.basename(args_item.workload)[:-4]
            summary["jcts"][name] = jct_dict
            summary["avgs"][name] = sum(jct_dict.values()) / len(jct_dict)
        summary["mean"] = sum(summary["avgs"].values()) / len(summary["avgs"])
        with open(args.output + "/summary.json", "w") as f:
            json.dump(summary, f, indent=4)
    else:
        # 导读 单个 CSV 直接把解析得到的 args 交给模拟主函数。
        multi_simulate(args)