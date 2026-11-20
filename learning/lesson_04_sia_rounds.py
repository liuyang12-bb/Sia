"""第四课用原版 Sia 观察调度周期。

教学输入有一个在第 75 秒提交的 CIFAR-10 作业和一台 4 GPU 节点。
每次运行观察前 180 秒模拟时间，逐轮打印资源分配和训练进度。
在项目根目录运行 python learning/lesson_04_sia_rounds.py --interval 60
"""

import argparse
from pathlib import Path
import sys

import pandas


# 从项目的原版模拟器目录加载 Cluster 和 SiaPolicy。
project_root = Path(__file__).resolve().parent.parent
simulator_dir = project_root / "official-artifacts" / "sia-simulator"
sys.path.insert(0, str(simulator_dir))

from cluster import Cluster
from sia import SiaPolicy


def make_cluster():
    workload = pandas.DataFrame([
        {
            "name": "cifar10-lesson",
            "time": 75,
            "application": "cifar10",
            "num_replicas": 1,
            "batch_size": 128,
            "category": "rigid",
        }
    ])
    num_nodes = {"aws": 1}
    gpus_per_node = {"aws": 4}
    policy = SiaPolicy()
    policy.populate_valid_configs(num_nodes, gpus_per_node, num_nodes)
    cluster = Cluster(workload, policy, num_nodes, gpus_per_node, num_nodes)
    # 和官方入口一样，为固定 GPU 数与 batch 大小的作业加载性能画像。
    cluster.jobs[0].seed_profiles_rigid(gpus_per_node)
    return cluster


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, choices=(30, 60), default=60)
    args = parser.parse_args()
    cluster = make_cluster()
    job = cluster.jobs[0]

    for _ in range(180 // args.interval):
        start = cluster.current_time
        gpus_before = sum(job.placement)
        cluster.step(args.interval)
        print("TRACE alloc", cluster.current_time, cluster.allocations, job.placement)
        gpus_after = sum(job.placement)
        eligible = job.name in cluster.get_job_infos()
        print(
            f"TRACE {start:3d}-{cluster.current_time:3d}s "
            f"eligible_at_end={str(eligible):5s} "
            f"gpus_before={gpus_before} gpus_after={gpus_after} "
            f"progress={job.progress:.3f} startup_left={job.rescale_time:.1f}s"
        )


if __name__ == "__main__":
    main()
