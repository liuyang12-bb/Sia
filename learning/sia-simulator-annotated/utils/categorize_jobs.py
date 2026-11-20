# 导读 读取 workload 表并补充作业类别，用于区分固定作业和可调整作业。
# 导读 来源 utils/categorize_jobs.py。以下带「导读」的行是阅读注释。
import pandas as pd
import numpy as np
import argparse
import random


# 导读 process_workload（原源码第 7 行）
# 导读 读取 workload 并按脚本规则填充作业类别。
# 导读 参数 workload 表示作业输入表或工作负载路径，取决于调用入口，args 表示命令行解析结果。
# 导读 从返回表达式可以继续追踪 workload。
def process_workload(workload, args):
    # add a new column to the workload to indicate the job type
    categories = ["adaptive", "scale-gpus", "rigid"]
    percent_adaptive = args.percent_adaptive
    percent_strong_scaling = args.percent_strong_scaling
    percent_rigid = args.percent_rigid
    # normalize percentages/weights to 1
    total = percent_adaptive + percent_strong_scaling + percent_rigid
    p_array = [percent_adaptive / total, percent_strong_scaling /
               total, percent_rigid / total]
    workload["category"] = np.random.choice(
        categories, size=len(workload), p=p_array)
    # print the number of jobs in each category
    print("Number of jobs in each category:")
    print(workload["category"].value_counts())
    return workload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="path to input workload csv")
    parser.add_argument("--output", type=str,
                        help="path to output workload csv")
    parser.add_argument("--percent_adaptive", type=float, default=100,
                        help="percentage of adaptive jobs")
    parser.add_argument("--percent_strong_scaling", type=float, default=0,
                        help="percentage of strong scaling jobs")
    parser.add_argument("--percent_rigid", type=float, default=0,
                        help="percentage of rigid jobs")

    args = parser.parse_args()
    print(f"Reading workload from {args.input}")
    input_workload = pd.read_csv(args.input)
    processed_workload = process_workload(input_workload, args)
    print(f"Writing processed workload to {args.output}")
    processed_workload.to_csv(args.output, index=False, header=True)
