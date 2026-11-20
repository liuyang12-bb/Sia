"""第五课用原版 Sia 的性能数据比较 GPU 配置。

固定 CIFAR-10、全局 batch 为 128、梯度累积次数为 0。
从项目根目录运行 python learning/lesson_05_throughput.py。
"""

from pathlib import Path
import sys


project_root = Path(__file__).resolve().parent.parent
simulator_dir = project_root / "official-artifacts" / "sia-simulator"
sys.path.insert(0, str(simulator_dir))

from applications import APPLICATIONS


def main():
    global_batch = 128
    placements = [(1,), (2,), (4,)]

    print("type  placement  GPUs  local_batch  step_ms  sync_ms  samples/s")
    for cluster_type in ("aws", "rtx"):
        application = APPLICATIONS[cluster_type]["cifar10"]
        for placement in placements:
            # 元组中每个数字是对应节点上的 GPU 数量。
            num_gpus = sum(placement)
            local_batch = global_batch // num_gpus
            step_time, sync_time = application.get_throughput(placement, local_batch)
            throughput = global_batch / step_time
            print(
                f"{cluster_type:<5} {str(placement):<10} {num_gpus:>4} "
                f"{local_batch:>12} {step_time * 1000:>8.2f} "
                f"{sync_time * 1000:>8.2f} {throughput:>10.1f}"
            )


if __name__ == "__main__":
    main()
