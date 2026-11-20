"""第一课：为什么执行顺序会改变平均作业完成时间？

这是独立教学模型，不调用 Sia，不生成正式实验结果。
假设：只有 1 张 GPU；所有作业在第 0 秒到达；每次只运行一个作业；
作业运行后一直执行到结束；速度固定且已知；没有启动或切换开销。
因此这里的 JCT 恰好等于完成时刻。真实项目会逐步放宽这些假设。

学习顺序：先手算，再运行，最后修改 JOBS 中的数字并解释结果。
"""


# 工作量用任意统一的“进度单位”表示，速度单位为“进度单位/秒”。
# 教学数字是人为设定的，不是项目性能测量。
JOBS = [
    {"name": "A", "work": 100, "rate": 1},
    {"name": "B", "work": 10, "rate": 1},
]


def duration(job):
    """固定速度下，执行时间 = 工作量 / 速度。"""
    return job["work"] / job["rate"]


def simulate(ordered_jobs):
    """按指定顺序执行，返回每个作业的时间记录。"""
    current_time = 0.0  # 模拟时钟，与电脑实际运行脚本的耗时不同。
    records = []

    for job in ordered_jobs:
        arrival = 0.0
        start = current_time
        finish = start + duration(job)
        records.append({
            "name": job["name"],
            "start": start,
            "finish": finish,
            "waiting": start - arrival,
            "jct": finish - arrival,
        })
        current_time = finish

    return records


def show_result(title, records):
    print(title)
    for row in records:
        print(
            f"  {row['name']}: start={row['start']:.1f}s, "
            f"finish={row['finish']:.1f}s, "
            f"wait={row['waiting']:.1f}s, JCT={row['jct']:.1f}s"
        )
    mean_jct = sum(row["jct"] for row in records) / len(records)
    makespan = max(row["finish"] for row in records)
    print(f"  mean JCT={mean_jct:.1f}s; makespan={makespan:.1f}s\n")


def main():
    # 同时到达时，用列表中的先后顺序打破 FIFO 的平局。
    fifo_order = JOBS
    # key=duration 表示按 duration(job) 的值升序排列。
    shortest_first_order = sorted(JOBS, key=duration)

    show_result("FIFO (A then B)", simulate(fifo_order))
    show_result("Shortest job first", simulate(shortest_first_order))


if __name__ == "__main__":
    main()

# 练习（先预测，再运行）：
# 1. 保持原参数，为什么两种顺序的 makespan 相同，但平均 JCT 不同？
# 2. 把 A 的 rate 改成 20：两种顺序还会不同吗？
# 3. 在列表末尾添加 C，work=20、rate=2，手算两种顺序的平均 JCT。
# 注意：最短作业优先在本例中的表现，不能直接证明真实项目中的最优性。
