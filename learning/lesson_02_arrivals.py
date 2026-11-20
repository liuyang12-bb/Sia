"""第二课：只能从已经到达的作业中做调度选择。

假设：1 张 GPU；速度固定且已知；没有启动开销；
作业可以在不同时刻到达，每次选中的作业一直运行到结束。
这里的调度发生在 GPU 空闲时，与真实 Sia 的周期调度有所不同。
这是独立教学模型，不调用项目算法，不生成正式实验结果。
"""


JOBS = [
    {"name": "A", "arrival": 0, "work": 8, "rate": 1},
    {"name": "B", "arrival": 1, "work": 4, "rate": 1},
    {"name": "C", "arrival": 2, "work": 1, "rate": 1},
    {"name": "D", "arrival": 20, "work": 2, "rate": 1}

]


def duration(job):
    return job["work"] / job["rate"]


def arrival_time(job):
    return job["arrival"]


# 两个策略只负责选择作业，不负责推进时钟。
def choose_fifo(ready_jobs):
    # 到达时刻相同则保留列表顺序。
    return min(ready_jobs, key=arrival_time)


def choose_shortest(ready_jobs):
    # 只在“当前已经到达”的作业里选运行时间最短的。
    # 运行时间相同则保留列表顺序。
    return min(ready_jobs, key=duration)

def choose_longest(ready_jobs):
    return max(ready_jobs,key=duration)

def simulate(jobs, choose_job):
    # 复制列表，保证 remove 不会删除原 JOBS 中的元素。
    # 本例不修改作业字典内部的数据，因此复制列表就足够了。
    pending = list(jobs)
    current_time = 0.0
    records = []

    while pending:
        # pending 是尚未运行的作业，可能包含未来才到达的作业。
        # ready 是现在已经到达、可以选择的作业。
        ready = []
        for job in pending:
            if job["arrival"] <= current_time:
                ready.append(job)

        if not ready:
            # 没有可运行作业时，GPU 空闲到下一次到达。
            next_job = min(pending, key=arrival_time)
            print(f"  GPU idle: {current_time:.1f}s -> {next_job['arrival']:.1f}s")
            current_time = next_job["arrival"]
            continue

        # 函数也可以作为参数传入。这里调用传入的调度策略。
        job = choose_job(ready)
        start = current_time
        finish = start + duration(job)
        records.append({
            "name": job["name"],
            "arrival": job["arrival"],
            "start": start,
            "finish": finish,
            "waiting": start - job["arrival"],
            "jct": finish - job["arrival"],
        })
        current_time = finish
        pending.remove(job)

    return records


def show_result(title, choose_job):
    print(title)
    records = simulate(JOBS, choose_job)
    for row in records:
        print(
            f"  {row['name']}: arrival={row['arrival']:.1f}s, "
            f"start={row['start']:.1f}s, finish={row['finish']:.1f}s, "
            f"wait={row['waiting']:.1f}s, JCT={row['jct']:.1f}s"
        )
    mean_jct = sum(row["jct"] for row in records) / len(records)
    print(f"  mean JCT={mean_jct:.2f}s\n")


def main():
    show_result("FIFO", choose_fifo)
    show_result("Shortest available job first", choose_shortest)
    show_result("Longest available job first",choose_longest)


if __name__ == "__main__":
    main()

# 练习：每题都从原始 JOBS 开始，先预测，再运行。
# 1. C 的运行时间最短，为什么第 0 秒先运行 A？
#因为第0秒只有A到了
# 2. 把 A 的 arrival 改成 5，预测两种策略的执行顺序和空闲区间。
# JOBS = [
#     {"name": "A", "arrival": 5, "work": 8, "rate": 1},
#     {"name": "B", "arrival": 1, "work": 4, "rate": 1},
#     {"name": "C", "arrival": 2, "work": 1, "rate": 1},
# ]
#FIFO:B->C->A
#短任务优先：B->C->A

# 3. 添加 D：arrival=20、work=2、rate=1。观察 GPU idle 输出。
# 4. 只修改调度策略，让已经到达的最长作业优先；不改 simulate。
# 思考：C 在第 2 秒到达时，哪段代码使它无法立即打断正在运行的 A？
