"""第三课：把第二课的职责组织成 Job、Policy、Cluster 对象。

本课保持第二课的调度规则：1 张 GPU，非抢占，固定且已知的运行速度，
没有启动成本。默认输入含第二课练习添加的 D。
先读 Job，再读两个策略，最后读 Cluster；本课不使用继承。
这是独立教学模型，不调用真实 Sia，不生成正式实验结果。
"""


class Job:
    def __init__(self, name, arrival, work, rate):
        # self 指向当前正在初始化的这个作业对象。
        self.name = name
        self.arrival = arrival
        self.work = work
        self.rate = rate
        # None 表示尚未开始 / 尚未完成，与时刻 0 不同。
        self.start = None
        self.finish = None

    def duration(self):
        # 方法可以读取当前这个作业自己的属性。
        return self.work / self.rate


# min 的 key 函数仍接收一个作业，现在通过属性和方法访问它。
def arrival_time(job):
    return job.arrival


def duration(job):
    return job.duration()


class FifoPolicy:
    def choose(self, ready_jobs):
        return min(ready_jobs, key=arrival_time)


class ShortestFirstPolicy:
    def choose(self, ready_jobs):
        return min(ready_jobs, key=duration)

class LongestFirstPolicy:
    def choose(self, ready_jobs):
        return max(ready_jobs, key=duration)

class Cluster:
    def __init__(self, jobs, policy):
        self.jobs = jobs
        self.pending = list(jobs)
        self.completed = []
        # 保存传入的策略对象；之后调用它的 choose 方法。
        self.policy = policy
        self.current_time = 0.0

    def run(self):
        while self.pending:
            ready = []
            for job in self.pending:
                if job.arrival <= self.current_time:
                    ready.append(job)

            if not ready:
                next_job = min(self.pending, key=arrival_time)
                print(f"  GPU idle: {self.current_time:.1f}s -> {next_job.arrival:.1f}s")
                self.current_time = next_job.arrival
                continue

            # 模拟器通过统一的 choose 接口使用策略。
            job = self.policy.choose(ready)
            job.start = self.current_time
            job.finish = job.start + job.duration()
            self.current_time = job.finish
            self.pending.remove(job)
            self.completed.append(job)


def make_jobs():
    # 每调用一次，创建一组新的作业对象。
    # 本课会修改作业的 start 和 finish，因此每种策略使用独立对象。
    return [
        Job(name="A", arrival=0, work=8, rate=1),
        Job(name="B", arrival=0, work=4, rate=1),
        Job(name="C", arrival=0, work=1, rate=1),
        Job(name="D", arrival=0, work=2, rate=1),
    ]


def show_result(title, policy):
    print(title)
    cluster = Cluster(jobs=make_jobs(), policy=policy)
    cluster.run()
    total_jct = 0.0
    for job in cluster.completed:
        waiting = job.start - job.arrival
        jct = job.finish - job.arrival
        total_jct += jct
        print(
            f"  {job.name}: arrival={job.arrival:.1f}s, "
            f"start={job.start:.1f}s, finish={job.finish:.1f}s, "
            f"wait={waiting:.1f}s, JCT={jct:.1f}s"
        )
    print(f"  mean JCT={total_jct / len(cluster.completed):.2f}s\n")


def main():
    show_result("FIFO", FifoPolicy())
    show_result("Shortest available job first", ShortestFirstPolicy())
    show_result("LongestFirst",LongestFirstPolicy())
    


if __name__ == "__main__":
    main()

# 练习：
# 1. 在单独的 scratch.py 中创建两个 Job，只改第一个的 rate，
#    打印两个对象的 duration()，观察它们是否互相影响。
# 2. 添加 LongestFirstPolicy，实现 choose，再在 main 中运行它。
#    保持 Cluster.run 的代码不变。
# 3. 把 make_jobs 中所有 arrival 改为 0，先预测三种策略的顺序。
# 4. 用自己的话解释 self.policy.choose(ready) 中的三个部分：
#    self 是谁？self.policy 是什么对象？choose 方法属于哪个类？
