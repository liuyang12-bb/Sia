# 导读 加载原始训练记录与性能画像。为 Job 提供单步计时、梯度统计、epoch 进度与完成判定所需的数据。
# 导读 来源 applications.py。以下带「导读」的行是阅读注释。
import collections
import glob
import math
import os
import pandas
import functools
import pdb

from scipy.interpolate import interp1d, LinearNDInterpolator

# 导读 get（原源码第 11 行）
# 导读 按名字从 APPLICATIONS 中取已加载的配置。
# 导读 参数 name 表示此对象或待查询条目的名称。
# 导读 从返回表达式可以继续追踪 APPLICATIONS[name]。
def get(name):
    return APPLICATIONS[name]


# 导读 memoize（原源码第 15 行）
# 导读 返回一个缓存包装函数。位置参数元组相同时复用已经计算的返回值。
# 导读 其余显式参数为 f，默认值见函数签名。
# 导读 从返回表达式可以继续追踪 helper。
def memoize(f):
    memo = {}

    # 导读 memoize.helper（原源码第 18 行）
    # 导读 先按参数检查缓存，缺失时调用原函数并保存结果。
    # 导读 从返回表达式可以继续追踪 memo[x]。
    def helper(*x):
        if x not in memo:
            memo[x] = f(*x)
        return memo[x]
    return helper


class Application(object):
    # 导读 Application.__init__（原源码第 26 行）
    # 导读 从应用目录加载验证曲线、逐放置计时和规模计时表，建立 batch 范围、应用类别及训练结束条件。
    # 导读 参数 trace_dir 表示应用性能与验证记录的目录，init_batch_size 表示应用的初始或参考全局 batch，max_batch_size 表示全局 batch 的上界。
    # 导读 参数 num_stages 表示流水线并行的阶段数。
    # 导读 其余显式参数为 cluster_suffix、min_local_bsz、max_local_bsz、max_epochs、target_metric，默认值见函数签名。
    # 导读 函数体中的相关调用包括 pandas.read_csv、self.placements.placement.apply、self.placements.local_bsz.min、self.placements.local_bsz.max、self.validation.values。
    # 导读 函数体直接赋值的实例字段包括 self.name、self.validation、self.placements、self.scalability、self.init_batch_size、self.max_batch_size、self.min_local_bsz、self.max_local_bsz、self.max_epochs。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, trace_dir, cluster_suffix=None,
                 init_batch_size=None, max_batch_size=None,
                 min_local_bsz=None, max_local_bsz=None,
                 max_epochs=None, target_metric=None,
                 num_stages=1):
        self.name = os.path.basename(trace_dir)
        # 导读 validation 的键是 batch 大小，值是该 batch 的训练与验证曲线表。
        validation = {}
        for path in glob.glob(os.path.join(trace_dir, "validation-*.csv")):
            batch_size = int(path.split("-")[-1].split(".")[0])
            validation[batch_size] = pandas.read_csv(path)
        self.validation = collections.OrderedDict(sorted(validation.items()))
        # 导读 按机器类型后缀选取 placements 表。每个 placement 编码描述各节点上的卡数，例如 21 对应 (2, 1)。
        placements_file = f"placements-{cluster_suffix}.csv" if cluster_suffix else "placements.csv"
        self.placements = \
            pandas.read_csv(os.path.join(trace_dir, placements_file))
        # 导读 placement 编码的数字位数是节点数，各位之和是 GPU 数。
        self.placements["num_nodes"] = \
            self.placements.placement.apply(lambda p: len(str(p)))
        self.placements["num_replicas"] = \
            self.placements.placement.apply(lambda p: sum(map(int, str(p))))
        # 导读 scalability 表为跨规模查询提供额外的计时点。
        scalability_file = f"scalability-{cluster_suffix}.csv" if cluster_suffix else "scalability.csv"
        self.scalability = \
            pandas.read_csv(os.path.join(trace_dir, scalability_file))
        self.init_batch_size = init_batch_size or min(self.validation)
        self.max_batch_size = max_batch_size or max(self.validation)
        self.min_local_bsz = min_local_bsz or self.placements.local_bsz.min()
        self.max_local_bsz = max_local_bsz or self.placements.local_bsz.max()
        if self.name == "deepspeech2" and cluster_suffix == "rtx":
            self.max_local_bsz = 57
        assert self.max_batch_size >= self.min_local_bsz
        self.max_epochs = max_epochs or min(map(len, self.validation.values()))
        self.target_metric = target_metric
        # 导读 恢复开销是按应用类型配置的模拟参数。CIFAR10 在这里取 50 秒。
        if self.name == "cifar10":
            self.rescale_time = 50
        elif self.name == "deepspeech2":
            self.rescale_time = 25
        elif self.name == "bert":
            self.rescale_time = 120
        elif self.name == "yolov3":
            self.rescale_time = 80
        elif self.name == "imagenet":
            self.rescale_time = 250
        elif self.name == "ncf":
            self.rescale_time = 15
        else:
            self.rescale_time = 30
        self.num_stages = num_stages
        if self.num_stages > 1:
            self.accum_steps = 48
        else:
            self.accum_steps = 1

    # 导读 Application._validated_batch_sizes（原源码第 76 行）
    # 导读 找到验证曲线中包围目标 batch 的两个已有 batch 值，供插值使用。
    # 导读 参数 batch_size 表示全局 batch 大小。
    # 导读 从返回表达式可以继续追踪 (lower_bsz, upper_bsz)。
    def _validated_batch_sizes(self, batch_size):
        # Find the lower-bound and upper-bound batch sizes (may be the same).
        lower_bsz = upper_bsz = None
        for bsz in self.validation:
            if bsz <= batch_size:
                lower_bsz = bsz
            if bsz >= batch_size:
                upper_bsz = bsz
                break
        assert lower_bsz is not None and upper_bsz is not None, \
            "{} {}".format(batch_size, list(self.validation))
        assert lower_bsz <= batch_size <= upper_bsz
        return lower_bsz, upper_bsz

    # 导读 Application.get_configurations（原源码第 90 行）
    # 导读 枚举候选 GPU 数和批量，按代码中的利用率区间及资源上限筛选配置。
    # 导读 参数 ngpus_per_node 表示每节点 GPU 数，异构接口可按类型索引。
    # 导读 其余显式参数为 lo_util、hi_util、max_gpus，默认值见函数签名。
    # 导读 函数体中的相关调用包括 self.validation.items、self.get_completion_epoch、self.get_throughput。
    # 导读 从返回表达式可以继续追踪 ret。
    def get_configurations(self, lo_util=0.5, hi_util=0.8, max_gpus=24, ngpus_per_node=4):
        # Assuming a cluster of max_gpus//ngpus_per_node nodes each with ngpus_per_node GPUs.
        ret = []
        base_jct = None
        base_batch_size = None
        num_replicas_list = [1]
        num_replicas_list.extend([2*x for x in range(1, max_gpus // 2)])
        for num_replicas in num_replicas_list:
            if num_replicas * self.min_local_bsz > self.max_batch_size:
                break
            placement = ()
            while sum(placement) < num_replicas:
                placement = (*placement, min(num_replicas -
                             sum(placement), ngpus_per_node))
            best_jct = None
            best_batch_size = None
            for batch_size, valid in self.validation.items():
                local_bsz = math.ceil(batch_size / sum(placement) - 1e-8)
                if local_bsz < self.min_local_bsz:
                    continue
                accum_steps = math.ceil(
                    local_bsz / self.max_local_bsz - 1e-8) - 1
                # if sum(placement) == 1 and batch_size > self.init_batch_size:
                #    accum_steps = max(1, accum_steps)
                atomic_bsz = math.ceil(local_bsz / (accum_steps + 1) - 1e-8)
                epoch = self.get_completion_epoch(batch_size)
                step_time, sync_time = self.get_throughput(
                    placement, atomic_bsz)
                step_time += accum_steps * (step_time - sync_time)
                jct = valid.iteration[epoch] * step_time
                if best_jct is None or jct < best_jct:
                    best_jct = jct
                    best_batch_size = batch_size
            if num_replicas == 1:
                base_jct = best_jct
                base_batch_size = best_batch_size
            elif best_jct < 12 * 3600 and \
                    lo_util < base_jct / best_jct / num_replicas < hi_util:
                ret.append((num_replicas, best_batch_size, best_jct))
        if not ret:
            ret.append((1, base_batch_size, base_jct))
        return ret

    # 导读 Application.get_best_batch_size（原源码第 133 行）
    # 导读 依据已有记录估算指定副本数下不同 batch 的完成时间，返回选出的 batch。
    # 导读 参数 num_replicas 表示配置中的训练副本数，数据并行时一般对应 GPU 数。
    # 导读 函数体中的相关调用包括 self.validation.items、self.get_completion_epoch、self.get_throughput。
    # 导读 从返回表达式可以继续追踪 best_batch_size。
    def get_best_batch_size(self, num_replicas):
        # Assuming a cluster of 16 nodes each with 4 GPUs.
        ret = []
        base_jct = None
        base_batch_size = None
        if num_replicas * self.min_local_bsz > self.max_batch_size:
            return None
        placement = ()
        while sum(placement) < num_replicas:
            placement = (*placement, min(num_replicas - sum(placement), 4))
        best_jct = None
        best_batch_size = None
        for batch_size, valid in self.validation.items():
            local_bsz = math.ceil(batch_size / sum(placement))
            if local_bsz < self.min_local_bsz:
                continue
            if local_bsz > self.max_local_bsz:
                break
            epoch = self.get_completion_epoch(batch_size)
            step_time, _ = self.get_throughput(placement, local_bsz)
            jct = valid.iteration[epoch] * step_time
            if best_jct is None or jct < best_jct:
                best_jct = jct
                best_batch_size = batch_size
        return best_batch_size

    # 导读 Application.get_epoch（原源码第 159 行）
    # 导读 从已有训练曲线的 progress 列查询给定累计进度对应的 epoch。
    # 导读 参数 progress 表示模拟训练的累计进度。
    # 导读 函数体中的相关调用包括 self.validation.values。
    # 导读 从返回表达式可以继续追踪 max((df.progress.searchsorted(progress, 'right') for df in self.validation.values()))。
    def get_epoch(self, progress):
        return max(df.progress.searchsorted(progress, "right")
                   for df in self.validation.values())

    # 导读 Application.get_progress（原源码第 164 行）
    # 导读 返回给定 epoch 边界的累计训练进度。第 0 个边界对应进度 0。
    # 导读 参数 epoch 表示训练曲线中的 epoch 或其边界编号。
    # 导读 函数体中的相关调用包括 self.validation.values。
    # 导读 从返回表达式可以继续追踪 0.0、min((df.progress[epoch - 1] for df in self.validation.values()))。
    @memoize
    def get_progress(self, epoch):
        if epoch == 0:
            return 0.0
        return min(df.progress[epoch - 1] for df in self.validation.values())

    # 导读 Application.get_completion_epoch（原源码第 170 行）
    # 导读 依据目标指标或固定训练长度，确定指定 batch 下作业的完成 epoch 阈值。
    # 导读 参数 batch_size 表示全局 batch 大小。
    # 导读 函数体中的相关调用包括 self.get_best_metric。
    # 导读 从返回表达式可以继续追踪 self.max_epochs - 1、epoch。
    @functools.lru_cache(maxsize=100000, typed=False)
    def get_completion_epoch(self, batch_size):
        if self.target_metric is None:
            return self.max_epochs - 1
        best_metric = None
        for epoch in range(self.max_epochs):
            next_metric = self.get_best_metric(batch_size, epoch)
            if best_metric is not None:
                sign = self.target_metric - best_metric
                if sign * (self.target_metric - next_metric) <= 0:
                    # Opposite signs, crossed target metric.
                    return epoch
        return epoch

    # 导读 Application.get_iteration（原源码第 184 行）
    # 导读 查表或插值获得指定 batch 和 epoch 对应的累计迭代次数。
    # 导读 参数 batch_size 表示全局 batch 大小，epoch 表示训练曲线中的 epoch 或其边界编号。
    # 导读 函数体中的相关调用包括 self._validated_batch_sizes。
    # 导读 从返回表达式可以继续追踪 lower_it、((batch_size - lower_bsz) * upper_it + (upper_bsz - batch_size) * lower_it) / (upper_bsz - lower_bsz)。
    @functools.lru_cache(maxsize=100000, typed=False)
    def get_iteration(self, batch_size, epoch):
        # Returns the number of iterations after completing a given epoch.
        lower_bsz, upper_bsz = self._validated_batch_sizes(batch_size)
        lower_it = self.validation[lower_bsz].iteration[epoch]
        upper_it = self.validation[upper_bsz].iteration[epoch]
        if lower_bsz == upper_bsz:
            assert lower_it == upper_it
            return lower_it
        # Linear interpolation between lower_bsz and upper_bsz.
        return ((batch_size - lower_bsz) * upper_it +
                (upper_bsz - batch_size) * lower_it) / (upper_bsz - lower_bsz)

    # 导读 Application.get_cur_iteration（原源码第 197 行）
    # 导读 用当前 epoch 与 progress 在相邻训练边界之间估算当前迭代号。
    # 导读 参数 batch_size 表示全局 batch 大小，cur_epoch 表示作业当前所处 epoch，cur_progress 表示作业当前的累计进度。
    # 导读 函数体中的相关调用包括 self.get_iteration、self.get_progress。
    # 导读 从返回表达式可以继续追踪 round(cur_iter)。
    @functools.lru_cache(maxsize=100000, typed=False)
    def get_cur_iteration(self, batch_size, cur_epoch, cur_progress):
        prev_iter = self.get_iteration(
            batch_size, cur_epoch-1) if cur_epoch > 0 else 0
        next_iter = self.get_iteration(batch_size, cur_epoch)
        prev_progress = self.get_progress(cur_epoch)
        next_progress = self.get_progress(cur_epoch+1)
        assert cur_progress >= prev_progress and cur_progress <= next_progress
        cur_iter = prev_iter + (cur_progress-prev_progress) / \
            (next_progress-prev_progress) * (next_iter-prev_iter)
        return round(cur_iter)

    # 导读 Application.get_best_metric（原源码第 209 行）
    # 导读 查表或在已有 batch 之间插值，得到指定 epoch 的最佳验证指标。
    # 导读 参数 batch_size 表示全局 batch 大小，epoch 表示训练曲线中的 epoch 或其边界编号。
    # 导读 函数体中的相关调用包括 self._validated_batch_sizes、self.validation.values、self.validation[lower_bsz].metric[:epoch].max、self.validation[upper_bsz].metric[:epoch].max、self.validation[lower_bsz].metric[:epoch].min、self.validation[upper_bsz].metric[:epoch].min。
    # 导读 从返回表达式可以继续追踪 lower_val、((batch_size - lower_bsz) * upper_val + (upper_bsz - batch_size) * lower_val) / (upper_bsz - lower_bsz)。
    @functools.lru_cache(maxsize=100000, typed=False)
    def get_best_metric(self, batch_size, epoch):
        # Returns the best observed validation metric before a given epoch.
        if epoch == 0:
            return None
        lower_bsz, upper_bsz = self._validated_batch_sizes(batch_size)
        if (next(iter(self.validation.values())).metric.values[0] <
                next(iter(self.validation.values())).metric.values[-1]):
            # Validation metric increases.
            lower_val = self.validation[lower_bsz].metric[:epoch].max()
            upper_val = self.validation[upper_bsz].metric[:epoch].max()
        else:
            lower_val = self.validation[lower_bsz].metric[:epoch].min()
            upper_val = self.validation[upper_bsz].metric[:epoch].min()
        if lower_bsz == upper_bsz:
            assert lower_val == upper_val
            return lower_val
        # Linear interpolation between lower_bsz and upper_bsz.
        return ((batch_size - lower_bsz) * upper_val +
                (upper_bsz - batch_size) * lower_val) / (upper_bsz - lower_bsz)

    # 导读 Application.get_grad_stats（原源码第 230 行）
    # 导读 按 batch 与 epoch 查得梯度信号平方和方差，返回二元组供效率计算。
    # 导读 参数 batch_size 表示全局 batch 大小，epoch 表示训练曲线中的 epoch 或其边界编号。
    # 导读 函数体中的相关调用包括 self._validated_batch_sizes。
    # 导读 从返回表达式可以继续追踪 (lower_sqr, lower_var)、(sqr, var)。
    @functools.lru_cache(maxsize=100000, typed=False)
    def get_grad_stats(self, batch_size, epoch):
        # Returns the gradient sqr and var estimates during a given epoch.
        lower_bsz, upper_bsz = self._validated_batch_sizes(batch_size)
        lower_sqr = self.validation[lower_bsz].grad_sqr[epoch]
        upper_sqr = self.validation[upper_bsz].grad_sqr[epoch]
        lower_var = self.validation[lower_bsz].grad_var[epoch]
        upper_var = self.validation[upper_bsz].grad_var[epoch]
        if lower_bsz == upper_bsz:
            assert lower_sqr == upper_sqr and lower_var == upper_var
            return lower_sqr, lower_var
        # Linear interpolation between lower_bsz and upper_bsz.
        sqr = ((batch_size - lower_bsz) * upper_sqr +
               (upper_bsz - batch_size) * lower_sqr) / (upper_bsz - lower_bsz)
        var = ((batch_size - lower_bsz) * upper_var +
               (upper_bsz - batch_size) * lower_var) / (upper_bsz - lower_bsz)
        return sqr, var

    # 导读 Application.get_throughput（原源码第 248 行）
    # 导读 输入 placement 和每副本局部 batch。优先查相同放置的记录并插值，缺失时尝试更通用的规模插值。返回 step_time 和 sync_time，单位为秒。
    # 导读 参数 placement 表示每个已使用节点上的 GPU 或副本数，local_bsz 表示传入当前方法的每副本局部 batch。
    # 导读 函数体中的相关调用包括 interp1d、self.placements.groupby(xs)[xs + ys].mean、self.placements.groupby、LinearNDInterpolator。
    # 导读 从返回表达式可以继续追踪 (accum_time.iloc[0], network_time.iloc[0])、ret。
    @functools.lru_cache(maxsize=100000, typed=False)
    def get_throughput(self, placement, local_bsz):
        # Normalize placement to the lexicographically smallest rotation.
        placement = tuple(filter(None, placement))
        placement = min(placement[i:] + placement[:i]
                        for i in range(len(placement)))
        # 导读 将节点卡数元组转成画像中使用的整数编码。
        placement_id = int("".join(map(str, placement)))
        xs = ["num_nodes", "num_replicas", "local_bsz"]
        ys = ["step_time", "sync_time"]
        if self.num_stages > 1 and placement_id not in self.placements.placement.values:
            return -1, -1
        elif self.num_stages > 1:
            # Found in placement traces
            df = self.placements[self.placements.placement == placement_id]
            atomic_bsz = local_bsz // self.num_microbatches
            df = df[df.local_bsz == atomic_bsz]
            fwd_stage = df['step_time']
            accum_time = 3 * ((self.num_stages - 1) *
                              fwd_stage + self.accum_steps * fwd_stage)
            network_time = df['sync_time']
            # accum_time and network_time are pandas series with one element.
            # return only the first element for each of them
            return accum_time.iloc[0], network_time.iloc[0]

        # 导读 有相同放置时，只沿 local_bsz 维度插值；输入恰好在已测点时取对应计时。
        if placement_id in self.placements.placement.values:
            # Found in placement traces, interpolate between local_bsz.
            df = self.placements[self.placements.placement == placement_id]
            interpolator = interp1d(df.local_bsz.values, df[ys].values, axis=0)
            try:
                ret = interpolator(local_bsz)
            except ValueError:
                return -1, -1
        else:
            # 导读 缺少这个放置的精确记录时，尝试在节点数、副本数、局部 batch 三维上插值。
            # Interpolate between num_nodes, num_replicas, and local_bsz.
            df = self.placements.groupby(xs)[xs + ys].mean()
            df = df.append(self.scalability, ignore_index=True)
            num_nodes, num_replicas = len(placement), sum(placement)
            num_nodes = min(num_nodes, 16)
            interpolator = LinearNDInterpolator(df[xs].values, df[ys].values)
            try:
                ret = interpolator([num_nodes, num_replicas, local_bsz])[0]
            except ValueError:
                return -1, -1
        assert sum(ret) == sum(ret), "{} {} {}".format(
            self.name, placement, local_bsz)
        return ret

    # 导读 Application.get_throughput_with_accum（原源码第 294 行）
    # 导读 把全局 batch 拆成每副本单次计算及累积步，再将样本数除以总迭代时间，返回样本每秒。
    # 导读 参数 placement 表示每个已使用节点上的 GPU 或副本数，batch_size 表示全局 batch 大小。
    # 导读 函数体中的相关调用包括 self.get_max_local_bsz、self.get_throughput。
    # 导读 从返回表达式可以继续追踪 xput。
    def get_throughput_with_accum(self, placement, batch_size):
        num_replicas = sum(placement)
        # otherwise, use the batch size specified by the user
        local_bsz = math.ceil(batch_size / num_replicas - 1e-8)
        max_local_bsz = self.get_max_local_bsz(placement)
        # 导读 单次 batch 容量不足时增加梯度累积，将一次更新拆成多次局部计算。
        accum_steps = math.ceil(local_bsz / max_local_bsz - 1e-8) - 1
        if num_replicas == 1 and batch_size > self.init_batch_size:
            accum_steps = max(1, accum_steps)
        atomic_bsz = math.ceil(local_bsz / (accum_steps + 1) - 1e-8)
        count = num_replicas * (accum_steps + 1)
        atomic_bsz = min(atomic_bsz, int(self.max_batch_size / count))
        placement = tuple(filter(None, placement))
        step_time, sync_time = self.get_throughput(placement, atomic_bsz)
        # 导读 用全局 batch 除以整次更新的秒数，得到样本每秒。
        xput = batch_size / (step_time + (step_time - sync_time) * accum_steps)
        return xput

    # 导读 Application.get_max_local_bsz（原源码第 310 行）
    # 导读 查看给定放置下画像覆盖的最大局部 batch，供计算批量上界。
    # 导读 参数 placement 表示每个已使用节点上的 GPU 或副本数。
    # 导读 从返回表达式可以继续追踪 max_local_bsz。
    def get_max_local_bsz(self, placement):
        # Normalize placement to the lexicographically smallest rotation.
        placement = tuple(filter(None, placement))
        placement = min(placement[i:] + placement[:i]
                        for i in range(len(placement)))
        # 导读 placement_id 是画像里存储的节点卡数编码，例如 21 表示节点卡数 (2, 1)。
        placement_id = int("".join(map(str, placement)))
        num_nodes, num_replicas = len(placement), sum(placement)
        if placement_id in self.placements.placement.values:
            df = self.placements[self.placements.placement == placement_id]
            max_local_bsz = df.local_bsz.max()
        else:
            sc_df = self.scalability[(self.scalability.num_nodes >= num_nodes) & (
                self.scalability.num_replicas >= num_replicas)]
            max_local_bsz = sc_df.local_bsz.max()
        return max_local_bsz


class PMPApplication(Application):
    # start with pass through calls to super class
    # 导读 PMPApplication.__init__（原源码第 329 行）
    # 导读 从应用目录加载验证曲线、逐放置计时和规模计时表，建立 batch 范围、应用类别及训练结束条件。
    # 导读 函数体直接赋值的实例字段包括 self.num_stages、self.max_local_bsz、self.microbatch_size、self.num_microbatches。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, *args, **kwargs):
        # must contain a num_stages attribute
        self.num_stages = 0
        super().__init__(*args, **kwargs)
        self.max_local_bsz = 48
        self.microbatch_size = 1
        self.num_microbatches = self.max_local_bsz // self.microbatch_size

    # 导读 PMPApplication.get_max_local_bsz（原源码第 337 行）
    # 导读 查看给定放置下画像覆盖的最大局部 batch，供计算批量上界。
    # 导读 参数 placement 表示每个已使用节点上的 GPU 或副本数。
    # 导读 从返回表达式可以继续追踪 self.max_local_bsz。
    def get_max_local_bsz(self, placement):
        return self.max_local_bsz


TRACES_DIR = os.path.join(os.path.dirname(__file__), "./traces")
APPLICATIONS = {
    "aws": {
        "bert": Application(os.path.join(TRACES_DIR, "bert"), cluster_suffix="aws", max_epochs=2),
        "cifar10": Application(os.path.join(TRACES_DIR, "cifar10"), cluster_suffix="aws", max_epochs=100),
        "ncf": Application(os.path.join(TRACES_DIR, "ncf"), cluster_suffix="aws", max_epochs=10),
        "imagenet": Application(os.path.join(TRACES_DIR, "imagenet"), cluster_suffix="aws", max_epochs=90),
        "deepspeech2": Application(os.path.join(TRACES_DIR, "deepspeech2"), cluster_suffix="aws", max_epochs=80),
        "yolov3": Application(os.path.join(TRACES_DIR, "yolov3"), cluster_suffix="aws", max_epochs=50)
    },
    "dgx": {
        "bert": Application(os.path.join(TRACES_DIR, "bert"), cluster_suffix="dgx", max_epochs=2),
        "cifar10": Application(os.path.join(TRACES_DIR, "cifar10"), cluster_suffix="dgx", max_epochs=100),
        "ncf": Application(os.path.join(TRACES_DIR, "ncf"), cluster_suffix="dgx", max_epochs=10),
        "imagenet": Application(os.path.join(TRACES_DIR, "imagenet"), cluster_suffix="dgx", max_epochs=90),
        "deepspeech2": Application(os.path.join(TRACES_DIR, "deepspeech2"), cluster_suffix="dgx", max_epochs=80),
        "yolov3": Application(os.path.join(TRACES_DIR, "yolov3"), cluster_suffix="dgx", max_epochs=50)
    },
    "rtx": {
        "bert": Application(os.path.join(TRACES_DIR, "bert"), cluster_suffix="rtx", max_epochs=2),
        "cifar10": Application(os.path.join(TRACES_DIR, "cifar10"), cluster_suffix="rtx", max_epochs=100),
        "ncf": Application(os.path.join(TRACES_DIR, "ncf"), cluster_suffix="rtx", max_epochs=10),
        "imagenet": Application(os.path.join(TRACES_DIR, "imagenet"), cluster_suffix="rtx", max_epochs=90),
        "deepspeech2": Application(os.path.join(TRACES_DIR, "deepspeech2"), cluster_suffix="rtx", max_epochs=80),
        "yolov3": Application(os.path.join(TRACES_DIR, "yolov3"), cluster_suffix="rtx", max_epochs=50),
        "gpt_pmp": PMPApplication(os.path.join(TRACES_DIR, "gpt_pmp"), cluster_suffix="rtx", max_epochs=2, num_stages=8),
    },
    "dgx-ext": {
        "bert": Application(os.path.join(TRACES_DIR, "bert"), cluster_suffix="dgx-ext", max_epochs=2),
        "cifar10": Application(os.path.join(TRACES_DIR, "cifar10"), cluster_suffix="dgx-ext", max_epochs=100),
        "ncf": Application(os.path.join(TRACES_DIR, "ncf"), cluster_suffix="dgx-ext", max_epochs=10),
        "imagenet": Application(os.path.join(TRACES_DIR, "imagenet"), cluster_suffix="dgx-ext", max_epochs=90),
        "deepspeech2": Application(os.path.join(TRACES_DIR, "deepspeech2"), cluster_suffix="dgx-ext", max_epochs=80),
        "yolov3": Application(os.path.join(TRACES_DIR, "yolov3"), cluster_suffix="dgx-ext", max_epochs=50),
        "gpt_pmp": PMPApplication(os.path.join(TRACES_DIR, "gpt_pmp"), cluster_suffix="dgx-ext", max_epochs=2, num_stages=2),
    },
    "azure": {
        "bert": Application(os.path.join(TRACES_DIR, "bert"), cluster_suffix="azure", max_epochs=2),
        "cifar10": Application(os.path.join(TRACES_DIR, "cifar10"), cluster_suffix="azure", max_epochs=100),
        "ncf": Application(os.path.join(TRACES_DIR, "ncf"), cluster_suffix="azure", max_epochs=10),
        "imagenet": Application(os.path.join(TRACES_DIR, "imagenet"), cluster_suffix="azure", max_epochs=90),
        "deepspeech2": Application(os.path.join(TRACES_DIR, "deepspeech2"), cluster_suffix="azure", max_epochs=80),
        "yolov3": Application(os.path.join(TRACES_DIR, "yolov3"), cluster_suffix="azure", max_epochs=50)
    },
    "quad": {
        "bert": Application(os.path.join(TRACES_DIR, "bert"), cluster_suffix="quad", max_epochs=2),
        "cifar10": Application(os.path.join(TRACES_DIR, "cifar10"), cluster_suffix="quad", max_epochs=100),
        "ncf": Application(os.path.join(TRACES_DIR, "ncf"), cluster_suffix="quad", max_epochs=10),
        "imagenet": Application(os.path.join(TRACES_DIR, "imagenet"), cluster_suffix="quad", max_epochs=90),
        "deepspeech2": Application(os.path.join(TRACES_DIR, "deepspeech2"), cluster_suffix="quad", max_epochs=80),
        "yolov3": Application(os.path.join(TRACES_DIR, "yolov3"), cluster_suffix="quad", max_epochs=50)
    }


}
# TODO:: daiyaan -- add gpt-pmp
