# 导读 单作业或两作业组合的可哈希标识，支持比较、集合运算和分解成单作业。
# 导读 来源 job_id_pair.py。以下带「导读」的行是阅读注释。
import hashlib

class JobIdPair():

    # 导读 JobIdPair.__init__（原源码第 5 行）
    # 导读 构造 JobIdPair 时执行。为实例建立 self._job0、self._job1、self._hash_value、self._is_pair、self._as_tuple、self._singletons 等状态。
    # 导读 其余显式参数为 job0、job1，默认值见函数签名。
    # 导读 函数体直接赋值的实例字段包括 self._job0、self._job1、self._hash_value、self._is_pair、self._as_tuple、self._singletons、self._as_set、self._repr。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, job0, job1):
        if job0 is None and job1 is None:
            raise ValueError('Cannot form JobIdPair with both ids None')
        elif job0 is None and job1 is not None:
            raise ValueError('First job id in a JobIdPair cannot be None')
        elif job0 is not None and job1 is not None:
            self._job0 = min(job0, job1)
            self._job1 = max(job0, job1)
        else:
            self._job0 = job0
            self._job1 = None

        a = self._job0
        b = self._job1
        if b is None:
            self._hash_value = a
        else:
            self._hash_value = a * a + a + b if a > b else a + b * b

        self._is_pair = self._job0 is not None and self._job1 is not None
        self._as_tuple = (self._job0, self._job1)
        if self._job1 is None:
            self._singletons = (self,)
        else:
            self._singletons = (JobIdPair(self._job0, None),
                                JobIdPair(self._job1, None))
    
        self._as_set = set([self._job0, self._job1])

        if self._job1 is None:
            self._repr = '%d' % (self._job0)
        else:
            self._repr = '(%d, %d)' % (self._job0, self._job1)

    # 导读 JobIdPair.__getitem__（原源码第 39 行）
    # 导读 按索引返回标识中对应的作业编号。
    # 导读 参数 index 表示将字典转换成矩阵时保存的行列索引。
    # 导读 从返回表达式可以继续追踪 self._job0、self._job1。
    def __getitem__(self, index):
        if index == 0:
            return self._job0
        elif index == 1:
            return self._job1
        else:
            raise ValueError('Attempting to access invalid JobIdPair '
                             'index %d' % index)

    # 导读 JobIdPair.__lt__（原源码第 48 行）
    # 导读 为两个作业标识定义排序所需的小于比较。
    # 导读 其余显式参数为 other，默认值见函数签名。
    # 导读 从返回表达式可以继续追踪 self._job1 < other._job1、self._job0 < other._job0。
    def __lt__(self, other):
        if other._job1 is not None:
            if self._job1 is None:
                return True
            elif self._job0 == other._job0:
                return self._job1 < other._job1
        elif self._job1 is not None:
            return False
        return self._job0 < other._job0

    # 导读 JobIdPair.__eq__（原源码第 58 行）
    # 导读 按代码中的编号规则比较两个作业标识是否相同。
    # 导读 其余显式参数为 other，默认值见函数签名。
    # 导读 从返回表达式可以继续追踪 self._job0 == other._job0 and self._job1 == other._job1。
    def __eq__(self, other):
        return self._job0 == other._job0 and self._job1 == other._job1

    # 导读 JobIdPair.__hash__（原源码第 61 行）
    # 导读 生成能与相等比较配套的哈希，用于字典键和集合。
    # 导读 从返回表达式可以继续追踪 self._hash_value。
    def __hash__(self):
        return self._hash_value

    # 导读 JobIdPair.__repr__（原源码第 64 行）
    # 导读 返回作业标识的可读字符串，供日志和调试显示。
    # 导读 从返回表达式可以继续追踪 self._repr。
    def __repr__(self):
        return self._repr
    
    # 导读 JobIdPair.as_tuple（原源码第 67 行）
    # 导读 把作业标识转换为元组表示。
    # 导读 从返回表达式可以继续追踪 self._as_tuple。
    def as_tuple(self):
        return self._as_tuple

    # 导读 JobIdPair.as_set（原源码第 70 行）
    # 导读 把标识包含的作业转换为集合，方便检查交集。
    # 导读 从返回表达式可以继续追踪 self._as_set。
    def as_set(self):
        return self._as_set

    # 导读 JobIdPair.overlaps_with（原源码第 73 行）
    # 导读 检查两个标识是否包含共同作业。
    # 导读 其余显式参数为 other，默认值见函数签名。
    # 导读 从返回表达式可以继续追踪 self._job0 in other._as_set。
    def overlaps_with(self, other):
        if self._is_pair:
            raise ValueError('Can only call overlaps_with on a '
                             'single job id')
        return self._job0 in other._as_set

    # 导读 JobIdPair.is_pair（原源码第 79 行）
    # 导读 判断标识是否包含两份作业。
    # 导读 从返回表达式可以继续追踪 self._is_pair。
    def is_pair(self):
        return self._is_pair

    # 导读 JobIdPair.singletons（原源码第 82 行）
    # 导读 把组合拆成各自的单作业标识。
    # 导读 从返回表达式可以继续追踪 self._singletons。
    def singletons(self):
        return self._singletons
