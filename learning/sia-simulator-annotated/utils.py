# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified: Suhas Jayaram Subramanya (suhasjs@cs.cmu.edu)

# 导读 Cluster 和策略之间交换的数据对象。JobInfo 描述作业摘要，NodeInfo 描述节点资源。
# 导读 来源 utils.py。以下带「导读」的行是阅读注释。
class JobInfo(object):
    # 导读 JobInfo.__init__（原源码第 18 行）
    # 导读 把资源需求、性能函数、提交时刻、已服务量和副本限制打包给策略。
    # 导读 参数 resources 表示资源名到数量的映射，preemptible 表示描述当前对象是否允许被抢占的布尔值，category 表示作业类别，例如 rigid 或 adaptive。
    # 导读 其余显式参数为 speedup_fn、creation_timestamp、attained_service、min_replicas、max_replicas、multipliers、scale_unit，默认值见函数签名。
    # 导读 函数体直接赋值的实例字段包括 self.resources、self.speedup_fn、self.creation_timestamp、self.attained_service、self.max_replicas、self.min_replicas、self.preemptible、self.multipliers、self.category。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, resources, speedup_fn, creation_timestamp, attained_service,
                 min_replicas, max_replicas, preemptible=True, multipliers=dict(),
                 category=None, scale_unit=1):
        """
        Args:
            resources (dict): Requested resources (eg. GPUs) of each replica.
            speedup_fn (SpeedupFunction): Speedup function for this job.
            creation_timestamp (datetime): Time when this job was created.
            min_replicas (int): Minimum number of replicas job's guaranteed.
            max_replicas (int): Maximum number of replicas. Maximum should be
                                greater or equal to Minimum
            preemptible (bool): Is the job preemptible?
            category (str) : [adaptive, rigid, scale-gpus]
            scale_unit (int) : number of GPUs to scale up or down by (default: 1)
        """
        for k, v in max_replicas.items():
            assert v > 0, f"max_replicas = 0 for cluster: {k}"
        for k in min_replicas.keys():
            assert min_replicas[k] <= max_replicas[k], f"min/max-replicas invalid"
        self.resources = resources
        self.speedup_fn = speedup_fn
        self.creation_timestamp = creation_timestamp
        self.attained_service = attained_service
        self.max_replicas = max_replicas
        self.min_replicas = min_replicas
        self.preemptible = preemptible
        self.multipliers = multipliers
        self.category = category


class JobInfoUnaware(object):
    # 导读 JobInfoUnaware.__init__（原源码第 49 行）
    # 导读 为异构性不感知的策略保存作业字段和各类资源限制。
    # 导读 参数 resources 表示资源名到数量的映射，preemptible 表示描述当前对象是否允许被抢占的布尔值，category 表示作业类别，例如 rigid 或 adaptive。
    # 导读 其余显式参数为 speedup_fn、creation_timestamp、attained_service、min_replicas、max_replicas、multipliers、scale_unit，默认值见函数签名。
    # 导读 函数体直接赋值的实例字段包括 self.resources、self.speedup_fn、self.creation_timestamp、self.attained_service、self.max_replicas、self.min_replicas、self.preemptible、self.multipliers、self.scale_unit。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, resources, speedup_fn, creation_timestamp, attained_service,
                 min_replicas, max_replicas, preemptible=True, multipliers=dict(),
                 category=None, scale_unit=1):
        """
        Args:
            resources (dict): Requested resources (eg. GPUs) of each replica.
            speedup_fn (SpeedupFunction): Speedup function for this job.
            creation_timestamp (datetime): Time when this job was created.
            min_replicas (int): Minimum number of replicas job's guaranteed.
            max_replicas (int): Maximum number of replicas. Maximum should be
                                greater or equal to Minimum
            preemptible (bool): Is the job preemptible?
            category (str) : [adaptive, rigid, scale-gpus]
            scale_unit (int) : number of GPUs to scale up or down by (default: 1)
        """
        assert max_replicas > 0, f"max_replicas = 0"
        assert min_replicas <= max_replicas, f"min/max-replicas invalid"
        self.resources = resources
        self.speedup_fn = speedup_fn
        self.creation_timestamp = creation_timestamp
        self.attained_service = attained_service
        self.max_replicas = max_replicas
        self.min_replicas = min_replicas
        self.preemptible = preemptible
        self.multipliers = multipliers
        self.scale_unit = 1


class NodeInfo(object):
    # 导读 NodeInfo.__init__（原源码第 78 行）
    # 导读 保存该节点的资源容量与抢占属性。
    # 导读 参数 resources 表示资源名到数量的映射，preemptible 表示描述当前对象是否允许被抢占的布尔值。
    # 导读 函数体直接赋值的实例字段包括 self.resources、self.preemptible。
    # 导读 调用完成后隐式返回 None，主要结果通过对象或输出操作留下。
    def __init__(self, resources, preemptible):
        """
        Args:
            resources (dict): Available resources (eg. GPUs) on this node.
            preemptible (bool): Whether this node is pre-emptible.
        """
        self.resources = resources
        self.preemptible = preemptible
