"""Derived mechanism attribution and scope, never modifies raw results."""
import json
from experiment import OUT,load,save
p=load();audit=json.loads((OUT/'independent_audit.json').read_text())
obs=json.loads((OUT/'outcome_analysis.json').read_text());ex=json.loads((OUT/'exchange_composition.json').read_text())
m=audit['results']['medium/JOINT_CC']['summary'];b=audit['results']['medium/JOINT_CC_P']['summary']
comparison=audit['comparisons']['medium']['JOINT_CC_P']
group={}
for app in ['cifar10','deepspeech2']:
    x=obs['medium/JOINT_CC']['application_times'][app];y=obs['medium/JOINT_CC_P']['application_times'][app]
    group[app]={k:x[k]-y[k] for k in x}
relative={key:m[key]/b[key]-1 for key in ['median_jct','p95_jct','p99_jct','makespan']}
resources={key:m['totals'][key]/b['totals'][key]-1 for key in ['gpu_hours','cpu_core_hours','ram_gb_hours']}
save(OUT/'interpretation.json',dict(status=audit['status'],algorithm_changes=False,
    source_fingerprint=p['source_fingerprint'],auxiliary_quota_increment_mean_jct_improvement=comparison['improvement'],
    auxiliary_ablation_time_delta=comparison['time_delta'],by_application_time_delta=group,
    tail_relative_change_positive_is_worse=relative,resource_relative_change_positive_is_more=resources,
    mechanism=dict(mixed_CF_DS_node_rounds=ex['composition'].get('cifar10+deepspeech2',0),
      CF_receives_from_DS_node_rounds=ex['exchanges_by_receiver_donor'].get('cifar10 <- deepspeech2',0),
      first_cross_application_exchange=ex['first_cross_application_exchange']),
    supported_conclusion='本冻结合成场景观察到辅助配额自由度的额外平均JCT收益；Neutral内部轨迹相同。不是所有任务/尾部/资源指标都改善。',
    explanation='CIFAR实际训练减少2093任务秒，并使后续CIFAR排队减少7980秒；DeepSpeech训练增加3464秒，另增加125秒checkpoint。总JCT净减少6584秒。',
    caveats=['CPU优化总效应包含随后的GPU、放置和队列反馈，不等于固定GPU的一次CPU加速率。',
      'SIA-ECT-AUX与CC还存在准入、目标和放置差异；45.85%不能全归于CPU，配额消融7.63%是更聚焦证据。',
      'RAM在Medium全CC所选列均125GB/GPU，未证明RAM差异化的额外价值。',
      'E-legacy保持不变：全CC所选Medium列无裁剪，CC-P有11/624列分母CPU裁剪；主资源交换见证均支持。',
      '新20任务同时改变组成与到达，不是只改变任务数量的隔离实验。',
      '跨硬件映射不是目标集群实测；一个受控队列也不是20次独立重复。'],
    recommendation='值得在同一固定算法上预注册独立负载复验，重点检验7.63%配额增量是否可复现并约束长任务退化；不再仅靠增加任务数解释差异。本轮不启动。'))
print('INTERPRETATION_READY')
