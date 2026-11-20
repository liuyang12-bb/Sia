"""Physical-domain adapter for one-node-per-type mechanism clusters.

Official Sia enumerates 4/8-GPU templates; remove impossible columns and
rebuild its index arrays consistently for ALL baseline/joint methods.
The official objective, solver, placement, and restart code are unchanged.
"""
import numpy as np

def restrict_single_node_domain(policy):
    if any(n!=1 for n in policy.cluster_num_nodes.values()):
        raise ValueError('PHYSICAL_ADAPTER_SINGLE_NODE_TYPES_ONLY')
    policy.idx_to_config_map={};policy.config_to_idx_map={}
    for t in policy.cluster_ordering:
        nn,gg=policy.configs[t]
        mask=(nn==1)&(gg>=1)&(gg<=policy.cluster_ngpus_per_node[t])
        nn,gg=nn[mask],gg[mask]
        if not len(gg):raise ValueError('EMPTY_PHYSICAL_GPU_DOMAIN:'+t)
        policy.configs[t]=(nn,gg)
        for n,g in zip(nn,gg):
            item=(t,n,g);index=len(policy.idx_to_config_map)
            policy.idx_to_config_map[index]=item;policy.config_to_idx_map[item]=index
    policy.num_configs={t:len(v[1]) for t,v in policy.configs.items()}
    policy.total_num_configs=sum(policy.num_configs.values())
    policy.adaptive_num_replicas=np.asarray([v[2] for v in policy.idx_to_config_map.values()],dtype=np.uint32).reshape(-1,1)
    policy.adaptive_reduce_matrix=np.zeros((policy.total_num_configs,len(policy.cluster_ordering)),dtype=np.uint32)
    for i,(t,n,g) in policy.idx_to_config_map.items():policy.adaptive_reduce_matrix[i,policy.cluster_ordering.index(t)]=g
    return policy
