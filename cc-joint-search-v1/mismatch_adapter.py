"""Explicit experimental capacities, installed before the first decision."""
import copy,json
from pathlib import Path
from core80 import build as original_build
from residual_capacity import install

def build(method,config,workload,response='medium'):
    cfg=json.loads(Path(config).read_text(encoding='utf8'))
    topology=cfg['cluster']['ngpus_per_node']
    caps=copy.deepcopy(cfg['cluster']['capacity_scenarios']['main'])
    c=original_build(method,response=response,workload_path=workload,config_path=config,topology=topology)
    assert c.current_time==0 and not c.policy.current_resources
    # JointCluster has no other CPU/RAM allocator: _nodes() and execution
    # commits both read this same map. No decisions have yet been made.
    c.policy.node_capacities=caps
    install(c)
    original=c.policy._scheduler
    def scheduler(nodes,specs):
        s=original(nodes,specs)
        for n in s.nodes:
            assert n.gpus==topology[n.gpu_type]
            assert (n.cpu_cores,n.ram_gb)==(caps[n.gpu_type]['cpu_cores'],caps[n.gpu_type]['ram_gb'])
        return s
    c.policy._scheduler=scheduler
    c.experimental_capacity_receipt=copy.deepcopy(caps)
    return c
