import unittest
import numpy as np
from types import SimpleNamespace
try:
 from joint_prototype.physical_domain import restrict_single_node_domain
except ModuleNotFoundError:
 def restrict_single_node_domain(p):return p

def policy(gpus):
 return SimpleNamespace(cluster_ordering=list(gpus),cluster_num_nodes={t:1 for t in gpus},cluster_ngpus_per_node=gpus,
    configs={t:(np.ones(4,dtype=np.uint32),np.array([1,2,4,8],dtype=np.uint32)) for t in gpus})

class PhysicalDomainTests(unittest.TestCase):
 def test_small_node_has_no_impossible_configs(self):
  p=restrict_single_node_domain(policy({'aws':2,'dgx-ext':2,'rtx':1}))
  self.assertEqual({t:list(v[1]) for t,v in p.configs.items()},{'aws':[1,2],'dgx-ext':[1,2],'rtx':[1]})
  self.assertEqual(p.total_num_configs,5)
  for i,(t,n,g) in p.idx_to_config_map.items():self.assertEqual(p.adaptive_reduce_matrix[i,p.cluster_ordering.index(t)],g)
 def test_supported_eight_gpu_domain_is_preserved(self):
  p=restrict_single_node_domain(policy({'dgx-ext':8}))
  self.assertEqual(list(p.configs['dgx-ext'][1]),[1,2,4,8])
 def test_multi_node_extension_is_not_silently_assumed(self):
  p=policy({'aws':2});p.cluster_num_nodes['aws']=2
  with self.assertRaises(ValueError):restrict_single_node_domain(p)
