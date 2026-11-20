"""Six isolated ablations; only restrict existing auxiliary freedom."""
from cc_factory import construct
METHODS=('SIA_P','SIA_AUX','CC_P','CC_CPU','CC_MEM','CC_FULL')
CAPACITIES={'R00':(12.,250.),'R10':(6.,250.),'R01':(12.,187.5),'R11':(6.,187.5)}
ORIGINAL_CPU=(1.,2.,3.,4.,6.,8.,9.,12.,24.)
ORIGINAL_RAM=(20.83,62.5,125.,187.5,250.)
def build(method,scenario,response='medium',workload_path=None):
 if method not in METHODS or scenario not in CAPACITIES:raise ValueError('UNREGISTERED_CELL')
 base={'SIA_P':'SIA_P','SIA_AUX':'SIA_ECT_AUX','CC_P':'JOINT_CC_P'}.get(method,'JOINT_CC_TAIL')
 c=construct(base,response,workload_path)
 if method=='CC_P':
  from joint_prototype.tail_recovery import install
  install(c)
 cpu,ram=CAPACITIES[scenario]
 c.policy.node_capacities={t:dict(cpu_cores=cpu,ram_gb=ram) for t in c.ngpus_per_node}
 if method=='CC_CPU':c.policy.aux_ram_points=(ram/2.,)
 if method=='CC_MEM':c.policy.aux_cpu_points=(cpu/2.,)
 return c
