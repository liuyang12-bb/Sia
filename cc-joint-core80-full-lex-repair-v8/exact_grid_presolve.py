"""Eliminate provably lexicographically dominated VARIABLES, not grid points.

Same job/config/placement implies identical GPU use and startup cost. If a
column uses no more CPU/RAM and has >= final utility, it can replace the other
column in any admission set. On equal primary/startup values the lower quota
also precedes the other in CleanCore's per-job geometry rank. Thus the removed
variable cannot occur in the final canonical solution. Full columns remain
available to the normalizer, candidate contract, records and precommit audit.
Only initiated on the base resource/job rows before utility/rank bands exist.
"""
from collections import defaultdict
import numpy as np
from scipy.optimize import LinearConstraint

def indices(columns):
 groups=defaultdict(list)
 for i,c in enumerate(columns):groups[c.job_id,c.config.config_id,c.placement.node_gpus].append(i)
 keep=[]
 for group in groups.values():
  ram=sorted({columns[i].query.ram_gb_per_gpu for i in group});rank={r:i+1 for i,r in enumerate(ram)};tree=[None]*(len(ram)+1)
  for i in sorted(group,key=lambda i:(columns[i].query.cpu_per_gpu,columns[i].query.ram_gb_per_gpu,-columns[i].scheduling_utility,i)):
   c=columns[i];cpu=c.query.cpu_per_gpu;m=c.query.ram_gb_per_gpu;score=c.scheduling_utility;k=rank[m];best=None
   while k:
    if tree[k] is not None and (best is None or tree[k]>best):best=tree[k]
    k-=k&-k
   dominated=best is not None and (best[0]>score or best[0]==score and (-best[1]<cpu or -best[2]<m))
   if not dominated:keep.append(i)
   entry=(score,-cpu,-m);k=rank[m]
   while k<len(tree):
    if tree[k] is None or entry>tree[k]:tree[k]=entry
    k+=k&-k
 return np.asarray(sorted(keep),dtype=int)

def reduced(self,cost,columns,constraints):
 cache=getattr(self,'_repair_grid_domain',None)
 if cache is None or cache[0] is not columns:
  # Do not apply to arbitrary captured/helper MIPs or an already constrained
  # primary/lex state. The actual Core80 controller supplies the scope token.
  if (not getattr(self,'_repair_scope',False) or len(constraints)!=1 or not columns
      or not all(hasattr(c,'config') and hasattr(c,'query') for c in columns)):
   return cost,columns,constraints,None
  kept=indices(columns);zero={}
  if getattr(self,'factors',(0,))[0]==1:
   by=defaultdict(list)
   for i,c in enumerate(columns):by[c.job_id].append(i)
   for name,own in by.items():
    j=self._repair_job_specs[name]
    if all(columns[i].scheduling_utility==0. and not(j.current_config_id==columns[i].config.config_id and j.current_placement==columns[i].placement.node_gpus) and self._delay(name,columns[i])>1e-7 for i in own):zero[name]=set(own)
  cache=(columns,kept,[columns[i] for i in kept],zero);self._repair_grid_domain=cache
 kept=cache[1]
 # Existing zero-utility/positive-startup theorem, now applied to solver
 # variables before solving rather than after the secondary optimum. Removing
 # such an OPTIONAL job preserves primary value, lowers secondary startup,
 # and cannot remove a canonical optimum. Every required FIFO/service/lock
 # job is retained for this particular solve. No candidate/admission change.
 lower=np.broadcast_to(constraints[0].lb,constraints[0].A.shape[0]);required={j for i,j in enumerate(self._repair_job_order) if lower[i]==1.}
 if required and cache[3]:
  removed=set().union(*(v for j,v in cache[3].items() if j not in required)) if any(j not in required for j in cache[3]) else set()
  if removed:
   kept=np.asarray([i for i in kept if i not in removed],dtype=int)
   self._repair_zero_variables_removed=max(getattr(self,'_repair_zero_variables_removed',0),len(removed))
 if len(kept)==len(columns):return cost,columns,constraints,None
 self._repair_variables_removed=max(getattr(self,'_repair_variables_removed',0),len(columns)-len(kept))
 return np.asarray(cost)[kept],[columns[i] for i in kept],[LinearConstraint(c.A[:,kept],c.lb,c.ub) for c in constraints],kept
