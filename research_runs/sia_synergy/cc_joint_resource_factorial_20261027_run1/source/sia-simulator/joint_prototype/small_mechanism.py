"""Small mechanism probe: full candidate evidence and fixed official GPU AUX.

No changes to historical Sia, L2, response, capacities or profile sources.
"""
import copy,math,time
from dataclasses import asdict,replace
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from sia import SiaPolicy,ZERO_ALLOC_GAIN
import simulator_config as sc
from .full160 import SiaDecisionR0Policy,FullJointPolicy
from .neutral_liveness import FactorScheduler
from .joint_scheduler import PolicyResult

def scheduler(policy,nodes,specs):
 return FactorScheduler(nodes,specs,policy.oracle,policy.aux_cpu_points,policy.aux_ram_points,
  allocation_bonus=0,verify_exact=False,minimum_cpu_per_gpu=policy.minimum_cpu_per_gpu,
  minimum_ram_gb_per_gpu=policy.minimum_ram_gb_per_gpu,candidate_policy='b3')

def utility(policy,jobs,specs,columns):
 by={j.job_id:j for j in specs};out=[]
 for c in columns:
  f=policy._restart_factor(c.job_id,jobs[c.job_id],c.config.config_id,by[c.job_id].current_config_id)
  v=max(c.query.effective_goodput*f,ZERO_ALLOC_GAIN)
  u=(-1 if policy.p_fairness<0 else 1)*(min(round(v**policy.p_fairness,2),sc.sia_goodput_clip_val)-(policy.lambda_n if policy.lambda_n else -1))
  out.append(replace(c,scheduling_utility=u))
 return out

def fixed_set_solve(s,columns,required):
 """Same fairness utility, equality admission; never drop official intent."""
 required=set(required);cols=[c for c in columns if c.job_id in required]
 if not required:
  audit=s.audit([])
  if not audit['valid']:raise RuntimeError('EMPTY_INTENT_LOCK_CONFLICT')
  return PolicyResult('SIA_AUX','EMPTY_INTENT',{},sorted(s.job_map),0,0,0,audit=audit)
 if not cols or any(not any(c.job_id==j for c in cols) for j in required):raise RuntimeError('MISSING_FIXED_INTENT_COLUMNS')
 m,upper=s._constraint_matrix(cols,('gpu','cpu','ram'));lower=np.full(len(upper),-np.inf)
 for i,j in enumerate(s.jobs):lower[i]=upper[i]=int(j.job_id in required)
 tick=time.perf_counter();res=milp(c=-np.array([c.scheduling_utility for c in cols]),integrality=np.ones(len(cols)),bounds=Bounds(0,1),constraints=LinearConstraint(m,lower,upper),options={'time_limit':30.,'mip_rel_gap':0.})
 if not res.success or res.status!=0 or res.x is None:raise RuntimeError(f'AUX_SOLVER_UNDECIDED:{res.status}:{res.message}')
 if np.any(abs(res.x-np.round(res.x))>1e-6):raise RuntimeError('NONINTEGRAL_AUX')
 chosen=[c for c,x in zip(cols,res.x) if x>.5];audit=s.audit(chosen)
 if not audit['valid'] or {c.job_id for c in chosen}!=required:raise RuntimeError(f'INVALID_FIXED_INTENT:{audit}')
 audit.update(solver_status=int(res.status),solver_message=str(res.message))
 return PolicyResult('SIA_AUX','OPTIMAL_FIXED_OFFICIAL_INTENT',{c.job_id:c for c in chosen},sorted(set(s.job_map)-required),sum(c.scheduling_utility for c in chosen),sum(c.query.effective_goodput for c in chosen),time.perf_counter()-tick,audit=audit)

class EvidenceMixin:
 def _job_specs(self,jobs,base):
  specs,provenance=super()._job_specs(jobs,base)
  self.probe_specs=specs;self.probe_provenance=copy.deepcopy(provenance)
  return specs,provenance
 def optimize(self,jobs,nodes,base):
  previous=copy.deepcopy(self.current_resources)
  live=getattr(self,'liveness',None)
  before=copy.deepcopy(dict(ready=live.ready,services=live.services,productive=self.productive)) if live else None
  result=super().optimize(jobs,nodes,base)
  s=scheduler(self,self._nodes(nodes),self.probe_specs)
  allcols=utility(self,jobs,self.probe_specs,s.build_joint_columns())
  props=utility(self,jobs,self.probe_specs,s.build_proportional_columns())
  # Count physical/floor exclusions without exposing them as legal candidates.
  s.enforce_floor=False; nofloor=s.build_joint_columns();s.enforce_floor=True
  state={j:dict(age=v.age,num_restarts=v.num_restarts,used_gpu_seconds=v.used_gpu_seconds,wasted_gpu_seconds=v.wasted_gpu_seconds) for j,v in jobs.items()}
  self.last_decision['probe']=dict(specs=[asdict(j) for j in self.probe_specs],provenance=self.probe_provenance,job_state=state,
   previous_resources=previous,l2_before=before,l2_after=copy.deepcopy(dict(ready=live.ready,services=live.services)) if live else None,
   joint_columns=[asdict(c) for c in allcols],proportional_columns=[asdict(c) for c in props],
   floor_rejected=[asdict(c) for c in nofloor if c.column_id not in {a.column_id for a in allcols}],
   restart_factors={j.job_id:{c.config_id:self._restart_factor(j.job_id,jobs[j.job_id],c.config_id,j.current_config_id) for c in j.configs} for j in self.probe_specs},
   contract=dict(p=self.p_fairness,lambda_n=self.lambda_n,allocation_bonus=0,progress_floor=True,response=self.oracle.response_exponent))
  return result

class ProbeSia(EvidenceMixin,SiaDecisionR0Policy):pass
class ProbeJoint(EvidenceMixin,FullJointPolicy):pass

class AuxBase(SiaDecisionR0Policy):
 def optimize(self,jobs,nodes,base):
  specs,provenance=self._job_specs(jobs,base);s=scheduler(self,self._nodes(nodes),specs);s.locked_columns()
  intent=SiaPolicy.optimize(self,jobs,nodes,base)
  prop=self._b1_result(intent,specs,s)
  fixed={j:c.config.config_id for j,c in prop.selected.items()}
  if self.canonical(self._allocations_from_result(prop,jobs),jobs)!=self.canonical(intent,jobs):raise RuntimeError('OFFICIAL_INTENT_LOCK_CONFLICT')
  cols=utility(self,jobs,specs,[c for c in s.build_joint_columns() if c.config.config_id==fixed.get(c.job_id)])
  try:result=fixed_set_solve(s,cols,fixed)
  except RuntimeError as exc:
   audit=s.audit(list(prop.selected.values()))
   if not audit['valid']:raise RuntimeError(f'NO_SAFE_AUX_FALLBACK:{audit}') from exc
   result=prop;result.status='FALLBACK_PROPORTIONAL';result.audit.update(fallback_reason=str(exc))
  if self.canonical(self._allocations_from_result(result,jobs),jobs)!=self.canonical(intent,jobs):raise RuntimeError('FIXED_GPU_INTENT_CHANGED')
  self._commit_metadata(result,provenance,fixed)
  self.last_decision.update(original_sia_allocations=copy.deepcopy(intent),strict_full_intent=True,admission='OFFICIAL_SIA_NO_L2_NO_SECOND_ADMISSION',gpu_optimizer='sia.SiaPolicy.optimize')
  return intent

class ProbeAux(EvidenceMixin,AuxBase):pass
