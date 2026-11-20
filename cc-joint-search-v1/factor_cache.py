"""Exact full-universe normalizer; materialize only the declared quota factor."""
from dataclasses import replace
from time import perf_counter

from recovery_precheck import potential_recovery_jobs

def full_max_rho(s, j, c, p, require_witness=False):
    from joint_prototype.joint_scheduler import _proportional_point

    proportional = _proportional_point(p, s.node_map)
    floor_cpu, floor_ram = s.allocation_floor(proportional)
    cpu, ram = s.resource_points(j, c, p)
    cpu_points = tuple(sorted(x for x in set(cpu + (floor_cpu,)) if x + 1e-9 >= s.minimum_cpu_per_gpu))
    ram_points = tuple(sorted(x for x in set(ram + (floor_ram,)) if x + 1e-9 >= s.minimum_ram_gb_per_gpu))
    nodes = tuple((n.node_id, n.gpus, n.cpu_cores, n.ram_gb) for n in s.nodes)
    key = (j.template_file, c.config_id, c.cpu_demand_scale, c.memory_demand_scale,
           p.node_gpus, cpu_points, ram_points, nodes, s.oracle.response_exponent)
    cache = getattr(s.oracle, '_repair_full_grid_rho', None)
    if cache is None:
        cache = s.oracle._repair_full_grid_rho = {}
    if key not in cache:
        best, best_point, count = None, None, 0
        for cpu in cpu_points:
            for ram in ram_points:
                if not s._fits_column(s._node_usage(p, cpu, ram)):
                    continue
                response = s.oracle.query(j, c, cpu, ram, proportional).rho
                count += 1
                if best is None or response > best:
                    best, best_point = response, (cpu, ram)
        cache[key] = (best, count, best_point)
    best, count, best_point = cache[key]
    if not require_witness:
        return best, count, None

    witnesses = getattr(s.oracle, '_repair_witness_rho', None)
    if witnesses is None:
        witnesses = s.oracle._repair_witness_rho = {}
    if key not in witnesses:
        witness = best
        if best_point is not None:
            # Original score evidence uses :g-formatted quota labels. The
            # slowest response among aliases gives a safe incumbent witness.
            cpu_label, ram_label = (f'{value:g}' for value in best_point)
            cpu_aliases = [value for value in cpu_points if f'{value:g}' == cpu_label]
            ram_aliases = [value for value in ram_points if f'{value:g}' == ram_label]
            if len(cpu_aliases) > 1 or len(ram_aliases) > 1:
                witness = min(
                    s.oracle.query(j, c, cpu, ram, proportional).rho
                    for cpu in cpu_aliases for ram in ram_aliases
                    if s._fits_column(s._node_usage(p, cpu, ram))
                )
        witnesses[key] = witness
    return best, count, witnesses[key]

def prepare_factor(self,jobs,specs,s,recovery_jobs):
 from joint_prototype.completion_time import predict_job,completion_cost
 from joint_prototype.completion_continuity import completion_priority
 from joint_prototype.joint_scheduler import enumerate_placements
 if s.enforce_floor:raise RuntimeError('UNEXPECTED_PROGRESS_FLOOR_IN_REPAIR_DOMAIN')
 pred={(j.job_id,c.config_id):predict_job(self.completion_jobs[j.job_id],c) for j in specs for c in j.configs}
 self.liveness.profile_sources={j:{c:v['goodput_source'] for c,v in r.items()} for j,r in self.probe_provenance.items()}
 prop=s.build_proportional_columns();keys={(c.job_id,c.config.config_id,c.placement.node_gpus,c.query.cpu_per_gpu,c.query.ram_gb_per_gpu) for c in prop}
 remaining={j.job_id:max(0.,self.declared_work[j.job_id]['total_progress']-self.completion_jobs[j.job_id].progress) for j in specs}
 wait={};proposals={}
 for j in specs:
  estimates=[completion_cost(remaining[j.job_id],pred[c.job_id,c.config.config_id]['raw_rate'],c.query.rho,self.completion_delay_costs[j.job_id][c.config.gpu_type]) for c in prop if c.job_id==j.job_id and pred[c.job_id,c.config.config_id]['raw_rate'] is not None]
  wait[j.job_id]=300.+min(estimates) if estimates else None
  held=self.liveness.services.get(j.job_id)
  if (held is not None and self.productive.get(j.job_id,0.)-held['productive_base']<self.liveness.quantum-1e-9) or s.lock_active(j):proposals[j.job_id]=j.current_config_id
  else:
   feasible={c.config.config_id for c in prop if c.job_id==j.job_id};cfg=[c for c in j.configs if c.config_id in feasible]
   if cfg:proposals[j.job_id]=min(cfg,key=lambda c:(-c.sia_goodput*self._restart_factor(j.job_id,jobs[j.job_id],c.config_id,j.current_config_id),c.config_id)).config_id
 factor=self.liveness.resource_factor;scale=0.;max_counts=0;known=set()
 bounds={name:{} for name in recovery_jobs}
 # For fixed job/config/placement, delay and work are quota-independent and
 # 1/(delay+work/(raw*rho)) is monotone in rho. Exactly the same maximum.
 for j in specs:
  job=self.completion_jobs[j.job_id]
  for c in j.configs:
   pr=pred[j.job_id,c.config_id]
   if pr['raw_rate'] is None or wait[j.job_id] is None:continue
   placements=enumerate_placements(c,s.nodes) if c.legal else []
   if s.lock_active(j):placements=[p for p in placements if c.config_id==j.current_config_id and p.node_gpus==j.current_placement]
   for p in placements:
    same=j.current_config_id==c.config_id and j.current_placement==p.node_gpus
    if s.lock_active(j):rho=s.locked_column(j).query.rho;count=1;witness=None
    else:rho,count,witness=full_max_rho(s,j,c,p,require_witness=same and j.job_id in bounds)
    if rho is None:continue
    known.add((j.job_id,c.config_id))
    delay=float(job.rescale_time) if same else self.completion_delay_costs[j.job_id][c.gpu_type]
    scale=max(scale,completion_priority(remaining[j.job_id],pr['raw_rate'],rho,delay));max_counts+=count
    if j.job_id in bounds:
     optimistic_eta=completion_cost(remaining[j.job_id],pr['raw_rate'],rho,delay)
     witness_eta=optimistic_eta if witness==rho else (
         completion_cost(remaining[j.job_id],pr['raw_rate'],witness,delay)
         if witness is not None else None)
     bounds[j.job_id][(c.config_id,p.node_gpus)]=(optimistic_eta,witness_eta)
 scale=scale or 1.;provider=getattr(s,'resource_point_provider',None)
 def points(current,j,c,p):
  from joint_prototype.joint_scheduler import _proportional_point
  floor=current.allocation_floor(_proportional_point(p,current.node_map));a,b=provider(current,j,c,p) if provider else (current.aux_cpu_points,current.aux_ram_points)
  if pred[j.job_id,c.config_id]['raw_rate'] is None or wait[j.job_id] is None:return (floor[0],),(floor[1],)
  if factor in ('PROP','MEM'):a=tuple(x for x in a if abs(x-floor[0])<1e-8)
  if factor in ('PROP','CPU'):b=tuple(x for x in b if abs(x-floor[1])<1e-8)
  return a,b
 return pred, keys, remaining, wait, scale, known, max_counts, points, proposals, bounds


def score_factor(self,s,columns,pred,keys,remaining,wait,scale):
 from joint_prototype.completion_time import completion_cost
 from joint_prototype.completion_continuity import completion_priority
 by=s.job_map;out=[];evidence=[]
 for col in columns:
  j=by[col.job_id];pr=pred[j.job_id,col.config.config_id];job=self.completion_jobs[j.job_id]
  same=j.current_config_id==col.config.config_id and j.current_placement==col.placement.node_gpus
  delay=float(job.rescale_time) if same else self.completion_delay_costs[j.job_id][col.config.gpu_type]
  if pr['raw_rate'] is None or wait[j.job_id] is None:
   if (j.job_id,col.config.config_id,col.placement.node_gpus,col.query.cpu_per_gpu,col.query.ram_gb_per_gpu) not in keys and not s.lock_active(j):continue
   eta=None;priority=0.
  else:eta=completion_cost(remaining[j.job_id],pr['raw_rate'],col.query.rho,delay);priority=completion_priority(remaining[j.job_id],pr['raw_rate'],col.query.rho,delay)
  value=priority/scale;out.append(replace(col,scheduling_utility=value))
  evidence.append(dict(job=j.job_id,column_id=col.column_id,raw_rate=pr['raw_rate'],rho=col.query.rho,delay=delay,remaining_work=remaining[j.job_id],estimated_completion=eta,
   wait_cost=None,utility=value,prediction_source=pr['source'],atomic_bsz=pr['atomic_bsz'],accum_steps=pr['accum_steps'],historical_wait_reference_ignored=wait[j.job_id],completion_priority=priority,common_numerical_scale=scale))
 return out,evidence


def fast_factor(self,jobs,specs,s):
 if self.clean_group=='A':return self._repair_literal_factor(jobs,specs,s)
 mode=getattr(self,'_search_mode',None)
 potential=potential_recovery_jobs(self,s) if mode is not None else ()
 use_precheck=bool(potential) and self._recovery_precheck
 prepared=prepare_factor(self,jobs,specs,s,potential if use_precheck else ())
 pred,keys,remaining,wait,scale,known,max_counts,points,proposals,bounds=prepared
 def score(columns):
  return score_factor(self,s,columns,pred,keys,remaining,wait,scale)
 search=None
 if mode is not None:
  from lazy_search import LazySearch
  search=LazySearch(self,s,points,score,potential)
  s.lazy_search=search
  if use_precheck:
   search.precheck(self,bounds)
 if search is not None and not search.recovery_eager:
  out,evidence=search.initial()
 else:
  started=perf_counter()
  provider=getattr(s,'resource_point_provider',None)
  s.resource_point_provider=points
  try:columns=s.build_joint_columns()
  finally:
   if provider is None:del s.resource_point_provider
   else:s.resource_point_provider=provider
  out,evidence=score(columns)
  if search is not None:
   search.stats["initial_columns"]=len(out)
   search.stats["initial_generation_seconds"]=perf_counter()-started
 self.completion_scores=evidence;self.continuity_scores=evidence;self.exchange_scores=[];self.exchange_columns=out
 s.current_known_configs=known;s.current_completion_scores={x['column_id']:x for x in evidence}
 self._repair_factor_certificate=dict(full_universe_scale=scale,full_grid_points_considered=max_counts,unknown_proportional_columns=sum(x['raw_rate'] is None for x in evidence),materialized_columns=len(out),factor=self.liveness.resource_factor,normalizer='EXACT_FULL_GRID_MAX_MONOTONE_ETA')
 return out,proposals
