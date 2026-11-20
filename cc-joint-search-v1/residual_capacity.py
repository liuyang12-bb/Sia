"""Opt-in deterministic execution-floor residuals, shared by FO and FULL.

One pass; no learned thresholds, recursive closure, future arrivals or job IDs.
The original breakpoint domain is a subset. Final audit uses this same provider.
"""
import math,types
from breakpoint_study import resource_points as original_points

VERSION='EXECUTION_FLOOR_RESIDUALS_V1'

def resource_points(scheduler,job,config,placement):
    base=original_points(scheduler,job,config,placement)
    floors=(scheduler.minimum_cpu_per_gpu,scheduler.minimum_ram_gb_per_gpu)
    enabled=(getattr(scheduler,'breakpoint_cpu_enabled',True),getattr(scheduler,'breakpoint_ram_enabled',True))
    out=[]
    for dim in range(2):
        values=set(base[dim])
        if enabled[dim]:
            # Every possible integral co-resident GPU count at the EXISTING
            # execution floor; no need to guess which job will be selected.
            # Uniform per-GPU quota across multiple nodes uses the tightest bound.
            maximum=min(scheduler.node_map[n].gpus-g for n,g in placement.node_gpus)
            for k in range(maximum+1):
                value=min(((scheduler.node_map[n].cpu_cores if dim==0 else scheduler.node_map[n].ram_gb)-k*floors[dim])/g for n,g in placement.node_gpus)
                if math.isfinite(value) and value>0 and value>=floors[dim]:values.add(value)
        out.append(tuple(sorted(values)))
    return tuple(out)

def install(cluster):
    original=cluster.policy._scheduler
    def make(self,nodes,specs):
        s=original(nodes,specs)
        s.resource_point_provider=resource_points
        s.breakpoint_candidate_version=VERSION
        return s
    cluster.policy._scheduler=types.MethodType(make,cluster.policy)
    return cluster
