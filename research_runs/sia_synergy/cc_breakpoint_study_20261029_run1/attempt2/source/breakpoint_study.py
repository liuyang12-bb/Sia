"""Candidate-only opt-in: inverse source knots and feasible peer complements.

No new response values, no target/arrival/work modification. Registry peers
are the full fixed offline application domain, never a future job queue.
"""
import math,types
import numpy as np
from resource_study import build as original_build
from joint_prototype.joint_profile import CPU_COORDS,RAM_COORDS

VERSION='SOURCE_SLOPE_KNOTS_PLUS_ONE_PEER_COMPLEMENTS_V1'
def slope_knots(matrix,axis,dimension):
    a=np.asarray(matrix,dtype=float)
    if dimension==1:a=a.T
    slopes=np.diff(a,axis=0)/np.diff(axis)[:,None]
    return tuple(float(axis[i]) for i in range(len(axis)) if i in (0,len(axis)-1)
                 or np.any(np.abs(slopes[i]-slopes[i-1])>1e-12))

def resource_points(scheduler,job,config,placement):
    oracle=scheduler.oracle
    cpu_enabled=getattr(scheduler,'breakpoint_cpu_enabled',True)
    ram_enabled=getattr(scheduler,'breakpoint_ram_enabled',True)
    own=oracle.reference(job.job_id,config.config_id)
    template=oracle.template(job.template_file)
    axes=(CPU_COORDS,RAM_COORDS)
    base=(scheduler.aux_cpu_points,scheduler.aux_ram_points)
    scales=(own['phi_c'],own['phi_m'])
    key=(job.template_file,config.config_id,scales,placement.node_gpus,
         tuple((n.node_id,n.cpu_cores,n.ram_gb,n.gpus) for n in scheduler.nodes),
         cpu_enabled,ram_enabled,base)
    cache=getattr(oracle,'_breakpoint_points_cache',None)
    if cache is None:cache={};oracle._breakpoint_points_cache=cache
    if key in cache:return cache[key]
    out=[]
    for dim,enabled in enumerate((cpu_enabled,ram_enabled)):
        values=set(base[dim])
        if enabled:
            values.update(k*scales[dim] for k in slope_knots(template.matrix,axes[dim],dim))
            for node_id,g in placement.node_gpus:
                node=scheduler.node_map[node_id]
                capacity=node.cpu_cores if dim==0 else node.ram_gb
                for ref in oracle.registry.values():
                    cid=ref['key'].split('/',1)[1];typ,nn,gg=cid.split(':');gg=int(gg)
                    if typ!=config.gpu_type or int(nn)!=1 or g+gg>node.gpus:continue
                    peer=oracle.template(ref['template'])
                    scale=ref['phi_c'] if dim==0 else ref['phi_m']
                    for knot in slope_knots(peer.matrix,axes[dim],dim):
                        values.add((capacity-gg*scale*knot)/g)
        # Preserve exact floats and original points. No result-based rounding.
        minimum=scheduler.minimum_cpu_per_gpu if dim==0 else scheduler.minimum_ram_gb_per_gpu
        out.append(tuple(sorted(v for v in values if math.isfinite(v) and v>=minimum and v>0)))
    cache[key]=tuple(out)
    return cache[key]

def install(cluster,method):
    if method in ('SIA_P','CC_P'):return cluster
    old=cluster.policy._scheduler
    def make(self,nodes,specs):
        s=old(nodes,specs)
        s.breakpoint_cpu_enabled=method!='CC_MEM'
        s.breakpoint_ram_enabled=method!='CC_CPU'
        s.resource_point_provider=resource_points
        s.breakpoint_candidate_version=VERSION
        return s
    cluster.policy._scheduler=types.MethodType(make,cluster.policy)
    return cluster

def build(method,scenario,response='medium',workload_path=None):
    return install(original_build(method,scenario,response,workload_path),method)
