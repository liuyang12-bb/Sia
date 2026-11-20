"""Offline arithmetic only: validate captured vectors, never call a solver."""
import gzip,json,hashlib
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_runs/sia_synergy/cc_core80_8h_20261030_run3/diagnostics_full_five_boundaries'

def decode(v):
    return float(v['nonfinite']) if set(v)=={'nonfinite'} else v

def main():
    source=OUT/'solver_full_diagnostics.json.gz'
    d=json.load(gzip.open(source,'rt',encoding='utf8'),object_hook=decode)
    model=d['model'];cost=np.asarray(model['c'],float);rows=[]
    for c in model['constraints']:
        a=c['A']
        if isinstance(a,dict):
            assert a['storage']=='COO'
            a=coo_matrix((a['data'],(a['row'],a['col'])),shape=tuple(a['shape'])).tocsr()
        else:a=np.atleast_2d(np.asarray(a,float))
        rows.append((a,np.broadcast_to(c['lb'],a.shape[0]),np.broadcast_to(c['ub'],a.shape[0])))
    results={}
    for name in ('original_x','retry_x'):
        x=d.get(name)
        if x is None:results[name]=None;continue
        x=np.asarray(x,float);entry={}
        for label,v in [('raw',x),('rounded',np.rint(x))]:
            violations=[];max_delta=0.
            for k,(a,lo,hi) in enumerate(rows):
                val=np.asarray(a@v).reshape(-1)
                narrow=np.isfinite(lo)&np.isfinite(hi)&(hi-lo<1e-6)
                tolerance=np.where(narrow,1e-10,1e-7)
                delta=np.maximum(lo-val,val-hi);max_delta=max(max_delta,float(delta.max()))
                for i in np.flatnonzero(delta>tolerance):
                    violations.append(dict(constraint=k,row=int(i),value=float(val[i]),lower=float(lo[i]),upper=float(hi[i]),excess=float(delta[i]),allowed=float(tolerance[i])))
            integral=float(np.max(abs(v-np.rint(v))));domain=max(float(np.max(-v)),float(np.max(v-1)),0.)
            entry[label]=dict(max_integrality_error=integral,domain_excess=domain,max_constraint_excess=max_delta,
                              original_rows_valid=not violations,strict_row_violation_count=len(violations),violations=violations[:30],objective=float(cost@v),
                              source_canonical_binary_guard_pass=not violations and domain==0 and integral==0)
        results[name]=entry
    result=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest().upper(),columns=len(cost),constraint_shapes=[list(a.shape) for a,_,_ in rows],
                original_status=d['original_status'],retry_status=d['retry_status'],validation=results,
                scope='No solve/no training. Original model rows and existing canonical tolerances reused only for diagnostic arithmetic; no proof of exact global optimality or a performance fix.')
    with (OUT/'independent_model_check.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
