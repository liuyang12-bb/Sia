"""Opt-in exact storage adapters; no candidate, objective or solver-option edits."""
import contextlib,dataclasses,hashlib,inspect,math,textwrap
import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint
from unittest.mock import patch

def matrix_bytes(value):
    if sparse.issparse(value):
        a=value.tocsc(copy=True);a.sort_indices()
        return b'CSC'+np.asarray(a.shape,dtype=np.int64).tobytes()+a.indptr.tobytes()+a.indices.tobytes()+a.data.tobytes()
    return np.asarray(value,dtype=float).tobytes()

def matrix_payload(value):
    if not sparse.issparse(value):return np.asarray(value,dtype=float).tolist()
    a=value.tocoo()
    return dict(storage='COO',shape=a.shape,row=a.row.tolist(),col=a.col.tolist(),data=a.data.tolist())

def sparse_constraints(self,columns,dimensions):
    # Exactly the historical job-row and node/dimension order, now O(nonzeros).
    key=(id(columns),len(columns),tuple(dimensions))
    cache=getattr(self,'_storage_constraint_cache',None)
    if cache and cache[0]==key and cache[1] is columns:return cache[2],cache[3].copy()
    indices={j.job_id:i for i,j in enumerate(self.jobs)}
    upper=[1.]*len(self.jobs);node_rows={}
    for n in self.nodes:
        for dim,cap in [('gpu',n.gpus),('cpu',n.cpu_cores),('ram',n.ram_gb)]:
            if dim in dimensions:node_rows[n.node_id,dim]=len(upper);upper.append(float(cap))
    rows=[];cols=[];data=[]
    for i,c in enumerate(columns):
        rows.append(indices[c.job_id]);cols.append(i);data.append(1.)
        for node,gpu,cpu,ram in c.node_usage:
            for dim,val in [('gpu',gpu),('cpu',cpu),('ram',ram)]:
                if dim in dimensions and val:
                    rows.append(node_rows[node,dim]);cols.append(i);data.append(val)
    matrix=sparse.csc_matrix((data,(rows,cols)),shape=(len(upper),len(columns)),dtype=float)
    upper=np.asarray(upper,dtype=float)
    self._storage_constraint_cache=(key,columns,matrix,upper.copy())
    return matrix,upper.copy()

def sparse_condition(constraints):
    result=[];exponents=[]
    for c in constraints:
        if not sparse.issparse(c.A):
            from joint_prototype.completion_continuity import _core80_original_condition
            a,p=_core80_original_condition([c]);result.extend(a);exponents.extend(p);continue
        matrix=c.A.tocsr();lo=np.broadcast_to(c.lb,(matrix.shape[0],)).copy();hi=np.broadcast_to(c.ub,(matrix.shape[0],)).copy()
        powers=np.zeros(matrix.shape[0],dtype=int)
        for row in range(matrix.shape[0]):
            width=hi[row]-lo[row]
            if np.isfinite(lo[row]) and np.isfinite(hi[row]) and 0<width<1e-6:powers[row]=math.ceil(-math.log2(width))
        a=matrix.copy();a.data=np.ldexp(a.data,np.repeat(powers,np.diff(a.indptr)))
        lower,upper=np.ldexp(lo,powers),np.ldexp(hi,powers)
        if not np.array_equal(np.ldexp(a.data,-np.repeat(powers,np.diff(a.indptr))),matrix.data):raise RuntimeError('NONREVERSIBLE_SPARSE_ROWS')
        if not np.array_equal(np.ldexp(lower,-powers),lo) or not np.array_equal(np.ldexp(upper,-powers),hi):raise RuntimeError('NONREVERSIBLE_SPARSE_BOUNDS')
        result.append(LinearConstraint(a.tocsc(),lower,upper));exponents.append(powers.tolist())
    return result,exponents

def slots_copy(cls):
    fields=[]
    for f in dataclasses.fields(cls):
        opts={}
        if f.default is not dataclasses.MISSING:opts['default']=f.default
        if f.default_factory is not dataclasses.MISSING:opts['default_factory']=f.default_factory
        fields.append((f.name,f.type,dataclasses.field(**opts)))
    namespace={k:v for k,v in cls.__dict__.items() if callable(v) and not k.startswith('__')}
    namespace['__module__']=__name__
    return dataclasses.make_dataclass('Compact'+cls.__name__,fields,namespace=namespace,frozen=True,slots=True)

@contextlib.contextmanager
def storage_context(oracle):
    from joint_prototype import joint_profile as jp,joint_scheduler as js,completion_continuity as ct,bundle_adaptation as ba
    from joint_prototype import adaptive_service
    # The Bundle numerical guard is byte-for-byte its existing source except
    # sparse serialization and an explicit super class (closure-free binding).
    src=textwrap.dedent(inspect.getsource(ba.BundleAdmission._milp))
    src=src.replace('super()._milp','super(BundleAdmission,self)._milp')
    src=src.replace('np.asarray(v.A,dtype=float).tolist()','matrix_payload(v.A)')
    src=src.replace('np.asarray(a,dtype=float).tobytes()','matrix_bytes(a)')
    env=dict(vars(ba));env.update(matrix_payload=matrix_payload,matrix_bytes=matrix_bytes)
    exec(compile(src,'<core80-equivalent-bundle-storage>','exec'),env)
    original_query=jp.JointProfileOracle.query
    compact_query=slots_copy(jp.ProfileQuery);compact_col=slots_copy(js.JointColumn)
    def cached(self,job,config,cpu,ram,prop):
        # Validate current ownership/legal state even on a response cache hit.
        if config not in job.configs or not config.legal or cpu<=0 or ram<=0 or config.gpus<=0 or config.num_nodes<=0 or config.sia_goodput<=0:
            return original_query(self,job,config,cpu,ram,prop)
        key=(job.template_file,config.config_id,config.cpu_demand_scale,config.memory_demand_scale,float(cpu),float(ram),tuple(prop),self.normalization_mode,self.fixed_reference,self.response_exponent)
        cache=getattr(self,'_core80_response_cache',None)
        if cache is None:cache={};self._core80_response_cache=cache
        q=cache.get(key)
        if q is None:q=original_query(self,job,config,cpu,ram,prop);cache[key]=q
        if q.job_id==job.job_id and q.raw_sia_goodput==config.sia_goodput:return q
        return dataclasses.replace(q,job_id=job.job_id,raw_sia_goodput=config.sia_goodput,effective_goodput=config.sia_goodput*q.rho)
    with contextlib.ExitStack() as s:
        s.enter_context(patch.object(jp,'ProfileQuery',compact_query));s.enter_context(patch.object(js,'JointColumn',compact_col))
        s.enter_context(patch.object(jp.JointProfileOracle,'query',cached))
        s.enter_context(patch.object(js.JointScheduler,'_constraint_matrix',sparse_constraints))
        s.enter_context(patch.object(ba.BundleAdmission,'_milp',env['_milp']))
        s.enter_context(patch.object(ct,'_core80_original_condition',ct.condition_constraint_bands,create=True))
        s.enter_context(patch.object(ct,'condition_constraint_bands',sparse_condition))
        yield
