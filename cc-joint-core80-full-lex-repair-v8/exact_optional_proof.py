"""Exact certificate for redundant lexicographic presence queries.

TRUSTED CONTEXT: CleanCore's fixed primary optimum, then minimum actual-startup
secondary optimum; required set nonempty. For a nonrequired job with every
candidate utility exactly0 and strictly positive added startup, selecting it
cannot occur at that secondary optimum: remove its one binary column. Every
required/lock and other job choice stays; resource upper bounds only relax,
primary is identical, and startup strictly decreases, contradicting optimality.
Previously fixed other-job lexicographic masks do not invalidate removal.
Thus its maximum-presence query is exactly0. Reuse the preceding independently
validated feasible point; all later job/rank ties are still executed unchanged.

This does NOT remove any column, change scores, choose a new job or impose new
admission. Unknown jobs still get exactly the historical required FIFO service.
Disable the opt-in to compare literal solver calls on identical input states.
"""
import numpy as np
from scipy.optimize import OptimizeResult

class ExactOptionalProofMixin:
    enable_exact_optional_proof=True
    def _decide(self,*args):
        self._proof_context=None;self._proof_last_x=None;self._proof_last_columns=None
        self._exact_optional_records=[]
        try:
            result=super()._decide(*args)
            result.audit['exact_zero_optional_certificates']=list(self._exact_optional_records)
            result.audit['exact_zero_optional_saved_solver_calls']=len(self._exact_optional_records)
            return result
        finally:
            # Ephemeral solver vectors must never enter persistent service
            # history, a later state, a fallback, or a trial controller copy.
            for k in ('_proof_context','_proof_last_x','_proof_last_columns','_exact_optional_records'):
                self.__dict__.pop(k,None)

    def _secondary_tie(self,s,columns,required,constraints,call,chosen):
        selected=super()._secondary_tie(s,columns,required,constraints,call,chosen)
        if (getattr(self,'enable_exact_optional_proof',False) and required and
                getattr(self,'factors',(0,))[0]==1 and len(constraints)>=3):
            startup=np.asarray(constraints[-1].A).reshape(-1)
            weights=np.asarray(constraints[1].A).reshape(-1)
            if (len(startup)==len(columns) and len(weights)==len(columns) and
                    np.array_equal(weights,np.asarray([c.scheduling_utility for c in columns]))):
                self._proof_context=(columns,constraints,set(required),startup)
        return selected

    def _milp(self,cost,columns,constraints):
        from joint_prototype.completion_continuity import canonical_integral_solution
        ctx=getattr(self,'_proof_context',None);values=np.asarray(cost);x=getattr(self,'_proof_last_x',None)
        if (ctx is not None and ctx[0] is columns and ctx[1] is constraints and
                getattr(self,'_proof_last_columns',None) is columns and x is not None):
            indices=np.flatnonzero(values)
            if len(indices) and np.all(values[indices]==-1.):
                names={columns[i].job_id for i in indices}
                if len(names)==1:
                    name=next(iter(names));own=[i for i,c in enumerate(columns) if c.job_id==name]
                    if (name not in ctx[2] and own==indices.tolist() and
                            all(columns[i].scheduling_utility==0. for i in own) and
                            np.all(ctx[3][own]>1e-7) and np.all(x[own]==0.)):
                        # This is an actual feasible allocation at the CURRENT
                        # primary/secondary/earlier-rank rows, not stale geometry.
                        valid=canonical_integral_solution(x,constraints)
                        record=dict(job=name,scope='CURRENT_MODEL_PRIMARY_AND_SECONDARY_OPTIMA',
                            maximum_presence=0.,minimum_added_startup=float(min(ctx[3][own])),
                            required_jobs=sorted(ctx[2]),columns_unchanged=len(columns),
                            certificate='REMOVE_ZERO_UTILITY_POSITIVE_STARTUP_OPTIONAL_COLUMN',
                            all_current_rows_revalidated=True)
                        self._exact_optional_records.append(record)
                        self._proof_last_x=valid.copy()
                        return OptimizeResult(x=valid,status=0,success=True,fun=0.,
                            message='CERTIFIED_ZERO_OPTIONAL: same primary/secondary optimum, current feasible vector revalidated')
        result=super()._milp(cost,columns,constraints)
        if result.status==0 and result.success and result.x is not None:
            self._proof_last_x=np.asarray(result.x).copy();self._proof_last_columns=columns
        else:self._proof_last_x=None;self._proof_last_columns=None
        return result
