"""Opt-in repair context; historical policy classes remain unchanged outside it."""
import contextlib
from unittest.mock import patch
from scale_storage import storage_context
from numeric_repair import canonical_bundle

@contextlib.contextmanager
def repair_context(oracle):
 from joint_prototype import bundle_adaptation as ba
 from core80 import Core80Admission
 from core80 import Core80Policy
 from factor_cache import fast_factor
 old=Core80Admission._decide
 old_factor=Core80Policy.factor_columns
 def decide(self,*args):
  ephemeral=('_repair_last_x','_repair_last_columns','_repair_grid_domain','_repair_scope','_repair_variables_removed','_repair_zero_variables_removed','_repair_bound_certificates','_repair_native_calls','_repair_native_seconds','_repair_lp_calls','_repair_lp_seconds','_repair_lp_variables_removed','_repair_support_trace','_repair_last_failure','_repair_job_specs','_repair_job_order')
  for key in ephemeral:self.__dict__.pop(key,None)
  extra_ephemeral=('_repair_primary_domain_cache','_repair_primary_band_lp_calls','_repair_primary_band_lp_seconds','_repair_primary_band_variables_removed','_repair_primary_certificates','_repair_integer_rows_tightened','_repair_capacity_cover_cache','_repair_capacity_cover_peak','_repair_capacity_cover_proofs','_repair_epsilon_trace')
  for key in extra_ephemeral:self.__dict__.pop(key,None)
  self._repair_scope=True
  self._repair_allow_cbc=True
  self._repair_job_specs=args[0].job_map;self._repair_job_order=[j.job_id for j in args[0].jobs]
  try:
   r=old(self,*args)
   r.audit['engineering_solver']={k:getattr(self,'_repair_'+k,0) for k in ('variables_removed','zero_variables_removed','bound_certificates','native_calls','native_seconds','lp_calls','lp_seconds','lp_variables_removed')}
   r.audit['engineering_solver']['support_trace']=getattr(self,'_repair_support_trace',[])
   r.audit['engineering_solver'].update({k:getattr(self,'_repair_'+k,0) for k in ('primary_band_lp_calls','primary_band_lp_seconds','primary_band_variables_removed','integer_rows_tightened')})
   r.audit['engineering_solver']['primary_certificates']=getattr(self,'_repair_primary_certificates',[])
   r.audit['engineering_solver']['epsilon_trace']=getattr(self,'_repair_epsilon_trace',[])
   r.audit['engineering_solver']['capacity_cover_rows']=getattr(self,'_repair_capacity_cover_peak',0)
   return r
  except Exception as error:
   if not hasattr(error,'solver_diagnostics') and hasattr(self,'_repair_last_failure'):error.solver_diagnostics=self._repair_last_failure
   raise
  finally:
   for key in ephemeral:self.__dict__.pop(key,None)
   for key in extra_ephemeral:self.__dict__.pop(key,None)
   self.__dict__.pop('_repair_allow_cbc',None)
 with storage_context(oracle),patch.object(ba.BundleAdmission,'_milp',canonical_bundle),patch.object(Core80Admission,'_decide',decide),patch.object(Core80Policy,'_repair_literal_factor',old_factor,create=True),patch.object(Core80Policy,'factor_columns',fast_factor):yield
