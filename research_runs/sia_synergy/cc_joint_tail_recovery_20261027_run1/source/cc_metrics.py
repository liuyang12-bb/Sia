"""AST-equivalent metrics_class from the validated CC-Joint experiment."""
import collections,math

def metrics_class():
    from run_stage2c_cell import Accumulator,independent_rho
    from joint_prototype.execution_audit import audit_execution
    class Metrics(Accumulator):
      def __init__(self,c):
        super().__init__(c);self.clip=collections.Counter();self.selected=0;self.fallback=collections.Counter();self.below_floor=0;self.lock_checks=0
      def consume(self,c,r):
        d=r['decision_for_next_interval'];res=d['result']
        if d['round']!=getattr(self,'last_decision_round',None):
          self.last_decision_round=d['round'];assert res['audit']['valid']
          self.status[res['status']]+=1;self.solve.append(res['solve_seconds'])
          self.events.update(e['event'] for e in d.get('liveness_events',[]))
          if res['status'].startswith('FALLBACK_'):self.fallback[res['status']]+=1
          totals={t+':0':[0.,0.,0.] for t in c.ngpus_per_node}
          specs={s['job_id']:s for s in d['probe']['specs']}
          for jid,col in res['selected'].items():
            cfg=col['config'];rho=independent_rho(c.policy.oracle,jid,cfg['config_id'],col['cpu_per_gpu'],col['ram_gb_per_gpu'])
            assert math.isclose(rho,col['rho'],rel_tol=1e-12)
            if c.policy.mode in ['b1','u_jp']:assert (col['cpu_per_gpu'],col['ram_gb_per_gpu'])==(6.,125.)
            self.below_floor+=rho+1e-9<independent_rho(c.policy.oracle,jid,cfg['config_id'],6.,125.)
            row=c.policy.oracle.reference(jid,cfg['config_id']);self.selected+=1
            assert cfg['cpu_demand_scale']==row['phi_c'] and cfg['memory_demand_scale']==row['phi_m']
            for label,cpu,ram in [('numerator',col['cpu_per_gpu'],col['ram_gb_per_gpu']),('denominator',3.,62.5)]:
              self.clip[label+'_cpu']+=not 1<=cpu/row['phi_c']<=24
              self.clip[label+'_ram']+=not 20.83<=ram/row['phi_m']<=250
            s=specs[jid]
            if s['checkpoint_locked'] and s['current_config_id']:
              assert cfg['config_id']==s['current_config_id'] and col['placement']==dict(s['current_placement'])
              assert col['cpu_per_gpu']==s['current_cpu_per_gpu'] and col['ram_gb_per_gpu']==s['current_ram_gb_per_gpu'];self.lock_checks+=1
            for node,g in col['placement'].items():
              totals[node][0]+=g;totals[node][1]+=g*col['cpu_per_gpu'];totals[node][2]+=g*col['ram_gb_per_gpu']
          for node,v in totals.items():
            assert all(x<=y+1e-7 for x,y in zip(v,[2,12,250]))
            self.idle_saturated+=v[1]>=12-1e-8 and v[0]<2
            for k,x in zip(['gpu','cpu','ram'],v):self.node_peaks[node][k]=max(self.node_peaks[node][k],x)
          if d.get('strict_full_intent'):
            assert c.policy.canonical(c.allocations,c.allocations)==c.policy.canonical(d['original_sia_allocations'],c.allocations)
        for e in r['execution']:assert audit_execution(e)['valid']
        # This parent invocation integrates execution only. Its old no-slowdown
        # decision check is replaced above; donor slowdown is intentional here.
        super().consume(c,r)
      def summary(self,c,p):
        out=super().summary(c,p)
        out.update(clipping_counts=dict(self.clip),selected_columns=self.selected,
          below_proportional_selected=self.below_floor,checkpoint_lock_checks=self.lock_checks,
          fallback_status=dict(self.fallback),fallback_count=sum(self.fallback.values()))
        return out
    return Metrics
