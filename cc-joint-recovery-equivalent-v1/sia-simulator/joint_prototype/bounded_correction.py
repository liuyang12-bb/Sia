"""One paid, spare-resource correction; no future arrival information.

The offered point is an EXISTING proportional column, shared by FO's GPU
proposal and FULL. Ordinary auxiliary optimization is unchanged. The existing
tail permit/ledger/atomic rollback commits it and freezes geometry afterwards.
"""
import math
from .adaptive_service import gpu_geometry

class BoundedCorrectionMixin:
    def shared_eligible(self,s,j,h,columns,productive,now):
        legal,evidence=super().shared_eligible(s,j,h,columns,productive,now)
        evidence['bounded_correction_rule']='ONE_EXISTING_PROP_POINT_SPARE_CAPACITY_PAID_ETA_V1'
        if (len(s.job_map)<=1 or self.ready or set(self.services)!=set(s.job_map)
            or h.get('tail_recovery_used',False) or s.lock_active(j)
            or j.current_config_id is None
            or productive.get(j.job_id,0.)<=h['last_geometry_productive']+1e-9):
            return legal,evidence
        # Reconstruct CURRENT peer geometry and quotas from current-round
        # trusted columns. Never reuse an old query/rho or displace a peer.
        peers=[]
        for name,peer in s.job_map.items():
            if name==j.job_id:continue
            matches=[c for c in columns if c.job_id==name
                and c.config.config_id==peer.current_config_id
                and c.placement.node_gpus==peer.current_placement
                and abs(c.query.cpu_per_gpu-peer.current_cpu_per_gpu)<1e-9
                and abs(c.query.ram_gb_per_gpu-peer.current_ram_gb_per_gpu)<1e-9]
            if not matches:
                evidence['bounded_correction_block']='CURRENT_PEER_COLUMN_UNAVAILABLE';return legal,evidence
            peers.append(matches[0])
        scores=getattr(s,'current_completion_scores',{})
        def eta(c):
            v=scores.get(c.column_id,{})
            e=v.get('estimated_completion');r=v.get('raw_rate')
            return e if e is not None and math.isfinite(e) and e>0 and r is not None and math.isfinite(r) and r>0 else None
        current=gpu_geometry(h['column'])
        incumbents=[eta(c) for c in legal if gpu_geometry(c)==current and eta(c) is not None]
        if not incumbents:return legal,evidence
        options=[]
        for c in columns:
            if c.job_id!=j.job_id or gpu_geometry(c)==current or not self.is_prop(c):continue
            if eta(c) is None or eta(c)>=min(incumbents)-1e-9:continue
            if self.profile_sources.get(j.job_id,{}).get(c.config.config_id) not in {'direct_profile','projected_profile'}:continue
            cost=self._delay(j.job_id,c)
            if not math.isfinite(cost) or not 0<=cost<=self.delay_bound:continue
            if scores[c.column_id].get('delay')!=cost:continue
            if not s.audit(peers+[c])['valid']:continue
            options.append(c)
        if not options:return legal,evidence
        best=min(options,key=lambda c:(eta(c),c.column_id))
        if best in legal:return legal,evidence
        cost=self._delay(j.job_id,best)
        self._pending_tail[j.job_id]=dict(column_id=best.column_id,
            extra_budget=max(0.,h['delay_charged']+cost-self.delay_bound),actual_startup=cost,
            eta_before=min(incumbents),eta_after=eta(best),time=now,
            recovery_scope='MULTI_JOB_SPARE_RESOURCE',
            fixed_peer_witness=[dict(job=c.job_id,column=c.column_id) for c in peers])
        legal.append(best)
        evidence.update(bounded_correction_offer=self._pending_tail[j.job_id],eligible_count=len(legal))
        return legal,evidence
