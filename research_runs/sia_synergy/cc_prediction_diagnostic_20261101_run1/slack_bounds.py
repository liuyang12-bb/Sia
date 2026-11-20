"""Descriptive physical slack bounds, explicitly NOT new policy candidates."""
import ast,json,math
from diagnose import OUT,OLD,SOURCE,read,save,rounds,matrix,response

def main():
    targets={r['time'] for r in read(OUT/'selected_state_receipt.json')};rows=[]
    for _,r in rounds('NEW_FULL'):
        if r['interval_end'] not in targets:continue
        d=r['decision_for_next_interval'];a=d['result']['audit'];specs={s['job_id']:s for s in d['probe']['specs']};chosen=d['result']['selected']
        for j,c in chosen.items():
            if specs[j]['checkpoint_locked']:continue
            assert len(c['placement'])==1
            node,g=next(iter(c['placement'].items()))
            others=[z for k,z in chosen.items() if k!=j and node in z['placement']]
            cpu=(6-sum(z['cpu_per_gpu']*z['placement'][node] for z in others))/g
            mem=(125-sum(z['ram_gb_per_gpu']*z['placement'][node] for z in others))/g
            mat=matrix(specs[j]['template_file']);old=response(mat,c['config'],c['cpu_per_gpu'],c['ram_gb_per_gpu'])
            allowed=[]
            for s in a.get('shared_eligibility',{}).get(j,{}).get('eligible_geometries',[]):
                job,cid,p,cc,mm=ast.literal_eval(s)
                p=ast.literal_eval(p) if isinstance(p,str) else p
                if cid==c['config']['config_id'] and dict(p)==c['placement']:allowed.append((cc,mm))
            for dimension,x,y in [('CPU_SLACK_ONLY',cpu,c['ram_gb_per_gpu']),('RAM_SLACK_ONLY',c['cpu_per_gpu'],mem),('BOTH_SLACK',cpu,mem)]:
                new=response(mat,c['config'],x,y)
                legal=any(math.isclose(x,xx,abs_tol=1e-9) and math.isclose(y,yy,abs_tol=1e-9) for xx,yy in allowed)
                rows.append(dict(time=r['interval_end'],job=j,config=c['config']['config_id'],node=node,dimension=dimension,
                    current_cpu=c['cpu_per_gpu'],current_ram=c['ram_gb_per_gpu'],current_rho=old['rho'],
                    hypothetical_cpu=x,hypothetical_ram=y,rho=new['rho'],relative_speedup=new['rho']/old['rho']-1,
                    within_capacity=True,checkpoint_locked=False,in_logged_eligible_grid=legal,
                    receiver_numerator_supported=not new['numerator_clipped'],receiver_denominator_supported=not new['denominator_clipped'],
                    peers_unchanged=True,peer_clipping=any(z['source_coordinates']['clipped'] or z['baseline_source_coordinates']['clipped'] for z in others),
                    status='PHYSICAL_ONLY_OUTSIDE_POLICY_DOMAIN' if not legal else 'EXISTING_POLICY_POINT',
                    scope='not executed; no grid or model change; no remaining-time or JCT extrapolation'))
    save('physical_slack_bounds.json',rows)
    print(json.dumps([r for r in rows if r['relative_speedup']>1e-9],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
