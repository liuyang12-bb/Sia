"""Evaluate ALREADY RECORDED model forecasts. No new forecast or simulation."""
import collections,json,math
from diagnose import OUT,rounds,save,numeric_summary

def main():
    rows=[];summaries={}
    for method in ('NEW_FULL','NEW_FO'):
        # Keep compact actual training segments only, avoiding large forecast logs in memory.
        actual=collections.defaultdict(list);checkpoints=collections.defaultdict(list);last_end=0
        for _,r in rounds(method):
            last_end=r['interval_end']
            for x in r['execution']:
                for z in x['training_segments']:
                    actual[x['job_id']].append(z)
        last=None
        for _,r in rounds(method):
            d=r['decision_for_next_interval']
            if d['round']==last:continue
            last=d['round'];ev=d['result']['audit'].get('release_completion',{})
            baseline=ev.get('baseline',{});origin=r['interval_end']
            if baseline.get('status')!='OK':continue
            pred=collections.defaultdict(list)
            for z in baseline.get('model_execution',[]):pred[z['job']].append(z)
            for h in (60,120,300):
                end=origin+h
                # Beyond final actual frame there is no remaining productive work, not censored.
                for j,zs in pred.items():
                    predicted=0.
                    for z in zs:
                        if z.get('raw_rate') is None:continue
                        start=z['time']+z['checkpoint'];dt=max(0,min(end,start+z['training'])-max(origin,start))
                        predicted+=dt*z['raw_rate']*z['rho']
                    observed=0.
                    for z in actual[j]:
                        start=z['start_time'];dt=max(0,min(end,start+z['seconds'])-max(origin,start))
                        observed+=dt*z['raw_rate_before_rho']*z['rho']
                    rows.append(dict(method=method,origin=origin,horizon=h,job=j,predicted_integrated_work=predicted,
                        observed_integrated_work=observed,ape=abs(predicted-observed)/observed if observed>0 else None,
                        zero_actual=observed==0,zero_actual_but_predicted=observed==0 and predicted>1e-7,
                        note='existing original-policy forecast vs later original-policy log; hindsight only; excludes integer epoch rounding residual'))
        summaries[method]={str(h):dict(ape=numeric_summary([z['ape'] for z in rows if z['method']==method and z['horizon']==h and z['ape'] is not None]),
            zero_actual=sum(z['zero_actual'] for z in rows if z['method']==method and z['horizon']==h),
            zero_actual_but_predicted=sum(z['zero_actual_but_predicted'] for z in rows if z['method']==method and z['horizon']==h)) for h in (60,120,300)}
    save('existing_forecast_horizon_rows.json',rows);save('existing_forecast_horizon_summary.json',summaries)
    print(json.dumps(summaries,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
