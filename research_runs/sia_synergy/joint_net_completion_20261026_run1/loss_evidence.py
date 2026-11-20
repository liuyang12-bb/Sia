"""Extract observed migration/budget interaction without new counterfactual runs."""
import gzip, json
from experiment import OUT, load, save


def main():
    load()
    rows = []
    with gzip.open(OUT/'cells/JOINT_NC/rounds.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line); d = r['decision_for_next_interval']
            if d['round'] not in [0, 1, 3, 15, 17, 18]:
                continue
            audit = d['result']['audit']; j = 'cifar10-0'
            col = d['result']['selected'][j]
            filt = audit['adaptive_candidate_filters'].get(j, {})
            rows.append(dict(round=d['round'], time=r['interval_end'],
                config=col['config']['config_id'], cpu=col['cpu_per_gpu'], ram=col['ram_gb_per_gpu'],
                ledger=audit['service_delay_ledgers'][j],
                rejection_counts=filt.get('rejection_counts', {}),
                event_reasons=audit['geometry_event_reasons']))
    assert rows[-1]['ledger']['delay_charged'] == 250.
    assert rows[-1]['rejection_counts']['SERVICE_DELAY_BUDGET'] > 0
    save(OUT/'loss_evidence.json', dict(valid=True, job='cifar10-0', rows=rows,
        interpretation='Observed sequence and constraint activation, not isolated causation of all JCT loss.',
        arithmetic_counterexample=dict(work=[1., 2.], rates=[2., 1.], startup=0.,
            static_eta_fast_to_long=2., static_eta_fast_to_short=2.5,
            dynamic_sum_completion_fast_to_long=2., dynamic_sum_completion_fast_to_short=1.75,
            assumptions='Two jobs concurrently use one device each. On first completion, remaining job may switch to released fast device, no startup, constant rates. Analytical illustration only; not a new workload or tuning datum.')))
    print('LOSS_EVIDENCE_READY', rows[-1]['rejection_counts'])


if __name__ == '__main__':
    main()
