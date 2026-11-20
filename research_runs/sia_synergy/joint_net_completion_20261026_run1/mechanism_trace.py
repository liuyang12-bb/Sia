"""Read completed logs only; compress realized geometry/quota changes."""
import collections, gzip, json
from experiment import OUT, load, save


def main():
    load()
    result = {}
    for method in ['JOINT_CC_P', 'JOINT_CC', 'JOINT_NC_P', 'JOINT_NC']:
        trace = collections.defaultdict(list)
        duration = collections.defaultdict(collections.Counter)
        progress = collections.defaultdict(collections.Counter)
        last = {}
        prior_round = None
        with gzip.open(OUT / 'cells' / method / 'rounds.jsonl.gz', 'rt', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                for e in r['execution']:
                    a = e['resource_allocation']
                    if a:
                        key = f"{a['config_id']} CPU={a['cpu_per_gpu']} RAM={a['ram_gb_per_gpu']}"
                        duration[e['job_id']][key] += e['training_seconds']
                        progress[e['job_id']][key] += e['progress_delta']
                d = r['decision_for_next_interval']
                if d['round'] == prior_round:
                    continue
                prior_round = d['round']
                scores = {v['column_id']: v for v in d['completion_scores']}
                for job, col in d['result']['selected'].items():
                    state = (col['config']['config_id'], col['cpu_per_gpu'], col['ram_gb_per_gpu'],
                             tuple(sorted(col['placement'].items())))
                    if last.get(job) != state:
                        score = scores[col['column_id']]
                        trace[job].append(dict(time=r['interval_end'], round=d['round'],
                            config=state[0], cpu=state[1], ram=state[2], placement=col['placement'],
                            rho=col['rho'], predicted_raw_rate=score['raw_rate'],
                            estimated_completion=score['estimated_completion'], delay=score['delay'],
                            event_reasons=d['result']['audit'].get('geometry_event_reasons')))
                        last[job] = state
        result[method] = dict(allocation_changes=dict(trace),
            productive_seconds_by_quota={j:dict(v) for j,v in duration.items()},
            realized_progress_by_quota={j:dict(v) for j,v in progress.items()})
    save(OUT / 'mechanism_trace.json', dict(results=result,
        warning='Post-hoc trajectory description, not a single-decision causal intervention. No new simulation.'))
    print('MECHANISM_TRACE_READY')


if __name__ == '__main__':
    main()
