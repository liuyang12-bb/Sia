"""Read-only R0 data audit. Never constructs Cluster, Job, or runs training."""
import argparse
import ast
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from joint_prototype.joint_profile import load_synergy_template, _bilinear
from applications import APPLICATIONS


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest().upper()


def dump(p, obj):
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def records(path, kind, gpus):
    rows = []
    with path.open(newline='', encoding='utf-8') as f:
        for line, row in enumerate(csv.DictReader(f), 2):
            if kind == 'placements':
                placement = str(row['placement'])
                match = len(placement) == 1 and sum(map(int, placement)) == gpus
            else:
                match = int(row['num_nodes']) == 1 and int(row['num_replicas']) == gpus
            if match:
                rows.append(dict(file=str(path), sha256=sha(path), line=line,
                    microbatch=int(row['local_bsz']), step_time=float(row['step_time']), source=kind))
    return rows


def boundaries(c, m):
    return dict(cpu_low=c < 1, cpu_high=c > 24, ram_low=m < 20.83, ram_high=m > 250)


def response(matrix, c, m, model):
    flags = boundaries(c, m)
    if model == 'E-strict' and any(flags.values()):
        return None
    r = _bilinear(matrix, c, m)[0]
    return r * min(1, c) * min(1, m/20.83) if model == 'E-linear' else r


def run(work, models, out):
    out.mkdir(parents=True, exist_ok=False)
    config_path = work/'joint_prototype/cpu_contention_20job_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    mapping = config['response']['application_template_rule']
    ngpus = config['cluster']['ngpus_per_node']
    capacities = config['cluster']['capacity_scenarios']['cpu_constrained']
    # Single-node DP domain follows the frozen topology, not an online profile.
    domain = [(app, typ, g) for app in sorted(mapping) for typ in sorted(ngpus)
              for g in (1, 2, 4, 8) if g <= ngpus[typ]]
    domain_hash = hashlib.sha256(json.dumps(domain,separators=(',',':')).encode()).hexdigest().upper()
    source_rows, support, registry, missing = {}, {}, [], []
    for app, typ, g in domain:
        key = f'{app}/{typ}:1:{g}'
        by_source = {kind:records(work/f'traces/{app}/{kind}-{typ}.csv',kind,g)
                     for kind in ('placements','scalability')}
        source_rows[key] = by_source
        support[key] = sorted({r['microbatch'] for rows in by_source.values() for r in rows})
    reference_batches = {}
    for app in sorted(mapping):
        keys = [k for k in support if k.startswith(app+'/')]
        intersection = set.intersection(*(set(support[k]) for k in keys))
        reference_batches[app] = dict(common_direct_batches=sorted(intersection),
                                     selected=min(intersection) if intersection else None)
        if not intersection:
            missing.append(dict(application=app, reason='EMPTY_COMMON_DIRECT_MICROBATCH_INTERSECTION'))
            continue
        b = min(intersection)
        entries = []
        for key in keys:
            rows = [r for r in source_rows[key]['placements'] if r['microbatch'] == b]
            if not rows:
                rows = [r for r in source_rows[key]['scalability'] if r['microbatch'] == b]
            if not rows or any(not math.isfinite(r['step_time']) or r['step_time'] <= 0 for r in rows):
                missing.append(dict(key=key, reason='MISSING_OR_INVALID_STEP_TIME'))
                continue
            step = statistics.median(r['step_time'] for r in rows)
            g = int(key.rsplit(':',1)[1])
            entries.append(dict(key=key, application=app, config_id=key.split('/')[1],
                b_ref=b, accumulation=0, epoch='not_applicable_throughput_reference',
                domain_hash=domain_hash,
                aggregation='median_same_source_placements_preferred', source_rows=rows,
                step_time=step, T0=g*b/step, per_gpu_T0=b/step))
        if len(entries) != len(keys):
            continue  # No incomplete denominator domain allowed.
        denom = min(e['per_gpu_T0'] for e in entries)
        for e in entries:
            r = e['per_gpu_T0']/denom
            e.update(denominator=denom, denominator_domain=keys, r0=r,
                     phi_c=min(4,max(.5,r)), phi_m=min(2,max(.5,math.sqrt(r))),
                     template=mapping[app], reference_state='R0-v1', quality='direct_offline_synthetic_transfer')
            registry.append(e)
    queries = []
    for e in registry:
        typ, _, gs = e['config_id'].split(':')
        g = int(gs)
        cap = capacities[typ]
        prop = (max(1., cap['cpu_cores']/ngpus[typ]),max(20.83,cap['ram_gb']/ngpus[typ]))
        cpu = sorted(set(config['response']['aux_cpu_points'] + [prop[0]]))
        ram = sorted(set(config['response']['aux_ram_points'] + [prop[1]]))
        template = load_synergy_template(models/e['template'])
        d = (3/e['phi_c'],62.5/e['phi_m'])
        for c, m in itertools.product(cpu,ram):
            n = (c/e['phi_c'],m/e['phi_m'])
            models_out = {}
            for model in ('E-legacy','E-linear','E-strict'):
                den = response(template.matrix,*d,model)
                num = response(template.matrix,*n,model)
                pn = response(template.matrix,prop[0]/e['phi_c'],prop[1]/e['phi_m'],model)
                rho = math.sqrt(num/den) if den is not None and den > 0 and num is not None else None
                floor = rho is not None and pn is not None and rho >= math.sqrt(pn/den)-1e-10
                models_out[model] = dict(numerator=num,denominator=den,rho=rho,progress_floor=floor)
            queries.append(dict(key=e['key'],quota=[c,m],proportional=(c,m)==prop,
                capacity_feasible=g*c <= cap['cpu_cores']+1e-8 and g*m <= cap['ram_gb']+1e-8,
                numerator_coordinate=n,denominator_coordinate=d,numerator_oob=boundaries(*n),
                denominator_oob=boundaries(*d),models=models_out,
                template_sha256=template.source_sha256))
    # Every row in the three official traces, not only a selected policy/cohort.
    job_mapping = []
    for trace_id in (1,5,10):
        path = work/f'workloads/saturn/workload-{trace_id}.csv'
        with path.open(newline='',encoding='utf-8') as f:
            for line,row in enumerate(csv.DictReader(f),2):
                app = row['application']
                keys = [e['key'] for e in registry if e['application']==app]
                static = []
                for key in keys:
                    typ,_,g = key.split('/')[1].split(':')
                    ap = APPLICATIONS[typ][app]
                    # Frozen cohorts are adaptive DP with bsz tuning enabled.
                    # The GPU demand/batch CSV fields are preserved, not treated
                    # as rigid bounds absent the original Cluster category rule.
                    if ap.num_stages == 1 and int(g)*ap.min_local_bsz <= ap.max_batch_size:
                        static.append(key)
                valid = [q['key'] for q in queries if q['key'] in static and q['proportional'] and q['capacity_feasible']]
                strict = [q['key'] for q in queries if q['key'] in keys and q['proportional'] and q['capacity_feasible'] and q['models']['E-strict']['rho'] is not None]
                job_mapping.append(dict(trace=str(path),trace_sha256=sha(path),source_line=line,
                    record=row,template=mapping.get(app),registry_keys=keys,
                    physical_proportional_keys=valid,strict_proportional_keys=strict,
                    static_training_keys=static,
                    job_batch_legality='adaptive_DP_tuning_enabled_app_max_batch/min_local_batch; not online profiled bound'))
    # Read all assignments to the actual countdown. No measured rollout is used.
    delay_sites = []
    job_ast = ast.parse((work/'job.py').read_text(encoding='utf-8'))
    for n in ast.walk(job_ast):
        if isinstance(n,(ast.Assign,ast.AugAssign)):
            targets = n.targets if isinstance(n,ast.Assign) else [n.target]
            if any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and
                   t.value.id=='self' and t.attr=='rescale_time' for t in targets):
                delay_sites.append(dict(line=n.lineno,expression=ast.unparse(n)))
    delays = {f'{a}/{t}':float(APPLICATIONS[t][a].rescale_time or 30) for a,t,_ in domain}
    dbar = max(delays.values())
    delay_audit = dict(domain='five mapped DP applications, frozen three GPU types; PMP excluded',
        application_delays=delays,D_bar=dbar,interval=60,Q=60*max(1,math.ceil(dbar/60)),
        countdown_assignments=delay_sites,
        source_hashes={n:sha(work/n) for n in ('applications.py','job.py','cluster.py','simulator_config.py')},
        reasoning='Cluster calls exactly one migrate OR reallocate on changed allocation; each resets to destination app delay, not a sum. Job.step subtracts delay before training. Full geometry service protection forbids resets until Q productive seconds. Epoch updates do not reset countdown.',
        limitation='No arbitrary caller resets, failures, topology changes or PMP guarantee.')
    coverage = {}
    for stage in ('all_quota_points','capacity_feasible','proportional_capacity_feasible'):
        subset = [q for q in queries if stage=='all_quota_points' or (q['capacity_feasible'] and (stage=='capacity_feasible' or q['proportional']))]
        coverage[stage] = dict(candidates=len(subset),configurations=len({q['key'] for q in subset}),
            boundaries={side:{axis:dict(candidate_count=sum(q[side][axis] for q in subset),
                config_count=len({q['key'] for q in subset if q[side][axis]}))
                for axis in ('cpu_low','cpu_high','ram_low','ram_high')} for side in ('numerator_oob','denominator_oob')},
            models={m:dict(defined=sum(q['models'][m]['rho'] is not None for q in subset),
                progress_floor=sum(q['models'][m]['progress_floor'] for q in subset)) for m in ('E-legacy','E-linear','E-strict')})
    template_summary=[]
    for path in sorted(models.glob('*_1.py')):
        template=load_synergy_template(path)
        template_summary.append(dict(file=str(path),sha256=template.source_sha256,
            used=path.name in mapping.values(),shape=list(template.matrix.shape),
            minimum=float(template.matrix.min()),maximum=float(template.matrix.max())))
    summary = dict(status='STOP_R0_MISSING' if missing else 'R0_COMPLETE_NEUTRAL_ONLY_BOUNDARY_NOT_SELECTED',
        config_sha256=sha(config_path),domain=domain,domain_hash=domain_hash,domain_size=len(domain),
        reference_batches=reference_batches,registry_entries=len(registry),missing=missing,
        coverage=coverage,jobs=len(job_mapping),strict_jobs_without_physical_proportional=sum(not j['strict_proportional_keys'] for j in job_mapping),
        training_executed=False,new_response_connected=False)
    dump(out/'registry.json',registry)
    dump(out/'source_support.json',dict(support=support,rows=source_rows))
    dump(out/'boundary_queries.json',queries)
    dump(out/'job_mapping.json',job_mapping)
    dump(out/'delay_audit.json',delay_audit)
    dump(out/'template_summary.json',template_summary)
    dump(out/'preflight_summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','domain_size','reference_batches','registry_entries','missing')}))
    return bool(missing)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True)
    a=p.parse_args()
    raise SystemExit(run(Path(__file__).resolve().parent,a.models.resolve(),a.output.resolve()))
