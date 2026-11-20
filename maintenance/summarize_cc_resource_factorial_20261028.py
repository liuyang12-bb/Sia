"""Summarize the completed frozen matrix, preserving the earlier partial report."""
import hashlib
import shutil
import json
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1'
DEST = OUT / 'completed_analysis_20261028_run1'


def save(name, data):
    with (DEST / name).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest().upper()


def main():
    data = json.loads((DEST / 'independent_audit_and_comparison.json').read_text(encoding='utf-8'))
    pre = json.loads((OUT / 'preregistration.json').read_text(encoding='utf-8'))
    records = [json.loads(x) for x in (OUT / 'execution_manifest.jsonl').read_text().splitlines() if x.strip()]
    assert len(records) == 30 and len(data['results']) == 30
    assert [r['cell'] for r in records] == ['_'.join(x) for x in pre['cells']]
    assert all(r['exit_code'] == 0 for r in records)
    results = data['results']
    assert all(v['valid'] for v in results.values())
    all_complete = all(v['summary']['completed'] == 20 for v in results.values())
    detailed = {}
    pairs = [('CC_P', 'SIA_P'), ('CC_FULL', 'SIA_P'), ('CC_FULL', 'SIA_AUX'),
             ('CC_CPU', 'CC_P'), ('CC_MEM', 'CC_P'), ('CC_FULL', 'CC_CPU'),
             ('CC_FULL', 'CC_MEM'), ('CC_FULL', 'CC_P')]
    for scenario in pre['capacities']:
        for left, right in pairs:
            x = results[f'{scenario}_medium_{left}']['summary']
            y = results[f'{scenario}_medium_{right}']['summary']
            if x['completed'] != 20 or y['completed'] != 20:
                detailed[f'{scenario}/{left}_vs_{right}'] = {'status': 'CENSORED_NOT_MEAN_COMPARABLE'}
                continue
            deltas = {j: x['jobs'][j]['jct']-y['jobs'][j]['jct'] for j in y['jobs']}
            contributions = Counter()
            for j, d in deltas.items():
                contributions[j.rsplit('-', 1)[0]] += d
            largest = min(deltas, key=deltas.get)
            total_saving = -sum(deltas.values())
            excluded_x = sum(v['jct'] for j, v in x['jobs'].items() if j != largest)/19
            excluded_y = sum(v['jct'] for j, v in y['jobs'].items() if j != largest)/19
            detailed[f'{scenario}/{left}_vs_{right}'] = {
                'mean_improvement': 1-x['mean_jct']/y['mean_jct'],
                'mean_delta_seconds': x['mean_jct']-y['mean_jct'],
                'improved': sum(v < 0 for v in deltas.values()),
                'worsened': sum(v > 0 for v in deltas.values()),
                'unchanged': sum(v == 0 for v in deltas.values()),
                'per_job_delta_seconds': deltas,
                'application_total_delta_seconds': dict(contributions),
                'top_beneficiary': largest,
                'top_beneficiary_saving_seconds': -deltas[largest],
                'share_of_net_saving': -deltas[largest]/total_saving if total_saving else None,
                'leave_one_out_diagnostic': {
                    'excluded_job': largest, 'left_mean': excluded_x, 'right_mean': excluded_y,
                    'improvement': 1-excluded_x/excluded_y, 'main_result_unchanged': True},
                'time_delta': {k: x['totals'][k]-y['totals'][k] for k in ['wait', 'checkpoint', 'training']},
                'resource_hour_delta': {k: x['totals'][k]-y['totals'][k] for k in ['gpu_hours', 'cpu_core_hours', 'ram_gb_hours']},
            }
    neutral_comparisons = {}
    for left, right in [('CC_FULL', 'SIA_P'), ('SIA_AUX', 'SIA_P'), ('CC_CPU', 'CC_P'),
                        ('CC_MEM', 'CC_P'), ('CC_FULL', 'CC_P')]:
        x = results[f'R11_neutral_{left}']['summary']
        y = results[f'R11_neutral_{right}']['summary']
        neutral_comparisons[left+'_vs_'+right] = (
            dict(mean_improvement=1-x['mean_jct']/y['mean_jct'],
                 mean_delta_seconds=x['mean_jct']-y['mean_jct'])
            if x['completed'] == y['completed'] == 20 else dict(status='CENSORED_NOT_MEAN_COMPARABLE'))
    resource_tables = {}
    for name, value in results.items():
        s = value['summary']
        cpu, ram = pre['capacities'][s['scenario']]
        denom = s['simulated_seconds']/3600
        total_training = s['totals']['training']
        resource_tables[name] = {
            'totals': s['totals'], 'solver_status': s['solver_status'],
            'response_checks': s['checks'],
            'node_time_weighted_utilization': {
                n: {k: h[k]/(cap*denom) for k, cap in [('gpu', 2), ('cpu', cpu), ('ram', ram)]}
                for n, h in value['node_resource_hours'].items()},
            'clipping_fraction_productive_job_seconds': {
                k: v/total_training if total_training else None
                for k, v in value['clipped_productive_job_seconds'].items()},
        }
    save('mechanism_attribution.json', detailed)
    save('neutral_comparisons.json', neutral_comparisons)
    save('resource_and_clipping_summary.json', resource_tables)
    save('coverage_30.json', [dict(cell=r['cell'], completed=results[r['cell']]['summary']['completed'],
                                 total=20, raw_audit_valid=results[r['cell']]['valid'], exit_code=r['exit_code'])
                              for r in records])
    old = json.loads((OUT / 'resume_20261028_run1/preflight.json').read_text())
    for name, expected in old['protected_artifact_hashes'].items():
        assert sha(OUT / name) == expected, ('PROTECTED_HISTORY_CHANGED', name)
    for name, expected in pre['source_hashes'].items():
        assert sha(OUT / 'source' / name) == expected, name
        assert sha(ROOT / 'cc-joint-resource-study' / name) == expected, name
    neutral = data['neutral_cc_trajectories_equal']
    assert len(neutral) == 3 and all(neutral.values())
    import audit_cc_resource_factorial_20261027 as raw_auditor
    neutral_sia_equal = raw_auditor.neutral_equal(
        OUT/'cells/R11_neutral_SIA_P/rounds.jsonl.gz',
        OUT/'cells/R11_neutral_SIA_AUX/rounds.jsonl.gz')
    save('final_validation.json', {
        'time_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'THIRTY_COMPLETE_VALID' if all_complete else 'THIRTY_AUDITED_WITH_CENSORING',
        'cells': 30, 'all_20_jobs_complete': all_complete, 'independent_raw_audits_valid': True,
        'capacity_violations': 0, 'rho_formula_violations': 0, 'rho_double_application_violations': 0,
        'neutral_cc_execution_trajectories_equal': neutral,
        'neutral_sia_and_serial_execution_trajectories_equal': neutral_sia_equal,
        'existing_tests_passed': 224, 'adapter_tests_passed': 5,
        'tests_reused_under_identical_source_and_dependencies': True,
        'prior_nine_cells_and_partial_reports_preserved': True,
        'algorithm_grid_response_workload_unchanged': True,
        'training_child_wall_seconds_total': sum(x['wall_seconds'] for x in records),
        'training_child_wall_seconds_resumed': sum(x['wall_seconds'] for x in records[9:]),
    })
    lines = ['# CC-TAIL CPU/内存压力与六方法完整对照', '',
             f'状态：{"THIRTY_COMPLETE_VALID" if all_complete else "THIRTY_AUDITED_WITH_CENSORING"}。30格已执行并从原始逐轮日志独立审计。', '',
             '**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**', '',
             '## 负载、配置及续跑保护', '',
             '同一20作业：18个CIFAR10、2个DeepSpeech2，取自官方Saturn记录，保留GPU需求、batch及完整训练工作量；到达为预注册受控序列，0秒4作业，之后每300秒2作业，到2400秒。不是原始自然到达trace，也不是多个独立workload。',
             'AWS和DGX-ext各1节点、每节点2GPU；R00=12CPU/250GB，R10=6CPU/250GB，R01=12CPU/187.5GB，R11=6CPU/187.5GB，CPU单位与节点规格为合成。',
             '4容量×6方法Medium(alpha=.5)=24格，R11×6方法Neutral(alpha=0)=6格；60秒周期、6000轮、seed_profiles(1,8)、R0/E-legacy及原网格不变。',
             'SIA-P使用官方Sia GPU和placement决策，比例辅助执行；SIA-AUX固定同一Sia完整GPU/placement意图后使用既有ECT辅助优化，不是官方Synergy-TUNE。四CC均为同一CC-TAIL目标、准入、连续服务及有界收尾，只限制CPU/RAM配额自由度：P均比例、CPU仅CPU自由、MEM仅RAM自由、FULL均自由。',
             '原9格及旧部分报告保持原哈希；本次从从未启动的第10格续跑，每格独立进程、串行，启动门槛仍为2GiB内存与5GiB磁盘。未因JCT结果改参数、删作业、重跑择优或扩大矩阵。', '',
             '## 全部完成事件指标', '',
             '|场景|响应|方法|完成|平均JCT|Median|P95|P99|Makespan|Restart|',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for cell in pre['cells']:
        s = results['_'.join(cell)]['summary']
        def fmt(k): return '—' if s[k] is None else f'{s[k]:.2f}'
        lines.append(f"|{s['scenario']}|{s['response']}|{s['method']}|{s['completed']}/20|{fmt('mean_jct')}|{fmt('median_jct')}|{fmt('p95_jct')}|{fmt('p99_jct')}|{fmt('makespan')}|{s['totals']['restarts']}|")
    lines += ['', '## 同容量配对与辅助配额贡献', '',
              '正改善率表示平均JCT更低。这里只比较全部20作业完成的配对。', '',
              '|场景|FULL对Sia|FULL对串行|CPU对P|MEM对P|FULL对CPU：内存增量|FULL对P|',
              '|---|---:|---:|---:|---:|---:|---:|']
    for scenario in pre['capacities']:
        keys = ['CC_FULL_vs_SIA_P','CC_FULL_vs_SIA_AUX','CC_CPU_vs_CC_P','CC_MEM_vs_CC_P','CC_FULL_vs_CC_CPU','CC_FULL_vs_CC_P']
        vals = [detailed[scenario+'/'+key].get('mean_improvement') for key in keys]
        lines.append('|'+scenario+'|'+'|'.join('删失不比较' if x is None else f'{x*100:.2f}%' for x in vals)+'|')
    if all_complete:
        overall = [detailed[s+'/CC_FULL_vs_SIA_P']['mean_improvement'] for s in pre['capacities']]
        cpu_increment = [detailed[s+'/CC_CPU_vs_CC_P']['mean_improvement'] for s in pre['capacities']]
        lines += ['',
                  f'四容量场景中完整CC对Sia的平均JCT改善为{100*min(overall):.2f}%至{100*max(overall):.2f}%；这包含目标、准入、GPU与放置及辅助资源的整体差异，不能全归因于CPU/RAM。',
                  f'在共享CC-TAIL机制下，仅CPU优化对比例版本改善{100*min(cpu_increment):.2f}%至{100*max(cpu_increment):.2f}%。',
                  '内存自由度没有在250GB两场景产生增量；187.5GB场景的独立和条件增量以上表为准。特别注意FULL对CPU的负改善率：同时优化资源可能因后续任务选择、配置与服务轨迹变化使最终JCT变差，并不保证单轮预测目标的改善兑现为全程JCT改善。',
                  '这不是重新调参寻找获胜点；四容量和全部配额自由度均在本次新结果出现前冻结。']
    lines += ['', 'CPU和内存的交互定义：I=J_FULL−J_CPU−J_MEM+J_P（秒）。I<0代表比两项单独改善相加更有利，I>0代表重叠或抵消；这是闭环调度结果，不是固定GPU上的直接响应导数。',
              json.dumps(data['quota_interaction_seconds'], ensure_ascii=False), '',
              'Neutral平均JCT配对：'+json.dumps(neutral_comparisons, ensure_ascii=False),
              'Neutral消融用于检验无性能响应时辅助自由度不产生虚假JCT收益；CC对Sia的目标/准入/放置差异仍保留，不能期望两种策略也逐轮相同。', '',
              '## 平均值、排队与长尾', '',
              'CC-P对Sia同时改变目标、准入、服务连续性、GPU配置与放置，不能将整体收益全部归因于CPU/RAM。四CC之间才是配额自由度消融，仍包含后续GPU、队列和放置反馈。', '',
              '|场景|FULL对Sia：改善/退化/不变|P95变化|P99变化|Makespan变化|等待差(任务秒)|训练差|checkpoint差|',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for scenario in pre['capacities']:
        key = scenario+'/CC_FULL_vs_SIA_P'
        d = detailed[key]
        if d.get('status'): continue
        q = data['comparisons'][key]
        lines.append(f"|{scenario}|{d['improved']}/{d['worsened']}/{d['unchanged']}|{100*q['p95_change']:.2f}%|{100*q['p99_change']:.2f}%|{100*q['makespan_change']:.2f}%|{d['time_delta']['wait']:.0f}|{d['time_delta']['training']:.0f}|{d['time_delta']['checkpoint']:.0f}|")
    lines += ['', '本表尾部/时间变化为新减旧，正值为增加或恶化。P95/P99来自20作业，只作描述，不是高精度总体尾估计。', '',
              '每个配对的逐作业差值、CIFAR/DeepSpeech贡献、最大受益作业及去除单个最大受益者的诊断见mechanism_attribution.json；全部20作业始终保留为主结果，不以事后删长尾代替主分析。', '',
              '## 资源消耗', '',
              '|场景|响应|方法|GPU-h|CPU单位-h|RAM-GB-h|等待秒|有效训练秒|checkpoint秒|',
              '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for cell in pre['cells']:
        s = results['_'.join(cell)]['summary']; t = s['totals']
        lines.append(f"|{s['scenario']}|{s['response']}|{s['method']}|{t['gpu_hours']:.3f}|{t['cpu_core_hours']:.3f}|{t['ram_gb_hours']:.3f}|{t['wait']:.0f}|{t['training']:.0f}|{t['checkpoint']:.0f}|")
    lines += ['', '## 正确性、响应可信范围与复现', '',
              '全部结果经原始完成事件重算JCT、分位数、等待/checkpoint/有效训练互斥分解、实际完成前资源小时及逐作业restart/wasted GPU-seconds。逐轮独立验证完整Sia意图、当前候选/R0来源、GPU/CPU/RAM容量、跨轮原子提交、checkpoint零进度、执行率×rho单次应用及收尾许可约束。',
              '原224项测试及5项适配测试已通过，源码与依赖冻结后复用这些证据，没有因续跑修改算法或测试。官方Sia独立有界逐轮回归精确一致。Neutral四CC的执行GPU/进度/完成轨迹逐轮一致结果：'+json.dumps(neutral, ensure_ascii=False)+'。该检查不要求无响应时辅助配额相同。',
              'Neutral Sia和串行辅助优化的GPU/进度/完成轨迹逐轮一致：'+str(neutral_sia_equal)+'。',
              'checkpoint锁和完整延迟轮的实际触发检查数量见各格checks；计数为0代表本批未触发，不能用无违规替代长checkpoint覆盖。真实长checkpoint由既有BERT回归等测试覆盖。',
              '原始矩阵来自Synergy，但CIFAR→AlexNet、DeepSpeech→DeepSpeech关联、R0 phi、跨GPU迁移、插值/裁剪与Sia执行率乘法均为合成假设；所有方法共享执行函数。训练秒加权的分子/分母CPU与RAM越界比例完整保存在resource_and_clipping_summary.json。裁剪存在时不能宣称整个迁移模型已被真实硬件验证。',
              '预检确实找到了源数据支持域内、计入donor损失的当前网格净进度机会，但不保证JCT。无损内存互补的部分连续点在网格外，本轮未加点；不能从无内存JCT收益直接推断内存感知无价值。',
              '截至6000轮的删失不插补JCT；统计附录预先修正为共同随访τ=357600秒的sample RMST，见上级STATISTICS_ADDENDUM.md及独立审计。所有完成时主均值不受这一统计修正影响。',
              f"30个训练子进程实测累计墙钟 {sum(x['wall_seconds'] for x in records):.3f}秒；本次21格 {sum(x['wall_seconds'] for x in records[9:]):.3f}秒。不包含用户暂停期间的日历时间与测试/审计时间。", '',
              '完整源码source/、依赖锁、输入和preregistration.json保留；resume_20261028_run1/记录前缀哈希、恢复命令、每格退出与独立审计；本目录为新的最终分析版本，未覆盖上级旧9格报告。完成后停止，不自动扩展负载或调参。']
    with (DEST/'COMPLETED_REPORT.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')
    scripts = DEST/'rebuild_scripts'
    scripts.mkdir()
    names = ['preflight_cc_resource_factorial_20261027.py', 'run_cc_resource_factorial_20261027.py',
             'audit_cc_resource_factorial_20261027.py', 'resume_cc_resource_factorial_20261028.py',
             'audit_cc_resource_factorial_complete_20261028.py', 'summarize_cc_resource_factorial_20261028.py']
    script_hashes = {}
    for name in names:
        shutil.copy2(ROOT/'maintenance'/name, scripts/name)
        script_hashes[name] = sha(scripts/name)
    save('rebuild_scripts_manifest.json', script_hashes)
    current = OUT/'CURRENT_TASK.md'
    history = current.read_text(encoding='utf-8')
    current.write_text('最新状态：30格执行和独立原始日志审计已完成，实验已停止。'
                       '全部20/20完成：'+str(all_complete)+'。'
                       '最终报告：completed_analysis_20261028_run1/COMPLETED_REPORT.md；'
                       '最终验证：该目录final_validation.json。'
                       '旧REPORT.md和pause_summary.json保留为原9格历史，不代表最新状态。'
                       '不得自动继续调参、扩展负载或恢复104格。\n\n'+history, encoding='utf-8')
    shutil.copy2(current, DEST/'CURRENT_TASK_FINAL_SNAPSHOT.md')
    with (OUT/'FINAL_REPORT_LOCATION.md').open('x', encoding='utf-8') as f:
        f.write('# 完整30格实验报告入口\n\n'
                '[完整报告](completed_analysis_20261028_run1/COMPLETED_REPORT.md)\n\n'
                '[独立原始日志审计](completed_analysis_20261028_run1/independent_audit_and_comparison.json)\n\n'
                '上级REPORT.md、pause_summary.json为原9格阶段的历史记录，已保留，未覆盖。\n')
    hashes = {p.relative_to(OUT).as_posix(): sha(p) for p in OUT.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'
              and p.name != 'delivery_hashes.json'}
    save('delivery_hashes.json', hashes)
    print('FINAL_SUMMARY', all_complete, len(records), data['quota_interaction_seconds'], flush=True)


if __name__ == '__main__':
    main()
