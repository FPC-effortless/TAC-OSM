"""Preregistered DEV065 gates and paired-seed factorial contrasts."""
import argparse
import csv
import io
import json
from pathlib import Path
import numpy as np
from analyze_belief_dev063 import metrics
from analyze_belief_dev062_r1 import reference
from summarize_memory_dev061 import interval, signflip, holm
from run_learned_stack_dev055 import write


def run(directory, output):
    root = Path(directory)
    summary = json.loads((root/'summary.json').read_text())
    assert summary['status'] == 'COMPLETE'
    protocol = json.loads(Path('research/dev065-semantic-stability/protocol.json').read_text())
    specs = json.loads((root/'specs.json').read_text())
    refs = {kind: [reference(protocol['primary_seed'], h, 64, kind) for h in protocol['histories']]
            for kind in ('public', 'privileged', 'shortcut')}
    # The inherited reference accepts the error/price values as explicit arguments.
    secondary = [reference(protocol['secondary_seed'], 32, 64, 'public', error=e, inspection=price)
                 for e, price in ((.2, .1), (.1, .4))]
    write(Path(output).with_name('public-reference.json'), dict(refs, secondary_public=secondary))
    tables, csvrows = {}, []
    for arm in specs:
        rows = {}
        for seed in protocol['seeds']:
            test = json.loads((root/f'group-{seed}'/arm/'test.json').read_text())
            m = metrics(test['cells'][:4])
            m['strata'] = [dict(history=h, **metrics([c])) for h, c in zip(protocol['histories'], test['cells'][:4], strict=True)]
            m['secondary'] = [metrics([c]) for c in test['cells'][4:]]
            m['fixed'] = test['fixed']
            m['passes_each_history'] = all(c['completion'] >= .95 for c in m['strata'])
            m['passes_each_secondary'] = all(c['completion'] >= .95 for c in m['secondary'])
            m['physical_guard'] = all(a['faults'] <= b['faults']
                for c, ref in zip(test['cells'], refs['public']+secondary, strict=True)
                for a, b in zip(c['rows'], ref['rows'], strict=True))
            m['public_cost_regret'] = m['cost'] - metrics(refs['public'])['cost']
            m['privileged_cost_regret'] = m['cost'] - metrics(refs['privileged'])['cost']
            rows[seed] = m
            csvrows.append(dict(arm=arm, seed=seed, **{k: m[k] for k in (
                'completion', 'physical_repair', 'faults', 'cost', 'actions', 'inspections',
                'early_stops', 'missing_stop', 'passes_each_history', 'passes_each_secondary',
                'physical_guard', 'stop_precision', 'stop_recall_at_healthy_decisions',
                'unnecessary_repairs', 'public_cost_regret', 'privileged_cost_regret')},
                fixed_p95_KL=m['fixed']['p95_KL'], fixed_p99_KL=m['fixed']['p99_KL'],
                secondary_faults=sum(c['faults'] for c in m['secondary'])))
        tables[arm] = rows
    aggregate = {}
    for arm, rows in tables.items():
        aggregate[arm] = dict(
            passing_seeds_each_history=sum(v['passes_each_history'] for v in rows.values()),
            passing_seeds_each_secondary=sum(v['passes_each_secondary'] for v in rows.values()),
            physical_guard_seeds=sum(v['physical_guard'] for v in rows.values()),
            faults=sum(v['faults'] for v in rows.values()),
            secondary_faults=sum(c['faults'] for v in rows.values() for c in v['secondary']),
            mean_completion=float(np.mean([v['completion'] for v in rows.values()])),
            mean_cost=float(np.mean([v['cost'] for v in rows.values()])),
            median_fixed_p95_KL=float(np.median([v['fixed']['p95_KL'] for v in rows.values()])),
            median_fixed_p99_KL=float(np.median([v['fixed']['p99_KL'] for v in rows.values()])),
            individual_tail_passes=sum(v['fixed']['p95_KL'] <= .05 and v['fixed']['p99_KL'] <= .1 for v in rows.values()),
            per_seed=rows)
    primary = protocol['primary']
    comparisons = {}
    for arm in ('gru_isolated_mixed', 'no_casm_isolated_mixed', 'tac_coupled_predicted'):
        delta = [tables[primary][s]['cost'] - tables[arm][s]['cost'] for s in protocol['seeds']]
        comparisons[arm] = dict(cost=interval(delta), p=signflip(delta))
    adjusted = holm({a: comparisons[a]['p'] for a in ('gru_isolated_mixed', 'no_casm_isolated_mixed')})
    for arm, value in adjusted.items():
        comparisons[arm]['holm_p'] = value
    effects = {}
    coefficients = {
        'isolation': {'tac_isolated_predicted': .5, 'tac_isolated_mixed': .5,
                      'tac_coupled_predicted': -.5, 'tac_coupled_mixed': -.5},
        'semantic_exposure': {'tac_coupled_mixed': .5, 'tac_isolated_mixed': .5,
                             'tac_coupled_predicted': -.5, 'tac_isolated_predicted': -.5},
        'interaction': {'tac_isolated_mixed': 1, 'tac_coupled_predicted': 1,
                        'tac_isolated_predicted': -1, 'tac_coupled_mixed': -1}}
    for name, coefs in coefficients.items():
        for metric in ('cost', 'p99_KL'):
            def value(arm, seed):
                m = tables[arm][seed]
                return m['cost'] if metric == 'cost' else m['fixed']['p99_KL']
            delta = [sum(coef*value(a, s) for a, coef in coefs.items()) for s in protocol['seeds']]
            effects[name+'/'+metric] = dict(estimate=interval(delta), p=signflip(delta))
    for name, value in holm({n: v['p'] for n, v in effects.items()}).items():
        effects[name]['holm_p'] = value
    delta = [tables[primary][s]['fixed']['p99_KL'] - tables['tac_coupled_predicted'][s]['fixed']['p99_KL'] for s in protocol['seeds']]
    mechanism = interval(delta)
    a = aggregate[primary]; baseline = aggregate['tac_coupled_predicted']
    gates = {
        '18of20_each_history': a['passing_seeds_each_history'] >= 18,
        '18of20_each_secondary': a['passing_seeds_each_secondary'] >= 18,
        'physical_no_excess_primary_and_secondary': a['physical_guard_seeds'] == 20,
        'median_fixed_p95_KL_005': a['median_fixed_p95_KL'] <= .05,
        'median_fixed_p99_KL_01': a['median_fixed_p99_KL'] <= .1,
        'economic_noninferiority_002': all(comparisons[n]['cost']['paired_seed_bootstrap95'][1] <= .02
            for n in ('gru_isolated_mixed', 'no_casm_isolated_mixed')),
        'mechanism_p99_gain_10percent': a['median_fixed_p99_KL'] <= .9*baseline['median_fixed_p99_KL']
            and mechanism['paired_seed_bootstrap95'][1] < 0,
        'replay_and_interventions': 'PENDING_SEPARATE_AUDIT',
    }
    result = dict(status='COMPLETE', primary=primary, run=summary, arms=aggregate,
                  references={k: metrics(v) for k, v in refs.items()}, secondary_public=[metrics([c]) for c in secondary],
                  comparisons=comparisons, factorial_effects=effects, mechanism_p99_difference=mechanism,
                  gates=gates, commercial='NOT_ESTABLISHED',
                  scope='Twenty initializations share64physical starts; posterior-supervised two-entity synthetic study. No CASM necessity or commercial claim.')
    write(output, result)
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=list(csvrows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(csvrows)
    Path(output).with_suffix('.csv').write_text(stream.getvalue())
    print(json.dumps(gates), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--directory', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); run(a.directory, a.output)
