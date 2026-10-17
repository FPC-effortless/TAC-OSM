"""Recheck frozen review evidence; paired fault uncertainty and lesion accounting."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from tac_osm.development.learned_belief_world import NEURAL
from tac_osm.development.research_statistics import paired_summary
from run_learned_belief_dev052b import write, digest
from audit_learned_belief_dev052b import native_replay


def run(review_path, old_path):
    folder, original = Path(review_path), Path(old_path)
    report = json.loads((folder/'review.json').read_text())
    common = json.loads((original/'common.json').read_text())
    seeds = list(map(int, report['experiments']))
    results = {'study': 'DEV-053A', 'fault_statistics': {}, 'lesions': {},
               'native_replay_executions': 0, 'native_replay_actions': 0,
               'method': 'Crossed paired worlds and global seed resampling; fault units are residual faults/world. Descriptive retrospective analysis.'}
    for cohort in ('in_regime', 'shifted'):
        results['fault_statistics'][cohort] = {}
        baseline = common['baselines']['estimated_bayes'][cohort]['rows']
        for arm in NEURAL:
            values = [[r['failures'] for r in json.loads((original/f'seed{s}/report.json').read_text())['arms'][arm]['evaluation'][cohort]['rows']] for s in seeds]
            stat = paired_summary(values, [r['failures'] for r in baseline], strata=[r['index'] % 4 for r in baseline] if cohort == 'shifted' else None)
            stat['mean_faults_per_world'] = stat.pop('mean_cost')
            stat['baseline_mean_faults_per_world'] = stat.pop('mean_baseline_cost')
            results['fault_statistics'][cohort][arm] = stat
        pooled = {name: [] for name in ('intact', 'zeroed_probabilities', 'shuffled_forecast_rows')}
        all_cases = {name: [] for name in pooled}
        for seed in seeds:
            meta = report['experiments'][str(seed)][cohort]
            path = folder/meta['artifact']; assert digest(path) == meta['sha256']
            data = json.loads(path.read_text())
            for mode, cell in data.items():
                pooled[mode].extend(cell['decision_diagnostics'])
                all_cases[mode].extend(cell['rows'])
                for row in cell['rows']:
                    results['native_replay_actions'] += native_replay(row, cohort == 'shifted')
                    results['native_replay_executions'] += 1
        results['lesions'][cohort] = {}
        for mode, rows in pooled.items():
            gaps = [r['top2_gap'] for r in rows if r['top2_gap'] is not None]
            expected = np.mean([r['forecast_expected_nll'] for r in rows])
            entropy = np.mean([r['oracle_entropy'] for r in rows])
            cases = all_cases[mode]
            results['lesions'][cohort][mode] = {
                'mean_cost_per_entity': sum(r['cost'] for r in cases)/(2*len(cases)),
                'mean_residual_faults_per_256_world_cohort': sum(r['failures'] for r in cases)/len(seeds),
                'decision_positions_including_seed_repeats': len(rows),
                'forecast_expected_nll': float(expected), 'oracle_entropy': float(entropy),
                'excess_nll': float(expected-entropy), 'top2_gap_median': float(np.median(gaps)),
                'gap_fraction_under_0_01': float(np.mean(np.array(gaps) < .01)),
                'gap_fraction_over_0_1': float(np.mean(np.array(gaps) > .1))}
    write(folder/'supplement.json', results)
    print(json.dumps({'native_replay_executions': results['native_replay_executions'],
                      'native_replay_actions': results['native_replay_actions']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review', required=True); p.add_argument('--original', required=True)
    a = p.parse_args(); run(a.review, a.original)
