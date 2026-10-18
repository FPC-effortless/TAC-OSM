"""Posthoc description of tail contexts and physical transitions; no refitting.

Not a new acceptance rule or confirmatory test. Uses already-open recorded data.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from run_learned_stack_dev055 import write


def physical(cells):
    actions = [a for c in cells for r in c['rows'] for a in r['actions']]
    return {
        'actions': len(actions),
        'fault_exposure_before_actions': sum(sum(a['before']) for a in actions),
        'new_fault_bit_transitions': sum(sum(not b and e for b, e in zip(a['before'], a['after'], strict=True)) for a in actions),
        'repairs_of_physically_healthy_targets': sum(a['kind'] == 'repair' and not a['before'][a['target']] for a in actions),
    }


def run(directory, statistics, output):
    root = Path(directory)
    assert json.loads((root / 'summary.json').read_text())['status'] == 'COMPLETE'
    summary = json.loads(Path(statistics).read_text())
    arm = summary['primary']
    records = torch.load(root / 'test-fixed.pt', weights_only=True)
    kl = np.array([summary['arms'][arm]['per_seed'][str(seed)]['fixed']['per_query_KL']
                   for seed in range(93401, 93421)])
    cases, groups = [], defaultdict(list)
    for i, record in enumerate(records):
        counts = record['events'][:, :6].sum(0).tolist()
        graph = int(record['posterior'].reshape(4, 3).sum(0).argmax())
        case = {'index': i, 'depth': record['depth'], 'graph': graph,
                'source_type_counts': counts, 'mean_KL_across_seeds': float(kl[:, i].mean()),
                'max_KL_across_seeds': float(kl[:, i].max())}
        cases.append(case)
        groups[str(record['depth'])].append(i)
    physical_rows = {}
    for name in summary['arms']:
        values = {}
        for seed in range(93401, 93421):
            cells = json.loads((root / f'group-{seed}' / name / 'test.json').read_text())['cells']
            values[str(seed)] = {'primary': physical(cells[:4]),
                                 'higher_noise': physical(cells[4:5]),
                                 'higher_inspection_price': physical(cells[5:6])}
        physical_rows[name] = values
    refs = json.loads(Path(statistics).with_name('public-reference.json').read_text())
    write(output, {
        'status': 'POSTHOC_DESCRIPTION_ONLY',
        'scope': 'Already-open fixed histories and recorded physical transitions. No new model, selection, test set or promotion rule. Twenty seeds reuse fixed cases/worlds.',
        'tail_primary_arm': arm,
        'tail_cases_ranked': sorted(cases, key=lambda v: -v['mean_KL_across_seeds']),
        'tail_by_recorded_action_depth': {
            depth: {'fixed_cases': len(ids), 'mean_KL_over_seeds_and_cases': float(kl[:, ids].mean()),
                    'p99_KL_over_seeds_and_cases': float(np.quantile(kl[:, ids], .99))}
            for depth, ids in groups.items()},
        'physical_transitions': physical_rows,
        'reference_physical_transitions': {name: physical(cells) for name, cells in refs.items()},
        'interpretation': 'Fault exposure counts pre-action faulty entity-steps, not independent failures. New-fault transitions count0->1 physical bits, not final residual faults. A physically healthy-target repair can be rational under the available uncertain public information; counts alone do not prove a dominated action. Public-reference totals cover256rollouts, whereas per-arm evidence has20training replicates of those starts. No transient-risk guarantee follows from zero final faults.',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--statistics', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.directory, args.statistics, args.output)
