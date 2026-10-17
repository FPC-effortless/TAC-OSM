"""Derive compute accounting for both registered studies from preserved records."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def account(folder):
    folder = Path(folder)
    common = read(folder/'common.json')
    probe_calls = sum(len(row['calls']) for arm in common['baselines'].values() for family in arm.values() for row in family['rows'])
    fit_seconds, parameter_bytes = 0., {}
    for seed in common['seeds']:
        report = read(folder/f'seed{seed}/report.json')
        for arm, cell in report['arms'].items():
            if cell['status'] != 'COMPLETED': continue
            probe_calls += sum(len(row['calls']) for family in cell['evaluation'].values() for row in family['rows'])
            fit_seconds += cell['training']['training_seconds']
            parameter_bytes[arm] = cell['training']['parameter_bytes']
    audit = read(folder/'audit.json')
    original_observation_calls = common['compute']['native_observation_actions']
    reconstruction_calls = original_observation_calls+probe_calls
    independent_calls = audit['primitive_probe_replay_actions']+audit['primitive_public_coupon_replay_actions']
    assert independent_calls == reconstruction_calls
    updates = common['compute']['successful_optimizer_updates']+common['compute'].get('failed_cell_completed_updates', 0)
    return {'source': common['source'], 'optimizer_updates_per_matrix': updates,
            'optimizer_updates_including_reconstruction': 2*updates,
            'native_observation_actions': original_observation_calls,
            'native_probe_actions': probe_calls,
            'native_actions_including_reconstruction_and_independent_replay': original_observation_calls+probe_calls+reconstruction_calls+independent_calls,
            'target_cells_constructed_original_and_reconstruction': 2*common['compute']['privileged_probability_target_cells_constructed'],
            'original_matrix_seconds': common['compute']['total_seconds'], 'audit_seconds': audit['audit_seconds'],
            'summed_fit_seconds_original': fit_seconds, 'parameter_bytes_by_arm': parameter_bytes}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adaptation', required=True); p.add_argument('--smoke', required=True); p.add_argument('--review', required=True)
    a = p.parse_args()
    main, smoke = account(a.adaptation), account(a.smoke)
    review, supplement = read(Path(a.review)/'review.json'), read(Path(a.review)/'supplement.json')
    result = {'study': 'DEV-053', 'adaptive_matrix': main, 'smoke_matrix': smoke,
              'retrospective_review': {'training_updates': 0,
                  'native_probe_executions': supplement['native_replay_executions'],
                  'native_calls_including_separate_replay': 2*supplement['native_replay_actions'],
                  'original_seconds': review['seconds']},
              'total_research_optimizer_updates': main['optimizer_updates_including_reconstruction']+smoke['optimizer_updates_including_reconstruction'],
              'total_registered_native_calls': main['native_actions_including_reconstruction_and_independent_replay']+smoke['native_actions_including_reconstruction_and_independent_replay']+2*supplement['native_replay_actions'],
              'total_constructed_target_cells': main['target_cells_constructed_original_and_reconstruction']+smoke['target_cells_constructed_original_and_reconstruction'],
              'summed_matrix_and_audit_process_seconds': main['original_matrix_seconds']+main['audit_seconds']+smoke['original_matrix_seconds']+smoke['audit_seconds']+review['seconds'],
              'definitions': 'Native action counts include collection, original policy evaluation, deterministic reconstruction and separate primitive replay. Privileged target cells count materialized probability labels, including validation/evaluation diagnostics; repeated training batch accesses are not fresh oracle queries. Regressions and interactive engineering smoke checks are separately logged, not model-selection trials. Summed elapsed times are not CPU-seconds. Public observation deployment expense is additionally charged in the total-cost curves even though paired arms reuse one generated observation prefix.',
              'test_count': 1315, 'gpu_used': False,
              'scope': 'MTSK component with explicit planning and addressing; no generic TAC-OSM or compute superiority claim.'}
    Path(a.adaptation,'resource-accounting.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
