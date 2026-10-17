"""Post-run accounting from preserved raw reports; performs no fitting or selection."""
import argparse
import json
from pathlib import Path



def counts(folder):
    common = json.loads((folder/'common.json').read_text())
    data = json.loads((folder/'training-data.json').read_text())
    ledger = json.loads((folder/'split-ledger.json').read_text())
    cal_worlds = len(ledger['calibration'])
    def summary_calls(s):
        return round((s['mean_inspections']+2*s['mean_services'])*cal_worlds)
    baseline_calls = sum(len(row['calls']) for arm in common['baselines'].values()
                         for cohort in arm.values() for row in cohort['rows'])
    calls = baseline_calls+summary_calls(common['calibration_baseline'])
    neural_calls = calibration_calls = student_calls = student_targets = student_inspections = 0
    seconds = 0.
    for seed in common['seeds']:
        report = json.loads((folder/f'seed{seed}/report.json').read_text())
        for arm, cell in report['arms'].items():
            neural_calls += sum(len(r['calls']) for cohort in cell['evaluation'].values() for r in cohort['rows'])
            calibration_calls += sum(summary_calls(t['summary']) for t in cell['calibration_trials'])
            seconds += cell['training']['training_seconds']
        path = folder/f'seed{seed}/student-data.json'
        if path.exists():
            student = json.loads(path.read_text())
            events = sum(map(len, student))
            student_calls += events-len(student)
            student_inspections += sum(int(row['x'][1]) for seq in student for row in seq)
            student_targets += 4*events
    collection_calls = common['compute']['common_collection_native_actions']
    collection_worlds = sum(len(ledger[k]) for k in ('train','validation','calibration'))
    common_targets = 4*sum(len(seq) for key in ('training','validation','calibration') for seq in data[key])+8*collection_worlds
    native = collection_calls+calls+neural_calls+calibration_calls+student_calls
    return dict(source=common['source'], model_fits=common['compute']['model_fits'],
                optimizer_updates=common['compute']['optimizer_updates'],
                initial_training_target_cells=common['compute']['training_target_oracle_queries'],
                total_constructed_target_cells=common_targets+student_targets+student_inspections,
                native_calls_collection=collection_calls, native_calls_baseline_eval=baseline_calls,
                native_calls_neural_eval=neural_calls, native_calls_neural_calibration=calibration_calls,
                native_calls_student_collection=student_calls,
                native_calls_all_phases=native,
                summed_fit_seconds=seconds, total_process_wall_seconds=common['compute']['total_seconds'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', required=True, help='Contains dev052b and dev052b-fresh original output folders')
    parser.add_argument('--smoke-v1', required=True)
    parser.add_argument('--smoke-fresh', required=True)
    args = parser.parse_args()
    root = Path(args.results_root)
    runs = {name:counts(root/name) for name in ('dev052b','dev052b-fresh')}
    audit = json.loads((root/'dev052b-fresh/audit.json').read_text())
    replay = dict(runs['dev052b-fresh'])
    replay['native_calls_all_phases'] += audit['primitive_native_replay_calls']
    replay['total_process_wall_seconds'] = audit['audit_seconds']
    replay['additional_independent_native_replay_calls'] = audit['primitive_native_replay_calls']
    smoke_paths = [Path(args.smoke_v1), Path(args.smoke_fresh)]
    preflights = []
    for smoke_path in smoke_paths:
        cell = counts(smoke_path)
        replay_audit = json.loads((smoke_path/'audit.json').read_text())
        preflights.append(dict(run=smoke_path.name,
            optimizer_updates_including_reconstruction=2*cell['optimizer_updates'],
            target_cells_including_reconstruction=2*cell['total_constructed_target_cells'],
            native_calls_including_reconstruction_and_replay=2*cell['native_calls_all_phases']+replay_audit['primitive_native_replay_calls'],
            summed_process_wall_seconds=cell['total_process_wall_seconds']+replay_audit['audit_seconds']))
    result = dict(full_runs=runs, accepted_run='dev052b-fresh', excluded_run='dev052b',
                  separate_reconstruction_and_replay=replay,
                  full_research_optimizer_updates=sum(v['optimizer_updates'] for v in runs.values())+replay['optimizer_updates'],
                  full_research_native_calls=sum(v['native_calls_all_phases'] for v in runs.values())+replay['native_calls_all_phases'],
                  full_research_target_cells=sum(v['total_constructed_target_cells'] for v in runs.values())+replay['total_constructed_target_cells'],
                  summed_full_process_wall_seconds=sum(v['total_process_wall_seconds'] for v in runs.values())+replay['total_process_wall_seconds'],
                  preflights=preflights,
                  all_research_optimizer_updates_including_smoke=sum(v['optimizer_updates'] for v in runs.values())+replay['optimizer_updates']+sum(v['optimizer_updates_including_reconstruction'] for v in preflights),
                  all_research_native_calls_including_smoke=sum(v['native_calls_all_phases'] for v in runs.values())+replay['native_calls_all_phases']+sum(v['native_calls_including_reconstruction_and_replay'] for v in preflights),
                  all_research_target_cells_including_smoke=sum(v['total_constructed_target_cells'] for v in runs.values())+replay['total_constructed_target_cells']+sum(v['target_cells_including_reconstruction'] for v in preflights),
                  tests='Two complete regression passes and targeted integrity tests are recorded separately; no trial or model selection from them.',
                  accounting_definition='Target queries count constructed scalar privileged supervision targets, including discarded validation/calibration target cells. Native calls count actual paid inspect/isolate/service, including calibration, DAgger and duplicate integrity run. Harness state reads for physical scoring are not learner queries.',
                  wall_clock_note='Summed process elapsed times include concurrent runs and are not CPU-seconds. No FLOP or compute advantage is claimed.')
    (root/'dev052b-fresh/resource-accounting.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
