"""Paired seed/group uncertainty and gates fixed before test opening."""
import argparse
import json
from pathlib import Path
import numpy as np
from run_learned_stack_dev055 import write


def run(directory, output):
    root = Path(directory); summary = json.loads((root/'summary.json').read_text())
    result = dict(status='COMPLETE', run_status=summary['status'], test_opened=summary['test_opened'],
        fit_count=summary['fit_count'], source=summary['source'], commercial_qualified=False,
        source_scope='same-survey recorded rehabilitation and measured 20L fill-time; no independent followup/price contract')
    reports = []
    for path in sorted(root.glob('*/report.json')):
        report = json.loads(path.read_text())
        reports.append(dict(path=str(path.relative_to(root)), arm=report['arm'], seed=report['seed'],
            recipe=report['recipe'], parameters=report['parameters'], validation_MAE_seconds=report['validation_MAE_seconds']))
    result['fits'] = reports
    if summary['status'] != 'TEST_COMPLETE':
        result.update(decision='NO_PROMOTION_'+summary['status'], gates={'validation_and_source':False})
        if summary['status'] in ('VALIDATION_REJECTED', 'SOURCE_HEADROOM_REJECTED'):
            assert not summary['test_opened'] and not (root/'test-labels.json').exists()
    else:
        labels = json.loads((root/'test-labels.json').read_text()); y = np.array(labels['target_seconds'])
        references = json.loads((root/'test-references.json').read_text())
        selection = json.loads((root/'selection.json').read_text())
        paths = sorted(root.glob('fresh-repair-*/test.json'))
        assert len(paths) == 20
        predictions = np.array([json.loads(p.read_text())['prediction_seconds'] for p in paths])
        errors = np.abs(predictions-y[None])
        groups = sorted(set(labels['groups'])); assert len(groups) >= 20
        group_ids = [np.flatnonzero(np.array(labels['groups']) == group) for group in groups]
        rng = np.random.default_rng(974004)
        seed_ids = rng.integers(20, size=(10000, 20))
        # Preserve ALL records per sampled group; repeats are not independent.
        group_samples = rng.integers(len(groups), size=(10000, len(groups)))
        comparisons = {}
        for name in sorted({'median', selection['reference']}):
            baseline = np.abs(np.array(references[name])-y)
            delta = errors-baseline[None]
            sampled = np.empty(10000)
            for b in range(10000):
                ids = np.concatenate([group_ids[i] for i in group_samples[b]])
                sampled[b] = delta[seed_ids[b]][:, ids].mean()
            upper = float(np.quantile(sampled, .975))
            ratio = float(errors.mean()/baseline.mean())
            seed_mae = errors.mean(-1); passing = int((seed_mae <= baseline.mean()).sum())
            comparisons[name] = dict(candidate_mean_MAE_seconds=float(errors.mean()), baseline_MAE_seconds=float(baseline.mean()),
                relative_MAE=ratio, mean_difference_seconds=float(delta.mean()), simultaneous_one_sided95_upper_seconds=upper,
                passing_individual_seeds=passing, gates=dict(improvement5=ratio <= .95, upper_no_regression=upper <= 0, seed18=passing >= 18))
        pre = np.array([json.loads(p.read_text())['prediction_seconds'] for p in sorted(root.glob('fresh-prestate-*/test.json'))])
        intervals = [json.loads(p.read_text()) for p in paths]
        result.update(comparisons=comparisons, point_gates_pass=all(all(c['gates'].values()) for c in comparisons.values()),
            test_rows=len(y), test_groups=len(groups), target_range_seconds=[float(y.min()), float(y.max())],
            per_seed_MAE_seconds=errors.mean(-1).tolist(), p95_absolute_error_seconds=float(np.quantile(errors, .95)),
            p99_absolute_error_seconds=float(np.quantile(errors, .99)),
            observational_action_feature_MAE_difference_seconds=float(errors.mean()-np.abs(pre-y[None]).mean()),
            interval_mean_row_coverage=float(np.mean([x['empirical_row_coverage'] for x in intervals])),
            interval_mean_width_seconds=float(np.mean([np.mean(np.array(x['upper_seconds'])-np.array(x['lower_seconds'])) for x in intervals])),
            independent_repair_or_commercial_qualified=False)
        result['decision'] = 'LOCAL_FACTUAL_POINT_FORECAST_GATES_PASS' if result['point_gates_pass'] else 'NO_PROMOTION_TEST_GATES_FAILED'
    write(output, result); print(json.dumps({k:v for k,v in result.items() if k != 'fits'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); run(args.directory, args.output)
