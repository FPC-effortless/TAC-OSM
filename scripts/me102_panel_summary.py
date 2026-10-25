"""No-fit posthoc ME102 panel summary and matched ME101 comparison."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ME102 = ROOT / 'research/replay-bundles/ME102'
ME101 = ROOT / 'research/replay-bundles/ME101'
DEST = ME102 / 'PANEL_SUMMARY.json'
RECIPES = ('light_floor', 'strong_reconstruction', 'physical_score')
SEEDS = tuple(range(102001, 102006))
PARENT_SEEDS = tuple(seed - 1000 for seed in SEEDS)
COHORTS = ('gap', 'general')
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 102102
RTOL = 1e-5
ATOL = 1e-6


def _json(path):
    if not path.is_file():
        raise FileNotFoundError(f"saved panel artifact is not available yet: {path}")
    return json.loads(path.read_text())


def _metric_row(q, truth, observed, nll):
    lo, mid, hi = (q[..., i] for i in range(3))
    ae = np.abs(mid - truth)
    coverage = (lo <= truth) & (truth <= hi)
    interval = hi - lo + 20 * np.maximum(lo - truth, 0) + 20 * np.maximum(truth - hi, 0)
    width = hi - lo
    per_channel = {}
    for channel in range(6):
        keep = observed[:, channel]
        if not keep.any():
            raise RuntimeError(f'channel {channel} has no observed truth in evaluation cohort')
        per_channel[str(channel)] = {
            'mae': float(ae[keep, channel].mean()),
            'coverage': float(coverage[keep, channel].mean()),
            'interval_score': float(interval[keep, channel].mean()),
            'mean_width': float(width[keep, channel].mean()),
            'p95_width': float(np.quantile(width[keep, channel], .95)),
        }
    return {
        'nll': float(nll),
        'by_channel': per_channel,
    }


def _check_metrics(recomputed, saved, label):
    for metric_name, source in (
        ('mae', 'mae_by_channel'),
        ('coverage', 'coverage_by_channel'),
        ('interval_score', 'interval_score_by_channel'),
    ):
        # Keep channel metrics separate because pollutants use different units.
        actual = [recomputed['by_channel'][str(i)][metric_name] for i in range(6)]
        np.testing.assert_allclose(actual, saved[source], rtol=RTOL, atol=ATOL,
                                   err_msg=f'{label}: forecast replay mismatch for {source}')


def _mean_dict(rows):
    out = {'nll': float(np.mean([row['nll'] for row in rows]))}
    out['by_channel'] = {}
    for channel in range(6):
        out['by_channel'][str(channel)] = {
            key: float(np.mean([row['by_channel'][str(channel)][key] for row in rows]))
            for key in ('mae', 'coverage', 'interval_score', 'mean_width', 'p95_width')
        }
    return out


def _bootstrap_gap_delta(recipe, forecasts, truth, observed, origins):
    """Week-cluster bootstrap of matched five-seed PM2.5 MAE difference."""
    weeks = np.asarray(origins, dtype=np.int64)[:, 1] // 168
    clusters = np.unique(weeks)
    if len(clusters) != 21:
        raise RuntimeError(f'gap cohort must preserve 21 global calendar weeks; found {len(clusters)}')
    candidate = np.stack([forecasts[f'gap_{recipe}_{s}'][:, 0, 1] for s in SEEDS])
    parent = np.stack([forecasts[f'gap_{s - 1000}_256'][:, 0, 1] for s in SEEDS])
    c_abs = np.abs(candidate - truth[None, :, 0])
    p_abs = np.abs(parent - truth[None, :, 0])
    mask = observed[None, :, 0]

    def delta(sampled_weeks):
        row_ix = np.concatenate([np.flatnonzero(weeks == w) for w in sampled_weeks])
        selected_mask = np.broadcast_to(mask[:, row_ix], (len(SEEDS), len(row_ix)))
        c_selected = c_abs[:, row_ix]
        p_selected = p_abs[:, row_ix]
        query_counts = selected_mask.sum(axis=1)
        if np.any(query_counts == 0):
            raise RuntimeError('bootstrap sample lacks an observed label for a model/channel')
        c_seed = np.where(selected_mask, c_selected, 0).sum(axis=1) / query_counts
        p_seed = np.where(selected_mask, p_selected, 0).sum(axis=1) / query_counts
        c = float(c_seed.mean())
        p = float(p_seed.mean())
        return c - p

    point = delta(clusters)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    for b in range(BOOTSTRAP_REPS):
        samples[b] = delta(rng.choice(clusters, size=len(clusters), replace=True))
    ci = np.quantile(samples, (.025, .975))
    return {'candidate_minus_parent_pm25_mae': point,
            'ci95': [float(ci[0]), float(ci[1])],
            'bootstrap_replicates': BOOTSTRAP_REPS,
            'cluster_unit': 'global_calendar_week (origin_hour // 168)',
            'week_count': len(clusters),
            'matched_model_count': len(SEEDS),
            'metric': 'PM2.5 (channel 0) raw-unit MAE; matched model mean across five seeds',
            'interpretation': 'conditional posthoc descriptive interval; not independent evidence, an acceptance gate, or a winner claim'}


def _numeric_replay(actual, expected, path='$'):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise RuntimeError(f'panel summary replay key mismatch at {path}')
        for key in expected:
            _numeric_replay(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise RuntimeError(f'panel summary replay length mismatch at {path}')
        for i, (a, e) in enumerate(zip(actual, expected)):
            _numeric_replay(a, e, f'{path}[{i}]')
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL,
                                   err_msg=f'panel summary numeric replay mismatch at {path}')
    elif actual != expected:
        raise RuntimeError(f'panel summary replay value mismatch at {path}: {actual!r} != {expected!r}')


def report():
    me102_results = _json(ME102 / 'RESULTS.json')
    me101_results = _json(ME101 / 'RESULTS.json')
    f102 = np.load(ME102 / 'FORECASTS.npz', allow_pickle=False)
    f101 = np.load(ME101 / 'FORECASTS.npz', allow_pickle=False)

    rows102 = {(row['recipe'], int(row['seed'])): row for row in me102_results['models']}
    if len(rows102) != len(RECIPES) * len(SEEDS):
        raise RuntimeError('ME102 results do not contain the complete fixed 15-fit panel')
    rows101 = {(row['seed'], row['context']): row
               for cohort in COHORTS for row in me101_results['cohorts'][cohort]['models']}

    cohort_summaries = {}
    cohort_inputs = {}
    for cohort, expected_count, expected_weeks in (('gap', 143, 21), ('general', 4288, 53)):
        origins_key = f'{cohort}_origins'
        truth_key = f'{cohort}_truth'
        observed_key = f'{cohort}_observed'
        for key in (origins_key, truth_key, observed_key):
            if key not in f101.files or key not in f102.files:
                raise RuntimeError(f'missing preserved original cohort array {key}')
            if not np.array_equal(f102[key], f101[key]):
                raise RuntimeError(f'ME102/ME101 original source array mismatch: {key}')
        origins = f102[origins_key]
        truth = f102[truth_key]
        observed = f102[observed_key].astype(bool, copy=False)
        weeks = np.unique(origins[:, 1] // 168)
        if len(origins) != expected_count or len(weeks) != expected_weeks:
            raise RuntimeError(f'{cohort} cohort count/week preservation failed')
        cohort_inputs[cohort] = (origins, truth, observed)

        parent_rows = {}
        for seed, parent_seed in zip(SEEDS, PARENT_SEEDS):
            key = f'{cohort}_{parent_seed}_256'
            if key not in f101.files:
                raise RuntimeError(f'missing ME101 matched 256-context forecast: {key}')
            saved = next((row for row in me101_results['cohorts'][cohort]['models']
                          if int(row['seed']) == parent_seed and int(row.get('context', -1)) == 256), None)
            if saved is None:
                raise RuntimeError(f'missing ME101 saved metrics for seed {parent_seed}, context 256')
            metric = _metric_row(f101[key], truth, observed, saved['raw_density_nll'])
            _check_metrics(metric, saved, f'ME101 {cohort} {parent_seed}')
            parent_rows[parent_seed] = metric

        recipes = {}
        for recipe in RECIPES:
            seed_rows = []
            full_gate_seeds = []
            for seed in SEEDS:
                key = f'{cohort}_{recipe}_{seed}'
                if key not in f102.files:
                    raise RuntimeError(f'missing ME102 forecast: {key}')
                result_row = rows102[(recipe, seed)]
                saved = next((row for row in me102_results['cohorts'][cohort]['models']
                              if row['recipe'] == recipe and int(row['seed']) == seed), None)
                if saved is None:
                    raise RuntimeError(f'missing ME102 saved metrics for {recipe} seed {seed}')
                metric = _metric_row(f102[key], truth, observed, saved['raw_density_nll'])
                _check_metrics(metric, saved, f'ME102 {cohort} {recipe} {seed}')
                seed_rows.append({'seed': seed, **metric})
                if result_row['full_gate']:
                    full_gate_seeds.append(seed)
            recipes[recipe] = {
                'seed_count': len(seed_rows),
                'full_gate_seed_count': len(full_gate_seeds),
                'full_gate_seeds': full_gate_seeds,
                'mean_across_five_seeds': _mean_dict(seed_rows),
                'per_seed': seed_rows,
            }
        cohort_summaries[cohort] = {
            'query_count': len(origins), 'calendar_week_count': len(weeks),
            'observed_label_count_by_channel': observed.sum(axis=0).astype(int).tolist(),
            'recipes': recipes,
            'matched_parent_me101_context_256_mean_across_five_seeds': _mean_dict(list(parent_rows.values())),
        }

    gap_origins, gap_truth, gap_observed = cohort_inputs['gap']
    bootstrap = {recipe: _bootstrap_gap_delta(recipe, {**{k: f102[k] for k in f102.files}, **{k: f101[k] for k in f101.files}}, f101['gap_truth'],
                                                f101['gap_observed'].astype(bool), gap_origins)
                 for recipe in RECIPES}
    summary = {
        'status': 'POSTHOC_NO_INDEPENDENT_CLAIM',
        'purpose': 'descriptive no-fit panel summary from saved ME102 and matched ME101 artifacts',
        'no_fit': True,
        'no_new_cohort_access': True,
        'report_loader_used': False,
        'protected_2016_2017_test_read': False,
        'original_forecast_arrays_modified': False,
        'metric_replay_tolerance': {'rtol': RTOL, 'atol': ATOL},
        'cohorts': cohort_summaries,
        'gap_pm25_mae_candidate_minus_parent_week_cluster_bootstrap': bootstrap,
        'bootstrap_note': 'Conditional posthoc descriptive comparison only; not independent evidence, not an acceptance gate, and not a winner claim.',
    }
    if DEST.exists():
        _numeric_replay(summary, _json(DEST))
        print('ME102_PANEL_SUMMARY_REPLAY=PASS')
    else:
        ME102.mkdir(parents=True, exist_ok=True)
        with DEST.open('x') as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print('ME102_PANEL_SUMMARY_SAVED=PASS')
    for recipe in RECIPES:
        gap = summary['cohorts']['gap']['recipes'][recipe]
        general = summary['cohorts']['general']['recipes'][recipe]
        ci = bootstrap[recipe]['ci95']
        print(f"{recipe}: gap PM2.5 MAE={gap['mean_across_five_seeds']['by_channel']['0']['mae']:.6g}, "
              f"NLL={gap['mean_across_five_seeds']['nll']:.6g}, "
              f"PM2.5 coverage={gap['mean_across_five_seeds']['by_channel']['0']['coverage']:.6g}, "
              f"daily PM2.5 MAE={general['mean_across_five_seeds']['by_channel']['0']['mae']:.6g}, "
              f"full_gates={gap['full_gate_seed_count']}/5, "
              f"gap ΔPM2.5 MAE={bootstrap[recipe]['candidate_minus_parent_pm25_mae']:.6g} "
              f"CI95=[{ci[0]:.6g}, {ci[1]:.6g}]")
    return summary


if __name__ == '__main__':
    report()
