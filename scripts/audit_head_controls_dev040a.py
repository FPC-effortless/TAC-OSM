"""DEV-040A stdlib target/action/cost audit and prespecified seed contrasts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tac_osm.development.schema_maintenance import make_pair, execute, ROLES, DEADLINES  # noqa: E402

ARMS = ('bounded_mtsk', 'masked_local', 'schema41')
HEADS = ('original', 'mlp_uniform', 'mlp_balanced', 'linear_uniform', 'linear_balanced')
THRESHOLDS = (0., .05, .1, .15, .2, .25, .3, .4, .5, .6, .7, .75, .8, .9, 1.)


def mean(xs):
    return sum(xs) / len(xs)


def predicted(logits):
    if len(logits) != 2 or not all(math.isfinite(v) for v in logits):
        raise AssertionError('nonfinite logits')
    return int(logits[1] > logits[0])


def services(p, threshold):
    return tuple(e for e, values in enumerate(p) if values[0] - values[1] > threshold)


def metrics(rows):
    cohorts = {'primary': [r for r in rows if r['deadline'] != 'before'],
               'cancellation': [r for r in rows if r['deadline'] != 'before' and 'cancel' in r['role']],
               'reissued': [r for r in rows if r['deadline'] != 'before' and r['role'] == 'reissued_cancel']}
    out = dict(worlds=len(rows), counts={k: len(v) for k, v in cohorts.items()}, heads={}, probes={})
    for name in HEADS:
        result = {k: mean([predicted(r['logits'][name][r['target_entity']][0]) == r['labels'][r['target_entity']][0]
                           for r in cohort]) for k, cohort in cohorts.items()}
        for mode in ('fixed', 'calibrated'):
            for field in ('total_cost', 'failures', 'unnecessary_services'):
                result[mode + '_' + field] = sum(r['controls'][name][mode][field] for r in rows) / (8 * len(rows))
        result['reissued_positive_rate'] = mean([predicted(r['logits'][name][r['target_entity']][0]) for r in cohorts['reissued']])
        out['heads'][name] = result
    critical = cohorts['reissued']
    for field in critical[0]['probes']:
        out['probes'][field] = mean([r['probes'][field] == r['labels'][r['target_entity']][0] for r in critical])
    out['reset_reissued'] = mean([r['reset_prediction'] == r['labels'][r['target_entity']][0] for r in critical])
    out['swap_reissued'] = mean([r['swapped_prediction'] == r['labels'][r['target_entity']][0] for r in critical])
    out['majority_reissued'] = max(mean([r['labels'][r['target_entity']][0] for r in critical]),
                                    mean([1 - r['labels'][r['target_entity']][0] for r in critical]))
    out['references'] = {mode: {field: sum(r['references'][mode][field] for r in rows) / (8 * len(rows))
                               for field in ('total_cost', 'failures', 'unnecessary_services')}
                         for mode in ('always', 'never', 'public')}
    return out


def pairs(split, spec):
    for delay in (32, 128, 512):
        for role in ROLES:
            for deadline in DEADLINES:
                for rep in range(spec['repetitions'][split]):
                    cases = make_pair(400000 + rep, 512, delay, role, deadline, domain=f'dev040a-{split}-v1')
                    for partner, case in enumerate(cases):
                        yield case, dict(delay=delay, role=role, deadline=deadline, rep=rep, partner=partner,
                                         history=512, target_entity=case.target_entity, fingerprint=case.fingerprint)


def audit(folder, spec):
    report = json.loads((folder / 'report.json').read_text())
    seed = report['seed']
    if report['study'] != 'DEV-040A' or report['smoke'] or seed not in spec['seeds'] or not report['all_three_architectures_frozen_before_evaluation']:
        raise AssertionError('study provenance mismatch')
    if report['protocol_sha256'] != hashlib.sha256((ROOT / 'research/dev040/pilot-protocol.json').read_bytes()).hexdigest():
        raise AssertionError('protocol changed')
    frozen = json.loads((folder / 'fit-manifest.json').read_text())
    cases = list(pairs('evaluation', spec))
    dev = list(pairs('calibration', spec))
    train = list(pairs('head-train', spec))
    all_fps = [{c.fingerprint for c, _ in group} for group in (cases, dev, train)]
    if any(all_fps[i] & all_fps[j] for i in range(3) for j in range(i)):
        raise AssertionError('head data split overlap')
    for arm in ARMS:
        record = report['arms'][arm]
        expected = dict(checkpoint_sha256=f'{arm}.pt', fit_features_sha256=f'{arm}-fit-features.pt',
                        evaluation_features_sha256=f'{arm}-evaluation-features.pt')
        for field, file in expected.items():
            if hashlib.sha256((folder / file).read_bytes()).hexdigest() != record[field]:
                raise AssertionError('evidence changed')
        if frozen[arm] != record['checkpoint_sha256'] or record['training']['updates'] != spec['backbone_updates']:
            raise AssertionError('fit freeze or training budget mismatch')
        backbone_fps = []
        for step in range(spec['backbone_updates']):
            backbone_fps.extend(c.fingerprint for c in make_pair(seed * 100000 + step, 512,
                (32, 128, 512)[(step // 4) % 3], ROLES[step % 4], DEADLINES[(step // 12) % 3],
                domain='dev040a-backbone-train-v1'))
        if backbone_fps != record['training']['fingerprints'] or any(set(backbone_fps) & f for f in all_fps):
            raise AssertionError('backbone exposure or data separation changed')
        for name in HEADS:
            calibration = record['calibrations'][name]
            probabilities = calibration['probabilities']
            scores = {}
            for threshold in THRESHOLDS:
                value = 0
                for p, (case, _) in zip(probabilities, dev):
                    chosen = services(p, threshold)
                    value += len(chosen) + 4 * sum(case.service_targets[e] if e in chosen else case.wait_targets[e] for e in range(8))
                scores[str(threshold)] = value
            if scores != calibration['scores'] or calibration['threshold'] != min(THRESHOLDS, key=lambda t: (scores[str(t)], abs(t - .25), t)):
                raise AssertionError('threshold not selected only from calibration cases')
        rows = report['rows'][arm]
        if len(rows) != len(cases):
            raise AssertionError('incomplete evaluation')
        for row, (case, condition) in zip(rows, cases):
            if any(row[k] != v for k, v in condition.items()):
                raise AssertionError('case census mismatch')
            truth = [[case.wait_targets[e], case.service_targets[e], case.current_targets[e]] for e in range(8)]
            if row['labels'] != truth:
                raise AssertionError('simulator target mismatch')
            for name in HEADS:
                p = row['probabilities'][name]
                for e in range(8):
                    for q in range(3):
                        z = row['logits'][name][e][q]
                        predicted(z)
                        expected_p = 1 / (1 + math.exp(max(-700, min(700, z[0] - z[1]))))
                        if abs(expected_p - p[e][q]) > 1e-6:
                            raise AssertionError('probability not derived from saved logits')
                for mode, threshold in (('fixed', .25), ('calibrated', record['calibrations'][name]['threshold'])):
                    actual = execute(case.world, case.horizon, services(p, threshold), 0)
                    if row['controls'][name][mode] != actual:
                        raise AssertionError('service outcome/cost mismatch')
            for mode, selected in (('always', tuple(range(8))), ('never', ()),
                                    ('public', tuple(e for e, value in enumerate(case.wait_targets) if value))):
                if row['references'][mode] != execute(case.world, case.horizon, selected, 0):
                    raise AssertionError('reference mismatch')
        summary = metrics(rows)
        if summary != record['metrics'] or summary['reset_reissued'] != .5 or summary['majority_reissued'] != .5:
            raise AssertionError('metric or reset/class balance invalid')
        if summary['probes']['query'] != .5 or summary['probes']['current'] != .5:
            raise AssertionError('memoryless paired-boundary control failed')
    return report


def interval(xs):
    m = mean(xs)
    half = 4.302652729749 * statistics.stdev(xs) / math.sqrt(3)
    return dict(mean=m, exploratory_95_t_df2=[m - half, m + half], seed_differences=xs)


def aggregate(folders, output):
    spec = json.loads((ROOT / 'research/dev040/pilot-protocol.json').read_text())
    reports = sorted((audit(f, spec) for f in folders), key=lambda x: x['seed'])
    if [r['seed'] for r in reports] != spec['seeds'] or len({r['source'] for r in reports}) != 1:
        raise AssertionError('incomplete seed/source matrix')
    contrasts = {}
    rows = []
    for arm in ARMS:
        contrasts[arm] = {}
        for left, right in (('mlp_balanced', 'mlp_uniform'), ('linear_balanced', 'linear_uniform'),
                            ('mlp_balanced', 'linear_balanced'), ('mlp_uniform', 'original')):
            contrasts[arm][left + '_minus_' + right] = interval([
                r['arms'][arm]['metrics']['heads'][left]['reissued'] -
                r['arms'][arm]['metrics']['heads'][right]['reissued'] for r in reports])
        for r in reports:
            for head, m in r['arms'][arm]['metrics']['heads'].items():
                rows.append(dict(seed=r['seed'], arm=arm, head=head, **m,
                                 calibrated_threshold=r['arms'][arm]['calibrations'][head]['threshold']))
    primary = contrasts['bounded_mtsk']['mlp_balanced_minus_mlp_uniform']
    utility = mean([r['arms']['bounded_mtsk']['metrics']['heads']['mlp_balanced']['calibrated_total_cost'] -
                    r['arms']['bounded_mtsk']['metrics']['heads']['mlp_uniform']['calibrated_total_cost'] for r in reports])
    out = dict(study='DEV-040A', status='ADAPTIVE_DEVELOPMENT_NOT_CONFIRMATORY', source=reports[0]['source'],
        primary_contrast=primary, primary_support=primary['mean'] >= .05 and
        primary['exploratory_95_t_df2'][0] > 0 and min(primary['seed_differences']) >= 0,
        calibrated_cost_difference=utility, mean_cost_nonincrease=utility <= 0,
        contrasts=contrasts, backbone_checkpoints=9, prediction_heads=45, probes=54,
        representation_worlds=sum(len(r['rows'][a]) for r in reports for a in ARMS),
        predictive_head_worlds=sum(len(r['rows'][a]) * len(HEADS) for r in reports for a in ARMS),
        by_seed=[dict(seed=r['seed'], arms={a: {k: r['arms'][a][k] for k in ('metrics', 'state_scalars', 'parameters', 'checkpoint_sha256')} for a in ARMS}) for r in reports],
        warning='Three fresh backbone seeds at 216 updates; no convergence, config-generalization, population reliability or full-system claim.')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'per-seed-heads.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'analysis.json').write_text(json.dumps(out, indent=2) + '\n')
    print('AUDIT PASSED; archive complete evidence before publishing summary', flush=True)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folders', nargs=3, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.folders, args.output)
