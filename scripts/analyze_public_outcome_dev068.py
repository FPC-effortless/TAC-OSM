"""Decision accounting; no candidate/test selection after fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
from run_learned_stack_dev055 import write, digest


def aggregate(cells):
    from data_belief_dev062 import BeliefWorld
    rows = [r for c in cells for r in c['rows']]
    kls = [x for r in rows for x in r['KL']]
    stopped = [r for r in rows if r['stopped']]
    physical = dict(fault_exposure_bits=0, healthy_target_repairs=0, newly_created_fault_bits=0,
                    non_stop_actions_after_physical_completion=0)
    # Hidden physical traces are evaluator diagnostics ONLY, never training or
    # policy inputs. They distinguish final repair from collateral damage.
    for cell in cells:
        for index, row in enumerate(cell['rows']):
            world = BeliefWorld(cell['seed'], index, cell['history'], cell['error'], cell['inspection'])
            for action in row['actions']:
                world.act(action); event = world.actions[-1]
                physical['fault_exposure_bits'] += sum(event['before'])
                physical['healthy_target_repairs'] += int(event['kind'] == 'repair' and not event['before'][event['target']])
                physical['newly_created_fault_bits'] += sum(max(0, b-a) for a, b in zip(event['before'], event['after'], strict=True))
                physical['non_stop_actions_after_physical_completion'] += int(not any(event['before']) and event['kind'] != 'stop')
            assert world.finish()['faults'] == row['faults']
    return dict(**physical, worlds=len(rows), completion=float(np.mean([r['completion'] for r in rows])),
        physical_repair=float(np.mean([r['physical_repair'] for r in rows])),
        fault_bits=sum(r['faults'] for r in rows), faulty_worlds=sum(r['faults'] > 0 for r in rows),
        physical_cost=float(np.mean([r['cost'] for r in rows])),
        actions=float(np.mean([len(r['actions']) for r in rows])),
        stop_precision=float(np.mean([r['completion'] for r in stopped])) if stopped else None,
        stop_recall=float(sum(r['completion'] for r in rows)/max(1, sum(r['physical_repair'] for r in rows))),
        false_stop_worlds=sum(r['stopped'] and not r['physical_repair'] for r in rows),
        healthy_budget_exhaustion=sum(not r['stopped'] and r['physical_repair'] for r in rows),
        all_budget_exhaustion=sum(not r['stopped'] for r in rows),
        median_KL=float(np.median(kls)), p95_KL=float(np.quantile(kls, .95)), p99_KL=float(np.quantile(kls, .99)),
        exact_posterior_action_flip=float(np.mean([x for r in rows for x in r['exact_flip']])),
        margin_below_0035=float(np.mean([x < .0035 for r in rows for x in r['margins']])),
        margin_below_epsilon=float(np.mean([x <= 1e-6 for r in rows for x in r['margins']])))


def run(directory, output):
    root = Path(directory); summary = json.loads((root/'summary.json').read_text())
    reports = sorted(root.glob('*/*-report.json'))
    per_fit = {}
    for path in reports:
        report = json.loads(path.read_text())
        item = dict(arm=report['arm'], seed=report['seed'], lr=report['lr'],
            parameter_count=report['spec']['parameters'], teacher_control=report['spec']['teacher'],
            fixed=report['validation_fixed'], validation=aggregate(report['validation']),
            validation_strata=[dict(history=c['history'], **aggregate([c])) for c in report['validation']])
        test_path = path.parent/f"{report['arm']}-test.json"
        if test_path.exists():
            test = json.loads(test_path.read_text())
            item['primary'] = aggregate(test['primary']); item['secondary'] = aggregate(test['secondary'])
            item['primary_strata'] = [dict(history=c['history'], **aggregate([c])) for c in test['primary']]
            item['secondary_strata'] = [dict(history=c['history'], **aggregate([c])) for c in test['secondary']]
        per_fit[str(path.relative_to(root))] = item
    result = dict(status='COMPLETE', run_status=summary['status'], per_fit=per_fit,
        fit_count=len(reports), tests_opened=summary['tests_opened'], source=summary['source'],
        acquisition={name: json.loads((root/'_datasets'/f'{name}-coverage.json').read_text()) for name in ('train', 'validation')},
        empirical_prior=json.loads((root/'prior.json').read_text()), commercial_qualified=False)
    if summary['status'] == 'VALIDATION_REJECTED':
        assert not summary['tests_opened']
        result.update(decision='NO_PROMOTION_VALIDATION_SCREEN_FAILED_TESTS_CLOSED', gates={'validation_entry': False})
    elif summary['status'] == 'TEST_COMPLETE':
        selection = json.loads((root/'selection.json').read_text()); primary = selection['primary']
        fresh = [r for name, r in per_fit.items() if name.startswith('group-')]
        candidate = sorted([r for r in fresh if r['arm'] == primary], key=lambda r: r['seed'])
        assert len(candidate) == 20
        reliable = sum(all(s['completion'] >= .95 for s in r['primary_strata']) for r in candidate)
        calibrated = sum(all(s['p95_KL'] <= .05 and s['p99_KL'] <= .1 for s in r['primary_strata']) for r in candidate)
        rng = np.random.default_rng(968003); ids = rng.integers(20, size=(10000, 20))
        contrasts = {}
        for control in ('gru_outcome', 'no_casm_outcome'):
            comparison = sorted([r for r in fresh if r['arm'] == control], key=lambda r: r['seed'])
            assert [r['seed'] for r in comparison] == [r['seed'] for r in candidate]
            delta = np.array([a['primary']['physical_cost']-b['primary']['physical_cost'] for a, b in zip(candidate, comparison, strict=True)])
            guard = all(sa['fault_bits'] <= sb['fault_bits'] for a, b in zip(candidate, comparison, strict=True)
                for sa, sb in zip(a['primary_strata'], b['primary_strata'], strict=True))
            upper = float(np.quantile(delta[ids].mean(-1), .975))
            contrasts[control] = dict(mean_cost_difference=float(delta.mean()), simultaneous_one_sided95_upper=upper,
                cost_noninferior=upper <= .02, no_additional_faults=guard)
        secondary = all(all(s['completion'] >= .95 and s['fault_bits'] == 0 for s in r['secondary_strata']) for r in candidate)
        gates = dict(reliable18=reliable >= 18, calibrated18=calibrated >= 18,
            controls=all(v['cost_noninferior'] and v['no_additional_faults'] for v in contrasts.values()),
            secondary_reliability=secondary)
        result.update(decision='LOCAL_SYNTHETIC_CAPABILITY_GATES_PASS' if all(gates.values()) else 'NO_PROMOTION_REGISTERED_GATES_FAILED',
            primary=primary, individual_reliable_seeds=reliable, individual_calibrated_seeds=calibrated,
            gates=gates, contrasts=contrasts, mechanism_superiority=False)
    else: result['decision'] = 'SMOKE_OR_INCOMPLETE_NOT_QUALIFIED'
    write(output, result); print(json.dumps({k: v for k, v in result.items() if k not in ('per_fit', 'acquisition')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args(); run(args.directory, args.output)
