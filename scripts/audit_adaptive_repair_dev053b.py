"""Same-author separate-process checkpoint reconstruction and native accounting."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from investigation_world.operational.models import EpisodeSubmission
from investigation_world.operational.runtime import OperationalRuntime
from tac_osm.development.adaptive_repair import (
    BASELINES, NEURAL, LEVELS, AdaptiveModel, native_episode, collect_streams, tensors,
    fit_parameters, fit_model, calibrate_baseline, calibrate_model,
    baseline_forecasts, evaluate,
)
from tac_osm.development.learned_belief_world import strip_timing
from tac_osm.development.research_statistics import paired_summary
from run_adaptive_repair_dev053b import PROTO, setup, specs, model_probabilities
from run_learned_belief_dev052b import digest, write


def replay_coupon(stream, step):
    original = stream['observations'][step]
    ep = native_episode(stream['seed'], stream['index'], step, stream['targets'][step], 'coupon', original['attempt'])
    runtime = OperationalRuntime(ep)
    paid = 0
    for call in original['calls']:
        result = dict(runtime.act(call['name'], **call['parameters']))
        assert result == call['receipt']
        paid += {'isolate': 2, 'service': 10}[call['name']]
    verification = runtime.submit(EpisodeSubmission(conclusion='DEV-053B public calibration fixture'))
    assert verification.model_dump(mode='json') == original['verification']
    assert int(verification.state == 1) == original['success']
    assert paid == runtime.budget.spent == original['native_spent'] == 12
    return len(original['calls'])


def replay_probe(row, stream):
    ep = native_episode(row['seed'], row['index'], row['level'], stream['targets'][row['level']])
    runtime = OperationalRuntime(ep)
    reads, paid, violations = [0, 0], 0, 0
    for call in row['calls']:
        name, args = call['name'], call['parameters']
        entity = args['entity']
        result = dict(runtime.act(name, **args))
        paid += {'inspect': 1, 'isolate': 2, 'service': 10}[name]
        if name == 'inspect':
            key = f"dev053b:sensor:{row['seed']}:{row['index']}:{row['level']}:{entity}:{reads[entity]}"
            truth = int(runtime.state[f'e{entity}.fault'])
            result['fault'] = truth if random.Random(key).random() < .85 else 1-truth
            reads[entity] += 1
        assert result == call['receipt']
        violations += int(runtime.state['plant.collateral'] != 0)
    faults = sum(int(runtime.state[f'e{e}.fault']) for e in (0, 1))
    violations += int(runtime.state['plant.slot'] != -1)
    verify = runtime.submit(EpisodeSubmission(conclusion='DEV-053B maintenance probe'))
    assert verify.model_dump(mode='json') == row['verification']
    assert paid == runtime.budget.spent == row['native_spent']
    assert faults == row['failures'] and row['cost'] == paid/10+4*faults
    assert row['total_cost_including_observations'] == row['cost']+1.2*row['level']
    assert row['blocked'] == sum(e.blocked for e in runtime.events)
    assert row['violations'] == violations
    return len(row['calls'])


def matrix(result, measure='cost'):
    rows = result['rows']
    n = len(rows)//len(LEVELS)
    values = np.array([r[measure]/2 if measure == 'cost' else r[measure] for r in rows])
    return values.reshape(n, len(LEVELS))


def statistics(common, reports):
    seeds = [r['seed'] for r in reports]
    result = {'families': {}, 'pooled': {}, 'guard': {}}
    for family in ('markov', 'drift'):
        result['families'][family] = {}
        baseline = matrix(common['baselines']['stationary_beta'][family])
        oracle = matrix(common['baselines']['oracle'][family])
        for arm in BASELINES+NEURAL:
            values = np.array([matrix(r['arms'][arm]['evaluation'][family]) for r in reports]) if arm in NEURAL else matrix(common['baselines'][arm][family])[None]
            result['families'][family][arm] = {
                'pooled_levels': paired_summary(values.mean(-1), baseline.mean(-1), oracle.mean(-1)),
                'by_level': {str(n): paired_summary(values[:, :, i], baseline[:, i], oracle[:, i]) for i, n in enumerate(LEVELS)}}
            faults = np.array([matrix(r['arms'][arm]['evaluation'][family], 'failures').mean(-1) for r in reports]) if arm in NEURAL else matrix(common['baselines'][arm][family], 'failures').mean(-1)[None]
            base_faults = matrix(common['baselines']['stationary_beta'][family], 'failures').mean(-1)
            fs = paired_summary(faults, base_faults)
            fs['mean_faults_per_probe'] = fs.pop('mean_cost')
            fs['baseline_mean_faults_per_probe'] = fs.pop('mean_baseline_cost')
            result['families'][family][arm]['paired_fault_statistics'] = fs
    strata = None
    for arm in BASELINES+NEURAL:
        per_family, baseline_family, oracle_family = [], [], []
        for family in ('markov', 'drift'):
            values = np.array([matrix(r['arms'][arm]['evaluation'][family]) for r in reports]) if arm in NEURAL else matrix(common['baselines'][arm][family])[None]
            per_family.append(values.mean(-1))
            baseline_family.append(matrix(common['baselines']['stationary_beta'][family]).mean(-1))
            oracle_family.append(matrix(common['baselines']['oracle'][family]).mean(-1))
        strata = [0]*per_family[0].shape[1]+[1]*per_family[1].shape[1]
        result['pooled'][arm] = paired_summary(np.concatenate(per_family, axis=1), np.concatenate(baseline_family), np.concatenate(oracle_family), strata)
    # Paired contrasts use matched initialization indices and whole stream clusters.
    for reference in ('gru128', 'hmm_mixture_dp', 'mtsk_frozen_kernel', 'mtsk_reset'):
        diffs, seed_guards = [], {}
        for family in ('markov', 'drift'):
            m = np.array([matrix(r['arms']['mtsk']['evaluation'][family]).mean(-1) for r in reports])
            b = np.array([matrix(r['arms'][reference]['evaluation'][family]).mean(-1) for r in reports]) if reference in NEURAL else matrix(common['baselines'][reference][family]).mean(-1)[None]
            diffs.append(m-b)
            seed_guards[family] = {}
            for report in reports:
                mr = report['arms']['mtsk']['evaluation'][family]['rows']
                br = report['arms'][reference]['evaluation'][family]['rows'] if reference in NEURAL else common['baselines'][reference][family]['rows']
                seed_guards[family][str(report['seed'])] = {
                    'no_additional_faults': sum(r['failures'] for r in mr) <= sum(r['failures'] for r in br),
                    'lower_cost': sum(r['cost'] for r in mr) < sum(r['cost'] for r in br),
                    'no_additional_blocked_or_path': sum(r['blocked']+r['violations'] for r in mr) <= sum(r['blocked']+r['violations'] for r in br)}
        diff = np.concatenate(diffs, axis=1)
        interval = paired_summary(diff, np.zeros(diff.shape[1]), strata=strata)
        passed = interval['paired_delta_95pct'][1] < 0 and all(all(g.values()) for group in seed_guards.values() for g in group.values())
        result['guard'][reference] = {'passed': passed, 'mtsk_minus_reference_cost': interval, 'per_seed_family': seed_guards}
    result['robust_mtsk_advantage'] = result['pooled']['mtsk']['oracle_gap'] > 1e-9 and all(r['passed'] for r in result['guard'].values())
    result['independent_evaluation_streams'] = len(strata)
    result['seed_initializations'] = seeds
    return result


def audit(directory, veritas_root, smoke=False):
    started = time.monotonic(); proto = setup(veritas_root, dirty=smoke)
    out = Path(directory)
    common = json.loads((out/'common.json').read_text())
    assert common['smoke'] == smoke and common['protocol_sha256'] == digest(PROTO)
    for path, sha in json.loads((out/'manifest.json').read_text()).items():
        assert digest(out/path) == sha, path
    assert not common['failures'], 'Training failures retained; full qualification audit cannot pass incomplete matrix'
    streams = json.loads((out/'streams.json').read_text())
    coupon_calls = 0
    for name, spec in specs(proto, smoke).items():
        print('rebuild observations', name, flush=True)
        regenerated = collect_streams(spec)
        assert regenerated == streams[name]
        for row in streams[name]:
            for step in range(32): coupon_calls += replay_coupon(row, step)
    ledger = json.loads((out/'split-ledger.json').read_text())
    for name, rows in streams.items(): assert ledger[name] == [r['id'] for r in rows]
    ids = [x for group in ledger.values() for x in group]
    assert len(ids) == len(set(ids))
    params = fit_parameters(streams['train']); assert params == common['parameters']
    settings = json.loads((out/'baseline-calibration.json').read_text())
    native_calls = native_executions = 0
    for arm in BASELINES:
        print('replay baseline', arm, flush=True)
        if arm != 'oracle':
            setting, trials = calibrate_baseline(streams['calibration'], arm, params, proto)
            assert {'setting': setting, 'trials': trials} == settings[arm]
        for family in ('markov', 'drift'):
            if arm == 'oracle': qs, weights = np.array([r['targets'] for r in streams[family]]), None
            else: qs, weights = baseline_forecasts(streams[family], arm, params, settings[arm]['setting'])
            actual = evaluate(streams[family], qs, arm, params, weights)
            expected = common['baselines'][arm][family]
            assert strip_timing(actual) == strip_timing(expected)
            for row in expected['rows']:
                native_calls += replay_probe(row, streams[family][row['index']]); native_executions += 1
    reports = []
    seeds = proto['model_seeds'][:1] if smoke else proto['model_seeds']
    updates = 3 if smoke else proto['training']['updates']
    for seed in seeds:
        report = json.loads((out/f'seed{seed}/report.json').read_text())
        assert report['source'] == common['source'] and set(report['arms']) == set(NEURAL)
        reports.append(report)
        for arm in NEURAL:
            print('reconstruct', seed, arm, flush=True)
            cell = report['arms'][arm]
            assert cell['status'] == 'COMPLETED'
            model = AdaptiveModel(seed, arm)
            training = fit_model(model, tensors(streams['train']), seed, updates)
            assert strip_timing(training) == strip_timing(cell['training'])
            checkpoint = out/f'seed{seed}/{arm}.pt'; assert digest(checkpoint) == cell['checkpoint_sha256']
            original = torch.load(checkpoint, map_location='cpu', weights_only=True)
            assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())
            temperature, trials = calibrate_model(model, streams['calibration'], proto)
            assert temperature == cell['temperature'] and trials == cell['calibration_trials']
            for family in ('markov', 'drift'):
                qs = model_probabilities(model, streams[family], temperature)
                actual = evaluate(streams[family], qs, arm, params)
                expected = cell['evaluation'][family]
                assert strip_timing(actual) == strip_timing(expected), (seed, arm, family)
                for row in expected['rows']:
                    native_calls += replay_probe(row, streams[family][row['index']]); native_executions += 1
    result = {'study': 'DEV-053B', 'status': 'AUDITED_DEVELOPMENT' if not smoke else 'AUDITED_SMOKE',
              'source': common['source'], 'protocol_sha256': digest(PROTO),
              'all_checkpoint_tensors_reconstructed': True, 'calibration_choices_reconstructed': True,
              'primitive_probe_replay_executions': native_executions, 'primitive_probe_replay_actions': native_calls,
              'primitive_public_coupon_replay_actions': coupon_calls,
              'statistics': statistics(common, reports), 'original_compute': common['compute'],
              'audit_seconds': time.monotonic()-started,
              'audit_independence': 'Same author, shared generator/model, separate process and separate primitive action accounting; not external scientific review',
              'scope': 'Actual MTSK kernel component vs generic GRUs. No CDL/CASM/full architecture superiority claim.'}
    write(out/'audit.json', result)
    print(json.dumps({'status': result['status'], 'robust_mtsk_advantage': result['statistics']['robust_mtsk_advantage'], 'seconds': result['audit_seconds']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory'); p.add_argument('--veritas-root', required=True); p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); audit(a.directory, a.veritas_root, a.smoke)
