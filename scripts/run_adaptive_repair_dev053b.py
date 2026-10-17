"""Run fixed DEV-053B meta-training and equal-information native learning curves."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from tac_osm.development.adaptive_repair import (
    BASELINES, NEURAL, AdaptiveModel, collect_streams, tensors,
    fit_parameters, fit_model, calibrate_baseline, calibrate_model,
    baseline_forecasts, expected_nll, evaluate,
)
from run_learned_belief_dev052b import write, digest

PROTO = ROOT/'research/dev053b/protocol.json'


def setup(veritas_root, dirty=False):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    import investigation_world
    vr = Path(veritas_root).resolve()
    assert Path(investigation_world.__file__).resolve().is_relative_to(vr)
    proto = json.loads(PROTO.read_text())
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=vr, text=True).strip() == proto['veritas_commit']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=vr, text=True).strip()
    if not dirty:
        assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip()
    return proto


def specs(proto, smoke=False):
    return {k: {**v, 'kind': k, 'seed': v['seed']+proto['smoke_seed_offset'] if smoke else v['seed'],
                'worlds': min(3, v['worlds']) if smoke else v['worlds']} for k, v in proto['splits'].items()}


def model_probabilities(model, streams, temperature):
    with torch.no_grad():
        return torch.sigmoid(model.sequence(tensors(streams)[0])/temperature).numpy()


def run(output, veritas_root, smoke=False):
    proto = setup(veritas_root, dirty=smoke)
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    started = time.monotonic()
    streams = {}
    for name, spec in specs(proto, smoke).items():
        print('collect', name, spec['worlds'], flush=True)
        streams[name] = collect_streams(spec)
    ids = {name: [r['id'] for r in rows] for name, rows in streams.items()}
    flattened = [x for group in ids.values() for x in group]
    assert len(flattened) == len(set(flattened))
    write(out/'streams.json', streams); write(out/'split-ledger.json', ids)
    params = fit_parameters(streams['train'])
    baseline_settings = {}
    for arm in BASELINES:
        if arm == 'oracle': continue
        setting, trials = calibrate_baseline(streams['calibration'], arm, params, proto)
        baseline_settings[arm] = {'setting': setting, 'trials': trials}
    write(out/'baseline-calibration.json', baseline_settings)
    common = {'study': 'DEV-053B', 'smoke': smoke, 'source': source, 'protocol_sha256': digest(PROTO),
              'streams_sha256': digest(out/'streams.json'), 'parameters': params,
              'seeds': [], 'baselines': {}, 'failures': [],
              'mechanism_sha256': digest(ROOT/'src/tac_osm/development/full_plm_reference.py')}
    for arm in BASELINES:
        print('native baseline', arm, flush=True)
        common['baselines'][arm] = {}
        for name in ('markov', 'drift'):
            if arm == 'oracle':
                probabilities, weights = np.array([r['targets'] for r in streams[name]]), None
            else:
                probabilities, weights = baseline_forecasts(streams[name], arm, params, baseline_settings[arm]['setting'])
            common['baselines'][arm][name] = evaluate(streams[name], probabilities, arm, params, weights)
    write(out/'common.json', common)
    data = tensors(streams['train'])
    seeds = proto['model_seeds'][:1] if smoke else proto['model_seeds']
    updates = 3 if smoke else proto['training']['updates']
    for seed in seeds:
        directory = out/f'seed{seed}'; directory.mkdir()
        report = {'seed': seed, 'smoke': smoke, 'source': source, 'protocol_sha256': digest(PROTO), 'arms': {}}
        for arm in NEURAL:
            print('fit', seed, arm, flush=True)
            start = time.monotonic()
            model = AdaptiveModel(seed, arm)
            try:
                fit = fit_model(model, data, seed, updates)
            except (FloatingPointError, RuntimeError) as error:
                failure = {'seed': seed, 'arm': arm, 'error': str(error), 'elapsed_seconds': time.monotonic()-start, 'updates_completed': model.completed_updates}
                common['failures'].append(failure)
                torch.save(model.state_dict(), directory/f'{arm}-FAILED.pt')
                report['arms'][arm] = {'status': 'FAILED', **failure}
                write(directory/'report.json', report); write(out/'common.json', common)
                continue
            checkpoint = directory/f'{arm}.pt'; torch.save(model.state_dict(), checkpoint)
            temp, trials = calibrate_model(model, streams['calibration'], proto)
            write(directory/f'{arm}-calibration.json', {'temperature': temp, 'trials': trials})
            results = {}
            for family in ('markov', 'drift'):
                print('evaluate', seed, arm, family, flush=True)
                qs = model_probabilities(model, streams[family], temp)
                results[family] = evaluate(streams[family], qs, arm, params)
            val = model_probabilities(model, streams['validation'], temp)
            report['arms'][arm] = {'status': 'COMPLETED', 'training': fit, 'temperature': temp,
                                   'calibration_trials': trials, 'checkpoint_sha256': digest(checkpoint),
                                   'validation_nll': expected_nll(val, [r['targets'] for r in streams['validation']]),
                                   'evaluation': results}
            write(directory/'report.json', report)
        common['seeds'].append(seed); write(out/'common.json', common)
    common['status'] = 'SMOKE_ONLY' if smoke else 'COMPLETE_UNAUDITED' if not common['failures'] else 'INCOMPLETE_CELLS_RETAINED'
    common['compute'] = {'total_seconds': time.monotonic()-started,
                          'successful_optimizer_updates': sum(cell['training']['updates'] for s in seeds for cell in json.loads((out/f'seed{s}/report.json').read_text())['arms'].values() if cell['status'] == 'COMPLETED'),
                          'fits_attempted': len(seeds)*len(NEURAL),
                          'failed_cell_completed_updates': sum(f['updates_completed'] for f in common['failures']),
                          'native_observation_episodes': sum(len(rows)*32 for rows in streams.values()),
                          'native_observation_actions': sum(len(o['calls']) for rows in streams.values() for row in rows for o in row['observations']),
                          'privileged_probability_target_cells_constructed': sum(len(rows)*33*3 for rows in streams.values()),
                          'evaluation_world_clusters': len(streams['markov'])+len(streams['drift']),
                          'note': 'Baseline and neural native actions derive from all saved rows; failures and calibration selections retained. Common observation prefixes charged to total-cost curves, reused equally across arms.'}
    write(out/'common.json', common)
    write(out/'manifest.json', {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(json.dumps({'status': common['status'], 'compute': common['compute']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True); p.add_argument('--veritas-root', required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); run(a.output, a.veritas_root, a.smoke)
