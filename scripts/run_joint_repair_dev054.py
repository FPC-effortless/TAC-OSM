"""Execute registered DEV-054; never select on held-out physical outcomes."""
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
from tac_osm.development import joint_repair as j
from tac_osm.development.adaptive_repair import fit_parameters, expected_nll
from run_learned_belief_dev052b import write, digest
PROTO = ROOT/'research/dev054/protocol.json'


def setup(veritas_root, smoke=False):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    import investigation_world
    vr = Path(veritas_root).resolve()
    p = json.loads(PROTO.read_text())
    assert Path(investigation_world.__file__).resolve().is_relative_to(vr)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=vr, text=True).strip() == p['veritas_commit']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=vr, text=True).strip()
    if not smoke: assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip()
    return p


def specs(p, smoke):
    return {k: {**v, 'kind': k, 'seed': v['seed']+(p['smoke_seed_offset'] if smoke else 0)} for k, v in p['splits'].items()}


def baseline_calibration(rows, params, p):
    chosen = {}
    for arm, settings in [('hmm', [{'hazard': h} for h in p['calibration']['hmm_hazards']]), ('stationary_beta', [{'strength': s} for s in p['calibration']['beta_strengths']])]:
        trials = []
        for setting in settings:
            qs, _ = j.forecasts(rows, arm, params, setting)
            trials.append({'setting': setting, 'nll': expected_nll(qs, [r['targets'] for r in rows])})
        chosen[arm] = {'setting': min(trials, key=lambda r: r['nll'])['setting'], 'trials': trials}
    return chosen


def run(output, veritas_root, smoke=False):
    p = setup(veritas_root, smoke); start = time.monotonic()
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    streams = {}
    for name, spec in specs(p, smoke).items():
        print('collect', name, flush=True); streams[name] = j.collect(spec, smoke)
    write(out/'streams.json', streams)
    ledger = {k: [r['id'] for r in rows] for k, rows in streams.items()}
    ids = [x for group in ledger.values() for x in group]; assert len(ids) == len(set(ids))
    write(out/'split-ledger.json', ledger)
    params = fit_parameters(streams['train'])
    settings = baseline_calibration(streams['calibration'], params, p)
    write(out/'baseline-calibration.json', settings)
    common = {'study': 'DEV-054', 'source': source, 'smoke': smoke, 'protocol_sha256': digest(PROTO), 'parameters': params, 'baselines': {}, 'failures': []}
    for arm in j.BASELINES:
        common['baselines'][arm] = {}
        for family in ('switching', 'drift'):
            print('baseline', arm, family, flush=True)
            qs, ws = j.forecasts(streams[family], arm, params, settings.get(arm, {}).get('setting'))
            common['baselines'][arm][family] = j.evaluate(streams[family], arm, params, qs, ws)
    write(out/'common.json', common)
    seeds = p['model_seeds'][:1] if smoke else p['model_seeds']
    for seed in seeds:
        directory = out/f'seed{seed}'; directory.mkdir()
        report = {'seed': seed, 'source': source, 'arms': {}}
        for arm in j.NEURAL:
            print('fit', seed, arm, flush=True)
            model = j.JointModel(seed, arm)
            ds = {k: j.data(rows, params, settings['hmm']['setting']['hazard'], arm == 'gru_hmm') for k, rows in streams.items()}
            try:
                fit = j.fit(model, ds['train'], ds['validation'], seed, 3 if smoke else p['training']['updates'], p['training']['log_updates'])
            except (RuntimeError, FloatingPointError) as error:
                failure = {'seed': seed, 'arm': arm, 'error': str(error), 'updates_completed': model.completed_updates}
                common['failures'].append(failure); report['arms'][arm] = {'status': 'FAILED', **failure}
                torch.save(model.state_dict(), directory/f'{arm}-FAILED.pt')
                write(out/'common.json', common); write(directory/'report.json', report); continue
            checkpoint = directory/f'{arm}.pt'; torch.save(model.state_dict(), checkpoint)
            temp, trials = j.calibrate(model, ds['calibration'], streams['calibration'], params, p['calibration']['temperatures'])
            result = {'status': 'COMPLETED', 'training': fit, 'temperature': temp, 'calibration_trials': trials, 'checkpoint_sha256': digest(checkpoint), 'evaluation': {}}
            for family in ('switching', 'drift'):
                print('evaluate', seed, arm, family, flush=True)
                w = j.weights(model, ds[family], temp)
                result['evaluation'][family] = j.evaluate(streams[family], arm, params, w@np.asarray(params['endpoints']), w)
            report['arms'][arm] = result; write(directory/'report.json', report)
    common['elapsed_seconds'] = time.monotonic()-start
    common['status'] = 'SMOKE_ONLY' if smoke else 'COMPLETE_UNAUDITED' if not common['failures'] else 'INCOMPLETE'
    write(out/'common.json', common)
    write(out/'manifest.json', {str(f.relative_to(out)): digest(f) for f in sorted(out.rglob('*')) if f.is_file()})
    print(common['status'], common['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True); parser.add_argument('--veritas-root', required=True); parser.add_argument('--smoke', action='store_true')
    a = parser.parse_args(); run(a.output, a.veritas_root, a.smoke)
