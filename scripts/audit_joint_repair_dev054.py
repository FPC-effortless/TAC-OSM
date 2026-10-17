"""Same-author deterministic reconstruction and separate primitive replay."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from run_joint_repair_dev054 import setup, specs, baseline_calibration, PROTO
from run_learned_belief_dev052b import write, digest
from tac_osm.development import joint_repair as j
from tac_osm.development.adaptive_repair import fit_parameters
from tac_osm.development.learned_belief_world import strip_timing
from audit_adaptive_repair_dev053b import replay_coupon, replay_probe


def audit(directory, veritas_root, smoke=False):
    start = time.monotonic(); p = setup(veritas_root, smoke)
    out = Path(directory); common = json.loads((out/'common.json').read_text())
    assert common['smoke'] == smoke and not common['failures']
    assert common['protocol_sha256'] == digest(PROTO)
    for name, sha in json.loads((out/'manifest.json').read_text()).items(): assert digest(out/name) == sha, name
    streams = json.loads((out/'streams.json').read_text()); coupons = probes = calls = 0
    for name, spec in specs(p, smoke).items():
        print('rebuild', name, flush=True)
        assert j.collect(spec, smoke) == streams[name]
        for row in streams[name]:
            for t in range(32): coupons += replay_coupon(row, t)
    ledger = json.loads((out/'split-ledger.json').read_text())
    assert ledger == {k: [r['id'] for r in rows] for k, rows in streams.items()}
    ids = [x for group in ledger.values() for x in group]; assert len(ids) == len(set(ids))
    params = fit_parameters(streams['train']); assert params == common['parameters']
    settings = baseline_calibration(streams['calibration'], params, p)
    assert settings == json.loads((out/'baseline-calibration.json').read_text())
    def check(actual, saved, family):
        nonlocal probes, calls
        assert strip_timing(actual) == strip_timing(saved)
        rows_by_index = {r['index']: r for r in streams[family]}
        for row in saved['rows']:
            calls += replay_probe(row, rows_by_index[row['index']]); probes += 1
    for arm in j.BASELINES:
        for family in ('switching', 'drift'):
            qs, ws = j.forecasts(streams[family], arm, params, settings.get(arm, {}).get('setting'))
            check(j.evaluate(streams[family], arm, params, qs, ws), common['baselines'][arm][family], family)
    for seed in (p['model_seeds'][:1] if smoke else p['model_seeds']):
        report = json.loads((out/f'seed{seed}/report.json').read_text())
        for arm in j.NEURAL:
            print('reconstruct', seed, arm, flush=True)
            cell = report['arms'][arm]; assert cell['status'] == 'COMPLETED'
            model = j.JointModel(seed, arm)
            ds = {k: j.data(rows, params, settings['hmm']['setting']['hazard'], arm == 'gru_hmm') for k, rows in streams.items()}
            fit = j.fit(model, ds['train'], ds['validation'], seed, 3 if smoke else p['training']['updates'], p['training']['log_updates'])
            assert {k:v for k,v in fit.items() if k != 'training_seconds'} == {k:v for k,v in cell['training'].items() if k != 'training_seconds'}
            checkpoint = out/f'seed{seed}/{arm}.pt'; assert digest(checkpoint) == cell['checkpoint_sha256']
            saved = torch.load(checkpoint, weights_only=True)
            assert saved.keys() == model.state_dict().keys()
            assert all(torch.equal(v, model.state_dict()[k]) for k,v in saved.items())
            temp, trials = j.calibrate(model, ds['calibration'], streams['calibration'], params, p['calibration']['temperatures'])
            assert temp == cell['temperature'] and trials == cell['calibration_trials']
            for family in ('switching', 'drift'):
                ws = j.weights(model, ds[family], temp)
                check(j.evaluate(streams[family], arm, params, ws@np.asarray(params['endpoints']), ws), cell['evaluation'][family], family)
    result = {'status': 'SMOKE_ONLY' if smoke else 'PASSED_DEVELOPMENT_REPLAY', 'source': common['source'], 'same_author_shared_code': True,
              'checkpoint_reconstructions': len(p['model_seeds'][:1] if smoke else p['model_seeds'])*len(j.NEURAL),
              'separate_primitive_probe_executions': probes, 'separate_probe_actions': calls, 'separate_coupon_actions': coupons,
              'elapsed_seconds': time.monotonic()-start}
    write(out/'audit.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--directory', required=True)
    parser.add_argument('--veritas-root', required=True); parser.add_argument('--smoke', action='store_true')
    a = parser.parse_args(); audit(a.directory, a.veritas_root, a.smoke)
