"""POSTHOC_FIT_ONLY generic observatory adapter, NO_NEW_MODEL_FITS.

Scope is original FIT augmentation -> six legal inputs -> actual ME109 heads.
No complete source/CSV/join/preprocessing or capability claim is established.

Replay preserves exact checkpoint/input digests and finding statuses. Each
within-run clean/poison prediction pair must have identical output hashes.
Only four runtime prediction/output hash fields are normalized across runs;
all numerical summaries retain the existing rtol=1e-5, atol=1e-6 comparison.
Saved original JSON and original model/score replay criteria remain unchanged.
"""
import argparse
import copy
import json
import numpy as np
import torch
from scripts import me109_residual_training as run
from tac_osm.research_observatory import architecture_inventory, gradient_conflict_atlas
from tac_osm.research_anticheat import (
    AuditLedger, Finding, audit_input_boundary, probe_future_invariance,
    probe_prediction_invariance, trace_torch_noninterference,
    trace_torch_modules, torch_gradient_diagnostics,
)

OUT = run.OUT / 'OBSERVATORY_ANTICHEAT.json'
INPUTS = ('values', 'observed', 'ages', 'clock', 'owners', 'hours')
OPEN = ('source_csv_joins', 'partition_preprocessing_provenance', 'nuisance_id_shuffle_controls',
        'native_memory_tomography', 'native_entity_isolation', 'native_dependencies', 'native_runtime')


OUTPUT_PAIRS = (('clean_prediction_sha256', 'poison_prediction_sha256'),
                ('legal_output_sha256', 'poison_output_sha256'))


def replay_comparison(actual, expected):
    """Normalize output digests only after exact scoped invariance checks."""
    assert len(actual['checks']) == len(expected['checks'])
    normalized = [copy.deepcopy(actual), copy.deepcopy(expected)]
    for current, original in zip(actual['checks'], expected['checks']):
        for key in ('seed', 'checkpoint_sha256', 'input_sha256', 'steps',
                    'recorded_batch', 'recorded_batch_seed'):
            assert current[key] == original[key], key
        current_findings = {f['check']: f for f in current['anticheat_ledger']['findings']}
        original_findings = {f['check']: f for f in original['anticheat_ledger']['findings']}
        assert current_findings.keys() == original_findings.keys()
        for name, finding in current_findings.items():
            assert finding['status'] == original_findings[name]['status'], name
        for name in ('input_boundary', 'future_invariance', 'prediction_invariance',
                     'graph_noninterference', 'gradient_health'):
            assert current_findings[name]['status'] == original_findings[name]['status'] == 'PASS'
    for report in normalized:
        for row in report['checks']:
            findings = {f['check']: f for f in row['anticheat_ledger']['findings']}
            for name, pair in zip(('prediction_invariance', 'graph_noninterference'), OUTPUT_PAIRS):
                evidence = findings[name]['evidence']
                left, right = pair
                assert isinstance(evidence[left], str) and len(evidence[left]) == 64
                assert evidence[left] == evidence[right], name
                evidence[left] = evidence[right] = '<within-run-exact-output-invariance>'
    # Input/future-input and checkpoint digests are deliberately not normalized.
    run._compare_numeric(*normalized)


def execute(replay=False):
    if OUT.exists() and not replay:
        raise RuntimeError('original observatory audit exists; no overwrite')
    torch.set_num_threads(1)
    index = run.budget_audit()
    recorded = json.loads((run.OUT/'POSTHOC_FIT_REGIME_GRADIENTS.json').read_text())
    d = run.fit_data()
    assert d['phase'] == 'fit'
    rawscale = torch.tensor(np.nanstd(d['raw'][..., :6].astype(np.float64), axis=(0,1)).clip(1e-4), dtype=torch.float32)
    rows = []
    for identity in index['models']:
        seed = identity['seed']
        batch = next(r for r in recorded['checks'] if r['seed'] == seed and r['batch'] == 0)
        origins = np.asarray(batch['origins'])
        rng = np.random.default_rng(batch['batch_seed'])
        # Original recorded RNG consumed gap/general index draws first.
        rng.integers(len(d['gap']), size=16); rng.integers(len(d['general']), size=16)
        x = run.tensor_batch(d, origins, 256, rng)
        rebuilt_digest = run.digest([run.sha_tensor(v) for v in x])
        if rebuilt_digest != batch['input_sha256']:
            # Diagnose against authenticated ORIGINAL exports before retaining
            # the original strict rejection. No hash/gate normalization here.
            from scripts import me109_observatory_inputs as frozen_inputs
            arrays, _ = frozen_inputs.load()
            diagnostics = [frozen_inputs.compare_field('batch0_' + name, value,
                            arrays['batch0_' + name])
                           for name, value in zip(frozen_inputs.FIELDS, x)]
            print(json.dumps(dict(status='ORIGINAL_INPUT_DIGEST_MISMATCH', seed=seed,
                current_input_sha256=rebuilt_digest, original_input_sha256=batch['input_sha256'],
                fields=diagnostics), indent=2), flush=True)
        assert rebuilt_digest == batch['input_sha256']
        path = run.OUT/'residual_guarded'/f'seed-{seed}'/'final.pt'
        digest = run.sha(path)
        assert digest == identity['checkpoint_sha256']
        cp = torch.load(path, map_location='cpu', weights_only=False)
        assert cp['steps'] == 4096 and cp['seed'] == seed and cp['architecture'] == 'residual_guarded'
        assert len(cp['optimizer']['state']) == 45
        assert {int(s['step']) for s in cp['optimizer']['state'].values()} == {4096}
        model = run.initialize(d, seed); model.load_state_dict(cp['model'], strict=True)
        versions = model.versions()
        old = [p.detach().clone() for p in model.parameters()]
        architecture = architecture_inventory(model, invocations={'actual_fit_forward': lambda m: m(*x[:6], state=None)})
        with trace_torch_modules(model) as observations:
            forecast, _, trace = model(*x[:6], state=None, return_trace=True)
        total, parts = run.role_objective(model, forecast, trace, x, rawscale)
        error = (model.raw_quantiles(forecast)[:,0,1] - x[6][:,0]).abs() / rawscale[0]
        def pm(start, stop):
            mask = x[7][start:stop,0]
            assert int(mask.sum()) > 0
            return torch.where(mask, error[start:stop], torch.zeros_like(error[start:stop])).sum()/mask.sum()
        losses = {'weighted_point': .5*parts['point'], 'weighted_nll': parts['nll'],
                  'weighted_interval': .25*parts['interval'],
                  'weighted_reconstruction': .3*parts['reconstruction_nll'],
                  'weighted_location_mse': .1*parts['location_mse'],
                  'gap_pm': pm(0,16), 'general_pm': pm(16,32)}
        atlas = gradient_conflict_atlas(model, losses)
        total.backward()
        prefixes = sorted({name.rsplit('.',1)[0] if '.' in name else name for name,_ in model.named_parameters()})
        health = torch_gradient_diagnostics(model, expected_prefixes=prefixes)
        ledger = AuditLedger(f'ME109-FIT-{seed}', ['input_boundary', 'future_invariance',
            'prediction_invariance', 'graph_noninterference', 'gradient_health', *OPEN])
        ledger.add(audit_input_boundary(dict(zip(INPUTS,x[:6])), INPUTS))
        ledger.add(health)
        sid, at = map(int, origins[0])
        original_pm = float(d['raw'][sid,at+6,0])
        assert np.isfinite(original_pm)
        clean = {'future_pm': original_pm}
        poisoned = {'future_pm': original_pm + 101.}
        def prepare(source):
            # The changed record enters the ORIGINAL augmentation producer.
            # All normalization caches and FIT statistics remain unchanged.
            alternate = dict(d)
            alternate['raw'] = d['raw'].copy()
            alternate['raw'][sid,at+6,0] = source['future_pm']
            return run.tensor_batch(alternate, origins[:1], 256,
                                    np.random.default_rng(batch['batch_seed']))[:6]
        ledger.add(probe_future_invariance(clean, poisoned, prepare, lambda source: source['future_pm']))
        def forward(m, source):
            torch.manual_seed(seed)
            return tuple(m(*prepare(source), state=None)[0])
        def predict(source):
            with torch.no_grad(): return forward(model, source)
        ledger.add(probe_prediction_invariance(predict, clean, poisoned))
        ledger.add(trace_torch_noninterference(model, forward, clean, poisoned))
        for name in OPEN:
            reason = ('native tool requires a source-specific ME109 state adapter; not run'
                      if name.startswith('native_') else 'outside augmentation-to-input-to-head scope; not verified')
            ledger.add(Finding(name, 'INCONCLUSIVE', reason, {}))
        assert model.versions() == versions and run.sha(path) == digest
        for p,before in zip(model.parameters(), old): assert torch.equal(p,before)
        assert {int(s['step']) for s in cp['optimizer']['state'].values()} == {4096}
        rows.append(dict(seed=seed, checkpoint_sha256=digest, steps=4096,
            recorded_batch=0, recorded_batch_seed=batch['batch_seed'], input_sha256=batch['input_sha256'],
            architecture=architecture, module_trace=observations, gradient_atlas=atlas,
            atlas_grouping='top-level parameter prefixes, not the 13 optimizer guard blocks; common live gradients only',
            anticheat_ledger=ledger.report(),
            loss_values={k:float(v.detach()) for k,v in losses.items()},
            poison_scope='one real FIT origin: raw PM2.5 at origin+6 only; original augmentation; fixed RNG; fresh state',
            poison_origin=[sid,at], no_optimizer_created=True, parameters_versions_checkpoint_unchanged=True))
    result = dict(status='POSTHOC_FIT_ONLY', policy='NO_NEW_MODEL_FITS', models=5,
        scope='actual original FIT augmentation-to-input-to-head only; source provenance and full end-to-end integrity INCONCLUSIVE',
        no_report_or_test_access=True, no_fit_or_optimizer_updates=True,
        interpretation='Local engineering diagnostics; no causal, winner, or capability claim.', checks=rows)
    if replay:
        expected = json.loads(OUT.read_text())
        replay_comparison(result,expected)
        print('Replay contract: exact identities/input digests/statuses and within-run output invariance; only four cross-run output digests normalized; numeric rtol=1e-5 atol=1e-6.')
        print('ME109_OBSERVATORY_ANTICHEAT_REPLAY_PASS')
    else:
        run.write(OUT,result)
        print('ME109_OBSERVATORY_ANTICHEAT_RECORDED')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','replay'))
    execute(p.parse_args().mode == 'replay')
