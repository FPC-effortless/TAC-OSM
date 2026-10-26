"""POSTHOC_FIT_ONLY generic observatory adapter, NO_NEW_MODEL_FITS.

Scope is original FIT augmentation -> six legal inputs -> actual ME109 heads.
No complete source/CSV/join/preprocessing or capability claim is established.

Replay preserves exact checkpoint/input digests and finding statuses. Each
within-run clean/poison prediction pair must have identical output hashes.
V2 uses authenticated frozen original batch inputs for model replay. Fresh
producer outputs must independently pass unchanged float tolerances and exact
shape/dtype/discrete checks; one-origin fresh/frozen model forecasts must also
match. Their producer fingerprint difference is disclosed in a separate V2
report before comparing the original future-invariance finding. The four
runtime prediction/output hash fields are normalized across runs;
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

V2_REPORT = run.OUT / 'PREPROCESS_REPLAY_V2.json'
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
    # Original frozen batch and checkpoint digests are never normalized here.
    # The separately validated current-producer exception is handled by V2 below.
    run._compare_numeric(*normalized)


def authenticated_inputs():
    from scripts import me109_observatory_inputs as frozen_inputs
    arrays, receipt = frozen_inputs.load()
    assert receipt['source_archive_sha256'] == run.data.SHA
    assert receipt['original_posthoc_sha256'] == run.sha(run.OUT/'POSTHOC_FIT_REGIME_GRADIENTS.json')
    assert receipt['original_observatory_sha256'] == run.sha(OUT)
    original=json.loads((run.OUT/'POSTHOC_FIT_REGIME_GRADIENTS.json').read_text())
    for batch in receipt['batches']:
        tensors=[torch.from_numpy(arrays[f"batch{batch['batch']}_{name}"]) for name in frozen_inputs.FIELDS]
        digest=run.digest([run.sha_tensor(v) for v in tensors])
        assert digest == batch['input_sha256']
        rows=[r for r in original['checks'] if r['batch']==batch['batch']]
        assert len(rows)==5
        for row in rows:
            assert row['input_sha256']==digest
            assert row['origins']==batch['origins'] and row['batch_seed']==batch['batch_seed']
    from tac_osm.research_anticheat import fingerprint
    clean=[torch.from_numpy(arrays['poison_clean_'+name]) for name in frozen_inputs.FIELDS[:6]]
    assert fingerprint(clean)==receipt['poison_clean_fingerprint']
    for row in json.loads(OUT.read_text())['checks']:
        finding=next(f for f in row['anticheat_ledger']['findings'] if f['check']=='future_invariance')
        assert finding['evidence']['inputs_sha256']==receipt['poison_clean_fingerprint']
    return arrays,receipt


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
    preprocess_rows = []
    frozen_arrays = frozen_receipt = None
    if replay:
        frozen_arrays, frozen_receipt = authenticated_inputs()
        from scripts import me109_observatory_inputs as frozen_inputs
        producer_diagnostics = frozen_inputs.diagnose()
        if producer_diagnostics['status'] != 'PASS':
            raise AssertionError('current producer exceeds unchanged numerical/discrete contract')
        rawscale = torch.from_numpy(frozen_arrays['rawscale'].copy())
    for identity in index['models']:
        seed = identity['seed']
        batch = next(r for r in recorded['checks'] if r['seed'] == seed and r['batch'] == 0)
        origins = np.asarray(batch['origins'])
        rng = np.random.default_rng(batch['batch_seed'])
        # Original recorded RNG consumed gap/general index draws first.
        rng.integers(len(d['gap']), size=16); rng.integers(len(d['general']), size=16)
        x = run.tensor_batch(d, origins, 256, rng)
        rebuilt_digest = run.digest([run.sha_tensor(v) for v in x])
        if replay:
            x = [torch.from_numpy(frozen_arrays['batch0_'+name].copy()) for name in frozen_inputs.FIELDS]
        assert run.digest([run.sha_tensor(v) for v in x]) == batch['input_sha256']
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
        if replay:
            fresh_clean=prepare(clean)
            frozen_clean=[torch.from_numpy(frozen_arrays['poison_clean_'+name].copy()) for name in frozen_inputs.FIELDS[:6]]
            checks=[frozen_inputs.compare_field('poison_clean_'+name,a,b.numpy())
                    for name,a,b in zip(frozen_inputs.FIELDS,fresh_clean,frozen_clean)]
            assert all(row['contract_pass'] for row in checks)
            with torch.no_grad():
                fresh_prediction=tuple(model(*fresh_clean,state=None)[0])
                frozen_prediction=tuple(model(*frozen_clean,state=None)[0])
            prediction_errors=[]
            for a,b in zip(fresh_prediction,frozen_prediction):
                torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
                prediction_errors.append(float((a-b).abs().max()))
            from tac_osm.research_anticheat import fingerprint
            preprocess_rows.append(dict(seed=seed,current_batch0_sha256=rebuilt_digest,
                frozen_original_batch0_sha256=batch['input_sha256'],
                current_poison_clean_sha256=fingerprint(fresh_clean),
                frozen_poison_clean_sha256=frozen_receipt['poison_clean_fingerprint'],
                poison_clean_field_checks=checks,
                fresh_vs_frozen_forecast_max_abs_errors=prediction_errors,
                numerical_contract=dict(rtol=1e-5,atol=1e-6),discrete_inputs_exact=True))
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
        # This versioned comparison records, rather than hides, producer drift.
        # Normalize this single producer fingerprint only after the independently
        # authenticated fieldwise and model-output comparisons above passed.
        compared=copy.deepcopy(result)
        for row, original, evidence in zip(compared['checks'],expected['checks'],preprocess_rows):
            assert row['seed']==original['seed']==evidence['seed']
            finding=next(f for f in row['anticheat_ledger']['findings'] if f['check']=='future_invariance')
            baseline=next(f for f in original['anticheat_ledger']['findings'] if f['check']=='future_invariance')
            assert finding['status']==baseline['status']=='PASS'
            assert finding['evidence']['inputs_sha256']==evidence['current_poison_clean_sha256']
            assert baseline['evidence']['inputs_sha256']==evidence['frozen_poison_clean_sha256']
            # Comparison copies explicitly omit host-specific producer digest equality;
            # both true digests remain in the immutable originals and V2 evidence.
            finding['evidence']['inputs_sha256']='<v2-separately-validated-producer-see-report>'
            baseline['evidence']['inputs_sha256']='<v2-separately-validated-producer-see-report>'
        replay_comparison(compared,expected)
        v2_result=dict(version='PREPROCESS_REPLAY_V2',status='PASS',
            scope='versioned diagnostic replay only; original receipts and model/score contracts unchanged',
            authenticated_export_sha256=frozen_receipt['file_sha256'],
            current_producer_diagnostics=producer_diagnostics,checks=preprocess_rows,
            original_observatory_sha256=run.sha(OUT),model_and_atlas_inputs='exact frozen original tensors',
            actual_current_result=result,
            future_poison='original source producer; within-run exact invariance retained',
            cross_run_exception='future-invariance producer input fingerprint compared numerically and disclosed; no checkpoint or original batch input identity normalized')
        if V2_REPORT.exists():
            prior=json.loads(V2_REPORT.read_text())
            for key in ('version','status','authenticated_export_sha256','original_observatory_sha256'):
                assert v2_result[key]==prior[key], key
            assert len(prior['checks'])==len(preprocess_rows)==5
            for current,original in zip(preprocess_rows,prior['checks']):
                for key in ('seed','frozen_original_batch0_sha256','frozen_poison_clean_sha256','numerical_contract','discrete_inputs_exact'):
                    assert current[key]==original[key], key
                assert all(r['contract_pass'] for r in current['poison_clean_field_checks'])
            assert all(r['contract_pass'] for r in producer_diagnostics['fields'])
        else:
            run.write(V2_REPORT,v2_result)
        # Host-dependent producer diagnostics are disclosed, never overwritten
        # into the original V2 receipt or mistaken for original tensor identity.
        print(json.dumps(dict(version='PREPROCESS_REPLAY_V2',status='PASS',
            differing_producer_fields=[r for r in producer_diagnostics['fields'] if not r['exact']],
            per_model=[dict(seed=r['seed'],current_batch0_sha256=r['current_batch0_sha256'],
                frozen_original_batch0_sha256=r['frozen_original_batch0_sha256'],
                current_poison_clean_sha256=r['current_poison_clean_sha256'],
                frozen_poison_clean_sha256=r['frozen_poison_clean_sha256'],
                forecast_max_abs_errors=r['fresh_vs_frozen_forecast_max_abs_errors']) for r in preprocess_rows]),sort_keys=True))
        print('V2 replay contract: exact frozen identities/input digests/statuses; fresh producer checked separately and disclosed; within-run poison invariance exact; four runtime output digests compared within-run; numerical rtol=1e-5 atol=1e-6.')
        print('ME109_OBSERVATORY_ANTICHEAT_REPLAY_PASS')
    else:
        run.write(OUT,result)
        print('ME109_OBSERVATORY_ANTICHEAT_RECORDED')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','replay'))
    args=p.parse_args()
    if args.mode != 'replay': raise RuntimeError('V2 adapter is replay-only; original run untouched')
    execute(True)
