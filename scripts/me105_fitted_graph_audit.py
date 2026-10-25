"""Read-only FP32/FP64 actual-source audit of the original ME105 panel.

Run only after the fixed panel checkpoints and receipts are Git-frozen. The
independent evaluator reconstructs ME105 outputs from the serial ME102
reference and does not call the model's forward method.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from scripts import me100_sensor_data as data
from scripts import me102_fitted_graph_audit as me102_audit
from scripts import me103_continuation_panel as me103_panel
from scripts import me105_protected_panel as panel
from scripts.me105_graph_reference import serial as independent_serial
from tac_osm.variants.protected_sensor_world import ProtectedSensorWorld


OUT = panel.OUT
AUDIT = OUT / 'FITTED_GRAPH_AUDIT.json'
EXPECTED_PARAMETERS = 100349
EXPECTED_PARAMETER_TENSORS = 33
EXPECTED_SEEDS = tuple(panel.SEEDS)
EXPECTED_RECIPES = tuple(panel.RECIPES)
NEW_STEPS = panel.STEPS
PARENT_STEPS = 3072


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _max_abs(actual, expected):
    return float((actual.detach() - expected.detach()).abs().max()) if actual.numel() else 0.0


def _close(name, actual, expected, *, rtol, atol, errors):
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=name)
    errors[name] = max(errors.get(name, 0.0), _max_abs(actual, expected))


def _as_dtype(batch, dtype):
    return [value.to(dtype=dtype) if value.is_floating_point() else value for value in batch]


def _state_parts(state):
    return state.source_sum, state.source_mass, state.evidence_sum, state.memory


def _parent_frozen_and_lineage(index, registration):
    """Freeze and validate all original ME103 source and parent artifacts."""
    parent_index = panel.PARENT / 'FIT_INDEX.json'
    parent_reg = panel.PARENT_DOC / 'FAIR_COMPARISON.json'
    if registration.get('parent_fit_index_sha256') != _sha(parent_index):
        raise RuntimeError('ME103 parent FIT_INDEX digest differs from ME105 registration')
    if registration.get('parent_registration_sha256') != _sha(parent_reg):
        raise RuntimeError('ME103 parent registration digest differs from ME105 registration')
    frozen = [parent_index, parent_reg]
    for seed in panel.PARENT_SEEDS:
        folder = panel.PARENT / 'strong_reconstruction' / f'seed-{seed}'
        frozen.extend((folder / 'final.pt', folder / 'FIT_RECEIPT.json', folder / 'live-budget.json'))
    me103_panel.prior.frozen(frozen)
    parent_index_json = json.loads(parent_index.read_text())
    parent_rows = {(row['architecture'], int(row['seed'])): row
                   for row in parent_index_json.get('models', [])}
    recorded = registration.get('parent_artifacts', {})
    for seed in EXPECTED_SEEDS:
        parent_seed = seed - 3000
        key = str(seed)
        if parent_seed not in panel.PARENT_SEEDS:
            raise RuntimeError(f'ME105 seed has no frozen ME103 strong parent: {seed}')
        folder = panel.PARENT / 'strong_reconstruction' / f'seed-{parent_seed}'
        cp_path = folder / 'final.pt'
        hashes = dict(checkpoint_sha256=_sha(cp_path),
                      fit_receipt_sha256=_sha(folder / 'FIT_RECEIPT.json'),
                      runtime_receipt_sha256=_sha(folder / 'live-budget.json'))
        if recorded.get(key) != hashes:
            raise RuntimeError(f'ME103 parent artifacts differ from ME105 registration: {seed}')
        row = parent_rows.get(('strong_reconstruction', parent_seed))
        if row is None or row.get('checkpoint_sha256') != hashes['checkpoint_sha256']:
            raise RuntimeError(f'ME103 FIT_INDEX parent checkpoint mismatch: {seed}')
        cp = torch.load(cp_path, map_location='cpu', weights_only=False)
        if cp.get('steps') != PARENT_STEPS or cp.get('seed') != parent_seed or cp.get('recipe') != 'strong_reconstruction':
            raise RuntimeError(f'ME103 parent checkpoint lineage mismatch: {seed}')
    return dict(parent_fit_index_sha256=_sha(parent_index),
                parent_registration_sha256=_sha(parent_reg),
                parent_artifacts=recorded)


def _audit_one(checkpoint_path, checkpoint_hash, row, d, origins, dtype):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rtol, atol = ((2e-5, 2e-6) if dtype == torch.float32 else (1e-9, 1e-10))
    cp = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = ProtectedSensorWorld()
    model.load_state_dict(cp['model'], strict=True)
    model = model.to(dtype=dtype).eval()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    deployed = sum(p.numel() for p in model.parameters())
    tensors = sum(1 for p in model.parameters() if p.requires_grad)
    if trainable != EXPECTED_PARAMETERS or deployed != EXPECTED_PARAMETERS or tensors != EXPECTED_PARAMETER_TENSORS:
        raise AssertionError(f'live model count mismatch: trainable={trainable}, deployed={deployed}, tensors={tensors}')
    if not all(p.requires_grad for p in model.parameters()):
        raise AssertionError('fitted model contains a frozen parameter')
    if cp.get('steps') != NEW_STEPS or cp.get('seed') != int(row['seed']) or cp.get('recipe') != row['architecture']:
        raise RuntimeError('ME105 checkpoint identity/step metadata mismatch')
    if cp.get('parent_checkpoint_sha256') != row.get('parent_checkpoint_sha256'):
        raise RuntimeError('ME105 checkpoint parent hash mismatch')
    states = cp.get('optimizer', {}).get('state', {})
    if len(states) != EXPECTED_PARAMETER_TENSORS or {int(state['step']) for state in states.values()} != {NEW_STEPS}:
        raise RuntimeError('ME105 fresh optimizer must hold 33 states at step 1,536')

    batch = _as_dtype([torch.from_numpy(value) for value in data.batch(d, origins, 64)], dtype)
    actual_inputs = [value.detach().clone().requires_grad_(value.is_floating_point())
                     if value.is_floating_point() else value.clone() for value in batch[:6]]
    reference_inputs = [value.detach().clone().requires_grad_(value.is_floating_point())
                        if value.is_floating_point() else value.clone() for value in batch[:6]]
    forecast, final, trace = model(*actual_inputs, return_trace=True)
    (ref_mu, ref_left, ref_right, ref_sum, ref_mass, ref_evidence, ref_memory,
     ref_trace) = independent_serial(model, *reference_inputs[:5])
    errors = {}
    for name, actual, expected in (
        ('forecast_location', forecast.log_location, ref_mu),
        ('forecast_left_scale', forecast.left_scale, ref_left),
        ('forecast_right_scale', forecast.right_scale, ref_right),
        ('state_source_sum', final.source_sum, ref_sum),
        ('state_source_mass', final.source_mass, ref_mass),
        ('state_evidence_sum', final.evidence_sum, ref_evidence),
        ('state_memory', final.memory, ref_memory),
        ('trace_width_features', trace['width_features'], ref_trace['width_features']),
        ('trace_belief_read', trace['belief_read'], ref_trace['belief_read']),
        ('trace_shared_write_key', trace['shared_write_key'], ref_trace['shared_write_key']),
        ('trace_base_log_scale', trace['base_log_scale'], ref_trace['base_log_scale']),
        ('trace_width_factors', trace['width_factors'], ref_trace['width_factors']),
        ('trace_median_correction', trace['median_correction'], ref_trace['median_correction']),
        ('trace_source_floor', trace['source_floor_scale'], ref_trace['source_floor_scale']),
        ('trace_missing_floor', trace['missing_floor_scale'], ref_trace['missing_floor_scale']),
        ('trace_protected_floor', trace['protected_floor_scale'], ref_trace['protected_floor_scale']),
    ):
        _close(name, actual, expected, rtol=rtol, atol=atol, errors=errors)

    for side in (forecast.left_scale, forecast.right_scale):
        if torch.any(side < trace['protected_floor_scale']):
            raise AssertionError('source-gap floor weakened')
    if torch.any(trace['median_correction'].abs() > 0.5):
        raise AssertionError('median innovation bound violated')

    actual_loss = forecast.log_location.square().sum() + forecast.left_scale.square().sum() + forecast.right_scale.square().sum()
    actual_grads = torch.autograd.grad(
        actual_loss, (*tuple(model.parameters()), *[x for x in actual_inputs if x.is_floating_point()]),
        retain_graph=False, allow_unused=True,
    )
    if any(grad is None or not torch.isfinite(grad).all() for grad in actual_grads):
        raise AssertionError('actual source graph has missing or nonfinite input/parameter gradients')
    ref_loss = ref_mu.square().sum() + ref_left.square().sum() + ref_right.square().sum()
    ref_grads = torch.autograd.grad(
        ref_loss, (*tuple(model.parameters()), *[x for x in reference_inputs if x.is_floating_point()]),
        allow_unused=True,
    )
    if any(grad is None or not torch.isfinite(grad).all() for grad in ref_grads):
        raise AssertionError('independent reference graph has missing or nonfinite input/parameter gradients')
    _close('gradient_mu_squared_plus_left_squared_plus_right_squared',
           torch.cat([g.reshape(-1) for g in actual_grads]),
           torch.cat([g.reshape(-1) for g in ref_grads]), rtol=rtol, atol=atol, errors=errors)

    # Independent raw-space density checks include censoring and exact median.
    target = batch[6].to(dtype).clamp_min(0)
    mask = batch[7].to(device=forecast.log_location.device, dtype=torch.bool)
    z = (torch.log1p(target) - model.target_mean) / model.target_scale
    side = torch.where(z < ref_mu, ref_left, ref_right)
    positive_nll = (0.5 * ((z - ref_mu) / side).square() + side.log()
                    + 0.5 * np.log(2 * np.pi) + model.target_scale.log()
                    + torch.log1p(target))
    threshold = -model.target_mean / model.target_scale
    threshold_scale = torch.where(threshold < ref_mu, ref_left, ref_right)
    zero_nll = -torch.special.log_ndtr((threshold - ref_mu) / threshold_scale)
    ref_nll = torch.where(target == 0, zero_nll, positive_nll)
    expected_nll = torch.where(mask, ref_nll, torch.zeros_like(ref_nll)).sum() / mask.sum().clamp_min(1)
    actual_nll = model.nll(forecast, target, mask)
    _close('censored_raw_nll', actual_nll, expected_nll, rtol=rtol, atol=atol, errors=errors)
    zero = torch.zeros_like(ref_mu)
    expected_atom = torch.special.ndtr((threshold - ref_mu) / threshold_scale)
    _close('cdf_zero_atom', model.cdf(forecast, zero), expected_atom,
           rtol=rtol, atol=atol, errors=errors)
    median = model.raw_quantiles(forecast)[..., 1]
    expected_median = torch.expm1(ref_mu * model.target_scale + model.target_mean).clamp_min(0)
    _close('default_median_quantile', median, expected_median,
           rtol=rtol, atol=atol, errors=errors)

    first_inputs = [value[:, :32] if value.ndim >= 2 else value for value in batch[:6]]
    later_inputs = [value[:, 32:] if value.ndim >= 2 else value for value in batch[:6]]
    _, first_state = model(*first_inputs)
    chunked, chunk_state, _ = model(*later_inputs, state=first_state, return_trace=True)
    for name, actual, expected in (
        ('chunk_location', forecast.log_location, chunked.log_location),
        ('chunk_left_scale', forecast.left_scale, chunked.left_scale),
        ('chunk_right_scale', forecast.right_scale, chunked.right_scale),
    ):
        _close(name, actual, expected, rtol=rtol, atol=atol, errors=errors)
    for name, actual, expected in zip(('source_sum', 'source_mass', 'evidence_sum', 'memory'),
                                      _state_parts(final), _state_parts(chunk_state)):
        _close(f'chunk_state_{name}', actual, expected, rtol=rtol, atol=atol, errors=errors)

    return dict(architecture=row['architecture'], seed=int(row['seed']),
                dtype='float32' if dtype == torch.float32 else 'float64',
                checkpoint_sha256=checkpoint_hash,
                parent_checkpoint_sha256=row['parent_checkpoint_sha256'],
                trainable_parameters=trainable, deployed_parameters=deployed,
                parameter_tensors=tensors, evaluated_origins=[list(map(int, origin)) for origin in origins],
                history=64, maximum_absolute_errors=errors,
                all_actual_input_and_parameter_gradients_present=True,
                all_reference_input_and_parameter_gradients_present=True,
                coherent_median_quantile=True, raw_nll_and_zero_atom_match=True,
                chunked_outputs_and_state_match=True)


def replay_identity(artifact):
    """Ignore only hardware-varying error magnitudes after live gates pass."""
    return me102_audit.replay_identity(artifact)


def run(mode):
    registration = json.loads(panel.REG.read_text())
    # Verify Git originals before the panel helper deserializes any fitted child.
    lineage = _parent_frozen_and_lineage(None, registration)
    index = panel.budget_audit()
    expected = {(recipe, seed) for recipe in EXPECTED_RECIPES for seed in EXPECTED_SEEDS}
    rows = index.get('models', [])
    seen = {(row.get('architecture'), int(row.get('seed', -1))) for row in rows}
    if len(rows) != 15 or seen != expected or len(seen) != len(rows):
        raise RuntimeError('ME105 fitted bundle must contain exactly all 15 recipe/seed originals')
    d = data.load('fit')
    origins = me102_audit._origins(d)
    observations = []
    for row in sorted(rows, key=lambda value: (value['architecture'], int(value['seed']))):
        recipe, seed = row['architecture'], int(row['seed'])
        folder = OUT / recipe / f'seed-{seed}'
        checkpoint_path = folder / 'final.pt'
        checkpoint_hash = _sha(checkpoint_path)
        if checkpoint_hash != row.get('checkpoint_sha256'):
            raise RuntimeError(f'ME105 checkpoint SHA mismatch: {recipe}/{seed}')
        receipt = json.loads((folder / 'FIT_RECEIPT.json').read_text())
        parent_seed = seed - 3000
        parent_hash = registration['parent_artifacts'][str(seed)]['checkpoint_sha256']
        if (receipt.get('architecture') != recipe or int(receipt.get('seed', -1)) != seed
                or receipt.get('actual_steps') != NEW_STEPS
                or receipt.get('optimizer_state_tensors') != EXPECTED_PARAMETER_TENSORS
                or receipt.get('parent_checkpoint_sha256') != parent_hash
                or row.get('parent_checkpoint_sha256') != parent_hash):
            raise RuntimeError(f'ME105 fit receipt/parent lineage metadata mismatch: {recipe}/{seed}')
        for dtype in (torch.float32, torch.float64):
            observations.append(_audit_one(checkpoint_path, checkpoint_hash, row, d, origins, dtype))
    artifact = dict(
        schema='tac-osm.me105.fitted_graph_audit.v1',
        status='PASS_SOURCE_REFERENCE_PARITY_ON_FROZEN_ME105_WEIGHTS',
        fit_source_sha256=data.SHA,
        registration_sha256=_sha(panel.REG),
        fit_index_sha256=_sha(OUT / 'FIT_INDEX.json'),
        parent_lineage=lineage,
        parent_variant='SHARED-RECONSTRUCTION-ME103-STRONG-RECONSTRUCTION',
        models=15, recipe_count=len(EXPECTED_RECIPES), seed_count=len(EXPECTED_SEEDS),
        parent_optimizer_steps=PARENT_STEPS, child_fresh_optimizer_steps=NEW_STEPS,
        trainable_parameters=EXPECTED_PARAMETERS, deployed_parameters=EXPECTED_PARAMETERS,
        parameter_tensors=EXPECTED_PARAMETER_TENSORS,
        dtype_checks=['float32', 'float64'],
        tolerances={'float32': {'rtol': 2e-5, 'atol': 2e-6},
                    'float64': {'rtol': 1e-9, 'atol': 1e-10}},
        forecast_origins=[list(map(int, origin)) for origin in origins],
        checks=observations,
        maximum_absolute_errors={
            key: max(row['maximum_absolute_errors'].get(key, 0.0) for row in observations)
            for key in sorted({key for row in observations for key in row['maximum_absolute_errors']})
        },
        no_training=True, no_score_generation=True, no_original_artifact_overwrite=True,
    )
    encoded = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if mode == 'replay':
        if not AUDIT.exists():
            raise RuntimeError('no saved ME105 fitted graph audit exists to replay')
        saved = json.loads(AUDIT.read_text())
        if replay_identity(saved) != replay_identity(artifact):
            raise RuntimeError('ME105 fitted graph audit replay provenance differs')
        print('ME105_FITTED_GRAPH_AUDIT_REPLAY=PASS')
    else:
        if AUDIT.exists():
            raise RuntimeError('ME105 fitted graph audit exists; use replay mode')
        with AUDIT.open('x') as stream:
            stream.write(encoded)
        print('ME105_FITTED_GRAPH_AUDIT=PASS')
    print(json.dumps(dict(models=len(rows), dtype_runs=len(observations),
                          maximum_absolute_errors=artifact['maximum_absolute_errors']), sort_keys=True))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', default='audit', choices=('audit', 'replay'))
    run(parser.parse_args().mode)
