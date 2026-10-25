"""Read-only source-equivalence audit for the frozen ME102 fitted panel.

Run ``python scripts/me102_fitted_graph_audit.py audit`` once after the
original 15 checkpoints have been committed. ``replay`` reruns the same
numerical checks against those immutable files and verifies the saved audit.
Neither mode trains or writes into an original checkpoint/score artifact.
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
from scripts import me102_reconstruction_panel as panel
from scripts.me102_graph_reference import serial as independent_serial
from tac_osm.variants.reconstruction_sensor_world import ReconstructionSensorWorld


OUT = panel.OUT
AUDIT = OUT / 'FITTED_GRAPH_AUDIT.json'
EXPECTED_PARAMETERS = 98495
EXPECTED_PARAMETER_TENSORS = 29
EXPECTED_SEEDS = tuple(range(102001, 102006))
EXPECTED_RECIPES = tuple(panel.RECIPES)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _max_abs(a, b):
    if a.numel() == 0:
        return 0.0
    return float((a.detach() - b.detach()).abs().max())


def _close(name, actual, expected, *, rtol, atol, errors):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=name)
        errors[name] = max(errors.get(name, 0.0), _max_abs(actual, expected))
        return
    if len(actual) != len(expected):
        raise AssertionError(f'{name}: gradient list length changed')
    for i, (a, b) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol, msg=f'{name}[{i}]')
        errors[name] = max(errors.get(name, 0.0), _max_abs(a, b))


def _origins(d):
    # Spread selections through each fixed FIT cohort while preserving two
    # general and two natural-gap rows. This selection uses no outcomes.
    def pair(rows):
        if len(rows) < 2:
            raise RuntimeError('FIT cohort has fewer than two origins')
        return [rows[0], rows[len(rows) // 2]]
    return pair(d['general']) + pair(d['gap'])


def _as_dtype(batch, dtype):
    return [x.to(dtype=dtype) if x.is_floating_point() else x for x in batch]


def _state_parts(state):
    return (state.source_sum, state.source_mass, state.evidence_sum, state.memory)


def _audit_one(checkpoint_path, checkpoint_hash, recipe, seed, d, origins, dtype):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dtype_name = 'float32' if dtype == torch.float32 else 'float64'
    if dtype == torch.float32:
        rtol, atol = 2e-5, 2e-6
    else:
        rtol, atol = 1e-9, 1e-10
    cp = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = ReconstructionSensorWorld()
    model.load_state_dict(cp['model'], strict=True)
    model = model.to(dtype=dtype).eval()
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameter_count != EXPECTED_PARAMETERS:
        raise AssertionError(f'expected {EXPECTED_PARAMETERS} trainable parameters, got {parameter_count}')
    parameter_tensors = sum(1 for p in model.parameters() if p.requires_grad)
    if parameter_tensors != EXPECTED_PARAMETER_TENSORS:
        raise AssertionError(
            f'expected {EXPECTED_PARAMETER_TENSORS} trainable tensors, got {parameter_tensors}'
        )

    x = _as_dtype([torch.from_numpy(v) for v in data.batch(d, origins, 64)], dtype)
    prediction, final, trace = model(*x[:6], return_trace=True)
    ref_loc, ref_scale, ref_sum, ref_mass, ref_evidence, ref_memory, ref_trace = independent_serial(
        model, *x[:5]
    )
    errors = {}
    _close('forecast_location', prediction.log_location, ref_loc, rtol=rtol, atol=atol, errors=errors)
    _close('forecast_scale', prediction.log_scale, ref_scale, rtol=rtol, atol=atol, errors=errors)
    for name, actual, expected in (
        ('state_source_sum', final.source_sum, ref_sum),
        ('state_source_mass', final.source_mass, ref_mass),
        ('state_evidence_sum', final.evidence_sum, ref_evidence),
        ('state_memory', final.memory, ref_memory),
        ('trace_belief_read', trace['belief_read'], ref_trace['belief_read']),
        ('trace_shared_write_key', trace['shared_write_key'], ref_trace['shared_write_key']),
        ('trace_current_mean', trace['current_mean'], ref_trace['current_mean']),
        ('trace_current_variance', trace['current_variance'], ref_trace['current_variance']),
    ):
        _close(name, actual, expected, rtol=rtol, atol=atol, errors=errors)

    source_loss = prediction.log_location.square().sum() + prediction.log_scale.square().sum()
    source_grads = torch.autograd.grad(source_loss, tuple(model.parameters()), retain_graph=False, allow_unused=True)
    if any(g is None for g in source_grads):
        missing = [name for (name, _), grad in zip(model.named_parameters(), source_grads) if grad is None]
        raise AssertionError(f'actual source graph has unused trainable parameters: {missing}')

    ref_loc2, ref_scale2, *_ = independent_serial(model, *x[:5])
    ref_loss = ref_loc2.square().sum() + ref_scale2.square().sum()
    ref_grads = torch.autograd.grad(ref_loss, tuple(model.parameters()), allow_unused=True)
    if any(g is None for g in ref_grads):
        missing = [name for (name, _), grad in zip(model.named_parameters(), ref_grads) if grad is None]
        raise AssertionError(f'independent reference graph has unused trainable parameters: {missing}')
    _close('gradient_location_squared_plus_scale_squared', source_grads, ref_grads,
           rtol=rtol, atol=atol, errors=errors)

    first_inputs = [v[:, :32] if v.ndim >= 2 else v for v in x[:6]]
    later_inputs = [v[:, 32:] if v.ndim >= 2 else v for v in x[:6]]
    _, first_state = model(*first_inputs)
    chunked, chunk_state, _ = model(*later_inputs, state=first_state, return_trace=True)
    _close('chunk_forecast_location', prediction.log_location, chunked.log_location,
           rtol=rtol, atol=atol, errors=errors)
    _close('chunk_forecast_scale', prediction.log_scale, chunked.log_scale,
           rtol=rtol, atol=atol, errors=errors)
    for name, actual, expected in zip(('source_sum', 'source_mass', 'evidence_sum', 'memory'),
                                      _state_parts(final), _state_parts(chunk_state)):
        _close(f'chunk_state_{name}', actual, expected, rtol=rtol, atol=atol, errors=errors)

    return dict(
        recipe=recipe,
        seed=int(seed),
        dtype=dtype_name,
        checkpoint_sha256=checkpoint_hash,
        trainable_parameters=parameter_count,
        parameter_tensors=parameter_tensors,
        evaluated_origins=[list(map(int, row)) for row in origins],
        history=64,
        maximum_absolute_errors=errors,
        all_trainable_gradients_present=True,
        chunked_forward_matches=True,
    )


def run(mode):
    # budget_audit() verifies every original checkpoint/receipt against Git
    # and the preregistered hashes before any checkpoint is deserialized.
    index = panel.budget_audit()
    expected = {(recipe, seed) for recipe in EXPECTED_RECIPES for seed in EXPECTED_SEEDS}
    rows = index.get('models', [])
    seen = {(row['architecture'], int(row['seed'])) for row in rows}
    if len(rows) != 15 or seen != expected or len(seen) != len(rows):
        raise RuntimeError('fitted bundle must contain the exact original 3 recipe x 5 seed panel')

    fit = data.load('fit')
    origins = _origins(fit)
    observations = []
    for row in sorted(rows, key=lambda value: (value['architecture'], value['seed'])):
        recipe, seed = row['architecture'], int(row['seed'])
        folder = OUT / recipe / f'seed-{seed}'
        checkpoint_path = folder / 'final.pt'
        checkpoint_hash = _sha(checkpoint_path)
        if checkpoint_hash != row['checkpoint_sha256']:
            raise RuntimeError(f'checkpoint digest mismatch: {recipe}/{seed}')
        receipt = json.loads((folder / 'FIT_RECEIPT.json').read_text())
        if (receipt.get('architecture') != recipe or int(receipt.get('seed', -1)) != seed
                or receipt.get('parameter_tensors') != EXPECTED_PARAMETER_TENSORS):
            raise RuntimeError(f'original receipt metadata mismatch: {recipe}/{seed}')
        for dtype in (torch.float32, torch.float64):
            observations.append(_audit_one(
                checkpoint_path, checkpoint_hash, recipe, seed, fit, origins, dtype
            ))

    artifact = dict(
        schema='tac-osm.me102.fitted_graph_audit.v1',
        status='PASS_SOURCE_EQUIVALENCE_ON_FROZEN_FIT_WEIGHTS',
        fit_source_sha256=data.SHA,
        original_panel_models=15,
        recipe_count=3,
        seed_count=5,
        dtype_checks=['float32', 'float64'],
        tolerances={'float32': {'rtol': 2e-5, 'atol': 2e-6},
                    'float64': {'rtol': 1e-9, 'atol': 1e-10}},
        forecast_origins=observations[0]['evaluated_origins'],
        checks=observations,
        maximum_absolute_errors={
            key: max(row['maximum_absolute_errors'].get(key, 0.0) for row in observations)
            for key in sorted({k for row in observations for k in row['maximum_absolute_errors']})
        },
        no_training=True,
        no_original_artifact_overwrite=True,
    )
    encoded = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if mode == 'replay':
        if not AUDIT.exists():
            raise RuntimeError('no saved fitted graph audit exists to replay')
        saved = json.loads(AUDIT.read_text())
        if saved != artifact:
            raise RuntimeError('fitted graph audit replay differs from saved evidence')
        print('ME102_FITTED_GRAPH_AUDIT_REPLAY=PASS')
    else:
        if AUDIT.exists():
            raise RuntimeError('fitted graph audit already exists; use replay mode')
        AUDIT.write_text(encoded)
        print('ME102_FITTED_GRAPH_AUDIT=PASS')
    print(json.dumps(dict(models=len(rows), dtype_runs=len(observations),
                          maximum_absolute_errors=artifact['maximum_absolute_errors']), sort_keys=True))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', default='audit', choices=('audit', 'replay'))
    run(parser.parse_args().mode)
