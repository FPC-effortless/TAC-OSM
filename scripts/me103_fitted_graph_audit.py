"""Read-only FP32/FP64 graph audit of all frozen ME103 continuations.

This reuses ME102's source-equivalence implementation because ME103 keeps the
same architecture and graph. It also verifies each child's exact ME102 parent
and cumulative optimizer step lineage before deserializing child weights.
"""
from pathlib import Path
import hashlib
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from scripts import me101_belief_sensor as prior
from scripts import me102_fitted_graph_audit as me102_audit
from scripts import me100_sensor_data as data
from scripts import me103_continuation_panel as panel


OUT = panel.OUT
AUDIT = OUT / 'FITTED_GRAPH_AUDIT.json'
EXPECTED_PARAMETERS = me102_audit.EXPECTED_PARAMETERS
EXPECTED_PARAMETER_TENSORS = me102_audit.EXPECTED_PARAMETER_TENSORS
EXPECTED_SEEDS = tuple(panel.SEEDS)
EXPECTED_RECIPES = tuple(panel.RECIPES)
TOTAL_STEPS = panel.TOTAL
PARENT = panel.PARENT
PARENT_DOC = panel.PARENT_DOC


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_complete_index(index):
    expected = {(recipe, seed) for recipe in EXPECTED_RECIPES for seed in EXPECTED_SEEDS}
    rows = index.get('models', [])
    seen = {(row.get('architecture'), int(row.get('seed', -1))) for row in rows}
    if len(rows) != 15 or seen != expected or len(seen) != len(rows):
        raise RuntimeError('fitted bundle must contain exactly all 15 ME103 lineages')


def _check_parent_lineage(index, registration):
    """Check original parent hashes and cumulative optimizer state for all arms."""
    parent_index_path = PARENT / 'FIT_INDEX.json'
    parent_reg_path = PARENT_DOC / 'FAIR_COMPARISON.json'
    if registration.get('parent_fit_index_sha256') != _sha(parent_index_path):
        raise RuntimeError('ME102 parent FIT_INDEX hash differs from ME103 registration')
    if registration.get('parent_registration_sha256') != _sha(parent_reg_path):
        raise RuntimeError('ME102 parent registration hash differs from ME103 registration')

    # Parent artifacts must themselves be committed originals before any child
    # checkpoint is loaded for numerical comparison.
    parent_files = [parent_index_path, parent_reg_path]
    for recipe in EXPECTED_RECIPES:
        for seed in EXPECTED_SEEDS:
            cp, guard, receipt = panel.parent_paths(recipe, seed)
            parent_files.extend((cp, guard, receipt))
    prior.frozen(parent_files)

    parent_index = json.loads(parent_index_path.read_text())
    parent_rows = {
        (row['architecture'], int(row['seed'])): row
        for row in parent_index.get('models', [])
    }
    registered = registration.get('parent_artifacts', {})
    if len(parent_rows) != 15:
        raise RuntimeError('ME102 parent index is not the complete original 15-model panel')

    for row in index['models']:
        recipe, seed = row['architecture'], int(row['seed'])
        key = f'{recipe}/{seed}'
        if (recipe, seed) not in parent_rows:
            raise RuntimeError(f'missing ME102 parent row: {key}')
        parent_cp, parent_guard, parent_receipt = panel.parent_paths(recipe, seed)
        current_parent = {
            'checkpoint_sha256': _sha(parent_cp),
            'runtime_receipt_sha256': _sha(parent_guard),
            'fit_receipt_sha256': _sha(parent_receipt),
        }
        if registered.get(key) != current_parent:
            raise RuntimeError(f'ME102 parent artifacts differ from registered hashes: {key}')
        if current_parent['checkpoint_sha256'] != parent_rows[(recipe, seed)].get('checkpoint_sha256'):
            raise RuntimeError(f'ME102 parent checkpoint differs from its FIT_INDEX row: {key}')

        child_path = OUT / recipe / f'seed-{seed}' / 'final.pt'
        child = torch.load(child_path, map_location='cpu', weights_only=False)
        parent = torch.load(parent_cp, map_location='cpu', weights_only=False)
        if child.get('parent_checkpoint_sha256') != current_parent['checkpoint_sha256']:
            raise RuntimeError(f'ME103 child names a different ME102 parent: {key}')
        if child.get('additional_steps') != panel.STEPS or child.get('steps') != TOTAL_STEPS:
            raise RuntimeError(f'ME103 child cumulative step metadata mismatch: {key}')
        if child.get('parent_optimizer_steps') != panel.STEPS:
            raise RuntimeError(f'ME103 child parent optimizer step metadata mismatch: {key}')
        if child.get('seed') != seed or child.get('recipe') != recipe:
            raise RuntimeError(f'ME103 child checkpoint lineage metadata mismatch: {key}')
        if parent.get('steps') != panel.STEPS or parent.get('seed') != seed or parent.get('recipe') != recipe:
            raise RuntimeError(f'ME102 parent checkpoint lineage metadata mismatch: {key}')
        parent_steps = {int(state['step']) for state in parent['optimizer']['state'].values()}
        child_steps = {int(state['step']) for state in child['optimizer']['state'].values()}
        if len(parent['optimizer']['state']) != EXPECTED_PARAMETER_TENSORS or parent_steps != {panel.STEPS}:
            raise RuntimeError(f'ME102 parent optimizer state is not complete at 1536: {key}')
        if len(child['optimizer']['state']) != EXPECTED_PARAMETER_TENSORS or child_steps != {TOTAL_STEPS}:
            raise RuntimeError(f'ME103 child optimizer state is not complete at 3072: {key}')
        if child.get('source_sha256') != data.SHA:
            raise RuntimeError(f'ME103 child FIT source hash mismatch: {key}')
        if child.get('source_files') != index.get('source_files'):
            raise RuntimeError(f'ME103 child source hash map differs from frozen index: {key}')
        if parent.get('source_sha256') != data.SHA:
            raise RuntimeError(f'ME102 parent FIT source hash mismatch: {key}')
        if parent.get('source_files') != {
                path: index['source_files'].get(path) for path in parent.get('source_files', {})}:
            raise RuntimeError(f'ME102 parent source map differs from unchanged ME103 sources: {key}')
        if (row.get('trainable_parameters') != EXPECTED_PARAMETERS
                or row.get('deployed_parameters') != EXPECTED_PARAMETERS):
            raise RuntimeError(f'ME103 parameter count differs from ME102 architecture: {key}')


def run(mode):
    # budget_audit first requires all child checkpoints and receipts to be
    # committed in Git, checks source hashes, and validates the live receipts.
    index = panel.budget_audit()
    _check_complete_index(index)
    registration = json.loads(panel.REG.read_text())
    _check_parent_lineage(index, registration)

    fit = data.load('fit')
    origins = me102_audit._origins(fit)
    observations = []
    for row in sorted(index['models'], key=lambda value: (value['architecture'], value['seed'])):
        recipe, seed = row['architecture'], int(row['seed'])
        folder = OUT / recipe / f'seed-{seed}'
        checkpoint_path = folder / 'final.pt'
        checkpoint_hash = _sha(checkpoint_path)
        if checkpoint_hash != row.get('checkpoint_sha256'):
            raise RuntimeError(f'ME103 checkpoint digest mismatch: {recipe}/{seed}')
        receipt = json.loads((folder / 'FIT_RECEIPT.json').read_text())
        if (receipt.get('architecture') != recipe or int(receipt.get('seed', -1)) != seed
                or receipt.get('parameter_tensors') not in (None, EXPECTED_PARAMETER_TENSORS)
                or receipt.get('cumulative_optimizer_steps') != TOTAL_STEPS
                or receipt.get('optimizer_final_steps') != TOTAL_STEPS
                or receipt.get('parent_checkpoint_sha256') != row.get('parent_checkpoint_sha256')):
            raise RuntimeError(f'ME103 fit receipt lineage metadata mismatch: {recipe}/{seed}')
        for dtype in (torch.float32, torch.float64):
            observations.append(me102_audit._audit_one(
                checkpoint_path, checkpoint_hash, recipe, seed, fit, origins, dtype
            ))

    artifact = dict(
        schema='tac-osm.me103.fitted_graph_audit.v1',
        status='PASS_UNCHANGED_ME102_GRAPH_ON_FROZEN_CONTINUED_WEIGHTS',
        fit_source_sha256=data.SHA,
        parent_variant='SHARED-RECONSTRUCTION-ME102',
        parent_tac_source_commit='bd7fe3c08f9979d34483d27dcd048eb5cec984e3',
        parent_pnds_commit='424938148dd2f81ef03022baab0bc838943d9a81',
        models=15,
        recipe_count=len(EXPECTED_RECIPES),
        seed_count=len(EXPECTED_SEEDS),
        cumulative_optimizer_steps=TOTAL_STEPS,
        trainable_parameters=EXPECTED_PARAMETERS,
        dtype_checks=['float32', 'float64'],
        tolerances={'float32': {'rtol': 2e-5, 'atol': 2e-6},
                    'float64': {'rtol': 1e-9, 'atol': 1e-10}},
        forecast_origins=observations[0]['evaluated_origins'],
        parent_artifact_hashes=registration['parent_artifacts'],
        checks=observations,
        maximum_absolute_errors={
            key: max(row['maximum_absolute_errors'].get(key, 0.0) for row in observations)
            for key in sorted({k for row in observations for k in row['maximum_absolute_errors']})
        },
        no_training=True,
        no_parent_artifact_overwrite=True,
    )
    encoded = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if mode == 'replay':
        if not AUDIT.exists():
            raise RuntimeError('no saved ME103 fitted graph audit exists to replay')
        saved = json.loads(AUDIT.read_text())
        # Fresh live tensor comparisons above use ME102's unchanged tolerances;
        # this identity check preserves provenance while allowing diagnostic
        # max-error values to vary across CPU implementations.
        if me102_audit.replay_identity(saved) != me102_audit.replay_identity(artifact):
            raise RuntimeError('ME103 fitted graph audit replay provenance differs')
        print('ME103_FITTED_GRAPH_AUDIT_REPLAY=PASS')
    else:
        if AUDIT.exists():
            raise RuntimeError('ME103 fitted graph audit already exists; use replay mode')
        with AUDIT.open('x') as stream:
            stream.write(encoded)
        print('ME103_FITTED_GRAPH_AUDIT=PASS')
    print(json.dumps(dict(models=len(index['models']), dtype_runs=len(observations),
                          maximum_absolute_errors=artifact['maximum_absolute_errors']), sort_keys=True))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', default='audit', choices=('audit', 'replay'))
    run(parser.parse_args().mode)
