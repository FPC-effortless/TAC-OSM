"""Restore authenticated original pre-update tensors instead of reseeding them."""
from pathlib import Path
import random
import torch

def restore(C, model, arm, seed, path, profile):
    registry_path = C.HERE/'ORIGINAL_INITIALIZATIONS.json'
    assert C.sha(registry_path) == profile['original_initializations_sha256']
    registry = C.read(registry_path)
    name = f'langmath101-fit-{arm}-{seed}-parameter_exposure_matched'
    record = next(r for r in registry['archives'] if r['name'] == name)
    path = Path(path)
    if path.is_dir(): path = path/name/'INITIAL_MODEL.pt'
    if path.is_symlink() or C.sha(path) != record['initial_checkpoint_sha256']:
        raise ValueError('Original zero-update checkpoint bytes differ')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    metadata = payload['metadata']
    frozen = next(r for r in profile['models'] if (r['architecture'],r['seed']) == (arm,seed))
    assert payload['optimizer'] is None
    assert (metadata['arm'],metadata['seed'],metadata['lane']) == (arm,seed,'parameter_exposure_matched')
    assert metadata['optimizer_steps'] == 0 and metadata['stage'] == 'INITIAL_BEFORE_DEV'
    assert metadata['recipe'] == C.RECIPE
    assert metadata['training_recipe_sha256'] == profile['training_recipe_sha256']
    assert metadata['tokenizer_receipt_sha256'] == profile['tokenizer_receipts_sha256'][str(seed)]
    original_sources = profile.get('instrumentation_rebinding',{}).get('original_source_hashes',profile['source_hashes'])
    assert metadata['source_hashes'] == original_sources
    assert metadata['implementation_sha256'] == C.digest(original_sources)
    model.load_state_dict(payload['model'], strict=True)
    assert C.state_digest(model) == metadata['model_sha256'] == frozen['weights_before']
    torch.random.set_rng_state(payload['rng']['torch'])
    random.setstate(payload['rng']['python'])
    # No NumPy draws occur in model construction or the fixed training graph;
    # retain setup(seed)'s complete NumPy state and verify its saved key array.
    import numpy as np
    assert np.array_equal(np.random.get_state()[1], payload['rng']['numpy_keys'].numpy())
    return {'source_run':registry['run_id'],'artifact_id':record['artifact_id'],
            'original_checkpoint_sha256':record['initial_checkpoint_sha256'],
            'original_archive_sha256':record['archive_sha256'],
            'initial_weights_sha256':frozen['weights_before']}
