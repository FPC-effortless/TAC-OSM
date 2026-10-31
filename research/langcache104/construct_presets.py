"""Instantiate preset sizes and check actual compact inference cache; zero fitting."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.langcache104.model import Config104, DeepCacheTAC104


def construct():
    torch.set_num_threads(2)
    declaration = json.loads((Path(__file__).with_name('configs.json')).read_text())
    rows = {}
    for name, entry in declaration['presets'].items():
        torch.manual_seed(104)
        config = Config104(**entry['config'])
        model = DeepCacheTAC104(config).eval()
        counts = model.counts()
        for key in ('trainable', 'deployed'):
            target = entry['target_'+key+'_parameters']
            if abs(counts[key]-target) / target > entry['relative_tolerance']:
                raise ValueError('construction count exceeds declared band')
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        with torch.inference_mode():
            logits, state = model(torch.arange(config.window+3).remainder(256)[None], owners=('construction-only',))
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('nonfinite constructed output')
        logical = state.state_bytes()
        physical = sum(t.untyped_storage().nbytes() for layer in state.layers for t in layer.tensors().values())
        expected = config.n_layers*config.n_heads*(2*(config.value_dim*config.key_dim+config.key_dim)+config.value_dim+(config.window-1)*(config.key_dim+config.value_dim))*4
        if logical != physical or logical != expected:
            raise ValueError('persistent tensor storage accounting mismatch')
        if not all(torch.equal(before[key], value) for key, value in model.state_dict().items()):
            raise ValueError('no-fit construction modified source weights')
        rows[name] = {'config': asdict(config), 'counts': counts,
            'inference_output_shape': list(logits.shape), 'observed_bytes': config.window+3,
            'fp32_state_tensor_bytes_per_stream': logical,
            'fp32_underlying_state_storage_bytes_per_stream': physical,
            'metadata_and_python_overhead_bytes': 'UNMEASURED_EXCLUDED',
            'exact_store_bytes': 'SEPARATE_EXPLICIT_COMPONENT_NOT_IN_NEURAL_LOGITS',
            'optimizer_updates': 0, 'parameters_and_buffers_unchanged': True}
    return {'schema': declaration['schema'], 'status': 'CONSTRUCTED_NO_FIT', 'hardware': 'CPU',
            'torch_version': torch.__version__, 'configs_sha256': hashlib.sha256(Path(__file__).with_name('configs.json').read_bytes()).hexdigest(),
            'presets': rows, 'training_authorized': False}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = construct()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({name: {'parameters': row['counts']['trainable'],
          'state_bytes': row['fp32_state_tensor_bytes_per_stream']}
          for name, row in result['presets'].items()}))
