"""Check upstream parameter count, then exercise real SISO prefill and cached LM decode."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS = {
    'mamba_ssm.models.mixer_seq_simple': 'e79b400dd2de7fcfe59001b6247bd817d1994d780c52b0857c18446827bfe700',
    'mamba_ssm.modules.mamba3': '930c3dfa04dea8444b1c9ee8b6ac9cbbc7ef492dc5ae1f4c83051ef953eca33c',
}


def count(cfg):
    ssm = cfg['ssm_cfg']
    width = cfg['d_model']
    state = ssm['d_state']
    inner = ssm['expand'] * width
    heads = inner // ssm['headdim']
    if (inner % ssm['headdim'] or ssm['is_mimo'] or ssm['is_outproj_norm']
            or cfg['d_intermediate'] or not cfg['tie_embeddings'] or not cfg['rms_norm']):
        raise ValueError('Formula only covers the specified SISO configuration')
    rotary_pairs = int(state * ssm['rope_fraction']) // 2
    projected = 2 * inner + 2 * state * ssm['ngroups'] + 3 * heads + rotary_pairs
    matrices = width * projected + inner * width
    # Bias for time delta, D skip, B/C head biases, B/C norm weights, block input norm.
    vectors = 2 * heads + 2 * heads * state + 2 * state + width
    return cfg['vocab_size'] * width + cfg['n_layer'] * (matrices + vectors) + width


def gpu_smoke(cfg, expected):
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This smoke needs a CUDA device with native BF16 support')
    for name, digest in PINS.items():
        path = Path(importlib.import_module(name).__file__)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Unexpected upstream source for {name}')
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from mamba_ssm.utils.generation import InferenceParams
    torch.manual_seed(10301)
    model = MambaLMHeadModel(MambaConfig(**cfg), device='cuda', dtype=torch.bfloat16).eval()
    measured = sum(p.numel() for p in model.parameters())
    if measured != expected or not all(p.requires_grad for p in model.parameters()):
        raise ValueError(f'Unexpected active parameter count {measured}')
    ids = torch.randint(0, 256, (1, 68), device='cuda')
    cache = InferenceParams(max_seqlen=68, max_batch_size=1)
    errors = []
    with torch.inference_mode():
        logits = model(ids[:, :64], inference_params=cache).logits
        if logits.shape != (1, 64, 256) or not torch.isfinite(logits).all():
            raise ValueError('Invalid prefill logits')
        for position in range(64, 68):
            cache.seqlen_offset = position
            streamed = model(ids[:, position:position+1], inference_params=cache).logits
            reference = model(ids[:, :position+1], num_last_tokens=1).logits
            if streamed.shape != (1, 1, 256) or not torch.isfinite(streamed).all() or not torch.isfinite(reference).all():
                raise ValueError('Invalid single-position logits')
            errors.append((streamed.float() - reference.float()).abs().max().item())
    torch.cuda.synchronize()
    return dict(device=torch.cuda.get_device_name(), torch=torch.__version__, parameters=measured,
                cached_vs_full_max_abs=errors, cached_steps=len(errors))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true', help='Run the official CUDA model')
    args = parser.parse_args()
    config = json.loads((HERE/'config.json').read_text())
    predicted = count(config)
    print(json.dumps(gpu_smoke(config, predicted) if args.gpu else
                     dict(source_derived_parameters=predicted, gpu_executed=False), indent=2))
