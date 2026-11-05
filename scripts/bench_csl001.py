"""No-fit timing: CPU/GPU forward/step only, not quality or TinyML validation."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tac_osm.csl001 import CSL001, csl20_config, theoretical_weight_bytes


def benchmark(fn, repeats, sync):
    for _ in range(3):
        fn()
    durations = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter_ns()
        fn()
        sync()
        durations.append((time.perf_counter_ns() - start) / 1e6)
    return {
        'median_ms': round(statistics.median(durations), 4),
        'min_ms': round(min(durations), 4),
        'p90_ms': round(sorted(durations)[int((len(durations)-1)*0.9)], 4),
        'samples_ms': [round(v, 4) for v in durations],
    }


def state_bytes(state):
    return sum(t.numel() * t.element_size() for s in state.layers for t in (s.fast, s.slow, s.conv_tail))


def CSL001_default():
    from tac_osm.csl001 import CSLConfig
    return CSLConfig()


def run(device_name='cpu', seq_len=128, repeats=20, decode_steps=32, variant='eight'):
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('requested CUDA is unavailable; no CPU substitution')
    device = torch.device(device_name)
    if device_name == 'cpu':
        torch.set_num_threads(1)
    torch.manual_seed(138)
    model = CSL001(csl20_config() if variant == "twenty" else CSL001_default()).eval().to(device)
    input_tokens = torch.randint(0, 256, (1, seq_len), device=device)
    step_tokens = torch.randint(0, 256, (1, decode_steps), device=device)
    sync = torch.cuda.synchronize if device_name == 'cuda' else (lambda: None)
    result = {
        'stage': 'NO_FIT_ENGINEERING_TIMING', 'device': device_name, 'variant': variant,
        'platform': platform.platform(), 'torch': torch.__version__,
        'cuda_gpu': torch.cuda.get_device_name(0) if device_name == 'cuda' else None,
        'torch_threads': torch.get_num_threads(), 'batch': 1,
        'sequence': seq_len, 'decode_steps': decode_steps,
        'repeats': repeats, 'optimizer_steps': 0,
        'model_sha256': hashlib.sha256((ROOT/'src/tac_osm/csl001.py').read_bytes()).hexdigest(),
        'model_parameters': model.parameter_count(),
        'storage_lower_bounds_bytes': {str(bits): theoretical_weight_bytes(model.parameter_count(), bits)
                                        for bits in (32, 8, 4, 2.125)},
        'rungs': {},
        'limits': 'No real model training, int8 or CQ2 kernel, language accuracy, effective FLOPs, RAM/RSS or energy measurement. Decode uses a PyTorch FP32 loop of individual steps.',
    }
    with torch.inference_mode():
        for depth in model.config.rungs:
            state = model.initial_state(1, device, model.embedding.weight.dtype,
                                        stream_ids=('benchmark',), depth=depth)
            prefill = benchmark(lambda: model(input_tokens, depth=depth), repeats, sync)
            # Run the same loop for each repeat so state cannot leak across timing samples.
            def decode():
                s = state
                for j in range(decode_steps):
                    _, s = model(step_tokens[:, j:j+1], depth=depth, state=s,
                                 stream_ids=('benchmark',))
                return s
            dec = benchmark(decode, max(5, repeats // 2), sync)
            result['rungs'][depth] = {
                'state_bytes_batch1': state_bytes(state),
                'prefill': prefill,
                'prefill_tokens_per_second': round(seq_len / (prefill['median_ms'] / 1000), 2),
                'decode_total': dec,
                'decode_tokens_per_second': round(decode_steps / (dec['median_ms'] / 1000), 2),
            }
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--seq', type=int, default=128)
    p.add_argument('--variant', choices=['eight','twenty'], default='eight')
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--decode-steps', type=int, default=32)
    p.add_argument('--output', default=str(ROOT/'research/mathematical-engineering/csl001/LOCAL_NO_FIT_BENCHMARK.json'))
    args = p.parse_args()
    data = run(args.device, args.seq, args.repeats, args.decode_steps, args.variant)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({k:{'state_bytes_batch1':v['state_bytes_batch1'],
                         'prefill_tokens_per_second':v['prefill_tokens_per_second'],
                         'decode_tokens_per_second':v['decode_tokens_per_second']}
                      for k,v in data['rungs'].items()}, indent=2))
