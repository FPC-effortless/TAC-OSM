"""LANGEFF108: source-faithful CPU profiling, no training or model selection.

Benchmarks the unchanged LANGDEPTH106 four-layer TAC and LANGBASE107
Transformer graphs at matched parameter scale. Synthetic token IDs are timing
fixtures only. Random initialization is NOT a learned-quality benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.profiler import ProfilerActivity, profile, record_function

from research.langcache104 import model as tac_module
from research.langcache104.model import Config104, DeepCacheTAC104
from research.langbase107.model import CausalTransformer107, TransformerConfig107

TAC_CFG = dict(d_model=120, n_layers=4, n_heads=4, key_dim=16,
               value_dim=16, write_rank=2, ffn_dim=224, window=128,
               chunk_size=32, rotary=True, local_rope=True, tied_embedding=True)


def measured(fn, warmups=3, repeats=9):
    # Paired runner timing, without profiler instrumentation.
    for _ in range(warmups):
        fn()
    values = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        values.append((time.perf_counter_ns() - start) * 1e-6)
    return dict(median_ms=statistics.median(values), mean_ms=statistics.mean(values),
                min_ms=min(values), max_ms=max(values), samples_ms=values)


def stack_state_bytes_tac(state):
    return state.tensor_bytes()


def build_model(name):
    if name == 'tac':
        model = DeepCacheTAC104(Config104(**TAC_CFG))
        assert sum(p.numel() for p in model.parameters()) == 998024
    else:
        model = CausalTransformer107(TransformerConfig107())
        assert sum(p.numel() for p in model.parameters()) == 998816
    return model.cpu()


def bench_model(name, seed=10801):
    torch.manual_seed(seed)
    model = build_model(name)
    params = sum(p.numel() for p in model.parameters())
    # Separate single-stream and batch training fixtures, always 128 tokens.
    train_tokens = torch.randint(0, 256, (8, 128), dtype=torch.long)
    targets = torch.randint(0, 256, (8, 128), dtype=torch.long)
    prefix = torch.randint(0, 256, (1, 128), dtype=torch.long)
    step_token = torch.randint(0, 256, (1, 1), dtype=torch.long)

    model.eval()
    with torch.inference_mode():
        if name == 'tac':
            def prefill():
                return model(train_tokens, mode='chunk')
            owners = ('owner-0',)
            _, initial_state = model(prefix, owners=owners, mode='chunk')
            carried_bytes = stack_state_bytes_tac(initial_state)
            def decode_series(n=16):
                state = initial_state
                for _ in range(n):
                    _, state = model.step(step_token, state=state, owners=owners)
                return state
        else:
            def prefill():
                return model(train_tokens)
            _, initial_state = model(prefix, use_cache=True)
            carried_bytes = initial_state.tensor_bytes()
            def decode_series(n=16):
                cache = initial_state
                for _ in range(n):
                    _, cache = model(step_token, cache=cache, use_cache=True)
                return cache
        prefill_timing = measured(prefill, warmups=3, repeats=9)
        # The first token at context 128 is deliberately included in every series.
        decode_timing = measured(lambda: decode_series(16), warmups=2, repeats=7)
        # Costs below are end-to-end including validation, cache update, projections.
        decode_timing['ms_per_byte'] = decode_timing['median_ms'] / 16
        decode_timing['bytes_per_second'] = 16000 / decode_timing['median_ms']

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, betas=(0.9, 0.95), weight_decay=0.1)
    def train_update():
        optimizer.zero_grad(set_to_none=True)
        if name == 'tac':
            logits, _ = model(train_tokens, mode='chunk')
        else:
            logits, _ = model(train_tokens)
        loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        return float(loss.detach())
    train_timing = measured(train_update, warmups=2, repeats=7)
    # Stopwatch measurement is complete. The profiler below adds overhead.
    return {
        'params': params, 'state_bytes_per_stream_at_128_prefix': carried_bytes,
        'batch8_sequence128_prefill': prefill_timing,
        'batch1_16_recurrent_decode': decode_timing,
        'batch8_sequence128_adamw_training_update': train_timing,
    }


@contextmanager
def annotated_tac(model):
    """Annotate unmodified computation to attribute inclusive CPU profiler costs."""
    originals = []
    def wrap(obj, name, label):
        original = getattr(obj, name)
        def labeled(*args, **kwargs):
            with record_function(label):
                return original(*args, **kwargs)
        setattr(obj, name, labeled)
        originals.append((obj, name, original))
    try:
        wrap(model, 'validate', 'TAC/validation')
        for i, block in enumerate(model.blocks):
            wrap(block.fast, 'forward', 'TAC/fast_bank')
            wrap(block.slow, 'forward', 'TAC/slow_bank')
            wrap(block, 'local_attention', 'TAC/local_attention')
        wrap(tac_module, 'rotary_scan', 'TAC/rotary_scan')
        yield
    finally:
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)


def op_profile(name, phase='prefill'):
    torch.manual_seed(10801)
    model = build_model(name).eval()
    tokens = torch.randint(0, 256, (8, 128), dtype=torch.long)
    prefix = torch.randint(0, 256, (1, 128), dtype=torch.long)
    step = torch.randint(0, 256, (1, 1), dtype=torch.long)
    with torch.inference_mode():
        if phase == 'decode':
            if name == 'tac':
                owners = ('owner-0',)
                _, state = model(prefix, owners=owners)
                def run():
                    return model.step(step, state=state, owners=owners)
            else:
                _, state = model(prefix, use_cache=True)
                def run():
                    return model(step, cache=state, use_cache=True)
        else:
            def run():
                if name == 'tac':
                    return model(tokens, mode='chunk')
                return model(tokens)
        for _ in range(2):
            run()
        from contextlib import nullcontext
        annotated = annotated_tac(model) if name == 'tac' else nullcontext()
        with annotated:
            with profile(activities=[ProfilerActivity.CPU],
                         record_shapes=False, profile_memory=True) as p:
                with record_function('PROFILE/forward'):
                    run()
    keys = p.key_averages()
    labels = []
    for e in keys:
        if e.key.startswith('TAC/'):
            labels.append(dict(name=e.key, calls=e.count,
                               inclusive_ms=e.cpu_time_total / 1000.0,
                               self_ms=e.self_cpu_time_total / 1000.0))
    labels.sort(key=lambda x: -x['inclusive_ms'])
    ops = sorted((e for e in keys if e.key.startswith('aten::')),
                 key=lambda x: -x.self_cpu_time_total)[:20]
    return dict(phase=phase, labels=labels,
                top_ops=[dict(name=e.key, calls=e.count,
                              self_cpu_ms=e.self_cpu_time_total / 1000.0,
                              total_cpu_ms=e.cpu_time_total / 1000.0,
                              self_cpu_memory_bytes=e.self_cpu_memory_usage)
                         for e in ops])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='ci-artifacts/langeff108/PROFILE.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    result = {
        'schema': 'langeff108.cpu.no-fit.v1',
        'scope': 'CPU ONLY; no fitted checkpoints, no learning/quality claims',
        'device': platform.platform(),
        'processor': platform.processor(),
        'torch': torch.__version__,
        'cpu_threads': torch.get_num_threads(),
        'cuda_available': torch.cuda.is_available(),
        'models': {},
        'operator_profiles': {},
    }
    for name in ('tac', 'transformer'):
        result['models'][name] = bench_model(name)
        result['operator_profiles'][name] = {
            'prefill': op_profile(name, 'prefill'),
            'decode': op_profile(name, 'decode'),
        }
        print('MODEL', name, json.dumps(result['models'][name], sort_keys=True), flush=True)
        print('OPERATORS', name, json.dumps(result['operator_profiles'][name], sort_keys=True), flush=True)
    for phase in ('batch8_sequence128_prefill',
                  'batch1_16_recurrent_decode',
                  'batch8_sequence128_adamw_training_update'):
        tac_ms = result['models']['tac'][phase]['median_ms']
        trans_ms = result['models']['transformer'][phase]['median_ms']
        result.setdefault('comparative_tac_over_transformer', {})[phase] = tac_ms / trans_ms
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
    print('RATIOS', json.dumps(result['comparative_tac_over_transformer'], sort_keys=True))
    print('RESULT', out.resolve())


if __name__ == '__main__':
    main()
