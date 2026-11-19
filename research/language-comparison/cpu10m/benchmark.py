"""Run the frozen, untrained CPU cost screen with independently installed sources."""
import hashlib
import importlib
import json
import math
import os
import platform
import random
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter_ns

import torch
from torch import nn
from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import KVCache
from nanochat.gpt import GPT, GPTConfig
from tac_osm.byte100k_001 import byte100k_config
from tac_osm.csl125_memory_precision import enable_precise_memory_
from tac_osm.csl125_r2 import CSL125R2

HERE = Path(__file__).resolve().parent
SEED = 1901
ARMS = ('tac_four', 'tac_two', 'nanochat', 'gru')


def verify_sources():
    expected = json.loads((HERE / 'source-hashes.json').read_text())
    for module, digest in expected.items():
        source = Path(importlib.import_module(module).__file__)
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Source pin differs for {module}')


class NativeByte(nn.Module):
    def __init__(self, depth):
        super().__init__()
        width, state_width, ff = (448, 64, 1392) if depth == 4 else (640, 96, 1920)
        cfg = replace(byte100k_config(), layers=depth, rungs=(depth,), width=width,
                      state_width=state_width, ff_width=ff, query_heads=8, kv_heads=4,
                      attention_layers=tuple(range(2, depth + 1, 2)), max_parameters=10_500_000)
        self.core = CSL125R2(cfg)
        self.core.tool_head = nn.Identity()
        self.core.field_head = nn.Identity()
        self.core.accept_head = nn.Identity()
        enable_precise_memory_(self.core)

    def forward(self, ids, state=None):
        outputs, state = self.core(ids, state=state, stream_ids=('request',))
        return outputs[self.core.config.layers]['token_logits'], state


class ByteGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(256, 896)
        self.recurrent = nn.GRU(896, 896, num_layers=2, batch_first=True)
        self.readout = nn.Linear(896, 256)

    def forward(self, ids, state=None):
        features, state = self.recurrent(self.embed(ids), state)
        return self.readout(features), state


def make(name):
    torch.manual_seed(SEED)
    if name == 'nanochat':
        model = GPT(GPTConfig(sequence_len=512, vocab_size=256, n_layer=4,
                              n_head=8, n_kv_head=8, n_embd=448))
        model.init_weights()
    elif name == 'gru':
        model = ByteGRU()
    else:
        model = NativeByte(4 if name == 'tac_four' else 2)
    return model.eval()


def request(model, prompt, verify=False):
    start = perf_counter_ns()
    nano = isinstance(model, GPT)
    if nano:
        cfg = model.config
        state = KVCache(1, cfg.n_kv_head, prompt.size(1) + 16, cfg.n_embd // cfg.n_head,
                        cfg.n_layer, 'cpu', COMPUTE_DTYPE)
    else:
        state = None
    current = prompt
    ends = []
    error = None
    for i in range(16):
        if nano:
            logits = model(current, kv_cache=state)
        else:
            logits, state = model(current, state)
        following = logits[:, -1].argmax(dim=-1, keepdim=True)
        ends.append(perf_counter_ns())
        if verify and i == 1:
            full_ids = torch.cat((prompt, current), dim=1)
            full = model(full_ids) if nano else model(full_ids)[0]
            torch.testing.assert_close(logits[:, -1], full[:, -1], atol=1e-4, rtol=1e-4)
            error = (logits[:, -1] - full[:, -1]).abs().max().item()
        current = following
    return {
        'request_ms': (ends[-1] - start) / 1e6, 'first_ms': (ends[0] - start) / 1e6,
        'step_ms': [(b - a) / 1e6 for a, b in zip(ends, ends[1:])],
    }, error


@torch.inference_mode()
def main():
    verify_sources()
    if COMPUTE_DTYPE != torch.float32:
        raise ValueError('nanochat CPU FP32 required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    models = {name: make(name) for name in ARMS}
    counts = {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()}
    trainable = {name: sum(p.numel() for p in m.parameters() if p.requires_grad) for name,m in models.items()}
    lo, hi = min(counts.values()), max(counts.values())
    if trainable != counts or not (9_500_000 <= lo <= hi <= 10_500_000 and hi / lo <= 1.01):
        raise ValueError(f'Matched live count gate failed: {counts}')
    data = (HERE / 'PROTOCOL.md').read_bytes()
    profiles = {}
    metrics = {}
    rng = random.Random(SEED)
    for length in (32, 256):
        prompt = torch.tensor([list(data[:length])])
        samples = {name: [] for name in ARMS}
        errors = {}
        for name, model in models.items():
            _, errors[name] = request(model, prompt, verify=True)
            for _ in range(2):
                request(model, prompt)
        for _ in range(24):
            order = list(ARMS)
            rng.shuffle(order)
            for name in order:
                sample, _ = request(models[name], prompt)
                samples[name].append(sample)
        summary = {}
        for name, rows in samples.items():
            steps = sorted(ms for row in rows for ms in row['step_ms'])
            summary[name] = {
                'median_request_ms': statistics.median(row['request_ms'] for row in rows),
                'median_first_ms': statistics.median(row['first_ms'] for row in rows),
                'p95_step_ms': steps[math.ceil(0.95 * len(steps)) - 1],
            }
            metrics[f'{name}_{length}_ms'] = summary[name]['median_request_ms']
        profiles[length] = dict(prompt_sha256=hashlib.sha256(data[:length]).hexdigest(),
                                cache_logit_max_error=errors, samples=samples, summary=summary)
    out = Path(os.environ['TRAINLAB_OUTPUT_DIR'])
    cpu = Path('/proc/cpuinfo').read_text()
    cpu_name = next(line.split(':', 1)[1].strip() for line in cpu.splitlines() if line.startswith('model name'))
    quota = Path('/sys/fs/cgroup/cpu.max')
    record = dict(parameters=counts, seed=SEED, torch=torch.__version__, cpu=cpu_name,
                  cpu_quota=quota.read_text().strip() if quota.exists() else None,
                  platform=platform.platform(), python=sys.version.split()[0], threads=torch.get_num_threads(),
                  protocol_sha256=hashlib.sha256(data).hexdigest(), profiles=profiles)
    for name, value in [('timings.json', record), ('metrics.json', metrics)]:
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
