"""Run or replay one arm of the owner-authorized, FIT-only CPU pilot."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'cpu10m'))
from benchmark import make, verify_sources

COUNTS = {'tac_selected': 10103943, 'nanochat': 10092770, 'gru': 10103552}
PROBES = (0, 16, 32, 48)


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def setup(arm):
    owner = json.loads((HERE / 'owner.json').read_text())
    if (owner['response'] != 'Authorize the FIT-only pilot' or arm not in owner['arms']
            or (owner['seed'], owner['steps'], owner['bytes_per_arm'], owner['seconds_per_arm']) != (10301, 64, 131072, 900)
            or sha(HERE / 'PROTOCOL.md') != owner['protocol_sha256']):
        raise ValueError('Pilot scope or authorization changed')
    pins = json.loads((HERE / 'pins.json').read_text())
    for name, expected in pins.items():
        if sha(HERE / name) != expected:
            raise ValueError(f'Changed implementation or input receipt: {name}')
    verify_sources()
    path = Path(os.environ['CSL_PILOT_FIT'])
    receipt = json.loads((HERE / 'INPUT.json').read_text())
    if sha(path) != receipt['packed_sha256']:
        raise ValueError('Input bytes changed')
    data = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8).long().reshape(64, 8, 257)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    m = make(arm, seed=owner['seed'])
    check_count(m, arm)
    return m, data, receipt


def check_count(model, arm):
    params = list(model.parameters())
    if sum(p.numel() for p in params) != COUNTS[arm] or not all(p.requires_grad for p in params):
        raise ValueError('Live model parameter count or trainability changed')


def logits(model, batch):
    x = batch[:, :-1].contiguous()
    values = model(x)
    z = values[0] if isinstance(values, tuple) else values
    if z.shape != (*x.shape, 256):
        raise ValueError('Unexpected byte logits')
    return z


@torch.inference_mode()
def probe(model, data):
    model.eval()
    predictions = hashlib.sha256()
    total = 0.
    count = 0
    for i in PROBES:
        batch = data[i]
        z = logits(model, batch)
        predictions.update(z.contiguous().view(torch.uint8).numpy().tobytes())
        target = batch[:, 1:].reshape(-1)
        total += F.cross_entropy(z.double().reshape(-1, 256), target, reduction='sum').item()
        count += target.numel()
    return dict(bits_per_byte=total / (math.log(2) * count), targets=count,
                logits_sha256=predictions.hexdigest())


def train(arm, model, data, receipt, out):
    identity = dict(arm=arm, seed=10301, updates=64, fit_sha256=receipt['packed_sha256'],
                    protocol_sha256=sha(HERE / 'PROTOCOL.md'), pins_sha256=sha(HERE / 'pins.json'))
    initial = probe(model, data)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=0.0003, betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=0.1, foreach=False)
    curve = []
    start = time.monotonic()
    for index, batch in enumerate(data):
        opt.zero_grad(set_to_none=True)
        z = logits(model, batch)
        loss = F.cross_entropy(z.double().reshape(-1, 256), batch[:, 1:].reshape(-1))
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        if any(p.grad is None for p in model.parameters()):
            raise ValueError('Missing gradient')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        curve.append(dict(step=index+1, fit_nats_per_byte=loss.item(),
                          gradient_norm=norm.item(), elapsed_s=time.monotonic()-start))
        if (index+1) % 8 == 0:
            print(json.dumps(curve[-1]), flush=True)
    check_count(model, arm)
    if any(int(s['step'].item()) != 64 for s in opt.state.values()):
        raise ValueError('Optimizer update count changed')
    final = probe(model, data)
    cp = out / 'checkpoint.pt'
    torch.save(dict(**identity, model=model.state_dict(), optimizer=opt.state_dict(),
                    rng=torch.get_rng_state(), next_order_index=512), cp)
    metrics = dict(initial_fit_bpb=initial['bits_per_byte'], final_fit_bpb=final['bits_per_byte'],
                   first8_fit_nats=statistics.mean(r['fit_nats_per_byte'] for r in curve[:8]),
                   last8_fit_nats=statistics.mean(r['fit_nats_per_byte'] for r in curve[-8:]))
    write_json(out/'curve.json', dict(**identity, parameters=COUNTS[arm], target_bytes=131072,
                initial=initial, final=final, curve=curve, torch=torch.__version__, cpu_threads=2,
                checkpoint_sha256=sha(cp)))
    write_json(out/'metrics.json', metrics)


def replay(arm, model, data, receipt, directory):
    cp = directory/'checkpoint.pt'
    record = json.loads((directory/'curve.json').read_text())
    if sha(cp) != record['checkpoint_sha256']:
        raise ValueError('Checkpoint identity changed')
    saved = torch.load(cp, weights_only=True, map_location='cpu')
    if (saved['fit_sha256'] != receipt['packed_sha256'] or saved['arm'] != arm
            or saved['updates'] != 64 or saved['seed'] != 10301 or saved['next_order_index'] != 512
            or saved['pins_sha256'] != sha(HERE/'pins.json')
            or saved['protocol_sha256'] != sha(HERE/'PROTOCOL.md')):
        raise ValueError('Checkpoint scope changed')
    model.load_state_dict(saved['model'], strict=True)
    check_count(model, arm)
    checked = probe(model, data)
    if checked != record['final']:
        raise ValueError('Original FIT predictions did not replay')
    return dict(arm=arm, checkpoint_sha256=record['checkpoint_sha256'], restored=checked)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('arm', choices=COUNTS)
    p.add_argument('--replay', type=Path)
    args = p.parse_args()
    model, data, receipt = setup(args.arm)
    if args.replay:
        print(json.dumps(replay(args.arm, model, data, receipt, args.replay), indent=2))
    else:
        train(args.arm, model, data, receipt, Path(os.environ['TRAINLAB_OUTPUT_DIR']))
