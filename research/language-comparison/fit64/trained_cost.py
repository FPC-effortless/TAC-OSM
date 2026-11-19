"""Fresh interleaved timing of the three replayed local pilot checkpoints."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'cpu10m'))
from benchmark import make, request, verify_sources
from pilot import COUNTS, check_count, sha


@torch.inference_mode()
def measure(runs, replay_dir):
    verify_sources()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    arms = ('tac_selected', 'nanochat', 'gru')
    models, checkpoint_hashes, seconds = {}, {}, {}
    packed = Path(os.environ['CSL_PILOT_FIT'])
    input_sha = json.loads((HERE/'INPUT.json').read_text())['packed_sha256']
    if sha(packed) != input_sha:
        raise ValueError('Changed FIT bytes')
    data = packed.read_bytes()
    for arm in arms:
        start = time.monotonic()
        run = json.loads((runs/f'{arm}-receipt.json').read_text())
        replay = json.loads((replay_dir/f'{arm}-replay.json').read_text())
        cp = runs/run['id']/'output/checkpoint.pt'
        expected = run['artifacts']['checkpoint.pt']
        if run['status'] != 'succeeded' or expected != replay['checkpoint_sha256'] or sha(cp) != expected:
            raise ValueError(f'Unverified weights for {arm}')
        state = torch.load(cp, map_location='cpu', weights_only=True)
        if state['arm'] != arm or state['fit_sha256'] != input_sha or state['updates'] != 64:
            raise ValueError('Pilot identity mismatch')
        model = make(arm, seed=10301)
        model.load_state_dict(state['model'], strict=True)
        del state
        check_count(model, arm)
        models[arm] = model.eval()
        checkpoint_hashes[arm] = expected
        seconds[arm] = run['elapsed_seconds'] + replay['elapsed_s'] + time.monotonic() - start

    def timed(arm, prompt, verify=False):
        if seconds[arm] >= 890:
            raise RuntimeError(f'Insufficient remaining pilot time for {arm}')
        start = time.monotonic()
        value = request(models[arm], prompt, verify=verify)
        seconds[arm] += time.monotonic() - start
        return value

    profiles = {}
    rng = random.Random(1901)
    for length in (32, 256):
        prompt = torch.tensor([list(data[:length])])
        errors = {}
        rows = {arm: [] for arm in arms}
        for arm in arms:
            _, errors[arm] = timed(arm, prompt, verify=True)
            for _ in range(2):
                timed(arm, prompt)
        for _ in range(24):
            order = list(arms)
            rng.shuffle(order)
            for arm in order:
                timing, _ = timed(arm, prompt)
                rows[arm].append(timing)
        summary = {}
        for arm, values in rows.items():
            times = sorted(t for row in values for t in row['step_ms'])
            summary[arm] = dict(median_request_ms=statistics.median(r['request_ms'] for r in values),
                                median_first_ms=statistics.median(r['first_ms'] for r in values),
                                p95_decode_ms=times[math.ceil(.95 * len(times)) - 1])
        tac = summary['tac_selected']['median_request_ms']
        savings = {arm: 1 - tac/summary[arm]['median_request_ms'] for arm in ('nanochat','gru')}
        profiles[length] = dict(prompt_sha256=hashlib.sha256(data[:length]).hexdigest(),
                                cache_max_error=errors, summary=summary, time_saving=savings,
                                clears_20_percent={arm: value >= .2 for arm, value in savings.items()},
                                raw=rows)
    cpu = Path('/proc/cpuinfo').read_text()
    cpu_model = next(line.split(':', 1)[1].strip() for line in cpu.splitlines() if line.startswith('model name'))
    return dict(fit_sha256=input_sha, checkpoint_sha256=checkpoint_hashes, parameters=COUNTS,
                cpu=cpu_model, torch=torch.__version__, platform=platform.platform(), cpu_threads=1,
                profiles=profiles, seconds_including_training_replay_and_measured_work=seconds)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', type=Path)
    p.add_argument('replays', type=Path)
    args = p.parse_args()
    print(json.dumps(measure(args.runs, args.replays), indent=2, allow_nan=False))
