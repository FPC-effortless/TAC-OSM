"""Fixed 1,928-byte raw-history control; Witten--Bell is established prior art.

The process timer precedes NumPy imports and input loading. Target bytes have
no predictor interface. This reference is not a parameter-matched neural arm.
"""
import time
PROCESS_START = time.perf_counter()

import argparse
import json
from pathlib import Path
import resource
import struct

import numpy as np

STATE_BYTES = 1928
HEADER_BYTES = 16
CAPACITY = STATE_BYTES - HEADER_BYTES
ORDER = 6


def distribution(observed):
    history = bytes(observed)
    probability = np.full(256, 1 / 256, dtype=np.float64)
    for order in range(min(ORDER, len(history)) + 1):
        suffix = history[-order:] if order else b""
        counts = np.zeros(256, dtype=np.int64)
        for end in range(order, len(history)):
            if order == 0 or history[end-order:end] == suffix:
                counts[history[end]] += 1
        total, types = int(counts.sum()), int(np.count_nonzero(counts))
        if total:
            probability = counts / (total + types) + types / (total + types) * probability
    if not np.isfinite(probability).all() or not np.isclose(probability.sum(), 1):
        raise ValueError('invalid Witten--Bell distribution')
    return probability


class RawRing:
    def __init__(self):
        self.storage = bytearray(STATE_BYTES)

    def admit(self, values):
        cursor, size = struct.unpack_from('<QQ', self.storage)
        for value in values:
            if type(value) is not int or not 0 <= value <= 255:
                raise ValueError('observed bytes required')
            self.storage[HEADER_BYTES + cursor] = value
            cursor = (cursor + 1) % CAPACITY
            size = min(CAPACITY, size + 1)
        struct.pack_into('<QQ', self.storage, 0, cursor, size)

    def observed(self):
        cursor, size = struct.unpack_from('<QQ', self.storage)
        values = self.storage[HEADER_BYTES:]
        return bytes(values[:size] if size < CAPACITY else values[cursor:] + values[:cursor])


def predict(acquisition, query):
    start = time.perf_counter()
    ring = RawRing()
    setup = time.perf_counter() - start
    start = time.perf_counter(); ring.admit(acquisition)
    acquisition_seconds = time.perf_counter() - start
    start = time.perf_counter(); ring.admit(query)
    probability = distribution(ring.observed())
    log_probs = np.log(probability)
    query_seconds = time.perf_counter() - start
    start = time.perf_counter(); generated = []
    for index in range(4):
        value = int(probability.argmax()); generated.append(value)
        if index < 3:
            ring.admit([value]); probability = distribution(ring.observed())
    return {'log_probs': log_probs, 'generated': np.asarray(generated, dtype=np.uint8),
            'cost': {'setup_seconds': setup, 'acquisition_seconds': acquisition_seconds,
                     'query_seconds': query_seconds,
                     'generation_seconds': time.perf_counter() - start,
                     'state_payload_bytes': len(ring.storage), 'model_bytes': 0}}


def copy_last_demonstration(acquisition):
    """Past-only external ARC reference; it does not infer a transformation."""
    pairs = [part for part in bytes(acquisition).split(b';') if b'>' in part]
    if not pairs:
        raise ValueError('a genuine observed demonstration is required')
    output = pairs[-1].split(b'>', 1)[1] + b';'
    if len(output) > 40:
        raise ValueError('demonstration exceeds the fixed output bound')
    return np.frombuffer(output.ljust(40, b' '), dtype=np.uint8).copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = json.loads(args.actor.read_text())
    allowed = {'id', 'domain', 'acquisition_bytes', 'query_bytes', 'generation_limit'}
    if any(set(row) != allowed for row in records):
        raise ValueError('actor-only input schema required')
    arc = [row for row in records if row['domain'] == 'arc']
    records = [row for row in records if row['domain'] == 'lexical']
    setup_seconds = time.perf_counter() - PROCESS_START
    results = [predict(row['acquisition_bytes'], row['query_bytes']) for row in records]
    arc_generated = [copy_last_demonstration(row['acquisition_bytes']) for row in arc]
    metadata = {'schema': 'langacquire097.raw-control.v1', 'optimizer_steps': 0,
                'setup_before_prediction_seconds': setup_seconds,
                'recorded_process_seconds': time.perf_counter() - PROCESS_START,
                'peak_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                'state_bytes_per_stream': STATE_BYTES,
                'arc_copy_reference_state_bytes': 48,
                'arc_copy_reference_scope': 'last observed output up to40bytes plus8-byte length; query ignored',
                'costs': [row['cost'] for row in results],
                'scope': 'cold isolated process; not a neural architecture or serving-latency match'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        np.savez_compressed(handle, log_probs=np.stack([r['log_probs'] for r in results]),
                            generated=np.stack([r['generated'] for r in results]),
                            ids=np.asarray([r['id'] for r in records]),
                            arc_ids=np.asarray([r['id'] for r in arc]),
                            arc_generated=np.stack(arc_generated) if arc_generated else np.empty((0,40),dtype=np.uint8),
                            metadata=np.asarray(json.dumps(metadata, sort_keys=True)))


if __name__ == '__main__':
    main()
