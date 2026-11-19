"""Read and authenticate only the first 64 FIT batches from the registered order."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def build(shards, order, output):
    plan = json.loads((HERE.parent / 'wikitext103/FIT_RECEIPT.json').read_text())
    owner = json.loads((HERE / 'owner.json').read_text())
    if digest(HERE.parent / 'wikitext103/FIT_RECEIPT.json') != owner['fit_receipt_sha256']:
        raise ValueError('FIT registration changed')
    seed = str(owner['seed'])
    if digest(order) != plan['order_sha256'][seed]:
        raise ValueError('FIT order changed')
    with order.open('rb') as reader:
        prefix = reader.read(owner['steps'] * plan['batch'] * 9)
    coords = list(struct.iter_unpack('<BII', prefix))
    if len(set(coords)) != 512:
        raise ValueError('Missing or repeated FIT positions')
    windows = [None] * len(coords)
    for shard, expected in enumerate(plan['shard_sha256']):
        path = shards / f'train-{shard:05d}-of-00002.parquet'
        if digest(path) != expected:
            raise ValueError('FIT shard changed')
        wanted = sorted((row, start, i) for i, (s, row, start) in enumerate(coords) if s == shard)
        offset = 0
        i = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=['text']):
            while i < len(wanted) and wanted[i][0] < offset + len(batch):
                row, start, position = wanted[i]
                raw = batch.column(0)[row - offset].as_py().encode('utf-8')
                windows[position] = raw[start:start+257]
                i += 1
            offset += len(batch)
            if i == len(wanted):
                break
    if any(w is None or len(w) != 257 for w in windows):
        raise ValueError('Window outside source row')
    packed = b''.join(windows)
    output.write_bytes(packed)
    return dict(seed=int(seed), steps=64, batch=8, context=256, windows=len(windows),
                target_bytes=131072, packed_bytes=len(packed), packed_sha256=digest(output),
                order_prefix_sha256=hashlib.sha256(prefix).hexdigest(),
                full_order_sha256=plan['order_sha256'][seed], shard_sha256=plan['shard_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('shards', 'order', 'output'):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.shards, args.order, args.output), indent=2))
