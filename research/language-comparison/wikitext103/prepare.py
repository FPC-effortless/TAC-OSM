"""Prepare deterministic FIT coordinates only. Each sample stays inside one source row."""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import random
import struct

import pyarrow.parquet as pq

REV = 'b08601e04326c79dfdd32d625aee71d232d685c3'
DIGESTS = [
    '74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212',
    'ba090ac30dbf5461e8dcbdd1a1b8e6f3cf9c2c756d64f0c1220450acd514f720',
]
STEPS = 16_384
BATCH = 8
CONTEXT = 256
SEEDS = (10301, 10302, 10303)
# Little-endian shard, row (zero-based within shard), and byte offset.
RECORD = struct.Struct('<BII')


def prepare(root, output):
    coordinates = array('Q')
    rows_per_shard = []
    for shard, expected in enumerate(DIGESTS):
        path = root / f'train-{shard:05d}-of-00002.parquet'
        with path.open('rb') as reader:
            actual = hashlib.file_digest(reader, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'Incorrect hash for {path.name}')
        rows = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=['text']):
            for row in batch.column(0).to_pylist():
                length = len(row.encode('utf-8'))
                if rows >= 2**31 or length >= 2**32:
                    raise ValueError('Source exceeds coordinate encoding')
                for offset in range(0, length - CONTEXT, CONTEXT):
                    coordinates.append((shard << 63) | (rows << 32) | offset)
                rows += 1
        rows_per_shard.append(rows)
    output.mkdir(parents=True, exist_ok=True)
    orders = {}
    for seed in SEEDS:
        indices = random.Random(seed).sample(range(len(coordinates)), STEPS * BATCH)
        path = output / f'seed-{seed}.bin'
        with path.open('wb') as writer:
            for index in indices:
                value = coordinates[index]
                writer.write(RECORD.pack(value >> 63, (value >> 32) & 0x7fffffff, value & 0xffffffff))
        with path.open('rb') as reader:
            orders[seed] = hashlib.file_digest(reader, 'sha256').hexdigest()
    return dict(dataset='Salesforce/wikitext', revision=REV, subset='wikitext-103-raw-v1',
                split='train', shard_sha256=DIGESTS, rows_per_shard=rows_per_shard,
                eligible_windows=len(coordinates), context=CONTEXT, batch=BATCH, steps=STEPS,
                unique_target_positions_per_seed=STEPS * BATCH * CONTEXT, order_sha256=orders)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parquet_directory', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.parquet_directory, args.output), indent=2))
