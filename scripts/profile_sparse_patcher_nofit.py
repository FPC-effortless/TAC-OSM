#!/usr/bin/env python3
"""Compare no-fit end-to-end CPU prototype paths on pinned source prefixes.

DENSE: causal feature production + 100400-weight fused nn.GRU + byte head.
SPARSE: directly supplied previous-byte IDs + causal stream + per-patch GRUCell +
        same 256-way byte prediction at all positions.
No optimizer, no GPU dispatch, no qualified predictive results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from diagnose_patch_sparse_nofit import PINNED_GIT_BLOBS, source_blob


def profile(root: Path, sample_bytes: int = 256, repeats: int = 3) -> dict:
    if not 32 <= sample_bytes <= 1024 or not 1 <= repeats <= 7:
        raise ValueError('preflight-only bounded input and repetitions required')
    import torch
    from tac_osm.causal_byte_probe import CausalByteGRU, predecision_examples, BOS_ID
    from tac_osm.patch_sparse_byte_probe import PatchSparseByteGRU
    from tac_osm.causal_patch_stream import STREAM_VARIANTS

    torch.set_num_threads(1)
    torch.manual_seed(91213)
    dense = CausalByteGRU().eval()
    sparse = PatchSparseByteGRU().eval()
    rows = []
    with torch.no_grad():
        for rel, expected in PINNED_GIT_BLOBS.items():
            f = root / rel
            if f.is_symlink() or not f.is_file():
                raise FileNotFoundError('missing or symlinked source: '+rel)
            full = f.read_bytes()
            if source_blob(full) != expected:
                raise ValueError('publisher Git blob mismatch: '+rel)
            raw = full[:sample_bytes]
            ids = torch.tensor([BOS_ID] + list(raw[:-1]), dtype=torch.long)[None]
            for variant in STREAM_VARIANTS:
                ts_dense, ts_sparse = [], []
                for run in range(repeats + 1):
                    start = perf_counter()
                    dids, features, _ = predecision_examples(raw, variant)
                    dlog = dense(dids[None], features[None])
                    dtime = perf_counter() - start
                    start = perf_counter()
                    slog = sparse(ids, variant=variant)
                    stime = perf_counter() - start
                    assert dlog.shape == slog.shape == (1, len(raw), 256)
                    if run > 0:
                        ts_dense.append(dtime)
                        ts_sparse.append(stime)
                rows.append({'file': rel, 'variant': variant,
                             'prefix_sha256': hashlib.sha256(raw).hexdigest(),
                             'dense_ms': round(median(ts_dense) * 1000, 6),
                             'sparse_ms': round(median(ts_sparse) * 1000, 6)})
    out = {'study': 'SENSE-TOK-003', 'evidence': 'NO_FIT_UNCONTROLLED_SINGLE_THREAD_CPU_TIMING',
           'source': 'python/cpython v3.13.5 full pinned Git blobs',
           'sample_bytes_per_source': sample_bytes, 'repeats_per_file_after_warmup': repeats,
           'torch': torch.__version__, 'trained_models': 0, 'optimizer_steps': 0,
           'conclusion': 'update-count reduction does not establish speedup; report raw measured direction',
           'caveats': ['Not optimized kernel parity or matched training FLOPs',
                       'Script-level Python timing on this host only',
                       'Dense includes source-to-feature preprocessing and fused GRU',
                       'Sparse includes internal stream processing and per-byte Python GRUCell',
                       'No model accuracy or useful token compression measured'],
           'per_source': rows,
           'summary': {}}
    for variant in STREAM_VARIANTS:
        sub = [r for r in rows if r['variant'] == variant]
        dense_ms = median(r['dense_ms'] for r in sub)
        sparse_ms = median(r['sparse_ms'] for r in sub)
        out['summary'][variant] = {'dense_median_ms_per_file': round(dense_ms, 5),
                                   'sparse_median_ms_per_file': round(sparse_ms, 5),
                                   'sparse_over_dense': round(sparse_ms / dense_ms, 4)}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', required=True, type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--max-bytes', type=int, default=256)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    report = profile(args.source_root, args.max_bytes, args.repeats)
    result = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding='utf8')
    else:
        print(result, end='')


if __name__ == '__main__':
    main()
