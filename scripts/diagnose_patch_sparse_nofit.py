#!/usr/bin/env python3
"""SENSE-TOK-003 no-fit source-to-logits and patch-update diagnosis.

No optimizer, data fitting, evaluator claims, TEST opening, or GPU dispatch.
Verify full original publisher blob identity before extracting a bounded prefix.
This runner is *not* an independent original source/reward/ACL attestation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

# Official python/cpython tag v3.13.5, tag points to commit
# 6cb20a219a860eaf687b2d968b41c480c7461909. These SHA1s are Git object
# identities, NOT collision-resistant independent publisher signatures.
PINNED_GIT_BLOBS = {
    'ast.py': '37b20206b8af4fda67c631fd010eda0008985511',
    'inspect.py': 'de2fe7e1ae0f6d345fdf1a53f4f0073079f50802',
    'csv.py': 'cd202659873811a2ea47c7374cc268d26a9ac5f3',
    'difflib.py': '33e7e6c165aa0c12c779331927c3a40f3b09d9e2',
    'textwrap.py': '7ca393d1c371aad54a2d1d70fae15efa889175bb',
    'email/parser.py': '06d99b17f2f9c4992301c432f498b0a32339171b',
}


def source_blob(raw: bytes) -> str:
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


def diagnose_source(root: Path, *, max_bytes: int = 256, seed: int = 20261031) -> dict:
    import torch
    from tac_osm.causal_byte_probe import CausalByteGRU, predecision_examples
    from tac_osm.patch_sparse_byte_probe import PatchSparseByteGRU, sparse_parameter_receipt
    from tac_osm.causal_patch_stream import STREAM_VARIANTS

    if max_bytes < 32 or max_bytes > 1024:
        raise ValueError('max_bytes must be 32..1024; no hidden expensive/full-file forward')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    baseline = CausalByteGRU().eval()
    model = PatchSparseByteGRU().eval()
    baseln = sum(p.numel() for p in baseline.parameters())
    param = sparse_parameter_receipt(model)
    assert baseln == 100400 and param['trainable_parameters'] == 100544
    assert param['baseline_100400_relative_gap'] < .01
    files = []
    for rel, expected in PINNED_GIT_BLOBS.items():
        path = root / rel
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError('missing or symlinked source: '+rel)
        raw = path.read_bytes()
        observed = source_blob(raw)
        if observed != expected:
            # Fail closed; the file is excluded, not silently substituted.
            files.append({'path': rel, 'source_status': 'EXCLUDED_BLOB_MISMATCH',
                          'observed_git_blob_sha1': observed, 'expected_git_blob_sha1': expected})
            continue
        prefix = raw[:max_bytes]
        if len(prefix) < 32:
            files.append({'path': rel, 'source_status': 'EXCLUDED_TOO_SHORT'})
            continue
        model_rows = []
        for arm in STREAM_VARIANTS:
            ids, features, target = predecision_examples(prefix, arm)
            with torch.no_grad():
                t0 = perf_counter_ns()
                out, (receipt,) = model(ids[None], variant=arm, audit=True)
                elapsed = perf_counter_ns()-t0
                # Native dense GRU isn't a per-arm sparse compute control.
                baseline_logits = baseline(ids[None], features[None])
            assert out.shape == baseline_logits.shape == (1, len(prefix), 256)
            assert bool(torch.isfinite(out).all())
            assert receipt.byte_readouts == len(prefix)
            assert all(e <= t for e,t in zip(receipt.committed_patch_ends,receipt.recurrent_update_times))
            model_rows.append({'variant':arm, 'byte_predictions':len(prefix),
                               'baseline_dense_recurrent_steps':len(prefix),
                               'sparse_recurrent_updates':receipt.recurrent_updates,
                               'recurrent_update_ratio_to_dense':round(receipt.recurrent_updates/len(prefix),6),
                               'delayed_boundaries':receipt.delayed_commits,
                               'elapsed_ns_uncontrolled_python_forward':elapsed,
                               'output_shape_pass':True,'future_noninterference_this_script':'NOT_CHECKED_SEPARATE_UNIT_TESTS'})
        files.append({'path':rel,'source_status':'FULL_GIT_BLOB_MATCH',
                      'expected_git_blob_sha1':expected,
                      'full_original_sha256':hashlib.sha256(raw).hexdigest(),
                      'prefix_sha256':hashlib.sha256(prefix).hexdigest(),
                      'tested_prefix_bytes':len(prefix),'arms':model_rows})
    verified = [r for r in files if r['source_status']=='FULL_GIT_BLOB_MATCH']
    if not verified:
        raise ValueError('no publisher-matching input: refusing diagnostic')
    return {
      'study_id':'SENSE-TOK-003', 'evidence_class':'NO_FIT_PREDECISION_GRAPH_DIAGNOSTIC',
      'authorization':{'gpu_spend':False,'model_fit':False,'sealed_test_opened':False},
      'model_fit_count':0, 'optimizer_steps':0,
      'source':{'publisher':'python/cpython','tag':'v3.13.5',
                'commit':'6cb20a219a860eaf687b2d968b41c480c7461909',
                'license':'PSF License 2.0; per-file notices require separate review',
                'independent_original_provenance':'NOT_ATTESTED',
                'predictive_task_real_world_qualified':False},
      'dense_model_parameters':baseln, 'sparse_model_parameters':param,
      'matched_architecture_parameter_gate':True,
      'matched_data_training_compute_and_evaluator_gate':'NOT_RUN',
      'files':files,'files_verified':len(verified),
      'limitations':['Model untrained; logit comparisons not predictive capability.',
                     'Only bounded prefixes of genuine code, not a held-out test.',
                     'Sparse GRU reduces recurrent update events, not necessarily actual total FLOPs or inference latency.',
                     'Full prefix copy, embedding, logits, source verification and Python dynamic graph cost must be included.',
                     '100k parameter parity does not imply FLOP parity; both must be measured prior to any superiority claim.',
                     'Research/PNDS formalization and independent signed source/goal/reward/ACL prereqs remain BLOCKED.']}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--max-bytes',type=int,default=256)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    output=diagnose_source(args.source_root,max_bytes=args.max_bytes)
    as_json=json.dumps(output,indent=2,sort_keys=True)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(as_json,encoding='utf8')
    else:
        print(as_json,end='')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
