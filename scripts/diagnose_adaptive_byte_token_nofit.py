#!/usr/bin/env python3
"""SENSE-TOK-004 real-source identity / no-fit model-graph diagnostic.

No optimizer step, model fitting, sealed TEST, reward, GPU dispatch or metric
of trained capability. Publisher Git blob checks match official CPython
v3.13.5. Random gate decisions must NOT be represented as learned selection.
"""
from __future__ import annotations

import argparse
from hashlib import sha1, sha256
import json
from pathlib import Path
import sys
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

SOURCE_TAG = 'v3.13.5'
SOURCE_COMMIT = '6cb20a219a860eaf687b2d968b41c480c7461909'
SOURCE_BLOBS = {
    'ast.py': '37b20206b8af4fda67c631fd010eda0008985511',
    'inspect.py': 'de2fe7e1ae0f6d345fdf1a53f4f0073079f50802',
    'csv.py': 'cd202659873811a2ea47c7374cc268d26a9ac5f3',
    'difflib.py': '33e7e6c165aa0c12c779331927c3a40f3b09d9e2',
    'textwrap.py': '7ca393d1c371aad54a2d1d70fae15efa889175bb',
    'email/parser.py': '06d99b17f2f9c4992301c432f498b0a32339171b',
}


def git_blob(raw: bytes) -> str:
    return sha1(b'blob ' + str(len(raw)).encode('ascii') + b'\0' + raw).hexdigest()


def diagnose(root: Path, *, max_bytes: int = 256, seed: int = 20261031) -> dict:
    import torch
    from tac_osm.adaptive_byte_token import AdaptiveByteTokenGRU, adaptive_parameter_receipt
    from tac_osm.causal_byte_probe import predecision_examples

    if not 32 <= max_bytes <= 1024:
        raise ValueError('restrict no-fit source-to-model audit to 32..1024 bytes per source')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = AdaptiveByteTokenGRU(max_token_bytes=8).eval()
    params = adaptive_parameter_receipt(model)
    if not (params['within_100k_band'] and params['within_1pct_vs_sparse']):
        raise AssertionError('live parameter count outside frozen no-fit budget')
    sources: list[dict] = []
    for rel, pinned in SOURCE_BLOBS.items():
        path = root / rel
        if not path.is_file() or path.is_symlink():
            sources.append({'file': rel, 'status': 'EXCLUDED_MISSING_OR_SYMLINK'})
            continue
        raw = path.read_bytes()
        actual = git_blob(raw)
        if actual != pinned:
            sources.append({'file': rel, 'status': 'EXCLUDED_PUBLISHER_BLOB_MISMATCH',
                            'actual_git_blob_sha1': actual, 'expected_git_blob_sha1': pinned})
            continue
        prefix = raw[:max_bytes]
        x, _, _ = predecision_examples(prefix, 'fixed')
        rows: list[dict] = []
        for override in (None, 'byte', 'token'):
            with torch.no_grad():
                start = perf_counter_ns()
                output, (audit,), proxy = model(x[None], mode='inference', force=override, audit=True)
                elapsed = perf_counter_ns() - start
            audit.verify(prefix[:-1])
            if output.shape != (1, len(prefix), 256) or not torch.isfinite(output).all():
                raise AssertionError('source-to-logits failed')
            counts = {'BYTE': sum(1 for e in audit.events if e.mode == 'BYTE'),
                      'TOKEN': sum(1 for e in audit.events if e.mode == 'TOKEN')}
            rows.append({
                'mode': override or 'untrained_adaptive_gate',
                'observed_bytes': audit.observed_bytes,
                'recurrent_updates_executed': audit.recurrent_updates,
                'byte_representations': counts['BYTE'],
                'token_representations': counts['TOKEN'],
                'remaining_uncommitted_bytes': len(audit.pending),
                'byte_decisions': audit.decisions.count('BYTE'),
                'token_accumulation_decisions': audit.decisions.count('TOKEN'),
                'mean_gate_p_byte_untrained': (
                    sum(audit.byte_probabilities) / len(audit.byte_probabilities)
                    if audit.byte_probabilities else None),
                'expected_update_proxy_not_flops': float(proxy.detach()),
                'byte_prediction_readouts': audit.byte_readouts,
                'elapsed_ns_uncontrolled_python_cpu': elapsed,
                'reconstruction_pass': True,
                'causal_event_availability_pass': True,
                'not_learned_semantic_routing': True,
            })
        sources.append({'file': rel, 'status': 'FULL_GIT_BLOB_MATCH',
                        'original_full_file_sha256': sha256(raw).hexdigest(),
                        'observed_prefix_sha256': sha256(prefix).hexdigest(),
                        'tested_prefix_bytes': len(prefix), 'arms': rows})
    verified = [r for r in sources if r['status'] == 'FULL_GIT_BLOB_MATCH']
    if len(verified) != len(SOURCE_BLOBS):
        raise RuntimeError('all six pinned original CPython sources required; cannot silently substitute inputs')
    return {
        'study_id': 'SENSE-TOK-004',
        'evidence_class': 'SOURCE_MATCHED_NO_FIT_ENGINEERING_ONLY',
        'torch_seed': seed, 'model_fit_count': 0, 'optimizer_steps': 0,
        'source': {'repository': 'python/cpython', 'tag': SOURCE_TAG, 'release_commit': SOURCE_COMMIT,
                   'publisher_blob_identity': 'MATCHED', 'independent_source_goal_acl_signer': 'ABSENT',
                   'research_split': 'EXPOSED_DEV_DIAGNOSTIC_NOT_SEALED_TEST'},
        'model_parameters': params, 'total_verified_files': len(verified),
        'source_rows': sources,
        'qualifies_architecture_superiority': False,
        'language_model_skill': 'NOT_EVALUATED_UNTRAINED',
        'training_authorized': False,
        'important_limits': [
            'Adaptive gate logits are random unfitted weights; observed decisions do not imply understanding.',
            'TOKEN refers to pooled multi-byte latent patch, not a discrete BPE vocabulary entry.',
            'Forced byte/token variants are structural interventions, not trained matched comparators.',
            'No real-world objective outcome, end-to-end trained BPB, runtime speedup or learned capacity measured.',
            'Python CPU time is an uncontrolled implementation measurement.',
            'Surrogate training computes both recurrent candidates; no training efficiency claim.',
            'Model still has byte-level readouts and stores raw input/metadata during audit.',
            'PNDS Lean/Python reciprocal mathematical formalization remains OPEN.',
            'Source/goal/reward/ACL independent signed attestation and Kaggle T4x2 approval absent.',
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-root', type=Path, default=Path('/usr/lib/python3.13'))
    ap.add_argument('--max-bytes', type=int, default=256)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    report = diagnose(args.source_root, max_bytes=args.max_bytes)
    encoded = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding='utf-8')
    else:
        print(encoded, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
