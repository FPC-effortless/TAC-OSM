#!/usr/bin/env python3
"""Reuses publisher TRAIN bytes; no tokenizers, models, effects or TEST access."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tac_osm.csl125_training_data import SOURCE_HASHES, SOURCE_REVISION, api_catalogue, digest, prepare_records, read_jsonl
from tac_osm.jsonl_byte_lineage import bind_dispositions, read_source_jsonl


def audit(publisher, output):
    if output.exists():
        raise FileExistsError('preserve prior lineage audit')
    protocol = json.loads((ROOT / 'research/tool_lineage_20261107/PROTOCOL.json').read_text())
    adapter = ROOT / 'src/tac_osm/csl125_training_data.py'
    if hashlib.sha256(adapter.read_bytes()).hexdigest() != protocol['source_adapter_sha256']:
        raise ValueError('frozen adapter source changed')
    # Exact allowlist; never enumerate/open other publisher files or TEST.
    for name, expected in SOURCE_HASHES.items():
        path = publisher / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('publisher original hash mismatch: ' + name)
    card = (publisher / 'README.md').read_text()
    if not re.search(r'license:\s*apache-2\.0', card, re.I):
        raise ValueError('license declaration changed')
    records, file_receipt = read_source_jsonl(publisher / 'DroidCall_train.jsonl', SOURCE_HASHES['DroidCall_train.jsonl'])
    legacy = read_jsonl(publisher / 'DroidCall_train.jsonl')
    if [r.value for r in records] != legacy:
        raise ValueError('new reader differs from original parsed source')
    tools, excluded = api_catalogue(read_jsonl(publisher / 'annotated_api.jsonl'))
    accepted, quarantine, duplicates, prior = prepare_records(legacy, tools)
    before = digest([accepted, quarantine, duplicates, prior])
    lineage = bind_dispositions(records, accepted, quarantine, duplicates)
    if digest([accepted, quarantine, duplicates, prior]) != before:
        raise ValueError('sidecar changed existing adapter decisions')
    raw = (publisher / 'DroidCall_train.jsonl').read_bytes()
    for row in lineage:
        if hashlib.sha256(raw[row['byte_begin']:row['byte_end']]).hexdigest() != row['raw_line_sha256']:
            raise ValueError('source span cannot replay exact raw line')
    if any(name in sys.modules for name in ('torch', 'tokenizers', 'xgrammar')):
        raise RuntimeError('source-only audit imported neural/tokenizer runtime')
    result = {'status': 'PASS_ORIGINAL_TRAIN_BYTE_LINEAGE', 'source_revision': SOURCE_REVISION,
              'publisher': 'mllmTeam/DroidCall', 'declared_license': 'Apache-2.0',
              'source_is_synthetic': True, 'independently_qualified': False,
              'file_receipt': file_receipt, 'prior_adapter_summary': prior,
              'disposition_counts': dict(Counter(r['disposition'] for r in lineage)),
              'raw_line_differs_from_legacy_canonical_hash': sum(r['raw_line_sha256'] != r['legacy_source_sha256'] for r in lineage),
              'original_adapter_decisions_sha256': before, 'decisions_unchanged': True,
              'catalogue_exclusions': len(excluded), 'model_executions': 0, 'tokenizer_fit_steps': 0,
              'optimizer_updates': 0, 'GPU_minutes': 0, 'protected_TEST_opened': False,
              'training_authorized': False, 'promotion_authorized': False,
              'neural_source_Lean_IEEE_AD': 'OPEN_UNCHANGED',
              'device': {k: 'NOT_RUN' for k in ('target','operators','numerical_agreement','p50_p95','RAM','energy','airplane_mode')},
              'limits': 'Publisher-byte lineage only. Synthetic labels are not observed executions; no tool capability, unseen-tool/paraphrase generalization or qualification follows.'}
    output.mkdir(parents=True)
    (output / 'RESULTS.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    (output / 'LINEAGE.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in lineage))
    print(json.dumps({'status': result['status'], 'source_rows': len(records), 'dispositions': result['disposition_counts'],
                      'decisions_unchanged': True, 'model_executions': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publisher-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.publisher_dir, args.output)
