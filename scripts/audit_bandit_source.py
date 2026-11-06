#!/usr/bin/env python3
"""Check the pinned publisher OBD example schema without parsing reward values.

A column called user_feature_* is NOT a persistent entity identifier. This
source-specific audit deliberately does not infer identity from feature hashes.
No model is fitted and no policy value is estimated here.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

PUBLISHER_COMMIT = '8cbd5fa4558b7ad2ba4781546d6604e4cc3e07c4'
SOURCE = f'https://raw.githubusercontent.com/st-tech/zr-obp/{PUBLISHER_COMMIT}/obd/random/all/all.csv'
EXPECTED_SHA256 = '7168295b6e0a9eabcf3392320a5dd434e542b68e705d5cd9491499af589812f1'


def audit(path):
    path = Path(path)
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            checksum.update(block)
    with path.open(newline='') as stream:
        header = next(csv.reader(stream))
    required = {'timestamp','item_id','position','click','propensity_score'}
    authentic = checksum.hexdigest() == EXPECTED_SHA256
    schema = required <= set(header)
    # Eligibility is specific to the pinned release; a renamed or injected
    # user_id column would change its hash and fail provenance.
    return {
        'source_url': SOURCE, 'license': 'CC-BY-4.0',
        'sha256': checksum.hexdigest(), 'bytes': path.stat().st_size,
        'pinned_publisher_bytes_verified': authentic,
        'required_bandit_fields_present': schema,
        'columns': header, 'reward_values_parsed': False,
        'persistent_entity_identifier_documented': False,
        'persistent_memory_study_eligible': False,
        'one_step_ope_schema_eligible': authentic and schema,
        'one_step_policy_qualification_complete': False,
        'decision': 'BLOCK_ENTITY_MEMORY_STUDY',
        'remaining_ope_checks': ['position-conditioned policy support',
            'independent policy-training and evaluation partitions',
            'cross-fitted reward nuisance predictions',
            'SNIPS/DR variance and effective sample size',
            'compulsory parameter/data/FLOP/CPU parity before fitting'],
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('csv')
    cli.add_argument('--output', required=True)
    cli.add_argument('--purpose', choices=('entity_memory','one_step_schema'), default='entity_memory')
    args = cli.parse_args()
    receipt = audit(args.csv)
    Path(args.output).write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: receipt[k] for k in ('sha256','decision','one_step_ope_schema_eligible')}))
    return 0 if args.purpose == 'one_step_schema' and receipt['one_step_ope_schema_eligible'] else 2

if __name__ == '__main__':
    raise SystemExit(main())
