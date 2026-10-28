"""Offline original-byte intake; never execute models or certify their results."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import re

SHA = re.compile(r'[0-9a-f]{64}')


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def safe_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str):
        raise ValueError('path must be a relative string')
    parts = PurePosixPath(relative).parts
    if (not parts or relative != PurePosixPath(relative).as_posix()
            or PurePosixPath(relative).is_absolute() or any(p in ('.', '..') for p in parts)
            or '\\' in relative):
        raise ValueError('noncanonical or escaped evidence path')
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('symlink evidence is not allowed')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('escaped evidence path')
    return path


def inspect(contract: dict, inventory: dict | None, root: Path,
            tracked: set[str]) -> dict:
    """The caller pins inventory bytes and supplies actual git-tracked paths.

    Hash agreement is agent-reviewed provenance, not independent authentication.
    Inventory must contain FIT/DEV only: reject TEST metadata before opening any
    payload. No pickle, archive extraction, source imports, scoring or network.
    """
    if (contract.get('schema') != 'tac-osm.replay-intake.v1'
            or contract.get('execution_level') != 'internal_research'
            or contract.get('protected_TEST') is not False
            or contract.get('new_fits') is not False):
        raise ValueError('internal no-fit contract required')
    report = dict(study_id=contract['study_id'],
                  evidence='engineering_intake_not_independent_qualification',
                  state='BLOCKED', REPO_REPLAY='BLOCKED',
                  training_steps=0, model_executions=0, protected_TEST=False,
                  missing=[], failures=[], verified=[])
    if inventory is None:
        report['missing'] = ['reviewed_original_inventory_with_sha256',
                             'publisher_license_and_original_document_hashes',
                             '36_original_FIT_DEV_documents',
                             '25_original_checkpoint_optimizer_RNG_payloads',
                             *contract['required_assets']]
        return report
    if (inventory.get('schema') != 'tac-osm.replay-inventory.v1'
            or inventory.get('study_id') != contract['study_id']):
        raise ValueError('inventory identity differs')
    sources = inventory.get('sources', [])
    checkpoints = inventory.get('checkpoints', [])
    assets = inventory.get('assets', [])
    if any(not isinstance(group, list) or any(not isinstance(r, dict) for r in group)
           for group in (sources, checkpoints, assets)):
        raise ValueError('evidence groups must contain records')
    records = sources + checkpoints + assets
    # Complete metadata validation precedes every filesystem read.
    expected_models = {(family, seed) for family in contract['checkpoint_families']
                       for seed in contract['seeds']}
    pairs = [(r.get('family'), r.get('seed')) for r in checkpoints]
    if (any(type(r.get('seed')) is not int for r in checkpoints)
            or len(pairs) != len(set(pairs)) or set(pairs) != expected_models):
        report['missing'].append('exact_original_family_seed_coverage')
    for partition, count in contract['source_counts'].items():
        if sum(r.get('partition') == partition for r in sources) != count:
            report['missing'].append(f'original_{partition}_document_coverage')
    if any(r.get('partition') not in ('fit', 'dev') for r in sources):
        raise ValueError('only FIT/DEV source metadata may be supplied')
    roles = [r.get('role') for r in assets]
    if len(roles) != len(set(roles)) or set(roles) != set(contract['required_assets']):
        report['missing'].append('original_recipe_protocol_metric_coverage')
    publisher = inventory.get('publisher', {})
    if not all(isinstance(publisher.get(k), str) and publisher[k].strip()
               for k in ('url', 'license', 'license_evidence')):
        report['missing'].append('publisher_license_record')
    source_records = [dict(path=p, sha256=h) for p, h in contract['source_files'].items()]
    records = source_records + records
    paths = [r.get('path') for r in records]
    if len(paths) != len(set(paths)):
        raise ValueError('duplicate evidence path')
    located = []
    for record in records:
        path = safe_file(root, record.get('path'))
        if not isinstance(record.get('sha256'), str) or not SHA.fullmatch(record['sha256']):
            raise ValueError('original byte sha256 required')
        located.append((record, path))
    identities = set()
    for record, path in located:
        name = record['path']
        if name not in tracked:
            report['missing'].append('not_repository_tracked:' + name)
            continue
        if not path.is_file():
            report['missing'].append('missing_original_bytes:' + name)
            continue
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in identities:
            raise ValueError('aliased evidence payload')
        identities.add(identity)
        if sha256(path) != record['sha256']:
            report['failures'].append('original_hash_mismatch:' + name)
        else:
            report['verified'].append(name)
    report['state'] = ('FAIL' if report['failures'] else
                       'BLOCKED' if report['missing'] else 'READY_FOR_REPLAY')
    # Readiness never claims actual score replay, scientific gates or provenance certification.
    return report
