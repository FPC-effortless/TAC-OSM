"""Additive source-byte lineage; existing canonical object hashes stay intact.

Hashes establish byte identity to a supplied pin, not publisher rights, factual
label correctness, evaluator independence or scientific qualification.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from tac_osm.csl125_training_data import DataContractError, digest, no_duplicates, query_group, split_for


@dataclass(frozen=True)
class SourceRecord:
    record_index: int
    physical_line: int
    byte_begin: int
    byte_end: int
    raw_line_sha256: str
    canonical_object_sha256: str
    value: dict


def read_source_jsonl(path, expected_sha256):
    if not isinstance(expected_sha256, str) or not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise DataContractError('immutable_file_sha256_required')
    path = Path(path)
    if path.is_symlink():
        raise DataContractError('source_symlink_not_allowed')
    raw = path.read_bytes()
    if len(raw) > 32_000_000:
        raise DataContractError('source_byte_budget_exceeded')
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise DataContractError('original_file_sha256_mismatch')
    records, begin, physical, blank = [], 0, 0, 0
    for physical, line in enumerate(raw.splitlines(keepends=True), 1):
        end = begin + len(line)
        text = line.decode('utf-8', errors='strict')
        if text.strip():
            value = json.loads(text, object_pairs_hook=no_duplicates,
                               parse_constant=lambda _: (_ for _ in ()).throw(DataContractError('nonfinite_json')))
            if not isinstance(value, dict):
                raise DataContractError('source_record_must_be_object')
            records.append(SourceRecord(len(records) + 1, physical, begin, end,
                                        hashlib.sha256(line).hexdigest(), digest(value), value))
        else:
            blank += 1
        begin = end
    return records, {'file_name': path.name, 'file_sha256': expected_sha256,
                     'file_bytes': len(raw), 'physical_lines': physical, 'blank_lines': blank,
                     'json_records': len(records), 'raw_line_hash_includes_line_ending': True,
                     'canonical_hash_definition': 'UTF8 ensure_ascii=False sort_keys=True separators=comma:colon allow_nan=False'}


def bind_dispositions(records, accepted, quarantine, duplicates):
    """Return a sidecar only; never mutate rows, labels, IDs or split decisions."""
    if not records or [r.record_index for r in records] != list(range(1, len(records) + 1)):
        raise DataContractError('complete_source_record_sequence_required')
    by_index = {r.record_index: r for r in records}
    dispositions = {}
    for category, rows, field in (('accepted', accepted, 'source_line'),
                                  ('quarantine', quarantine, 'line'), ('duplicate', duplicates, 'line')):
        for row in rows:
            index = row.get(field)
            if type(index) is not int or index not in by_index or index in dispositions:
                raise DataContractError('missing_duplicate_or_invalid_disposition_index')
            source = by_index[index]
            if digest(source.value) != source.canonical_object_sha256:
                raise DataContractError('parsed_source_mutated')
            if row.get('source_sha256') != source.canonical_object_sha256:
                raise DataContractError('legacy_canonical_object_hash_mismatch')
            if category == 'accepted':
                group = query_group(source.value.get('query'))
                if row.get('group') != group or row.get('split') != split_for(group):
                    raise DataContractError('accepted_group_or_partition_changed')
            dispositions[index] = (category, row)
    if set(dispositions) != set(by_index):
        raise DataContractError('source_record_missing_disposition')
    sidecar = []
    for index, source in by_index.items():
        category, row = dispositions[index]
        kept = None
        if category == 'duplicate':
            kept = row.get('kept_line')
            if (type(kept) is not int or kept not in by_index or kept >= index
                    or dispositions[kept][0] == 'duplicate'):
                raise DataContractError('invalid_duplicate_representative')
            original = by_index[kept].value
            if (query_group(original.get('query')) != query_group(source.value.get('query'))
                    or digest(original.get('answers')) != digest(source.value.get('answers'))):
                raise DataContractError('duplicate_group_or_label_mismatch')
        sidecar.append({'record_index': index, 'physical_line': source.physical_line,
                        'byte_begin': source.byte_begin, 'byte_end': source.byte_end,
                        'raw_line_sha256': source.raw_line_sha256,
                        'canonical_object_sha256': source.canonical_object_sha256,
                        'legacy_source_sha256': row['source_sha256'],
                        'disposition': category, 'duplicate_of_record': kept,
                        'partition': row.get('split') if category == 'accepted' else None})
    return sidecar
