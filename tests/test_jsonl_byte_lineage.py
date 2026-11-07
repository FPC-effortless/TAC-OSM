"""Constructed provenance controls, never model/corpus-quality evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tac_osm.csl125_training_data import DataContractError, api_catalogue, digest, prepare_records
from tac_osm.jsonl_byte_lineage import bind_dispositions, read_source_jsonl

ROOT = Path(__file__).resolve().parents[1]


def load(tmp_path, raw):
    path = tmp_path / 'fixture.jsonl'
    path.write_bytes(raw)
    return read_source_jsonl(path, hashlib.sha256(raw).hexdigest())


def row(query='Weather in Jos', city='Jos'):
    return {'query': query, 'answers': [{'name': 'weather', 'arguments': {'city': city}}], 'tools': []}


def table(tmp_path):
    values = [row(), row(' WEATHER  IN JOS '), row('Beta', 'A'), row(' beta ', 'B'),
              {'query': 'Gamma', 'answers': [{'name': 'unsupported', 'arguments': {}}], 'tools': []}]
    raw = b''.join((json.dumps(v) + '\n').encode() for v in values)
    records, receipt = load(tmp_path, raw)
    tools, _ = api_catalogue([{'name': 'weather', 'description': 'Weather',
                             'arguments': {'city': {'type': 'str', 'required': True}}, 'returns': {}}])
    accepted, quarantine, duplicates, summary = prepare_records([r.value for r in records], tools)
    return records, receipt, accepted, quarantine, duplicates, summary


def test_same_object_different_raw_serialization_has_distinct_source_identity(tmp_path):
    raw = b'{"a":1,"b":"\\u00e9"}\n{ "b": "\xc3\xa9", "a": 1 }\n'
    records, receipt = load(tmp_path, raw)
    assert records[0].value == records[1].value
    assert records[0].canonical_object_sha256 == records[1].canonical_object_sha256
    assert records[0].raw_line_sha256 != records[1].raw_line_sha256
    for record in records:
        assert hashlib.sha256(raw[record.byte_begin:record.byte_end]).hexdigest() == record.raw_line_sha256
    assert receipt['json_records'] == 2


def test_crlf_blank_lines_and_missing_final_newline_keep_exact_offsets(tmp_path):
    first = b'{"q":1}\r\n'
    blank = b'  \r\n'
    last = b'{"q":2}'
    records, receipt = load(tmp_path, first + blank + last)
    assert [r.physical_line for r in records] == [1, 3]
    assert records[1].byte_begin == len(first + blank)
    assert records[1].byte_end == len(first + blank + last)
    assert receipt['physical_lines'] == 3 and receipt['blank_lines'] == 1


def test_unchanged_decisions_have_total_unique_raw_lineage(tmp_path):
    records, _, accepted, quarantine, duplicates, _ = table(tmp_path)
    before = deepcopy((accepted, quarantine, duplicates))
    sidecar = bind_dispositions(records, accepted, quarantine, duplicates)
    assert (accepted, quarantine, duplicates) == before
    assert [r['record_index'] for r in sidecar] == list(range(1, 6))
    assert [r['disposition'] for r in sidecar] == ['accepted', 'duplicate', 'quarantine', 'quarantine', 'quarantine']
    assert sidecar[1]['duplicate_of_record'] == 1
    assert all(r['canonical_object_sha256'] == r['legacy_source_sha256'] for r in sidecar)
    assert all(r['raw_line_sha256'] != r['legacy_source_sha256'] for r in sidecar)


@pytest.mark.parametrize('fault', ['omitted', 'repeated', 'index_bool', 'bad_hash', 'wrong_split', 'bad_representative'])
def test_corrupt_or_incomplete_adapter_receipts_cannot_bind(tmp_path, fault):
    records, _, accepted, quarantine, duplicates, _ = table(tmp_path)
    if fault == 'omitted':
        quarantine.pop()
    elif fault == 'repeated':
        accepted.append(deepcopy(accepted[0]))
    elif fault == 'index_bool':
        accepted[0]['source_line'] = True
    elif fault == 'bad_hash':
        quarantine[0]['source_sha256'] = '0' * 64
    elif fault == 'wrong_split':
        accepted[0]['split'] = 'DEV' if accepted[0]['split'] == 'FIT' else 'FIT'
    else:
        duplicates[0]['kept_line'] = 3
    with pytest.raises(DataContractError):
        bind_dispositions(records, accepted, quarantine, duplicates)


@pytest.mark.parametrize('raw', [b'{"x":1,"x":2}\n', b'{"x":NaN}\n', b'{"x":1e309}\n', b'[]\n', b'\xff\n'])
def test_invalid_source_bytes_fail_without_repair(tmp_path, raw):
    with pytest.raises(ValueError):
        load(tmp_path, raw)


def test_original_file_pin_is_mandatory_and_not_canonicalized(tmp_path):
    path = tmp_path / 'fixture.jsonl'
    original = b'{"x":1}\n'
    path.write_bytes(original)
    with pytest.raises(DataContractError, match='immutable_file'):
        read_source_jsonl(path, None)
    path.write_bytes(b'{ "x": 1 }\n')
    with pytest.raises(DataContractError, match='original_file'):
        read_source_jsonl(path, hashlib.sha256(original).hexdigest())


def test_parsed_row_mutation_is_detected(tmp_path):
    records, _, accepted, quarantine, duplicates, _ = table(tmp_path)
    records[0].value['query'] = 'mutated'
    with pytest.raises(DataContractError, match='parsed_source_mutated'):
        bind_dispositions(records, accepted, quarantine, duplicates)


def test_import_remains_source_only_without_model_tokenizer_runtime():
    code = ("import sys;sys.path.insert(0,'src');import tac_osm.jsonl_byte_lineage;"
            "assert not any(n in sys.modules for n in ('torch','tokenizers','xgrammar'))")
    subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True, capture_output=True)
