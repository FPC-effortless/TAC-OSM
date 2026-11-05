"""Engineering task-validity controls, not neural or real-world evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tac_osm.copy_task_contract import (
    ABSENT, audit_cases, literal_answer, prospective_controls, validate_case,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def original():
    protocol = json.loads((ROOT / "research/copy_contract_20261105/PROTOCOL.json").read_text())
    raw = (ROOT / protocol["source_fixture"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == protocol["source_fixture_sha256"]
    return json.loads(raw)


@pytest.fixture
def controls(original):
    values = {validate_case(c)[2] for c in original}
    return prospective_controls(values), values


def test_original_is_consistent_copying_but_has_no_selection_witness(original):
    report = audit_cases(original, {bytes.fromhex(c["target_hex"]) for c in original})
    assert report["cases"] == report["literal_last_record_correct"] == 8
    assert report["same_prefix_query_selection_witnesses"] == 0
    assert report["distinct_query_keys"] == report["records_per_history_max"] == 1
    assert report["key_blind_empirical_max_correct"] == 8
    assert report["best_constant_exact_answer_correct"] == 1
    assert report["best_complete_answer_given_gold_first_byte_correct"] == 8
    assert report["binding_or_capacity_claim_authorized"] is False


def test_selection_and_overwrite_force_some_query_dependence(controls):
    rows, training = controls
    assert len(rows) == 64
    for family, count in (("query_selection", 32), ("latest_write", 16)):
        report = audit_cases([c for c in rows if c["family"] == family], training)
        assert report["cases"] == count
        assert report["key_blind_empirical_max_correct"] == count // 2
        assert report["literal_last_record_correct"] == count // 2
        assert report["same_prefix_query_selection_witnesses"] == count // 2
        assert report["present_value_overlap_with_training"] == 0
        assert report["binding_or_capacity_claim_authorized"] is False


def test_copy_novelty_and_absence_are_separate_controls(controls):
    rows, training = controls
    copying = audit_cases([c for c in rows if c["family"] == "unseen_value_copy"], training)
    absent = audit_cases([c for c in rows if c["family"] == "absence"], training)
    assert copying["cases"] == 8 and copying["present_value_overlap_with_training"] == 0
    assert copying["best_complete_answer_given_gold_first_byte_correct"] == 2
    assert copying["same_prefix_query_selection_witnesses"] == 0
    assert absent["absence_cases"] == 8 and absent["literal_last_record_correct"] == 0


def test_latest_write_and_distinct_key_order_semantics():
    a = b"@aa: abcdxxxx\n@bb: efghxxxx\n"
    b = b"@bb: efghxxxx\n@aa: abcdxxxx\n"
    assert literal_answer(a, b"Copy aa: ") == literal_answer(b, b"Copy aa: ") == b"abcd"
    assert literal_answer(a + b"@aa: hgfexxxx\n", b"Copy aa: ") == b"hgfe"
    assert literal_answer(a, b"Copy cc: ") == ABSENT


def test_poisoned_labels_do_not_enter_the_reference(original):
    c = deepcopy(original[0])
    before = validate_case(c)
    c["target_hex"] = b"hhhh".hex()
    assert literal_answer(before[0], before[1]) == before[2]
    with pytest.raises(ValueError, match="target differs"):
        validate_case(c)


def test_slab_future_suffix_does_not_enter_the_observed_input(original):
    c = deepcopy(original[0])
    before = validate_case(c)
    raw = bytearray.fromhex(c["slab_hex"])
    for i in range(c["first_position"] + 5, len(raw)):
        raw[i] = (raw[i] + 113) % 256
    c["slab_hex"] = raw.hex()
    assert validate_case(c) == before


@pytest.mark.parametrize("field", ["first_position", "records", "queried_key", "slab_hex"])
def test_declared_metadata_cannot_substitute_for_wire_bytes(original, field):
    c = deepcopy(original[0])
    c[field] = {"first_position": 111, "records": [["aa", b"hhhh".hex()]],
                "queried_key": "bb", "slab_hex": b"bad".hex()}[field]
    with pytest.raises(ValueError):
        validate_case(c)


def test_empty_or_duplicated_cohort_is_not_a_pass(original):
    with pytest.raises(ValueError, match="empty"):
        audit_cases([])
    with pytest.raises(ValueError, match="duplicate"):
        audit_cases([original[0], original[0]])


def test_malformed_query_and_record_are_rejected():
    with pytest.raises(ValueError, match="query"):
        literal_answer(b"", b"Copy aa: hhhh")
    with pytest.raises(ValueError, match="record"):
        literal_answer(b"@aa: BAD!xxxx\n", b"Copy aa: ")


def test_prospective_novelty_is_not_silently_relaxed():
    with pytest.raises(ValueError, match="overlap"):
        prospective_controls({b"aceg"})


def test_source_audit_is_torch_free_and_does_not_overwrite(tmp_path):
    out = tmp_path / "audit"
    command = [sys.executable, "-B", str(ROOT / "scripts/audit_copy_task_contract.py"), "--output", str(out)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    r = json.loads((out / "RESULTS.json").read_text())
    assert r["models_constructed"] == r["checkpoint_loads"] == r["optimizer_updates"] == 0
    assert r["707_allocation_changed_or_consumed"] is False
    again = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert again.returncode != 0 and "FileExistsError" in again.stderr


def test_independent_pnds_record_reference_matches_observed_wire(controls, original):
    import importlib.util
    from tac_osm.copy_task_contract import records_from_prefix
    path = ROOT / "research/copy_contract_20261105/pnds_reference.py"
    spec = importlib.util.spec_from_file_location("copy_contract_reference", path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    for c in original + controls[0]:
        prefix, query, answer = validate_case(c)
        result = ref.lookup(records_from_prefix(prefix), query[5:7])
        assert (ABSENT if result is None else result) == answer
    pairs = [(validate_case(c)[0], c["queried_key"], validate_case(c)[2])
             for c in controls[0] if c["family"] == "query_selection"]
    assert ref.query_blind_ceiling(pairs) == 16
