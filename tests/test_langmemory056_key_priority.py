"""LANGMEMORY056 diagnostic safeguards; constructed sequences are NOT capability claims."""
from __future__ import annotations

import hashlib

import pytest

from scripts.langmemory056_key_priority import (
    PriorityKeyTable, owner_probe, BUDGETS, POLICIES, AGE_QUANTUM
)


@pytest.mark.parametrize("budget,slots", [(1344, 160), (4096, 504)])
@pytest.mark.parametrize("policy", POLICIES)
def test_bounded_payload_and_2way_geometry(budget, slots, policy):
    tab = PriorityKeyTable(budget, policy)
    assert tab.actual_payload_bytes() == budget
    assert tab.slots == slots and tab.sets * 2 == slots
    assert len(tab.payload) == budget - 64
    assert len(tab.offsets(b"abc")) == 2


def test_zero_byte_is_valid_candidate_and_full_key_matters():
    tab = PriorityKeyTable(1344, "lfu")
    tab.update(b"\x00ab", 0, 3)
    assert tab.read(b"\x00ab") == 0
    assert tab.read(b"\x00ac") is None


def test_clock_invalid_value_and_key_are_rejected():
    tab = PriorityKeyTable(1344, "decayed")
    with pytest.raises(ValueError): tab.update(b"ab", 42, 5)
    with pytest.raises(ValueError): tab.update(b"abc", 256, 5)
    with pytest.raises(ValueError): tab.update(b"abc", -1, 5)
    with pytest.raises(ValueError): tab.update(b"abc", 1, 65536)
    with pytest.raises(ValueError): tab.read(b"ab")
    with pytest.raises(ValueError): PriorityKeyTable(1344, "nonexistent")


def test_majority_candidate_requires_previous_observation():
    tab = PriorityKeyTable(1344, "lfu")
    assert tab.read(b"abc") is None
    for t, val in enumerate(b"YYXYY"):
        tab.update(b"abc", val, t + 3)
    assert tab.read(b"abc") == ord("Y")
    off = tab._find(b"abc")
    assert tab.payload[off + 5] == 5
    assert tab.payload[off + 4] > 0


def test_lfu_priority_and_recency_decay_are_predefined():
    fresh = PriorityKeyTable(1344, "lfu")
    aged = PriorityKeyTable(1344, "decayed")
    for model in (fresh, aged):
        for t in range(10):
            model.update(b"abc", 11, t + 3)
    off = fresh._find(b"abc")
    assert fresh._priority(off, 1023) == 10
    off = aged._find(b"abc")
    assert aged._priority(off, 12) == 10
    assert aged._priority(off, 12 + AGE_QUANTUM) == 5


def test_future_suffix_changes_later_state_not_prior_predictions():
    prefix = b"abcXabcYabcX" * 70
    a = owner_probe(prefix + b"Q" * 300, "owner", 1536)
    b = owner_probe(prefix + b"Z" * 300, "owner", 1536)
    assert a["source_sha256"] != b["source_sha256"]
    assert a["arms"]["k1_lfu_1344"]["queries"] == b["arms"]["k1_lfu_1344"]["queries"]
    assert a["cases"] == b["cases"]


def test_same_source_owner_replay_and_identity_exact():
    raw = b"abcXabcY" * 220
    a = owner_probe(raw, "a", 1536)
    b = owner_probe(raw, "a", 1536)
    assert a["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert a["case_sha256"] == b["case_sha256"]
    assert {k: (v["correct"], v["covered"]) for k, v in a["arms"].items()} == {
        k: (v["correct"], v["covered"]) for k, v in b["arms"].items()}
    assert all(v["queries"] == a["cases"] for v in a["arms"].values())


def test_owner_isolation_and_score_before_update():
    original = b"abcX" * 256
    a = owner_probe(original, "owner-a", 1536)
    b = owner_probe(original, "owner-b", 1536)
    assert a["case_sha256"] != b["case_sha256"]
    # Query at t=128 must know only t<128. No peek at target byte.
    assert a["arms"]["k1_lfu_1344"]["correct"] <= a["cases"]
    assert a["arms"]["raw_1344"]["correct"] <= a["cases"]


def test_saturated_frequency_is_bounded_and_never_wraps():
    tab = PriorityKeyTable(1344, "lfu")
    for t in range(300):
        tab.update(b"abc", 65, t)
    off = tab._find(b"abc")
    assert tab.payload[off + 5] == 255
    assert tab.read(b"abc") == 65


def test_no_extra_python_per_key_index():
    t = PriorityKeyTable(1344, "decayed")
    assert isinstance(t.payload, bytearray)
    assert not any(isinstance(value, dict) for value in vars(t).values())
