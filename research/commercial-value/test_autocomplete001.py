"""Constructed-only evaluator diagnostics: NOT commercial or neural evidence."""
import importlib.util
import math
from collections import Counter
from pathlib import Path

import pytest
import torch

PROGRAM = Path(__file__).with_name("autocomplete001.py")
spec = importlib.util.spec_from_file_location("autocomplete001", PROGRAM)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_publisher_git_blob_function_identity():
    import hashlib
    b = b"abc"
    expected = hashlib.sha1(b"blob 3\0abc").hexdigest()
    assert m.git_blob(b) == expected
    assert m.sha256(b) == hashlib.sha256(b).hexdigest()


def test_fixed_score_and_offer_denominators():
    assert m.STOP - m.START == 512
    assert m.OFFER_LEN == 6
    assert m.OFFER_EVERY == 16
    assert len(m.SOURCE_BLOBS) == 3
    assert len(m.SEEDS) == 5 and len(m.ARMS) == 4
    assert len(set(m.SOURCE_BLOBS.values())) == 3


def test_greedy_rollout_consumes_only_predicted_bytes():
    # Constructed logits for 256 categories. No future string supplied.
    outputs = []
    def step(byte, state):
        outputs.append(byte)
        x = torch.full((1, 1, 256), -100.)
        x[0, 0, (byte + 1) % 256] = 9.
        return x, state + 1
    logits = torch.full((1, 1, 256), -100.)
    logits[0, 0, 50] = 9.
    suggestion = m.greedy_suggestion(step, logits, 0, 6)
    assert suggestion == bytes(range(50, 56))
    assert outputs == list(range(50, 55))


def test_fit_only_p3_never_changes_on_new_document():
    docs = {"fit_a": b"abcabcabcxyz", "fit_b": b"aaaaaabc"}
    p3 = m.FitOnlyP3(docs)
    before = p3.probabilities(b"abc")
    assert len(before) == 256 and math.isclose(sum(before), 1., abs_tol=1e-10)
    assert all(math.isfinite(x) and x > 0 for x in before)
    _ = m.score_count_doc(p3, "synthetic_fixture", b"abc123" * 120)
    assert p3.probabilities(b"abc") == before


def test_score_count_denominators_and_oracle_bound():
    p3 = m.FitOnlyP3({"f": b"abcdef" * 300})
    doc = m.score_count_doc(p3, "constructed_fixture", b"abcdef" * 120)
    assert doc["n"] == 512
    assert doc["offers"] == 32
    assert 0 <= doc["exact6_accepts_oracle"] <= 32
    assert doc["oracle_net_saved_keys"] == 5 * doc["exact6_accepts_oracle"]
    assert math.isfinite(doc["nll_sum"])
    assert doc["top1_count"] <= doc["top5_count"] <= 512


def test_aggregator_rejects_missing_owner():
    sample = {
        "owner": "fake", "n": 512, "nll_sum": 300.,
        "top1_count": 100, "top5_count": 200, "offers": 32,
        "exact6_accepts_oracle": 2, "oracle_net_saved_keys": 10,
        "persistent_state_bytes": 3,
    }
    with pytest.raises(ValueError, match="denominator"):
        m.aggregate([sample, sample])
    result = m.aggregate([sample, sample, sample])
    assert result["targets"] == 1536 and result["offers"] == 96
    assert result["oracle_net_saved_keys"] == 30


def test_archive_incompleteness_fails_closed_before_network(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ARCHIVE", tmp_path)
    with pytest.raises(FileNotFoundError):
        m.validate_archive()


def test_original_model_initial_state_is_compact():
    model = m.make_model("original")
    state = model.start(("constructed-owner",))
    assert m.state_bytes(state) == 1920
    assert sum(p.numel() for p in model.parameters()) == 100521
