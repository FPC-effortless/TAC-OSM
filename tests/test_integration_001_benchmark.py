"""Pre-model validity tests for frozen INTEGRATION-001 benchmark."""
import json
from pathlib import Path

import pytest

from tac_osm.integration_001_benchmark import (
    DELAYS, HISTORY_SIZES, fingerprint, make_pair, make_pool, model_inputs,
    validate_pair,
)


def test_contract_preregistered_and_complete():
    path = Path(__file__).resolve().parents[1] / "contracts/TACOSM-PLM-INTEGRATION-001.json"
    spec = json.loads(path.read_text())
    assert spec["status"] == "pre-registered"
    assert spec["experiment_id"] == "TACOSM-PLM-INTEGRATION-001"
    assert spec["seeds"] == spec["evaluation"]["seeds"] == [101, 211, 307, 401, 503]
    assert spec["h_levels"] == list(HISTORY_SIZES)
    assert spec["evaluation"]["intervening_writes"] == list(DELAYS)
    assert spec["evaluation"]["retrieval_budgets"] == [4, 8, 16]
    assert len(spec["arms"]) == 32
    assert len(spec["gates"]["void_if"]) >= 10
    assert len(spec["decision_rule"]) == 3
    assert spec["thresholds"]["minimum_persistent_accuracy"] == 0.9


@pytest.mark.parametrize("history_size", HISTORY_SIZES)
@pytest.mark.parametrize("delay", DELAYS)
def test_pair_is_causally_identifiable(history_size, delay):
    pair = make_pair(9_100_000 + history_size * 101 + delay, history_size, delay)
    validate_pair(pair, history_size, delay)
    left = model_inputs(pair.left)
    right = model_inputs(pair.right)
    assert left["writes"] == right["writes"]
    assert left["query_key"] == right["query_key"]
    assert left["current_bit"] == right["current_bit"]
    assert pair.left.target != pair.right.target
    assert sum(a != b for a, b in zip(left["history"], right["history"])) == 1
    assert len(left["writes"]) == delay
    for name in ("target", "target_slot", "gold", "oracle", "condition_label"):
        assert name not in left


def test_generator_reproducible_across_invocations():
    a = make_pair(12345, 128, 16)
    b = make_pair(12345, 128, 16)
    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a) != fingerprint(make_pair(12346, 128, 16))


def test_pool_fingerprints_unique_and_train_eval_disjoint():
    train = make_pool(11_000_000, (101, 211), 32, 4, 3)
    eval_ = make_pool(29_000_000, (101, 211), 32, 4, 3)
    fp_train = {fingerprint(x) for x in train}
    fp_eval = {fingerprint(x) for x in eval_}
    assert len(fp_train) == len(fp_eval) == 6
    assert fp_train.isdisjoint(fp_eval)


def test_unknown_condition_fails_closed():
    with pytest.raises(ValueError):
        make_pair(3, 7, 4)
    with pytest.raises(ValueError):
        make_pair(3, 32, 2)
    with pytest.raises(ValueError):
        make_pool(0, (0,), 32, 4, 0)


def test_reset_is_information_theoretically_limited_to_half():
    pair = make_pair(98, 32, 1)
    assert pair.left.target != pair.right.target
    # A reset classifier sees only the current observation and intervening
    # writes; both are byte-identical across the paired histories.
    a, b = model_inputs(pair.left), model_inputs(pair.right)
    assert (a["query_key"], a["current_bit"], a["writes"]) == (
        b["query_key"], b["current_bit"], b["writes"]
    )
    # Any deterministic prediction invariant to the initial history must
    # match exactly one of the two opposite labels.
    for guess in (0, 1):
        assert (guess == pair.left.target) + (guess == pair.right.target) == 1
