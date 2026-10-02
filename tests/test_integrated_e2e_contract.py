import json
from pathlib import Path

from scripts.run_integrated_e2e_001 import (
    BITS,
    HELDOUT,
    STEPS,
    BATCH_SIZE,
    TRAIN_COMBOS,
)


ROOT = Path(__file__).resolve().parents[1]


def test_registered_contract_matches_runner():
    contract = json.loads(
        (ROOT / "contracts" / "TACOSM-PLM-INTEGRATED-E2E-001.json").read_text()
    )
    assert contract["status"] == "registered"
    assert contract["protocol"]["seeds"] == [0, 1, 2, 3, 4]
    assert contract["protocol"]["train_steps"] == STEPS
    assert contract["protocol"]["batch_size"] == BATCH_SIZE
    assert contract["protocol"]["latent_bits"] == BITS == 12
    assert tuple(
        map(tuple, contract["protocol"]["query_pairs_train_excluded_from_test"])
    ) == HELDOUT
    assert contract["primary_endpoint"]["name"] == "heldout_q2_accuracy_all_modalities"
    assert contract["primary_endpoint"]["threshold"] == 0.80


def test_theoretical_operator_prior_baseline_is_above_random_chance():
    heldout_ops = [op for op, _, _ in HELDOUT]
    assert all(heldout_ops.count(op) == 2 for op in ("xor", "and", "or", "xnor"))
    baseline = (2 * 0.50 + 2 * 0.75 + 2 * 0.75 + 2 * 0.50) / 8
    assert abs(baseline - 0.625) < 1e-12


def test_training_pairs_are_unordered_and_holdout_disjoint():
    train_pairs = {(op, i, j) for op, i, j in TRAIN_COMBOS}
    holdout_pairs = {(op, i, j) for op, i, j in HELDOUT}
    assert all(i < j for _, i, j in TRAIN_COMBOS + HELDOUT)
    assert train_pairs.isdisjoint(holdout_pairs)
    assert all((op, j, i) not in train_pairs for op, i, j in HELDOUT)
