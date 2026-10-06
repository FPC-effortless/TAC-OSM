import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from tac_osm.integrated_e2e_benchmark import BITS, BATCH_SIZE, HELDOUT, TRAIN_COMBOS
from tac_osm.integrated_e2e_benchmark import EVAL_EPISODES as EVAL_EPISODES_BENCHMARK
from tac_osm.contract import load_contract

ROOT = Path(__file__).resolve().parents[1]


def test_invalidated_e2e001_contract_is_not_executable():
    contract = json.loads(
        (ROOT / "contracts" / "TACOSM-PLM-INTEGRATED-E2E-001.json").read_text()
    )
    assert contract["status"] == "invalidated"
    assert contract["invalidation"]["confirmatory_result_admissible"] is False
    assert contract["invalidation"]["runner_allowed_in_confirmatory_ci"] is False


def test_e2e003_contract_matches_corrected_benchmark():
    contract = load_contract("TACOSM-PLM-INTEGRATED-E2E-003")
    assert contract.seeds == (0, 1, 2, 3, 4)
    assert contract.steps == 300
    assert contract.eval_steps == EVAL_EPISODES_BENCHMARK == 400
    assert contract.primary_endpoint.name == "heldout_q2_accuracy_all_modalities"
    assert contract.primary_endpoint.threshold == 0.80
    assert BITS == 12
    assert BATCH_SIZE == 96


def test_training_pairs_are_unordered_and_holdout_disjoint():
    train_pairs = set(TRAIN_COMBOS)
    holdout_pairs = set(HELDOUT)
    assert all(i < j for _, i, j in TRAIN_COMBOS + HELDOUT)
    assert train_pairs.isdisjoint(holdout_pairs)
    assert all((op, j, i) not in train_pairs for op, i, j in HELDOUT)


def test_heldout_query_space_is_cross_modal():
    modality = lambda bit: 0 if bit < 4 else 1 if bit < 8 else 2
    assert all(modality(i) != modality(j) for _, i, j in HELDOUT)
