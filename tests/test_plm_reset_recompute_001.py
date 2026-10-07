import inspect
import json
import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch as th

from tac_osm.contract import load_contract
from tac_osm.integrated_e2e_008_benchmark import sample_evaluation_episodes
from tac_osm.integrated_e2e_005 import FunctionalConfig, FunctionalMultimodalPLM
from scripts.run_plm_reset_recompute_001 import (
    BATCH_SIZE,
    PARAM_COUNT,
    REFERENCE_FINGERPRINTS,
    EVAL_N,
    write_memory,
)

ROOT = Path(__file__).resolve().parents[1]


def test_reset_contract_is_current_and_sound():
    c = load_contract("TACOSM-PLM-PERSISTENCE-RESET-RECOMPUTE-001")
    assert c.check_consistency() == []
    assert c.h_levels == (3,)
    assert c.seeds == (0, 1, 2, 3, 4)
    assert c.steps == 300
    assert c.eval_steps == 600
    assert [a.name for a in c.arms] == ["normal", "reset_recompute"]
    assert c.primary_endpoint() == "heldout_q2_accuracy"


def test_reference_fingerprints_are_pinned_for_all_seeds():
    assert set(REFERENCE_FINGERPRINTS) == {0, 1, 2, 3, 4}
    assert len({*REFERENCE_FINGERPRINTS.values()}) == 5


def test_reset_recompute_uses_observations_only():
    signature = inspect.signature(write_memory)
    assert list(signature.parameters) == ["model", "rows"]
    source = inspect.getsource(write_memory)
    assert "outcome" not in source
    assert "target" not in source
    assert "verifier" not in source
    assert "correct" not in source


def test_recomputed_state_is_identical_before_the_action_boundary():
    model = FunctionalMultimodalPLM(
        config=FunctionalConfig(hidden_dim=64, state_write_mode="residual_linear")
    )
    assert sum(p.numel() for p in model.parameters()) == PARAM_COUNT
    episodes = sample_evaluation_episodes(random.Random(181000), 2)
    for ep in episodes:
        a = write_memory(model, ep[0])
        b = write_memory(model, ep[0])
        assert th.equal(a, b)


def test_reset_experiment_does_not_change_reference_evaluation_protocol():
    contract = json.loads(
        (ROOT / "contracts/TACOSM-PLM-PERSISTENCE-RESET-RECOMPUTE-001.json").read_text()
    )
    legacy = contract["legacy_reference"]
    assert legacy["experiment_id"] == "TACOSM-PLM-INTEGRATED-E2E-008"
    assert legacy["parameter_count"] == PARAM_COUNT
    assert len(legacy["evaluation_fingerprints"]) == 5
    assert contract["leakage_rules"][-1].startswith(
        "this experiment is an attribution follow-up"
    )
