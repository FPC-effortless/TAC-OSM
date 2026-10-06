import inspect

import pytest

torch = pytest.importorskip("torch")
from tac_osm.functional_repair_signed_casm import (
    FunctionalRepairPLM,
    SignedBooleanCASM,
    signed_boolean_margins,
)


def test_signed_boolean_margin_truth_table():
    cases = (
        (-2.0, -3.0, [0, 0, 0, 1]),
        (-2.0, 3.0, [1, 0, 1, 0]),
        (2.0, -3.0, [1, 0, 1, 0]),
        (2.0, 3.0, [0, 1, 1, 1]),
    )
    margins = []
    for a, b, expected in cases:
        m = signed_boolean_margins(
            torch.tensor([a]),
            torch.tensor([b]),
        ).squeeze(0)
        predicted = (m >= 0).to(torch.long).tolist()
        margins.append(predicted)
    assert margins == [case[2] for case in cases]


def test_signed_casm_returns_boolean_actions_at_threshold():
    torch.manual_seed(5006)
    model = SignedBooleanCASM(hidden=40, latent_bits=12)
    state = torch.randn(4, 40)
    i = torch.tensor([0, 0, 0, 0])
    j = torch.tensor([1, 1, 1, 1])
    op = torch.tensor([0, 1, 2, 3])
    action, logits, bits = model(state, i, j, op)
    assert action.shape == (4,)
    assert logits.shape == (4, 2)
    assert bits.shape == (4, 12)
    assert torch.equal((action >= 0.5), (logits[:, 1] >= logits[:, 0]))


def test_repair_model_preserves_label_free_pre_action_api():
    assert list(inspect.signature(FunctionalRepairPLM.query).parameters) == [
        "self", "memory", "entity", "i", "j", "op"
    ]
    assert "answer" not in inspect.getsource(FunctionalRepairPLM.query)
    assert not hasattr(FunctionalRepairPLM, "forward_episode")


def test_development_runner_never_imports_retired_holdout():
    source = inspect.getsource(
        __import__(
            "scripts.run_plm_functional_repair_dev",
            fromlist=["sample_dev_episodes"],
        )
    )
    assert "HELDOUT" not in source
    assert "TRAIN_COMBOS" in source
    assert "development_only" in source


def test_development_sampling_uses_only_training_compositions():
    import random
    from tac_osm.integrated_e2e_005_benchmark import TRAIN_COMBOS
    from scripts.run_plm_functional_repair_dev import sample_dev_episodes

    episodes = sample_dev_episodes(random.Random(7006), 128)
    for episode in episodes:
        for q in episode[1:3]:
            assert (q[3], q[1], q[2]) in TRAIN_COMBOS

def test_completion_run_is_fixed_and_development_only():
    from pathlib import Path
    source = (Path(__file__).resolve().parent.parent / "scripts" / "run_plm_functional_completion_dev.py").read_text(
        encoding="utf-8"
    )
    assert "steps=1200" in source
    assert "TRAIN_COMBOS" not in source
    assert "HELDOUT" not in source
    assert "not_scientific_evidence" in source
    assert "retired_e2e005_holdout_used" in source


def test_functional_runner_default_horizon_remains_300():
    import scripts.run_plm_functional_repair_dev as dev
    assert dev.STEPS == 300
