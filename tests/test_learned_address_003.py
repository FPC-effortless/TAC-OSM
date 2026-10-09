"""Pre-measurement integrity and identifiability tests for Address-003."""
import pytest
import torch

from scripts import run_plm_learned_address_003 as runner
from tac_osm.learned_address_002 import LearnedAddressMetric002
from tac_osm.learned_address_002_benchmark import make_trial_batch


def test_contract_is_frozen_and_runtime_constants_match():
    contract = runner.validate_contract()
    assert contract["status"] == "pre-registered"
    assert len(contract["arms"]) == 6
    assert contract["thresholds"]["mixed_isotropic_learned_minus_raw_mean_min"] == -0.02
    assert contract["thresholds"]["mixed_structured_M32_high_noise_gain_mean_min"] == 0.03


def test_channel_mix_is_deterministic_and_has_no_eval_label():
    assert [runner.training_condition(i, "mixed_unlabeled") for i in range(8)] == [
        (True, 0.2), (False, 0.2),
        (True, 0.4), (False, 0.4),
        (True, 0.2), (False, 0.2),
        (True, 0.4), (False, 0.4),
    ]


def test_reference_schedule_is_exactly_structured_only():
    assert [runner.training_condition(i, "structured_only") for i in range(4)] == [
        (True, 0.2), (True, 0.4),
        (True, 0.2), (True, 0.4),
    ]


def test_nonregistered_training_arm_rejected():
    with pytest.raises(ValueError, match="unregistered"):
        runner.training_condition(0, "infer_channel")


def test_training_budgets_and_model_architecture_match():
    for arm in ("structured_only", "mixed_unlabeled"):
        model, meta = runner.train(seed=3, arm=arm, steps=4)
        assert meta["steps"] == 4 and meta["train_batch"] == runner.TRAIN_BATCH
        assert meta["parameter_changed"] and meta["training_gradient_pass"]
        assert isinstance(model, LearnedAddressMetric002)
    assert runner.train(3, "structured_only", steps=1)[1]["initial_parameter_hash"] == (
        runner.train(3, "mixed_unlabeled", steps=1)[1]["initial_parameter_hash"]
    )


def test_mixed_condition_has_no_corruption_label_input():
    model = LearnedAddressMetric002(address_dim=16)
    gen = torch.Generator().manual_seed(44)
    keys, query, _target, _payload = make_trial_batch(
        gen, batch=4, memory=8, sigma=0.2, structured=True
    )
    # The scorer accepts only query and keys. It cannot read a channel label.
    assert model(query, keys).shape == (4, 8)
    with pytest.raises(TypeError):
        model(query, keys, True)


def test_identical_evaluation_batch_for_both_models_and_all_controls(monkeypatch):
    # A very small smoke run; the *confirmatory* 10k trials are CI-only.
    monkeypatch.setattr(runner, "TRIALS", 20)
    torch.manual_seed(55)
    model_a = LearnedAddressMetric002(address_dim=16)
    model_b = LearnedAddressMetric002(address_dim=16)
    a = runner.evaluate_condition(
        {"structured_only": model_a, "mixed_unlabeled": model_b},
        seed=0, memory=8, sigma=0.2, structured=True
    )
    b = runner.evaluate_condition(
        {"structured_only": model_a, "mixed_unlabeled": model_b},
        seed=0, memory=8, sigma=0.2, structured=True
    )
    assert a["evaluation_fingerprint"] == b["evaluation_fingerprint"]
    assert a["raw_dot_accuracy"] == b["raw_dot_accuracy"]
    assert a["arms"]["structured_only"] == a["arms"]["mixed_unlabeled"]
    for arm in a["arms"].values():
        assert arm["permutation_identity_accuracy"] == 1.0
        assert 0 <= arm["oracle_query_accuracy"] <= 1
        assert 0 <= arm["wrong_key_query_accuracy"] <= 1


def test_distinct_evaluation_conditions_have_distinct_fingerprints(monkeypatch):
    monkeypatch.setattr(runner, "TRIALS", 20)
    m = LearnedAddressMetric002()
    models = {"structured_only": m, "mixed_unlabeled": m}
    a = runner.evaluate_condition(
        models, seed=4, memory=8, sigma=0.2, structured=False
    )
    b = runner.evaluate_condition(
        models, seed=4, memory=8, sigma=0.2, structured=True
    )
    assert a["evaluation_fingerprint"] != b["evaluation_fingerprint"]


def test_previous_experiment_generator_remains_unmodified():
    assert runner.BENCHMARK_PATH.name == "learned_address_002_benchmark.py"
    assert runner.MODEL_PATH.name == "learned_address_002.py"
    assert runner.EVAL_NAMESPACE == 8_300_000
    assert runner.TRAIN_NAMESPACE == 1_900_000


def test_permutation_control_matches_relabelled_unique_winner():
    scores = torch.tensor([[0.1, 0.4, 0.9], [0.8, 0.1, 0.4]])
    permutation = torch.tensor([2, 0, 1])
    runner.require_slot_permutation_identity(
        scores, scores[:, permutation], permutation,
        seed=0, memory=3, sigma=0.2, structured=True, arm="mixed_unlabeled",
    )


def test_permutation_control_fails_closed_even_on_exact_logit_ties():
    scores = torch.tensor([[0.4, 0.4, 0.1]])
    permutation = torch.tensor([1, 0, 2])
    with pytest.raises(
        AssertionError, match="original_top2_margin=0"
    ):
        runner.require_slot_permutation_identity(
            scores, scores[:, permutation], permutation,
            seed=0, memory=3, sigma=0.4, structured=True,
            arm="structured_only",
        )
