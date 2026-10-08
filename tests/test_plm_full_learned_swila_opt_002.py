import json
from pathlib import Path

CONTRACT = Path("contracts/TACOSM-PLM-FULL-LEARNED-SWILA-OPT-002.json")
SCRIPT = Path("scripts/run_plm_full_learned_swila_opt_002.py")


def test_diagnostic_contract_has_three_registered_arms():
    c = json.loads(CONTRACT.read_text())
    assert [a["name"] for a in c["arms"]] == [
        "untrained", "trained_full", "trained_physics_off"
    ]


def test_diagnostic_contract_has_one_primary_endpoint():
    c = json.loads(CONTRACT.read_text())
    assert sum(bool(e["primary"]) for e in c["endpoints"]) == 1
    assert c["endpoints"][0]["name"] == "trained_minus_untrained_action_accuracy"


def test_exact_original_training_schedule_is_frozen():
    c = json.loads(CONTRACT.read_text())
    p = c["protocol"]
    assert p["train_steps"] == 220
    assert p["batch_size"] == 8
    assert p["learning_rate"] == 0.001
    assert p["weight_decay"] == 0.0001
    assert p["seeds"] == [0, 1, 2, 3, 4]


def test_only_registered_physics_weight_differs():
    text = SCRIPT.read_text()
    assert "FULL_PHYSICS_WEIGHT = 0.03" in text
    assert 'return train_seed(seed, 0.0)' in text
    assert "WORLD_WEIGHT = 0.20" in text
    assert "VERIFIER_WEIGHT = 0.50" in text
    assert "REPAIR_WEIGHT = 0.15" in text
    assert "SECOND_VERIFIER_WEIGHT = 0.25" in text


def test_untrained_arm_has_no_optimizer_update():
    text = SCRIPT.read_text()
    assert 'if arm == "untrained":' in text
    assert '"parameter_changed": False' in text
    assert 'first20_mean_total_loss": None' in text


def test_diagnostic_never_accepts_binary_success_as_model_input():
    text = SCRIPT.read_text()
    assert "reward_label" in text  # external supervision remains allowed
    assert "hidden_state" not in text
    assert "target_after" in text  # benchmark-side environment bookkeeping only
