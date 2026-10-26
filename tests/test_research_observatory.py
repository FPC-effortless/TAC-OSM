"""No-fit adversarial tests; constructed tensors only, not capability evidence."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tac_osm.predictive_state_agent import PredictiveStateAgent
from tac_osm.research_observatory import (
    architecture_inventory, memory_tomography, input_dependency_map,
    gradient_conflict_atlas, interpret_snapshot,
)
from tac_osm.research_evidence import compare_evidence

ROOT = Path(__file__).resolve().parents[1]
H = lambda letter: letter * 64


def case():
    torch.manual_seed(211)
    model = PredictiveStateAgent(5, 3, 1).eval()
    history = torch.randn(2, 5, 5)
    actions = torch.randn(2, 3, 3)
    return model, history, actions


def test_inventory_counts_real_shared_parameters_not_calls():
    model, history, actions = case()
    original = copy.deepcopy(model.state_dict())
    def step(net):
        return net.observe(net.initial(2), history[:, 0])
    result = architecture_inventory(model, invocations={"observe": step})
    assert result["total_trainable"] == sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert result["total_deployed_learned"] == sum(p.numel() for p in model.parameters())
    assert "enc" in result["component_parameters"]
    calls = [item["module"] for item in result["phases"]["observe"]["ordered_leaf_calls"]]
    assert "observation" in calls and "next_observation" not in calls
    assert "next_observation" in result["phases"]["observe"]["unobserved_modules"]
    assert "not a computational DAG" in result["phases"]["observe"]["meaning"]
    for name, p in model.state_dict().items():
        assert torch.equal(p, original[name])


def test_memory_tomography_tracks_native_fast_slow_associative():
    model, history, actions = case()
    result = memory_tomography(model, history, actions)
    assert result["events"] == history.shape[1]
    assert all(set(("fast", "slow", "memory")).issubset(row) for row in result["timeline"])
    assert result["query_does_not_write_state"]
    assert result["candidate_permutation_max_abs_error"] < 1e-6
    assert "all_state" in result["interventions"]
    assert all(0 <= metrics["outcome_mean_absolute_change"] for metrics in result["interventions"].values())


def test_memory_probe_rejects_nonfinite_or_wrong_dimensions():
    model, history, actions = case()
    with pytest.raises(ValueError, match="nonfinite"):
        poisoned = history.clone()
        poisoned[0, 0, 0] = float("nan")
        memory_tomography(model, poisoned, actions)
    with pytest.raises(ValueError, match="feature dimensions"):
        memory_tomography(model, history[:, :, :-1], actions)


def test_dependency_map_marks_real_local_graph_connections():
    model, history, actions = case()
    deps = input_dependency_map(model, history, actions)["heads"]
    assert deps["outcome"]["associative_memory"]["connected"]
    assert not deps["next_observation"]["associative_memory"]["connected"]
    assert deps["next_observation"]["candidate_action"]["connected"]
    assert deps["next_observation"]["fast_state"]["connected"]
    assert deps["outcome"]["current_context"]["connected"]
    assert deps["next_observation"]["current_context"]["connected"] is False


def test_opposed_loss_gradients_detected_in_actual_shared_block():
    m = torch.nn.Linear(1, 1, bias=False)
    prediction = m(torch.ones(3, 1)).sum()
    report = gradient_conflict_atlas(m, {
        "objective_a": prediction,
        "objective_b": -prediction,
    })
    block = report["pairs"][0]["components"]["weight"]
    assert block["status"] == "CONFLICT_AT_THIS_POINT"
    assert block["cosine"] == pytest.approx(-1., abs=1e-7)


def test_disjoint_heads_are_not_mislabelled_as_agreement():
    class Separate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(1, 1)
            self.b = torch.nn.Linear(1, 1)
    m = Separate()
    x = torch.ones(1, 1)
    report = gradient_conflict_atlas(m, {
        "a": m.a(x).sum(), "b": m.b(x).sum(),
    })
    blocks = report["pairs"][0]["components"]
    assert blocks["<all_shared>"]["cosine"] is None
    assert blocks["<all_shared>"]["status"] == "NO_SHARED_GRADIENT"


def test_loss_atlas_rejects_nondifferentiable_or_nan_terms():
    m = torch.nn.Linear(1, 1)
    good = m(torch.ones(1, 1)).sum()
    with pytest.raises(ValueError, match="differentiable"):
        gradient_conflict_atlas(m, {"valid": good, "bad": torch.tensor(1.0)})
    with pytest.raises(ValueError, match="finite"):
        gradient_conflict_atlas(m, {"valid": good, "bad": good * float("nan")})


def test_interpretation_is_hypothesis_only():
    report = {"memory": {
        "query_does_not_write_state": False, "candidate_permutation_max_abs_error": .01,
        "interventions": {}}, "gradients": {
            "pairs": [{"left": "forecast", "right": "decision",
                       "components": {"encoder": {"status": "CONFLICT_AT_THIS_POINT"}}}]}}
    flags = interpret_snapshot(report)
    assert {x["area"] for x in flags["alerts"]} == {"prediction", "action_order", "losses"}
    assert "hypotheses" in flags["rule"]


def receipt(run_id, checkpoint, losses=(.5, .4)):
    return {
        "run_id": run_id, "source_sha256": H("a"), "fit_split_sha256": H("b"),
        "dev_split_sha256": H("c"), "test_split_sha256": H("d"),
        "feature_contract_sha256": H("e"), "target_contract_sha256": H("f"),
        "evaluation_contract_sha256": H("a"),
        "training_sample_order_sha256": H("b"),
        "optimizer_schedule_sha256": H("c"), "tuning_policy_sha256": H("d"),
        "checkpoint_sha256": checkpoint, "evaluation_metric": "absolute-error",
        "direction": "lower", "seed": 7, "examples_seen": 2000,
        "optimizer_steps": 100, "trainable_parameters": 100000,
        "deployed_parameters": 100000, "training_flops": 5e9,
        "observations": [
            {"id_sha256": H("1"), "value": losses[0], "group": "site-a", "horizon": "h1"},
            {"id_sha256": H("2"), "value": losses[1], "group": "site-b", "horizon": "h4"},
        ],
    }


def test_paired_receipt_comparison_does_not_claim_scientific_superiority():
    result = compare_evidence(receipt("A", H("8")), receipt("B", H("9"), (.2, .5)))
    assert result["status"] == "RECEIPT_COMPATIBLE"
    assert result["paired_sample_count"] == 2
    assert result["right_minus_left"]["mean"] == pytest.approx(-.1)
    assert result["right_minus_left"]["right_better_count"] == 1
    assert result["right_minus_left"]["left_better_count"] == 1
    assert result["eligible_for_architecture_superiority"] is False


@pytest.mark.parametrize("field,value", [
    ("source_sha256", H("f")),
    ("training_sample_order_sha256", H("f")),
    ("optimizer_steps", 101),
    ("trainable_parameters", 104000),
    ("training_flops", 6e9),
    ("seed", 8),
])
def test_scientific_receipt_mismatches_block_comparison(field, value):
    left, right = receipt("A", H("8")), receipt("B", H("9"))
    right[field] = value
    report = compare_evidence(left, right)
    assert report["status"] == "BLOCKED"
    assert not report.get("right_minus_left")


def test_missing_and_duplicate_paired_sample_ids_block():
    left, right = receipt("A", H("8")), receipt("B", H("9"))
    right["observations"][1]["id_sha256"] = H("1")
    assert compare_evidence(left, right)["status"] == "BLOCKED"
    right = receipt("B", H("9"))
    right["observations"][1]["id_sha256"] = H("3")
    assert compare_evidence(left, right)["status"] == "BLOCKED"


def test_mismatched_group_or_horizon_blocks_without_silent_stratum_change():
    a, b = receipt("A", H("8")), receipt("B", H("9"))
    b["observations"][0]["horizon"] = "h512"
    report = compare_evidence(a, b)
    assert "unmatched_group_or_horizon" in report["blockers"]


def test_nonfinite_metric_rejected_before_report():
    a, b = receipt("A", H("8")), receipt("B", H("9"))
    b["observations"][0]["value"] = float("nan")
    assert compare_evidence(a, b)["status"] == "BLOCKED"


def test_fixture_cli_emits_correct_scientific_limit():
    run = subprocess.run([sys.executable, str(ROOT / "scripts/run_research_observatory.py"),
                          "--fixture"], cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr + run.stdout[-1000:]
    result = json.loads(run.stdout)
    assert result["source_provenance"] == "constructed_untrained_fixture_no_capability"
    assert result["gradient_conflicts"]["status"] == "NOT_RUN_NO_AUTHENTIC_SUPERVISED_LOSSES"


def test_cli_pinned_input_and_checkpoint_rejects_wrong_hash(tmp_path):
    input_file = tmp_path / "input.json"
    input_file.write_text('{"history": [], "candidates": []}')
    run = subprocess.run([sys.executable, str(ROOT / "scripts/run_research_observatory.py"),
                          "--input-json", str(input_file), "--checkpoint", str(input_file),
                          "--expected-input-sha256", "0" * 64,
                          "--expected-checkpoint-sha256", "0" * 64], cwd=ROOT,
                         capture_output=True, text=True)
    assert run.returncode == 2
    assert json.loads(run.stdout)["status"] == "BLOCKED"
