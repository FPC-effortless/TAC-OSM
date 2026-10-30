"""Constructed receipts test resource enforcement, never model capability."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from tac_osm.research_objective_budget import (
    OperatorLedger, canonical, digest, validate_comparison, validate_program,
)
from scripts.validate_research_parity import check_manifest, ParityError


def ledger(stage="forward_backward", calls=1, name="aten::mm"):
    return [{"stage": stage, "signature": json.dumps([name, [["torch.float32", [4, 8]], ["torch.float32", [8, 4]]],
                                                               [["torch.float32", [4, 4]]]], separators=(",", ":")),
             "calls": calls}]


def fixture(root):
    source = root / "fixture_source.py"
    source.write_text("# Constructed source binding, no model fit.\n")
    sources = {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}
    recipe = {"steps": 3, "batch_size": 4, "context": 272, "learning_rate": .001,
              "weight_decay": .01, "grad_clip": 1., "betas": [.9, .999], "eps": 1e-8,
              "foreach": False}
    optimizer = {"name": "AdamW", "learning_rate": .001, "weight_decay": .01, "grad_clip": 1.,
                 "betas": [.9, .999], "eps": 1e-8, "amsgrad": False, "foreach": False,
                 "clip_foreach": False, "fused": False}
    source_hash = "a" * 64
    rows, profiles = [], []
    common = {key: digest(key) for key in ("train_split_sha256", "dev_split_sha256",
              "feature_contract_sha256", "target_contract_sha256", "optimizer_schedule_sha256",
              "tuning_policy_sha256", "selection_policy_sha256", "evaluation_contract_sha256")}
    for seed in (1701, 1702):
        for objective in ("full", "query"):
            budget = {"steps": 3, "examples_seen": 12, "tokens_seen": 1200, "batch_size": 4,
                      "history_events": 272, "dense_compute_tokens": 3264, "tuning_trials": 0, "tuning_flops": 0}
            model = {"name": objective, "objective": objective, "seed": seed, "architecture_family": "tac",
                     "trainable_parameters": 100000, "deployed_parameters": 100000,
                     "dataset_sha256": source_hash, "training_sample_order_sha256": digest(seed),
                     **common, "budget": budget}
            operators = ledger()
            p = {"family": "tac", "objective": objective, "seed": seed,
                 "trainable_parameters": 100000, "deployed_parameters": 100000,
                 "fit_sha256": common["train_split_sha256"], "schedule_sha256": digest(seed),
                 "weights_before": digest([seed, "weights"]), "weights_after": digest([seed, "weights"]),
                 "all_gradients_connected_and_finite": True, "input_shape": [4, 272],
                 "planned_steps": 3, "planned_examples_seen": 12, "planned_tokens_seen": 1200,
                 "planned_dense_compute_tokens": 3264, "operators": operators,
                 "operator_sha256": digest(operators), "optimizer_steps": 0}
            first = ledger("optimizer", name="aten::mul")
            warm = ledger("optimizer", 2, name="aten::mul")
            model["measured"] = {"trainable_parameters": 100000, "deployed_parameters": 100000,
                "checkpoint_sha256": digest([seed, objective]), "actual_steps": 3, "actual_examples": 12,
                "actual_tokens": 1200, "actual_dense_compute_tokens": 3264,
                "actual_sample_order_sha256": digest(seed),
                "forward_backward_operator_step_sha256": [p["operator_sha256"]] * 3,
                "optimizer_operator_step_sha256": [digest(first), digest(warm), digest(warm)],
                "optimizer_operators_first": first, "optimizer_operators_warm": warm}
            rows.append(model)
            profiles.append(p)
    profile = {"schema": "tac-osm.objective-operator-profile.v1", "family": "tac",
               "source_sha256": source_hash, "recipe_sha256": digest(recipe), "training_recipe": recipe,
               "implementation_sha256": digest(sources), "source_hashes": sources,
               "optimizer_contract": optimizer, "optimizer_steps": 0, "profiles": profiles}
    profile["hardware"] = {"torch_threads": 1, "cuda_used": False}
    path = root / "PROFILE_tac.json"
    path.write_bytes(canonical(profile))
    manifest = {"schema_version": "tac-osm.parity.v1", "study_id": "LANGACQUIRE097_BUDGET_FIXTURE",
        "study_type": "within_family_objective_comparison", "architecture_family": "tac",
        "execution_level": "internal_research", "registered_before_first_fit": True,
        "no_architecture_superiority_claim": True, "seeds": [1701, 1702],
        "compute_accounting": "within_family_complete_operator_ledger",
        "real_world_source": {"sha256": source_hash, "observed_real_outcomes": True,
            "name": "Constructed software source declaration", "license": "Constructed license declaration",
            "publisher_url": "https://example.org", "license_evidence_url": "https://example.org/license",
            "source_receipt": "fixture-software-only.json"},
        "research_scope": {"publicly_licensed": True, "partitions": ["FIT", "DEV"],
            "protected_evaluation": False, "reward_integrity_certification": False,
            "safety_claim": False, "checkpoint_promotion": False},
        "split_policy": {"frozen": True, "cohort_sha256": "f" * 64},
        "causal_protocol": "Constructed byte-only software contract", "checkpoint_policy": "Constructed hash-only fixture",
        "operator_profile_path": path.name, "operator_profile_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "training_recipe_sha256": digest(recipe), "implementation_sha256": digest(sources),
        "optimizer_contract": optimizer, "models": rows}
    manifest["literature_gap"] = {
        "unresolved_TAConly_question": "Constructed objective-specific comparison fixture",
        "what_is_not_being_rediscovered": "Recurrent acquisition and masked language losses are established",
        "established_work": [{"paper": "Recurrent burn-in learning", "url": "https://example.org/one",
                              "established_result": "Prior observed history can update recurrent state"},
                             {"paper": "Masked conditional learning", "url": "https://example.org/two",
                              "established_result": "Loss masks can focus later conditional prediction"}]}
    return manifest, profile


def rewrite_profile(root, manifest, profile):
    path = root / manifest["operator_profile_path"]
    path.write_bytes(canonical(profile))
    manifest["operator_profile_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_strict_gate_passes_pair_without_architecture_or_absolute_flop_claim(tmp_path, monkeypatch):
    m, _ = fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    for phase in ("preflight", "final"):
        result = check_manifest(m, phase)
        assert result["status"] == "PASS_WITHIN_FAMILY_OPERATOR_PARITY"
        assert result["eligible_for_architecture_superiority"] is False
        assert result["absolute_training_flops"] == "INCONCLUSIVE_NOT_CLAIMED"


@pytest.mark.parametrize("mutation", ["parameter", "data", "schedule", "steps", "optimizer", "family", "seed"])
def test_pair_mismatch_fails_closed(tmp_path, mutation):
    m, _ = fixture(tmp_path)
    r = m["models"][1]
    if mutation == "parameter":
        r["trainable_parameters"] = r["deployed_parameters"] = 100001
    elif mutation == "data":
        r["feature_contract_sha256"] = "0" * 64
    elif mutation == "schedule":
        r["training_sample_order_sha256"] = "0" * 64
    elif mutation == "steps":
        r["budget"]["steps"] = 4
    elif mutation == "optimizer":
        m["optimizer_contract"]["eps"] = 1e-6
    elif mutation == "family":
        r["architecture_family"] = "transformer"
    else:
        r["seed"] = 9
    with pytest.raises(ValueError):
        validate_comparison(m, root=tmp_path)


def test_changed_or_missing_profile_and_source_cannot_pass(tmp_path):
    m, _ = fixture(tmp_path)
    original = (tmp_path / "PROFILE_tac.json").read_bytes()
    (tmp_path / "PROFILE_tac.json").write_bytes(original + b" ")
    with pytest.raises(ValueError, match="profile changed"):
        validate_comparison(m, root=tmp_path)
    (tmp_path / "PROFILE_tac.json").write_bytes(original)
    (tmp_path / "fixture_source.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="source changed"):
        validate_comparison(m, root=tmp_path)


def test_opaque_differing_operator_is_never_estimated_as_zero(tmp_path):
    m, p = fixture(tmp_path)
    row = p["profiles"][1]
    row["operators"] += ledger(name="aten::unsupported_fused_magic")
    row["operator_sha256"] = digest(row["operators"])
    rewrite_profile(tmp_path, m, p)
    with pytest.raises(ValueError, match="operator signatures"):
        validate_comparison(m, root=tmp_path)


def test_prefit_weight_update_or_optimizer_step_blocks(tmp_path):
    m, p = fixture(tmp_path)
    p["optimizer_steps"] = 1
    rewrite_profile(tmp_path, m, p)
    with pytest.raises(ValueError, match="optimizer"):
        validate_comparison(m, root=tmp_path)
    p["optimizer_steps"] = 0
    p["profiles"][0]["weights_after"] = "0" * 64
    rewrite_profile(tmp_path, m, p)
    with pytest.raises(ValueError, match="changed model weights"):
        validate_comparison(m, root=tmp_path)


def test_postfit_requires_every_actual_step_and_optimizer_pair(tmp_path):
    m, _ = fixture(tmp_path)
    m["models"][0]["measured"]["forward_backward_operator_step_sha256"][1] = "0" * 64
    with pytest.raises(ValueError, match="every actual"):
        validate_comparison(m, "final", root=tmp_path)
    m, _ = fixture(tmp_path)
    measured = m["models"][1]["measured"]
    measured["optimizer_operators_warm"][0]["calls"] = 3
    measured["optimizer_operator_step_sha256"][1:] = [digest(measured["optimizer_operators_warm"])] * 2
    with pytest.raises(ValueError, match="optimizer warm"):
        validate_comparison(m, "final", root=tmp_path)


def test_no_independent_qualification_or_parameter_scale_bypass(tmp_path):
    m, _ = fixture(tmp_path)
    m["execution_level"] = "independent_qualification"
    with pytest.raises(ValueError, match="internal research"):
        validate_comparison(m, root=tmp_path)
    m, _ = fixture(tmp_path)
    m["models"][0]["trainable_parameters"] = 94999
    with pytest.raises(ValueError, match="100K"):
        validate_comparison(m, root=tmp_path)


def test_actual_dispatch_ledger_captures_forward_backward_and_mask_identity():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    torch.manual_seed(97001)
    weight = torch.randn(8, 4, requires_grad=True)
    inputs = torch.randn(4, 8)
    rows = []
    for mask in (torch.ones(4, 4), torch.eye(4)):
        weight.grad = None
        with OperatorLedger("forward_backward") as actual:
            y = inputs @ weight
            loss = (y.square() * mask).sum() / mask.sum()
            loss.backward()
        rows.append(actual.rows())
        assert actual.sha256() == digest(actual.rows())
        assert any("aten::mm" in item["signature"] for item in actual.rows())
    assert rows[0] == rows[1]


def program_fixture(root):
    original, profile = fixture(root)
    paths = {}
    for family in ("tac", "gru", "ssm", "transformer"):
        m, p = deepcopy(original), deepcopy(profile)
        m["architecture_family"] = p["family"] = family
        m["operator_profile_path"] = f"PROFILE_{family}.json"
        for row in m["models"]:
            row["architecture_family"] = family
        for row in p["profiles"]:
            row["family"] = family
        rewrite_profile(root, m, p)
        path = root / f"FAIR_{family}.json"
        path.write_bytes(canonical(m))
        paths[family] = path.name
    return {"study_id": "LANGACQUIRE097_PROGRAM_FIXTURE", "study_type": "within_family_objective_program",
            "registered_before_first_fit": True, "execution_level": "internal_research",
            "no_architecture_superiority_claim": True, "seeds": [1701, 1702],
            "real_world_source": original["real_world_source"], "family_manifests": paths}


def test_program_recursively_checks_all_families_and_cross_family_exposure(tmp_path):
    p = program_fixture(tmp_path)
    report = validate_program(p, root=tmp_path)
    assert report["models"] == 16
    assert report["eligible_for_architecture_superiority"] is False
    assert validate_program(p, "final", root=tmp_path)["status"] == "PASS_WITHIN_FAMILY_OBJECTIVE_PROGRAM"
    path = tmp_path / p["family_manifests"]["gru"]
    child = json.loads(path.read_text())
    for row in child["models"]:
        row["feature_contract_sha256"] = "9" * 64
    path.write_bytes(canonical(child))
    with pytest.raises(ValueError, match="cross-family data"):
        validate_program(p, root=tmp_path)
    del p["family_manifests"]["transformer"]
    with pytest.raises(ValueError, match="all four"):
        validate_program(p, root=tmp_path)
