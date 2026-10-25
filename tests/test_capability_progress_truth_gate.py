"""Synthetic JSON declarations test software policy only; NOT model evidence."""
import copy
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "capability_progress_gate", Path(__file__).resolve().parents[1] / "scripts/validate_capability_progress.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fixture(tier="DEV_GAIN"):
    return {
        "schema_version": "tac-osm.capability-progress.v1",
        "study_id": "FUTURE_PROSPECTIVE_STUDY",
        "tier": tier,
        "what_improved": "lower next-event NLL",
        "what_regressed": "none",
        "what_remains_unknown": "transfer and real economic benefit",
        "source": {
            "authentic_real_data": True,
            "exposure": "EXPOSED_DEV",
            "sha256": "a" * 64,
            "eligible_ids_sha256": "b" * 64,
            "holdout_unopened_before_freeze": False,
        },
        "task": {
            "prediction_time_validated": True,
            "independent_cases": 150,
            "actual_future_targets": 135,
            "target": "next genuinely recorded activity",
            "query_and_horizon": "after six observations, one event ahead",
            "same_case_ids_target_horizon_metric": True,
        },
        "model": {
            "prior_sha256": "c" * 64,
            "candidate_sha256": "d" * 64,
        },
        "integrity": {
            "git_original_checkpoints": True,
            "independent_clean_replay": True,
            "raw_case_scores_preserved": True,
            "registered_before_evaluation": True,
            "independent_raw_case_audit": False,
        },
        "primary_comparison": {
            "direction": "lower_is_better",
            "prior": 0.55,
            "candidate": 0.45,
            "minimum_useful_gain": 0.03,
            "paired_case_cluster_95ci_gain": [0.05, 0.16],
            "paired_seed_count": 5,
            "beneficial_seed_count": 5,
        },
        "guardrails": [
            {"metric": "interval coverage", "status": "PASS"},
            {"metric": "early-branch risk", "status": "PASS"},
        ],
        "practical_reference_scored": True,
        "independent_real_distributions": 1,
    }


def independent():
    x = fixture("INDEPENDENT_GAIN")
    x["source"]["exposure"] = "UNTOUCHED_REAL"
    x["source"]["holdout_unopened_before_freeze"] = True
    x["integrity"]["independent_raw_case_audit"] = True
    return x


def test_development_gain_is_provisional_and_structurally_valid():
    assert mod.check(fixture()) == []


def test_independent_gain_requires_real_holdout_and_ci():
    assert mod.check(independent()) == []
    a = independent()
    a["source"]["exposure"] = "EXPOSED_TEST"
    assert any("cannot use" in s for s in mod.check(a))
    a = independent()
    a["primary_comparison"]["paired_case_cluster_95ci_gain"][0] = 0.01
    assert any("lower CI" in s for s in mod.check(a))
    a = independent()
    a["primary_comparison"]["beneficial_seed_count"] = 3
    assert any("4/5" in s for s in mod.check(a))


def test_no_fake_independence_or_task_changes():
    a = independent()
    a["task"]["independent_cases"] = 10
    assert any("power justification" in s for s in mod.check(a))
    a = independent()
    a["task"]["same_case_ids_target_horizon_metric"] = False
    assert any("identical cases" in s for s in mod.check(a))
    a = independent()
    a["task"]["actual_future_targets"] = 1000
    assert any("future target count" in s for s in mod.check(a))


def test_material_gain_and_bad_calibration_block_promotion():
    a = independent()
    a["primary_comparison"]["candidate"] = 0.535
    assert any("material threshold" in s for s in mod.check(a))
    a = independent()
    a["guardrails"][0]["status"] = "FAIL"
    assert any("guardrail" in s for s in mod.check(a))
    a = independent()
    a["integrity"]["git_original_checkpoints"] = False
    assert any("checkpoint" in s for s in mod.check(a))


def test_mixed_tradeoff_requires_a_visible_failure():
    a = fixture("MIXED_TRADEOFF")
    assert any("failing guardrail" in s for s in mod.check(a))
    a["guardrails"][0]["status"] = "FAIL"
    assert mod.check(a) == []


def test_transfer_and_customer_claims_require_extra_evidence():
    a = independent()
    a["tier"] = "TRANSFER_REPLICATED"
    assert any("independent real distributions" in s for s in mod.check(a))
    a["independent_real_distributions"] = 2
    assert mod.check(a) == []
    a = independent()
    a["tier"] = "OPERATIONAL_VALUE_VERIFIED"
    assert any("prospective_pilot" in s for s in mod.check(a))
    a["prospective_pilot"] = {
        "authorized_external_customer": True,
        "real_outcomes_and_costs_verified": True,
        "preregistered_net_benefit_confirmed": True,
    }
    assert mod.check(a) == []


def test_unverifiable_replay_and_fake_source_fail():
    a = fixture()
    a["integrity"]["independent_clean_replay"] = False
    assert any("replay" in s for s in mod.check(a))
    a = fixture()
    a["source"]["sha256"] = "unknown"
    assert any("SHA256" in s for s in mod.check(a))


def test_nonprogress_and_blocked_claims_are_not_forced_positive():
    a = fixture("BLOCKED")
    assert mod.check(a) == []
    a["tier"] = "NOT_REAL"
    assert any("evidence tier" in s for s in mod.check(a))
