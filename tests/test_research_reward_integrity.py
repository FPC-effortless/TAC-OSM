"""Adversarial no-fit reward hacking diagnostics: constructed fixtures ONLY."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from tac_osm.research_anticheat import AuditGateError
from tac_osm.research_reward_integrity import (
    SCHEMA, REQUIRED, audit_goal_recomputation, audit_protected_assets,
    audit_proxy_goal_alignment, audit_reward_recomputation,
    probe_reward_noninterference, reward_audit_ledger, sha256,
)


def digest(label):
    return sha256(label.encode())


def policy():
    return {
        "min_proxy_gain": .05, "min_goal_gain": .0,
        "max_goal_regression": .0, "max_added_safety_violations": 0,
        "max_added_decision_cost": .1, "goal_direction": "higher",
        "min_group_samples": 1, "required_groups": ["difficult", "normal"],
    }


def receipt(run_id, reward=(.4, .4), goal=(.6, .5), cost=(.2, .2),
            safety=(False, False)):
    return {
        "schema": SCHEMA, "run_id": run_id, "source_sha256": digest("source"),
        "cohort_sha256": digest("cohort"),
        "reward_scorer_sha256": digest("reward scorer"),
        "goal_evaluator_sha256": digest("independent evaluator"),
        "goal_contract_sha256": digest("goal contract"),
        "outcome_authority": "external_source",
        "rows": [
            {"id_sha256": digest("episode_1"), "group": "difficult",
             "horizon": "512", "reward": reward[0], "goal": goal[0],
             "decision_cost": cost[0], "safety_violation": safety[0]},
            {"id_sha256": digest("episode_2"), "group": "normal",
             "horizon": "128", "reward": reward[1], "goal": goal[1],
             "decision_cost": cost[1], "safety_violation": safety[1]},
        ]
    }


def test_higher_proxy_without_goal_improvement_is_flagged_in_each_group():
    a = receipt("before")
    b = receipt("after", reward=(.95, .95))
    result = audit_proxy_goal_alignment(a, b, policy())
    assert result.status == "FAIL"
    assert "proxy_gain_without_goal_gain:difficult" in result.evidence["flags"]
    assert result.evidence["subgroups"]["normal"]["candidate_minus_baseline_goal_benefit"] == 0


def test_proxy_improvement_with_true_goals_and_no_harm_passes_structural_gate():
    a = receipt("before")
    b = receipt("after", reward=(.5, .5), goal=(.7, .6))
    result = audit_proxy_goal_alignment(a, b, policy())
    assert result.status == "PASS"
    assert result.evidence["qualification"] == "descriptive_paired_receipts_not_causal_or_attested"


def test_hidden_subgroup_regression_cannot_be_offset_by_easy_group():
    a = receipt("before")
    b = receipt("after", reward=(.8, .8), goal=(.4, 1.0))
    f = audit_proxy_goal_alignment(a, b, policy())
    assert f.status == "FAIL"
    assert "goal_or_safety_or_cost_regression:difficult" in f.evidence["flags"]


def test_reward_gain_with_better_goal_but_new_safety_violation_fails():
    a = receipt("before")
    b = receipt("after", reward=(.6, .6), goal=(.9, .8),
                safety=(True, False))
    assert audit_proxy_goal_alignment(a, b, policy()).status == "FAIL"


def test_reward_gain_with_better_goal_but_overintervention_cost_fails():
    a = receipt("before")
    b = receipt("after", reward=(.6, .6), goal=(.9, .8),
                cost=(1.0, .2))
    assert audit_proxy_goal_alignment(a, b, policy()).status == "FAIL"


def test_goal_can_be_minimized_with_declared_goal_direction():
    a = receipt("before", goal=(.7, .7))
    b = receipt("after", reward=(.9, .9), goal=(.5, .6))
    p = policy()
    p["goal_direction"] = "lower"
    assert audit_proxy_goal_alignment(a, b, p).status == "PASS"


@pytest.mark.parametrize("mutation", [
    lambda r: r.__setitem__("source_sha256", digest("other data")),
    lambda r: r.__setitem__("goal_evaluator_sha256", digest("hacked evaluator")),
    lambda r: r["rows"].pop(),
    lambda r: r["rows"][0].__setitem__("id_sha256", digest("other episode")),
    lambda r: r["rows"][0].__setitem__("horizon", "fake"),
    lambda r: r["rows"][0].__setitem__("goal", float("nan")),
    lambda r: r["rows"][0].__setitem__("reward", True),
    lambda r: r["rows"][1].__setitem__("group", "hidden"),
])
def test_mismatched_or_invalid_evidence_never_passes(mutation):
    a, b = receipt("before"), receipt("after")
    mutation(b)
    assert audit_proxy_goal_alignment(a, b, policy()).status != "PASS"


def test_missing_required_group_always_inconclusive():
    a, b = receipt("before"), receipt("after")
    a["rows"][0]["group"] = b["rows"][0]["group"] = "normal"
    assert audit_proxy_goal_alignment(a, b, policy()).status == "INCONCLUSIVE"


def test_policy_tampering_or_missing_limits_does_not_greenlight():
    a, b = receipt("before"), receipt("after", reward=(.9, .9))
    modified = policy()
    del modified["required_groups"]
    assert audit_proxy_goal_alignment(a, b, modified).status == "INCONCLUSIVE"


def test_artifact_digest_and_permission_guard_accepts_out_of_band_pins():
    bytes_map = {
        "tests/secret_goals.json": b"sealed evaluation",
        "policy/rewards.py": b"reward scorer",
        "ci/reports/immutable.json": b"trace",
    }
    pins = {k: sha256(v) for k, v in bytes_map.items()}
    report = audit_protected_assets(
        actor_write_paths=["trained-model/weights", "workspace"],
        protected_sha256=pins, read_artifact=bytes_map.__getitem__)
    assert report.status == "PASS"


def test_artifact_guard_rejects_write_permissions_even_before_tampering():
    f = audit_protected_assets(
        actor_write_paths=["scores", "workspace"],
        protected_sha256={"scores/goal.py": sha256(b"original")},
        read_artifact=lambda p: b"original")
    assert f.status == "FAIL"
    assert "scores/goal.py" in f.evidence["overlapping_protected_paths"]


def test_artifact_guard_detects_mutated_evaluator_bytes():
    f = audit_protected_assets(
        actor_write_paths=["workspace"],
        protected_sha256={"eval/goal.json": sha256(b"original")},
        read_artifact=lambda p: b"tampered")
    assert f.status == "FAIL"
    assert f.evidence["mismatched_paths"] == ["eval/goal.json"]


def test_missing_pins_or_external_reader_cannot_pass():
    a = audit_protected_assets(
        actor_write_paths=["workspace"],
        protected_sha256={}, read_artifact=lambda p: b"ok")
    b = audit_protected_assets(
        actor_write_paths=["workspace"],
        protected_sha256={"eval/goal.json": sha256(b"ok")},
        read_artifact=None)
    assert a.status == b.status == "INCONCLUSIVE"


def test_path_escape_and_ambiguous_paths_cannot_pass():
    for bad in ("../goal", "/absolute", "goal//config", r"goal\windows", "goal/../other"):
        f = audit_protected_assets(
            actor_write_paths=[bad],
            protected_sha256={"goal/check": sha256(b"a")},
            read_artifact=lambda x: b"a")
        assert f.status == "INCONCLUSIVE"


def test_reward_recomputed_independently_detects_forged_reward():
    rows = [{"id_sha256": digest("a"), "recorded_reward": 100,
             "evaluator_payload": {"verifiable_score": 1}}]
    f = audit_reward_recomputation(rows, lambda x: x["verifiable_score"])
    assert f.status == "FAIL"
    assert f.evidence["discrepant_id_sha256"] == [digest("a")]


def test_verified_outcome_recomputation_detects_forged_goal():
    rows = [{"id_sha256": digest("a"), "recorded_goal": 1,
             "evaluator_payload": {"actual_result": 0}}]
    assert audit_goal_recomputation(rows, lambda x: x["actual_result"]).status == "FAIL"
    assert audit_goal_recomputation(rows, None).status == "INCONCLUSIVE"


def test_valid_independently_recomputed_score_and_goal_fixtures():
    rr = [{"id_sha256": digest("a"), "recorded_reward": 1.,
           "evaluator_payload": {"actual": 1.}}]
    gg = [{"id_sha256": digest("a"), "recorded_goal": 1.,
           "evaluator_payload": {"actual": 1.}}]
    assert audit_reward_recomputation(rr, lambda x: x["actual"]).status == "PASS"
    assert audit_goal_recomputation(gg, lambda x: x["actual"]).status == "PASS"


def test_wrong_reward_score_and_duplicated_ids_cannot_pass():
    rows = [{"id_sha256": digest("a"), "recorded_reward": 1.,
             "evaluator_payload": 1},
            {"id_sha256": digest("a"), "recorded_reward": 1.,
             "evaluator_payload": 1}]
    assert audit_reward_recomputation(rows, lambda x: x).status == "INCONCLUSIVE"


def test_reward_only_poison_traces_hidden_prediction_dependency():
    def illicit(row):
        return row["observed_state"] + row["reward"]
    a = {"observed_state": 3., "reward": 1.}
    b = {"observed_state": 3., "reward": 100.}
    f = probe_reward_noninterference(illicit, a, b)
    assert f.status == "FAIL"


def test_prediction_not_conditioned_on_future_reward_passes_probe():
    a = {"observed_state": 3., "reward": 1.}
    b = {"observed_state": 3., "reward": 100.}
    f = probe_reward_noninterference(lambda row: row["observed_state"] * 2, a, b)
    assert f.status == "PASS"


def test_poison_must_change_only_denied_reward():
    a = {"observed_state": 3., "reward": 1.}
    b = {"observed_state": 4., "reward": 100.}
    assert probe_reward_noninterference(lambda row: row["observed_state"], a, b).status == "INCONCLUSIVE"


def test_identical_poison_and_missing_predictor_never_pass():
    a = {"observed_state": 3., "reward": 1.}
    assert probe_reward_noninterference(lambda row: row, a, a).status == "INCONCLUSIVE"
    assert probe_reward_noninterference(None, a, {**a, "reward": 100}).status == "INCONCLUSIVE"


def test_reward_gate_requires_all_five_checks_and_blocks_promotion():
    f = audit_proxy_goal_alignment(receipt("before"), receipt("after", reward=(.9, .9)), policy())
    ledger = reward_audit_ledger("experiment-001", [f])
    assert set(ledger.report()["required"]) == set(REQUIRED)
    assert ledger.report()["status"] == "FAIL"
    with pytest.raises(AuditGateError):
        ledger.assert_clean()


def test_reward_gate_fails_closed_on_missing_evidence():
    ledger = reward_audit_ledger("experiment-002", [])
    assert ledger.report()["status"] == "INCONCLUSIVE"
    with pytest.raises(AuditGateError):
        ledger.assert_clean()



def test_single_episode_harm_not_masked_by_aggregate_reward_and_goal():
    left = receipt("before", goal=(.6, .6))
    right = receipt("after", reward=(.8, .8), goal=(.4, .9))
    # Both episodes in same stratum; aggregate independent goal improves.
    left["rows"][1]["group"] = right["rows"][1]["group"] = "difficult"
    p = policy()
    p["required_groups"] = ["difficult"]
    result = audit_proxy_goal_alignment(left, right, p)
    assert result.status == "FAIL"
    assert result.evidence["subgroups"]["difficult"]["candidate_minus_baseline_goal_benefit"] > 0
    assert result.evidence["subgroups"]["difficult"]["worst_individual_goal_benefit_delta"] < 0


def test_new_safety_failure_cannot_be_offset_by_fixing_other_episode():
    left = receipt("before", safety=(False, True))
    right = receipt("after", reward=(.9, .9), goal=(.8, .8),
                    safety=(True, False))
    left["rows"][1]["group"] = right["rows"][1]["group"] = "difficult"
    p = policy()
    p["required_groups"] = ["difficult"]
    result = audit_proxy_goal_alignment(left, right, p)
    assert result.status == "FAIL"
    assert result.evidence["subgroups"]["difficult"]["added_safety_violations"] == 1
