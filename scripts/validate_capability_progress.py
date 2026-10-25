#!/usr/bin/env python3
"""Prospective, fail-closed STRUCTURAL guard for claimed TAC-OSM capability gains.

JSON declarations are not evidence of true data provenance, training, or statistics.
An auditor MUST verify original real per-case scores, frozen Git checkpoints,
actual source/time access, preregistration and independent replay separately.
See research/CAPABILITY_PROGRESS_TRUTH_STANDARD.md.
"""
from __future__ import annotations
import argparse
import json
import math
import re
import sys
from pathlib import Path

TIERS = {
    "DESIGN_ONLY", "ENGINEERING_VERIFIED", "TRAINED_REPLAY_VERIFIED",
    "DEV_CAPABILITY_OBSERVED", "DEV_GAIN", "MIXED_TRADEOFF",
    "INDEPENDENT_GAIN", "TRANSFER_REPLICATED",
    "OPERATIONAL_VALUE_VERIFIED", "REGRESSION", "INCONCLUSIVE", "BLOCKED",
}
SHA = re.compile(r"^[0-9a-f]{64}$")
INDEPENDENT = {"INDEPENDENT_GAIN", "TRANSFER_REPLICATED", "OPERATIONAL_VALUE_VERIFIED"}
MEASURED = {
    "TRAINED_REPLAY_VERIFIED", "DEV_CAPABILITY_OBSERVED", "DEV_GAIN",
    "MIXED_TRADEOFF", "INDEPENDENT_GAIN", "TRANSFER_REPLICATED",
    "OPERATIONAL_VALUE_VERIFIED",
}


def check(record: dict) -> list[str]:
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    def obj(name):
        value = record.get(name)
        if not isinstance(value, dict):
            errors.append(name + " must be an object")
            return {}
        return value

    def finite(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

    require(record.get("schema_version") == "tac-osm.capability-progress.v1", "invalid schema_version")
    tier = record.get("tier")
    require(tier in TIERS, "invalid evidence tier")
    require(bool(record.get("study_id")), "missing study_id")
    require(bool(record.get("what_improved")), "state what improved (or 'none')")
    require(bool(record.get("what_regressed")), "state what regressed (or 'none')")
    require(bool(record.get("what_remains_unknown")), "state what remains unknown")
    source, task, model, integrity = (obj(k) for k in ("source", "task", "model", "integrity"))

    if tier in MEASURED:
        require(source.get("authentic_real_data") is True, "measured skill requires authentic real observations")
        require(source.get("exposure") in {"FIT", "EXPOSED_DEV", "EXPOSED_TEST", "UNTOUCHED_REAL", "PROSPECTIVE_PILOT"},
                "unknown real source exposure")
        for k in ("sha256", "eligible_ids_sha256"):
            require(isinstance(source.get(k), str) and SHA.fullmatch(source[k]) is not None,
                    "source." + k + " must be a verified-length SHA256 identifier")
        require(task.get("prediction_time_validated") is True, "prediction-time input availability not asserted")
        require(type(task.get("independent_cases")) is int and task.get("independent_cases", 0) > 0,
                "missing independent real-case denominator")
        require(type(task.get("actual_future_targets")) is int and
                0 <= task.get("actual_future_targets", -1) <= task.get("independent_cases", -1),
                "invalid actual future target count; repeated seeds/paths do not add cases")
        require(bool(task.get("target")) and bool(task.get("query_and_horizon")),
                "must specify factual target and query/horizon")
        require(integrity.get("git_original_checkpoints") is True, "original checkpoint bytes not verified")
        require(integrity.get("independent_clean_replay") is True, "independent original-score replay absent")
        require(integrity.get("raw_case_scores_preserved") is True, "raw original case scores missing")
        require(isinstance(model.get("candidate_sha256"), str) and SHA.fullmatch(model["candidate_sha256"]) is not None,
                "candidate model checkpoint SHA256 absent")
        require(integrity.get("registered_before_evaluation") is True,
                "preregistration-before-evaluation not asserted")
    if tier in {"DEV_GAIN", "MIXED_TRADEOFF"} | INDEPENDENT:
        primary = obj("primary_comparison")
        require(task.get("same_case_ids_target_horizon_metric") is True,
                "before/after comparisons MUST use identical cases, target, horizon and score")
        require(isinstance(model.get("prior_sha256"), str) and SHA.fullmatch(model["prior_sha256"]) is not None,
                "missing frozen prior complete-model checkpoint SHA256")
        require(model.get("prior_sha256") != model.get("candidate_sha256"),
                "prior and candidate model checkpoint identities must differ")
        for k in ("prior", "candidate", "minimum_useful_gain"):
            require(finite(primary.get(k)), "primary_comparison." + k + " must be finite")
        require(primary.get("direction") in ("lower_is_better", "higher_is_better"),
                "missing primary direction")
        if all(finite(primary.get(k)) for k in ("prior", "candidate", "minimum_useful_gain")):
            delta = primary["prior"] - primary["candidate"] if primary["direction"] == "lower_is_better" else primary["candidate"] - primary["prior"]
            if tier in {"DEV_GAIN"} | INDEPENDENT:
                require(primary["minimum_useful_gain"] > 0, "improvement requires preregistered nonzero material gain")
                require(delta > primary["minimum_useful_gain"],
                        "measured gain does not exceed declared material threshold")
            if tier == "MIXED_TRADEOFF":
                require(delta > 0, "mixed tradeoff requires a documented primary positive gain")
        guardrails = record.get("guardrails")
        require(isinstance(guardrails, list) and bool(guardrails), "explicit guardrail outcomes required")
        if isinstance(guardrails, list):
            require(all(isinstance(g, dict) and g.get("status") in ("PASS", "FAIL") and g.get("metric") for g in guardrails),
                    "each guardrail requires metric and PASS/FAIL")
            if tier in INDEPENDENT:
                require(all(g.get("status") == "PASS" for g in guardrails if isinstance(g, dict)),
                        "critical guardrail failure forbids independent gain")
            if tier == "MIXED_TRADEOFF":
                require(any(g.get("status") == "FAIL" for g in guardrails if isinstance(g, dict)),
                        "mixed tradeoff must explicitly disclose at least one failing guardrail")
        if tier in INDEPENDENT:
            require(source.get("exposure") in ("UNTOUCHED_REAL", "PROSPECTIVE_PILOT"),
                    "INDEPENDENT_GAIN cannot use FIT/exposed DEV or old TEST")
            require(source.get("holdout_unopened_before_freeze") is True,
                    "heldout cohort was not preregistered as sealed")
            ci = primary.get("paired_case_cluster_95ci_gain")
            require(isinstance(ci, list) and len(ci) == 2 and all(finite(v) for v in ci),
                    "requires numeric paired independent-case 95% CI for benefit")
            if isinstance(ci, list) and len(ci) == 2 and all(finite(v) for v in ci):
                require(ci[0] <= ci[1], "CI bounds are reversed")
                if finite(primary.get("minimum_useful_gain")):
                    require(ci[0] > primary["minimum_useful_gain"],
                            "95% paired-case benefit lower CI does not exceed material gain")
            seeds = primary.get("paired_seed_count")
            wins = primary.get("beneficial_seed_count")
            require(type(seeds) is int and seeds >= 5, "default requires >=5 paired original seeds")
            require(type(wins) is int and type(seeds) is int and 0 <= wins <= seeds and wins >= 4,
                    "default requires >=4/5 independently fitted paired seed wins")
            require(record.get("practical_reference_scored") is True,
                    "no honest same-task practical reference")
            require(task.get("independent_cases", 0) >= 100 or record.get("preregistered_small_cohort_power_justification") is True,
                    "under 100 independent real cases needs preregistered power justification")
            require(integrity.get("independent_raw_case_audit") is True,
                    "independent raw case-level data/score audit not asserted")
    if tier == "TRANSFER_REPLICATED":
        require(type(record.get("independent_real_distributions")) is int and
                record.get("independent_real_distributions", 0) >= 2,
                "transfer replication requires >=2 independent real distributions")
    if tier == "OPERATIONAL_VALUE_VERIFIED":
        pilot = obj("prospective_pilot")
        require(pilot.get("authorized_external_customer") is True and
                pilot.get("real_outcomes_and_costs_verified") is True and
                pilot.get("preregistered_net_benefit_confirmed") is True,
                "operator value requires verified authorized external pilot, real outcomes/costs, and net benefit")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON capability evidence claim")
    args = parser.parse_args()
    try:
        record = json.loads(args.manifest.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError("manifest must be object")
        errors = check(record)
    except (OSError, ValueError) as exc:
        errors = [str(exc)]
    for message in errors:
        print("FAIL: " + message, file=sys.stderr)
    if errors:
        raise SystemExit(1)
    print("DECLARATION_STRUCTURE_PASS_ONLY; original data and scoring still require independent audit")


if __name__ == "__main__":
    main()
