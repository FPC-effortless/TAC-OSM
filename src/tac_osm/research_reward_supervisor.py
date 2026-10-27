"""Privileged evaluator-side evidence assembly; NEVER in an optimizing actor.

A separate, access-controlled audit service invokes source-authentication and
real-score callbacks, holds the private Ed25519 signing operation, and returns
only signed statements. Untrusted training jobs receive neither a signing
capability nor source-only target data. Supplying fake callbacks in a test
proves *only* the harness mechanics, not real evaluator independence.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from typing import Any

from .research_anticheat import Finding
from .research_reward_integrity import (
    audit_protected_assets, audit_reward_recomputation,
    audit_goal_recomputation, probe_reward_noninterference,
    audit_proxy_goal_alignment,
)
from .research_reward_runner import (
    SCHEMA, REQUIRED_PREFLIGHT, REQUIRED_FINAL, RewardEvidenceError,
    binding, verify_reward_attestation,
)


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _passed(f: Finding, expected: str) -> None:
    if not isinstance(f, Finding) or f.check != expected or f.status != "PASS":
        raise RewardEvidenceError(
            f"trusted external supervisor check {expected} did not PASS")


def issue_privileged_runner_attestation(
        manifest: Mapping[str, Any], phase: str, *,
        verified_source: Finding,
        verified_actor_access: Finding,
        independently_verified_goals: Finding,
        read_protected_artifact: Callable[[str], bytes],
        actual_actor_write_paths: Sequence[str],
        external_protected_assets_sha256: Mapping[str, str],
        reward_scorer_sha256: str,
        goal_evaluator_sha256: str,
        access_policy_sha256: str,
        evaluation_policy_sha256: str,
        sign_external: Callable[[bytes], bytes],
        actual_reward_rows: Sequence[Mapping[str, Any]] | None = None,
        actual_goal_rows: Sequence[Mapping[str, Any]] | None = None,
        score_reward_independently: Callable[[Any], float] | None = None,
        score_goal_independently: Callable[[Any], float] | None = None,
        predict_at_fixed_checkpoint: Callable[[Mapping[str, Any]], Any] | None = None,
        clean_decision_row: Mapping[str, Any] | None = None,
        future_reward_poison: Mapping[str, Any] | None = None,
        original_baseline: Mapping[str, Any] | None = None,
        candidate_receipt: Mapping[str, Any] | None = None,
        preregistered_policy: Mapping[str, Any] | None = None,
        source_verified_model_outcomes: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create an independently signed prefit/final proof in the *auditor*.

    The caller must actually possess independently trusted capabilities:
    source verification, goal source verification, access inspection, trained
    prediction replay, actual FIT/DEV outcomes and private signing. Never
    expose these capabilities to a model training/policy-agent process.
    """
    if phase not in ("preflight", "final"):
        raise RewardEvidenceError("invalid privilege-gate phase")
    b = binding(manifest)
    _passed(verified_source, "source_provenance")
    _passed(verified_actor_access, "actor_access_isolation")
    _passed(independently_verified_goals, "goal_evaluator_separation")
    protected = audit_protected_assets(
        actor_write_paths=actual_actor_write_paths,
        protected_sha256=external_protected_assets_sha256,
        read_artifact=read_protected_artifact)
    _passed(protected, "protected_assets")
    hashes = (reward_scorer_sha256, goal_evaluator_sha256,
              access_policy_sha256, evaluation_policy_sha256)
    if any(not isinstance(x, str) or len(x) != 64 or
           any(c not in "0123456789abcdef" for c in x.lower()) for x in hashes):
        raise RewardEvidenceError("external scorer/goal/access/policy hash missing")
    if reward_scorer_sha256 == goal_evaluator_sha256:
        raise RewardEvidenceError("reward and external goal evaluator must differ")
    assets_digest = hashlib.sha256(_canonical(
        dict(sorted(external_protected_assets_sha256.items())))).hexdigest()
    findings = {"protected_assets": protected.status}
    payload: dict[str, Any] = {
        "schema": SCHEMA, "binding": b, "phase": phase,
        "source_authority": "independently_verified_real_source",
        "evaluator_executed_outside_actor": True,
        "reward_scorer_sha256": reward_scorer_sha256,
        "goal_evaluator_sha256": goal_evaluator_sha256,
        "access_policy_sha256": access_policy_sha256,
        "protected_assets_sha256": assets_digest,
        "evaluation_policy_sha256": evaluation_policy_sha256,
    }
    if phase == "preflight":
        findings.update({
            "source_provenance": verified_source.status,
            "goal_evaluator_separation": independently_verified_goals.status,
            "actor_access_isolation": verified_actor_access.status})
    else:
        if not (actual_reward_rows and actual_goal_rows and
                original_baseline and candidate_receipt and
                preregistered_policy and source_verified_model_outcomes and
                clean_decision_row and future_reward_poison):
            raise RewardEvidenceError("independent postfit reward/goal/source evidence incomplete")
        measured = [
            audit_reward_recomputation(actual_reward_rows, score_reward_independently),
            audit_goal_recomputation(actual_goal_rows, score_goal_independently),
            probe_reward_noninterference(
                predict_at_fixed_checkpoint, clean_decision_row,
                future_reward_poison),
            audit_proxy_goal_alignment(original_baseline, candidate_receipt,
                                       preregistered_policy),
        ]
        for item in measured:
            _passed(item, item.check)
            findings[item.check] = item.status
        b_rows = b["variant_seeds"]
        evaluated = []
        for entry in source_verified_model_outcomes:
            if set(entry) != {"variant", "seed", "checkpoint_sha256",
                              "outcome_evidence_sha256", "observation_count"}:
                raise RewardEvidenceError("model-specific independent outcome receipt invalid")
            evaluated.append(dict(entry))
        evaluated.sort(key=lambda r: (r["variant"], r["seed"]))
        if ([{"variant": r["variant"], "seed": r["seed"]} for r in evaluated] != b_rows):
            raise RewardEvidenceError("independent outcomes missing registered model/seed")
        payload["evaluated"] = evaluated
    mandatory = REQUIRED_PREFLIGHT if phase == "preflight" else REQUIRED_FINAL
    if set(findings) != set(mandatory) or any(findings[x] != "PASS" for x in mandatory):
        raise RewardEvidenceError("privileged audit findings do not satisfy required gate")
    payload["checks"] = findings
    raw = sign_external(_canonical(payload))
    if not isinstance(raw, bytes) or len(raw) != 64:
        raise RewardEvidenceError("external signer must return 64-byte Ed25519 signature")
    return {"payload": payload, "signature_hex": raw.hex()}
