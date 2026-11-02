"""Independently signed source/outcome/access attestation gate for real runners.

Source authority is *not* established by a text receipt. The trusted
Ed25519 verification key must be provisioned independently of model agents,
dataset/trainable worktrees, manifests and PR content. Private signing keys
never belong in model processes or this repository. This module VERIFIES
attestations; it does not issue or self-sign them.

A successful signature proves a trusted key signed the contents, not that
the underlying real-world observations are causally sufficient. Fail closed.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

SCHEMA = "tac-osm.reward-runner-attestation.v1"
FAMILIES = ("System 1", "Enterprise", "Temporal", "Physical", "Edge-Lite",
            "Counterfactual")
REQUIRED_FINAL = ("protected_assets", "reward_recomputation",
                  "goal_recomputation", "reward_noninterference",
                  "proxy_goal_alignment")
REQUIRED_PREFLIGHT = ("protected_assets", "source_provenance",
                      "goal_evaluator_separation", "actor_access_isolation")
_HEX = set("0123456789abcdef")


class RewardEvidenceError(ValueError):
    """Evidence absent, untrusted, mismatched, stale or independently failed."""


def _digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        x in _HEX for x in value.lower())


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _key(public_key_hex: str):
    if not isinstance(public_key_hex, str) or len(public_key_hex) != 64:
        raise RewardEvidenceError("trusted external Ed25519 public key required")
    try:
        raw = bytes.fromhex(public_key_hex)
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey)
        return Ed25519PublicKey.from_public_bytes(raw)
    except (ValueError, ImportError) as exc:
        raise RewardEvidenceError("invalid/unavailable Ed25519 trusted key") from exc


def binding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Stable frozen prefit identity, independent of postfit score receipts."""
    if not isinstance(manifest, Mapping):
        raise RewardEvidenceError("frozen study manifest required")
    source = manifest.get("real_world_source")
    if not isinstance(source, Mapping) or not _digest(source.get("sha256")):
        raise RewardEvidenceError("genuine real-world source hash required")
    study = manifest.get("study_id")
    if not isinstance(study, str) or not study:
        raise RewardEvidenceError("registered study ID required")
    kind = manifest.get("study_type")
    rows = manifest.get("models") if kind == "single_model_training" else manifest.get("arms")
    if kind not in ("single_model_training", "architecture_comparison") or not isinstance(rows, list) or not rows:
        raise RewardEvidenceError("registered trainable model arms required")
    entries = []
    for row in rows:
        if not isinstance(row, Mapping) or type(row.get("seed")) is not int:
            raise RewardEvidenceError("each model requires frozen seed")
        variant = row.get("architecture") if kind == "architecture_comparison" else row.get("variant")
        if not isinstance(variant, str) or not variant.strip():
            raise RewardEvidenceError("each model/seed must declare an explicit variant")
        entries.append((variant, row["seed"]))
    if len(set(entries)) != len(entries):
        raise RewardEvidenceError("duplicate variant/seed in run registration")
    return {"study_id": study, "source_sha256": source["sha256"].lower(),
            "study_type": kind, "variant_seeds": sorted(
                [{"variant": variant, "seed": seed} for variant, seed in entries],
                key=lambda x: (x["variant"], x["seed"]))}


def _signing_payload(attestation: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(attestation, Mapping) or set(attestation) != {"payload", "signature_hex"}:
        raise RewardEvidenceError("signed envelope must have payload and signature_hex")
    payload = attestation["payload"]
    signature = attestation["signature_hex"]
    if not isinstance(payload, Mapping) or not isinstance(signature, str):
        raise RewardEvidenceError("invalid signed envelope contents")
    if len(signature) != 128:
        raise RewardEvidenceError("Ed25519 signature required")
    return payload


def verify_reward_attestation(
        manifest: Mapping[str, Any], phase: str,
        attestation: Mapping[str, Any] | None,
        *, trusted_public_key_hex: str | None) -> dict[str, Any]:
    """Check signed exact-match prefit/final evidence for EVERY registered arm.

    This is a trusted *verification* boundary: callers must supply a public
    key pinned by repository operator/CI outside the actor-controlled repo.
    A key embedded in the model manifest or the attestation is untrusted.
    """
    if phase not in ("preflight", "final"):
        raise RewardEvidenceError("unrecognized audit phase")
    if attestation is None or not trusted_public_key_hex:
        raise RewardEvidenceError("independent reward/outcome/access attestation missing")
    payload = _signing_payload(attestation)
    try:
        sig = bytes.fromhex(attestation["signature_hex"])
        _key(trusted_public_key_hex).verify(sig, _canonical(payload))
    except (ValueError, TypeError, OverflowError) as exc:
        raise RewardEvidenceError("Ed25519 evidence signature invalid") from exc
    except Exception as exc:
        # cryptography InvalidSignature is not a builtin ValueError.
        raise RewardEvidenceError("Ed25519 evidence signature untrusted") from exc
    b = binding(manifest)
    if payload.get("schema") != SCHEMA or payload.get("phase") != phase:
        raise RewardEvidenceError("wrong evidence schema or stage")
    if payload.get("binding") != b:
        raise RewardEvidenceError("evidence bound to a different source/study/variant matrix")
    if payload.get("source_authority") != "independently_verified_real_source":
        raise RewardEvidenceError("independent source identity not verified")
    if payload.get("evaluator_executed_outside_actor") is not True:
        raise RewardEvidenceError("independent evaluator execution boundary unverified")
    for key in ("reward_scorer_sha256", "goal_evaluator_sha256",
                "access_policy_sha256", "protected_assets_sha256",
                "evaluation_policy_sha256"):
        if not _digest(payload.get(key)):
            raise RewardEvidenceError("missing independent pinned: " + key)
    if payload["reward_scorer_sha256"] == payload["goal_evaluator_sha256"]:
        raise RewardEvidenceError("same reward/goal evaluator independence conflict")
    checks = payload.get("checks")
    mandatory = REQUIRED_PREFLIGHT if phase == "preflight" else REQUIRED_FINAL
    if not isinstance(checks, Mapping) or set(checks) != set(mandatory):
        raise RewardEvidenceError("missing or extra mandatory independent check")
    if any(checks[name] != "PASS" for name in mandatory):
        raise RewardEvidenceError("independent audit gate is FAIL or INCONCLUSIVE")
    if phase == "final":
        evaluated = payload.get("evaluated")
        if not isinstance(evaluated, list) or len(evaluated) != len(b["variant_seeds"]):
            raise RewardEvidenceError("per-model independently evaluated evidence missing")
        got = []
        for row in evaluated:
            if not isinstance(row, Mapping) or (
                    set(row) != {"variant", "seed", "checkpoint_sha256",
                                "outcome_evidence_sha256", "observation_count"}):
                raise RewardEvidenceError("incomplete final model outcome receipt")
            if type(row["seed"]) is not int or type(row["observation_count"]) is not int or (
                    row["observation_count"] <= 0):
                raise RewardEvidenceError("invalid per-model outcome count/seed")
            for key in ("checkpoint_sha256", "outcome_evidence_sha256"):
                if not _digest(row[key]):
                    raise RewardEvidenceError("invalid postfit source/checkpoint digest")
            got.append({"variant": row["variant"], "seed": row["seed"]})
            models = manifest["models"] if b["study_type"] == "single_model_training" else manifest["arms"]
            match = [r for r in models if r["seed"] == row["seed"] and (
                r.get("architecture", r.get("variant")) == row["variant"])]
            if len(match) != 1 or not isinstance(match[0].get("measured"), Mapping) or (
                    match[0]["measured"].get("checkpoint_sha256") != row["checkpoint_sha256"]):
                raise RewardEvidenceError("certified checkpoint does not equal measured trained model")
        if got != b["variant_seeds"]:
            raise RewardEvidenceError("final attestation incomplete or wrong variant/seed ordering")
    return {
        "status": "PASS_SIGNED_EXTERNAL_AUDIT",
        "phase": phase, "study_id": b["study_id"],
        "variant_seed_count": len(b["variant_seeds"]),
        "key_sha256": hashlib.sha256(bytes.fromhex(trusted_public_key_hex)).hexdigest(),
        "evidence_sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
        "qualification": "signed_auditor_receipt_not_independently_proven_causality",
    }



def verify_stage_continuity(preflight: Mapping[str, Any],
                            postfit: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze independent judge/scorer/ACL/policy across the fitted run.

    Call this *after* each signed envelope has been separately verified using
    the SAME independently pinned public key and original run registration.
    Signed but changed reward rules must never silently count as improvement.
    """
    a = _signing_payload(preflight)
    b = _signing_payload(postfit)
    if a.get("phase") != "preflight" or b.get("phase") != "final":
        raise RewardEvidenceError("reward stage order wrong")
    if a.get("schema") != SCHEMA or b.get("schema") != SCHEMA:
        raise RewardEvidenceError("reward evidence schema changed")
    for field in ("binding", "source_authority", "reward_scorer_sha256",
                  "goal_evaluator_sha256", "access_policy_sha256",
                  "protected_assets_sha256", "evaluation_policy_sha256"):
        if a.get(field) != b.get(field):
            raise RewardEvidenceError("reward/goal/source/ACL/policy changed across fit: " + field)
    if a.get("evaluator_executed_outside_actor") is not True or b.get(
            "evaluator_executed_outside_actor") is not True:
        raise RewardEvidenceError("evaluator isolation not confirmed across stages")
    return {"status": "PASS_UNCHANGED_INDEPENDENT_SCORING_CONTRACT",
            "frozen_contract_digest": hashlib.sha256(_canonical({
                field: a[field] for field in (
                    "binding", "reward_scorer_sha256", "goal_evaluator_sha256",
                    "access_policy_sha256", "protected_assets_sha256",
                    "evaluation_policy_sha256")})).hexdigest()}
