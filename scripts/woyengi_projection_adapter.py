#!/usr/bin/env python3
"""Read-only adapter for actual Woyengi ClaimLedger.projectAt output SHAPE.

No Woyengi network/ledger write; no authentication of source, licenses or policy
by this module. A trusted caller MUST execute Woyengi's read authorization first
and provide a separate record-specific authorizer and allowlisted feature mapper.
The returned candidate observations are NOT training-qualified by themselves.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Callable, Mapping, Any

from validate_ecosystem_boundary import BoundaryError, _obj, _text, _sha, _when, _safe_features


def _contains(interval: object, at: datetime, label: str) -> None:
    span = _obj(interval, label)
    if not set(span) <= {"from", "to"} or "from" not in span:
        raise BoundaryError(f"{label}: invalid Woyengi time interval")
    start = _when(span["from"], f"{label}.from")
    if start > at:
        raise BoundaryError(f"{label}: not yet valid/known at projection time")
    if "to" in span and _when(span["to"], f"{label}.to") <= at:
        raise BoundaryError(f"{label}: expired at projection time")


def _projection_candidate(raw: object, projection: dict, valid_at: datetime,
                          recorded_at: datetime, cutoff: datetime) -> dict:
    item = _obj(raw, "projected claim")
    if set(item) != {"claim", "effectiveLifecycle"}:
        raise BoundaryError("projected claim: unexpected or missing fields")
    claim = _obj(item["claim"], "Woyengi claim")
    if claim.get("kind") != "claim":
        raise BoundaryError("only Woyengi native ClaimRecord is supported")
    if item["effectiveLifecycle"] not in {"provisional", "verified"}:
        raise BoundaryError("non-governing lifecycle must not reach adapter")
    if claim.get("lifecycle") not in {
        "provisional", "verified", "rejected", "superseded", "retracted", "invalidated", "archived"
    }:
        raise BoundaryError("unknown original claim lifecycle")
    if claim.get("subject") != projection["subject"] or claim.get("predicate") != projection["predicate"]:
        raise BoundaryError("source identity/predicate collision")
    for key in ("id", "subject", "predicate"):
        _text(claim.get(key), f"claim.{key}")
    _contains(claim.get("validTime"), valid_at, "claim.validTime")
    _contains(claim.get("transactionTime"), recorded_at, "claim.transactionTime")
    # Source-available transaction date cannot be later than the prediction cutoff.
    if _when(claim["transactionTime"]["from"], "claim.recordedAt") > cutoff:
        raise BoundaryError("future knowledge in native claim")
    if not isinstance(claim.get("confidence"), (float, int)) or isinstance(claim.get("confidence"), bool):
        raise BoundaryError("native claim confidence must be numeric")
    if not math.isfinite(claim["confidence"]) or not 0 <= claim["confidence"] <= 1:
        raise BoundaryError("invalid claim confidence")
    auth = _obj(claim.get("authority"), "native authority")
    # Authority is preserved by Woyengi, NEVER interpreted as a model grant here.
    if not isinstance(auth.get("level"), (int, float)) or not math.isfinite(auth["level"]):
        raise BoundaryError("invalid native authority level")
    _text(auth.get("basis"), "native authority basis")
    if not isinstance(claim.get("observationIds"), list) or not isinstance(claim.get("evidenceIds"), list):
        raise BoundaryError("claim evidence/observation IDs must be explicit")
    for reference in (*claim["observationIds"], *claim["evidenceIds"]):
        _text(reference, "claim source reference")
    _obj(claim.get("provenance"), "native provenance")
    return claim


def adapt_native_projection(
    projection: Mapping[str, Any],
    *,
    model_cutoff: str,
    source_artifact_sha256: str,
    read_authorized: Callable[[str, str, list[str], str], bool],
    feature_allowlist: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> dict:
    """Convert one Woyengi StateProjection into bounded, read-only research features.

    Caller MUST provide an authorization callback that uses its real Woyengi
    identity/purpose/read policy and a feature allowlist. Passing a permissive
    callback is NOT an authenticated permission receipt; this adapter reports
    only a structural transformation and never upgrades scientific trust.
    No single 'selected' projection can suppress a competing candidate.
    """
    p = _obj(projection, "Woyengi StateProjection")
    if set(p) != {"subject", "predicate", "validAt", "recordedAt", "selected",
                  "candidates", "conflicts", "trace"}:
        raise BoundaryError("requires complete Woyengi StateProjection including conflicts/trace")
    _text(p["subject"], "projection.subject")
    _text(p["predicate"], "projection.predicate")
    _sha(source_artifact_sha256, "original source artifact reference")
    cutoff = _when(model_cutoff, "prediction cutoff")
    valid_at = _when(p["validAt"], "projection.validAt")
    recorded_at = _when(p["recordedAt"], "projection.recordedAt")
    if valid_at > cutoff or recorded_at > cutoff:
        raise BoundaryError("projection asks about future time or future knowledge")
    if not isinstance(p["candidates"], list) or not isinstance(p["conflicts"], list):
        raise BoundaryError("source candidates and conflicts must be explicit arrays")
    if not isinstance(p["trace"], list) or not p["trace"]:
        raise BoundaryError("projection trace missing")
    if not p["candidates"]:
        raise BoundaryError("no source-eligible candidates; abstain, do not fabricate state")
    by_id: dict[str, dict] = {}
    for candidate in p["candidates"]:
        claim = _projection_candidate(candidate, p, valid_at, recorded_at, cutoff)
        if claim["id"] in by_id:
            raise BoundaryError("duplicate source claim ID")
        by_id[claim["id"]] = claim
    selected = p["selected"]
    if selected is None:
        raise BoundaryError("source projection has no governing selection; defer")
    selected_claim = _projection_candidate(selected, p, valid_at, recorded_at, cutoff)
    if selected_claim["id"] not in by_id or by_id[selected_claim["id"]] != selected_claim:
        raise BoundaryError("selected claim omitted or changed in candidate inventory")
    conflicting_ids = []
    for candidate in p["conflicts"]:
        claim = _projection_candidate(candidate, p, valid_at, recorded_at, cutoff)
        if claim["id"] not in by_id or by_id[claim["id"]] != claim:
            raise BoundaryError("conflict omitted from candidate inventory")
        if claim["id"] == selected_claim["id"] or claim["id"] in conflicting_ids:
            raise BoundaryError("invalid repeated/selected conflict")
        conflicting_ids.append(claim["id"])
    # Value differences require conflicts to be represented. Canonical JSON is
    # just comparison; cannot assert the values are real or independently verified.
    def value_key(x):
        return json.dumps(x["object"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    selected_value = value_key(selected_claim)
    changed = {cid for cid, claim in by_id.items()
               if cid != selected_claim["id"] and value_key(claim) != selected_value}
    if changed != set(conflicting_ids):
        raise BoundaryError("conflict suppression or false conflict in source projection")
    ids = sorted(by_id)
    if not callable(read_authorized) or not callable(feature_allowlist):
        raise BoundaryError("trusted host must provide read authorization and field selector")
    # Explicit read authorization on all candidates, not just selected one.
    # The host's policy must account for exact principal/purpose; this adapter
    # cannot independently verify the callback's provenance.
    if read_authorized(p["subject"], p["predicate"], ids, model_cutoff) is not True:
        raise BoundaryError("Woyengi source read not authorized by host callback")
    observations = []
    for cid in ids:
        claim = by_id[cid]
        features = _obj(feature_allowlist(claim), "explicit feature allowlist")
        _safe_features(features, "features")
        # Local deterministic digest identifies this derived feature projection,
        # not the native publisher artifact or an independently verified source.
        claim_bytes = json.dumps(claim, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        observations.append({
            "id": "observation:" + cid,
            "entity_id": p["subject"],
            "valid_at": claim["validTime"]["from"],
            "recorded_at": claim["transactionTime"]["from"],
            "available_at": claim["transactionTime"]["from"],  # known-by proxy; do not claim sensor availability
            "source_record_id": cid,
            "record_sha256": hashlib.sha256(claim_bytes).hexdigest(),
            "conflicts_count": len(conflicting_ids),
            "conflicts_preserved": True,
            "features": features,
        })
    return {
        "adapter": "woyengi-claim-projection-v1",
        "status": "STRUCTURAL_ONLY",
        "original_source_artifact_sha256": source_artifact_sha256,
        "derived_hash_kind": "SHA256_WOYENGI_CLAIM_JSON_NOT_PUBLISHER_AUTHENTICATION",
        "availability_semantics": "Woyengi.transactionTime.from; not upstream sensor arrival",
        "claim_ids": ids,
        "conflicting_claim_ids": sorted(conflicting_ids),
        "selected_claim_id": selected_claim["id"],
        "observations": observations,
        "trusted_source_authentication": "NOT_VERIFIED_BY_ADAPTER",
        "Woyengi_permission_provenance": "CALLBACK_RESULT_NOT_INDEPENDENTLY_VERIFIED",
        "scientific_qualification": "NOT_EVALUATED",
        "governing_authority": "NOT_GRANTED",
    }
