"""Strict, *transport-neutral* Woyengi -> TAC forecast envelope (candidate).

This is a validation/serialization boundary, NOT a Woyengi HTTP connector or an
authorization implementation. The Woyengi side MUST verify permissions and
receipts before issuing the envelope and MUST revalidate before execution.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Mapping

SCHEMA = "tac.woyengi.forecast.v1"
MAX_HISTORY = 65  # 64 observed transitions
MAX_SCENARIOS = 5
MAX_HORIZON = 16


def _timestamp(value: Any, where: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: offset-aware ISO timestamp required")
    try:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{where}: invalid timestamp") from exc
    if date.tzinfo is None or date.utcoffset() is None:
        raise ValueError(f"{where}: timezone required")
    return date.astimezone(timezone.utc)


def _str(mapping: Mapping, key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing {key}")
    return value


def _vector(values: Any, expected_width: int, where: str) -> None:
    if not isinstance(values, list) or len(values) != expected_width:
        raise ValueError(f"{where}: expected {expected_width} values")
    if any(isinstance(n, bool) or not isinstance(n, (int, float)) or
           not math.isfinite(n) for n in values):
        raise ValueError(f"{where}: finite numeric values required")


def validate_prediction_request(payload: Mapping[str, Any], *, width: int = 14) -> str:
    """Validate an authorized *claim* and return a canonical input SHA-256.

    The presence of authorization_receipt_id is NOT proof of authorization:
    it is an externally verified reference required from Woyengi.
    No source credentials, private claims, or raw authoritative records belong
    in a model output. Input fingerprint is for immutable trace binding only.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("request must be an object")
    if payload.get("schema_version") != SCHEMA:
        raise ValueError("unsupported schema")
    for name in ("request_id", "trace_id", "principal_id", "workspace_id",
                 "entity_id", "feature_schema_id", "model_id",
                 "authorization_receipt_id", "reconstruction_receipt_id",
                 "state_revision"):
        _str(payload, name)
    checkpoint = _str(payload, "checkpoint_sha256")
    if len(checkpoint) != 64 or any(c not in "0123456789abcdef" for c in checkpoint):
        raise ValueError("checkpoint must be a lower-case SHA256")
    valid_cutoff = _timestamp(payload.get("valid_at"), "valid_at")
    known_cutoff = _timestamp(payload.get("recorded_at"), "recorded_at")
    rows = payload.get("observed_history")
    if not isinstance(rows, list) or not 2 <= len(rows) <= MAX_HISTORY:
        raise ValueError("observed_history requires 2..65 observed states")
    last = None
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("history row must be an object")
        observed_at = _timestamp(row.get("valid_at"), "history.valid_at")
        recorded_at = _timestamp(row.get("recorded_at"), "history.recorded_at")
        if observed_at > valid_cutoff or recorded_at > known_cutoff:
            raise ValueError("future information or retrospective leakage")
        if last is not None and observed_at <= last:
            raise ValueError("history must be strictly ordered by valid time")
        last = observed_at
        _str(row, "provenance_ref")
        _vector(row.get("state"), width, "history.state")
        if index:
            _vector(row.get("executed_action"), width, "history.executed_action")
        elif row.get("executed_action") is not None:
            raise ValueError("first history state has no preceding action")
        if not isinstance(row.get("human_takeover"), bool):
            raise ValueError("history.human_takeover must be boolean")
    if last != valid_cutoff:
        raise ValueError("last state must equal requested valid_at")

    scenarios = payload.get("candidate_scenarios")
    if not isinstance(scenarios, list) or not 1 <= len(scenarios) <= MAX_SCENARIOS:
        raise ValueError("candidate_scenarios requires 1..5 action sequences")
    ids = set()
    for scenario in scenarios:
        if not isinstance(scenario, Mapping):
            raise ValueError("scenario must be an object")
        key = _str(scenario, "scenario_id")
        if key in ids:
            raise ValueError("duplicate scenario_id")
        ids.add(key)
        actions = scenario.get("proposed_actions")
        if not isinstance(actions, list) or not 1 <= len(actions) <= MAX_HORIZON:
            raise ValueError("actions require 1..16 proposed steps")
        for action in actions:
            _vector(action, width, "scenario.proposed_actions")
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def prediction_response(request: Mapping[str, Any], *,
                        forecast_summaries: list[Mapping[str, Any]],
                        model_replay_commit: str, width: int = 14) -> dict:
    """Return non-authoritative proposals; no field grants tool authority."""
    fingerprint = validate_prediction_request(request, width=width)
    if len(forecast_summaries) != len(request["candidate_scenarios"]):
        raise ValueError("forecast count must match scenario count")
    expected_ids = [s["scenario_id"] for s in request["candidate_scenarios"]]
    actual_ids = [s.get("scenario_id") for s in forecast_summaries]
    if actual_ids != expected_ids:
        raise ValueError("forecast identity/order mismatch")
    _ = model_replay_commit.strip() if isinstance(model_replay_commit, str) else ""
    if len(_) != 40 or any(c not in "0123456789abcdef" for c in _):
        raise ValueError("pinned 40-hex Git commit required")
    return {
        "schema_version": SCHEMA,
        "request_id": request["request_id"],
        "trace_id": request["trace_id"],
        "state_revision": request["state_revision"],
        "recorded_at": request["recorded_at"],
        "input_sha256": fingerprint,
        "checkpoint_sha256": request["checkpoint_sha256"],
        "model_replay_commit": model_replay_commit,
        "status": "forecast_only_unqualified",
        "authority": "none",
        "execute": False,
        "forecasts": [dict(row) for row in forecast_summaries],
    }
