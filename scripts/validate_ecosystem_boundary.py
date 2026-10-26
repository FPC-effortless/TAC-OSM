#!/usr/bin/env python3
"""Fail-closed structural check of an UNATTESTED cross-product research bundle.

Constructed tests only. This does NOT authenticate publishers, rights, signatures,
model capability, independent Veritas verification, Woyengi authority, or any effect.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

_HEX = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN = {"future", "oracle", "gold", "label", "hidden", "reward", "target",
              "outcome", "acceptance", "authorized", "verified", "secret"}
_ALLOWED_OUTPUTS = {"predicate": {"kind", "probability"},
                    "choice": {"kind", "probabilities"},
                    "score": {"kind", "value", "units"},
                    "forecast": {"kind", "estimate", "units"},
                    "defer": {"kind", "reason"}}


class BoundaryError(ValueError):
    """No trusted interpretation available."""


def _obj(x, label):
    if not isinstance(x, dict):
        raise BoundaryError(f"{label}: object required")
    return x


def _keys(d, exact, label):
    if set(d) != set(exact):
        raise BoundaryError(f"{label}: required fields only; missing {sorted(set(exact)-set(d))}; extra {sorted(set(d)-set(exact))}")


def _text(x, label):
    if not isinstance(x, str) or not x.strip():
        raise BoundaryError(f"{label}: nonempty text required")
    return x


def _sha(x, label):
    if not isinstance(x, str) or not _HEX.fullmatch(x):
        raise BoundaryError(f"{label}: lowercase SHA256 required")
    return x


def _when(x, label):
    value = _text(x, label)
    if not re.search(r"(?:[Zz]|[+-][0-9]{2}:[0-9]{2})$", value):
        raise BoundaryError(f"{label}: timezone required")
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00").replace("z", "+00:00"))
        if dt.tzinfo is None or dt.utcoffset() is None:
            raise ValueError("naive")
        return dt.astimezone(timezone.utc)
    except ValueError as exc:
        raise BoundaryError(f"{label}: invalid timestamp") from exc


def _num(x, label):
    if type(x) not in (int, float) or not math.isfinite(x):
        raise BoundaryError(f"{label}: finite number required")
    return x


def _safe_features(value, label, depth=0):
    if depth > 12:
        raise BoundaryError(f"{label}: overdeep observation feature")
    if isinstance(value, dict):
        if len(value) > 2048:
            raise BoundaryError(f"{label}: feature too large")
        for k, v in value.items():
            if not isinstance(k, str) or not k:
                raise BoundaryError(f"{label}: named feature required")
            norm = re.sub(r"[^a-z0-9]+", "_", k.lower())
            segments = set(norm.split("_"))
            if segments & _FORBIDDEN:
                raise BoundaryError(f"{label}: forbidden future/authority field {k}")
            _safe_features(v, label + "." + k, depth + 1)
    elif isinstance(value, list):
        if len(value) > 4096:
            raise BoundaryError(f"{label}: list too long")
        for child in value:
            _safe_features(child, label, depth + 1)
    elif value is not None and type(value) not in (bool, str, int, float):
        raise BoundaryError(f"{label}: invalid feature type")
    elif type(value) is float and not math.isfinite(value):
        raise BoundaryError(f"{label}: nonfinite feature")


def _output(raw):
    d = _obj(raw, "output")
    kind = d.get("kind")
    if kind not in _ALLOWED_OUTPUTS:
        raise BoundaryError("output.kind invalid")
    _keys(d, _ALLOWED_OUTPUTS[kind], "output")
    if kind == "predicate":
        p = _num(d["probability"], "predicate probability")
        if not 0 <= p <= 1:
            raise BoundaryError("predicate probability outside [0,1]")
    elif kind == "choice":
        ps = _obj(d["probabilities"], "choice probabilities")
        if not ps or not all(isinstance(k, str) and k for k in ps):
            raise BoundaryError("invalid choice labels")
        if any(not 0 <= _num(p, "choice probability") <= 1 for p in ps.values()):
            raise BoundaryError("choice probability outside [0,1]")
        if abs(sum(ps.values()) - 1) > 1e-5:
            raise BoundaryError("choice probabilities not normalized")
    elif kind in {"score", "forecast"}:
        _num(d["value"] if kind == "score" else d["estimate"], "output number")
        _text(d["units"], "output units")
    else:
        _text(d["reason"], "defer reason")


def check_bundle(bundle: dict) -> dict:
    b = _obj(bundle, "bundle")
    _keys(b, {"contract_version", "study_id", "split", "source_artifact_sha256",
              "source_rights_reference", "as_of_at", "observations", "predictions",
              "outcomes", "verification_status"}, "bundle")
    if type(b["contract_version"]) is not int or b["contract_version"] != 1:
        raise BoundaryError("unsupported version")
    _text(b["study_id"], "study_id")
    if b["split"] not in {"FIT", "DEV", "TEST"}:
        raise BoundaryError("invalid split")
    _sha(b["source_artifact_sha256"], "source artifact")
    _text(b["source_rights_reference"], "rights reference (not verification)")
    cutoff = _when(b["as_of_at"], "as_of_at")
    if b["verification_status"] != "UNATTESTED":
        raise BoundaryError("structural bundle cannot self-certify Veritas or Woyengi verification")
    if b["split"] == "TEST":
        raise BoundaryError("structural preflight refuses protected TEST rows; independent sealed evaluator required")
    ids = set()
    def unique(value):
        _text(value, "record ID")
        if value in ids:
            raise BoundaryError("duplicate record ID")
        ids.add(value)
        return value
    obs = {}
    if not isinstance(b["observations"], list) or not b["observations"]:
        raise BoundaryError("observations must be nonempty")
    for raw in b["observations"]:
        o = _obj(raw, "observation")
        _keys(o, {"id", "entity_id", "valid_at", "recorded_at", "available_at",
                  "source_record_id", "record_sha256", "conflicts_count",
                  "conflicts_preserved", "features"}, "observation")
        oid = unique(o["id"])
        _text(o["entity_id"], "entity_id")
        _text(o["source_record_id"], "source_record_id")
        _sha(o["record_sha256"], "observation hash")
        _when(o["valid_at"], "valid_at")
        if _when(o["recorded_at"], "recorded_at") > cutoff:
            raise BoundaryError("recorded future source before model cutoff")
        if _when(o["available_at"], "available_at") > cutoff:
            raise BoundaryError("future information became available after model cutoff")
        if type(o["conflicts_count"]) is not int or o["conflicts_count"] < 0:
            raise BoundaryError("conflicts_count invalid")
        if o["conflicts_preserved"] is not True:
            raise BoundaryError("source conflicts must be preserved explicitly")
        _safe_features(_obj(o["features"], "features"), "features")
        obs[oid] = o
    predictions = {}
    if not isinstance(b["predictions"], list) or not b["predictions"]:
        raise BoundaryError("predictions must be nonempty")
    for raw in b["predictions"]:
        p = _obj(raw, "prediction")
        _keys(p, {"id", "observation_ids", "checkpoint_sha256", "predicted_at",
                  "horizon_seconds", "target_id", "output", "authority"}, "prediction")
        pid = unique(p["id"])
        _sha(p["checkpoint_sha256"], "model checkpoint")
        if _when(p["predicted_at"], "predicted_at") != cutoff:
            raise BoundaryError("prediction cutoff mismatch")
        if type(p["horizon_seconds"]) is not int or p["horizon_seconds"] <= 0:
            raise BoundaryError("prediction horizon invalid")
        _text(p["target_id"], "prediction target")
        if p["authority"] != "advisory":
            raise BoundaryError("model prediction cannot grant governing authority")
        refs = p["observation_ids"]
        if not isinstance(refs, list) or not refs or len(set(refs)) != len(refs):
            raise BoundaryError("invalid or duplicate observation references")
        if any(ref not in obs for ref in refs):
            raise BoundaryError("prediction accesses unregistered observation")
        _output(p["output"])
        predictions[pid] = p
    if not isinstance(b["outcomes"], list):
        raise BoundaryError("outcomes must be a list")
    outcomes_for = set()
    for raw in b["outcomes"]:
        o = _obj(raw, "outcome")
        _keys(o, {"id", "prediction_id", "observed_at", "recorded_at",
                  "available_at", "source_record_id", "record_sha256", "target_value"}, "outcome")
        unique(o["id"])
        ref = o["prediction_id"]
        if ref not in predictions or ref in outcomes_for:
            raise BoundaryError("outcome must match one registered prediction once")
        outcomes_for.add(ref)
        _text(o["source_record_id"], "independent original outcome source ref")
        _sha(o["record_sha256"], "outcome content hash")
        observed = _when(o["observed_at"], "outcome observed_at")
        recorded = _when(o["recorded_at"], "outcome recorded_at")
        available = _when(o["available_at"], "outcome available_at")
        if observed <= cutoff or recorded <= cutoff or available <= cutoff:
            raise BoundaryError("future outcome must not exist at prediction cutoff")
        if recorded < observed or available < observed:
            raise BoundaryError("outcome knowledge predates its valid observation")
        if type(o["target_value"]) not in (bool, int, float, str) and o["target_value"] is not None:
            raise BoundaryError("unsupported outcome target type")
        if type(o["target_value"]) is float and not math.isfinite(o["target_value"]):
            raise BoundaryError("nonfinite outcome target")
    return {
        "structure": "PASS",
        "source_legality": "UNVERIFIED",
        "source_cryptographic_authenticity": "UNVERIFIED",
        "model_scientific_qualification": "NOT_EVALUATED",
        "veritas_independent_outcome_attestation": "NOT_PRESENT",
        "woyengi_governing_authority": "NOT_GRANTED",
        "confidential_test_gate": "NOT_OPENED",
        "observation_count": len(obs),
        "prediction_count": len(predictions),
        "outcome_count": len(outcomes_for),
        "study_id": b["study_id"],
        "split": b["split"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path)
    args = parser.parse_args(argv)
    try:
        def no_constant(value):
            raise BoundaryError(f"non-JSON numeric constant: {value}")
        data = json.loads(args.input_json.read_text(encoding="utf-8"), parse_constant=no_constant)
        print(json.dumps(check_bundle(data), sort_keys=True))
        return 0
    except (BoundaryError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"structure": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
