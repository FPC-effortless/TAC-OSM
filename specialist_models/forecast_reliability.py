#!/usr/bin/env python3
"""No-fit reliability layer for pre-existing temporal forecast intervals.

Fits ONLY nonnegative interval expansions on a distinct, source-time-valid
calibration cohort. It does not train TAC-OSM, authenticate source records,
open protected labels, or guarantee conditional coverage under drift.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import sys


class EvidenceError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def iso_utc(value):
    if not isinstance(value, str):
        raise EvidenceError("UTC timestamp required")
    try:
        t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceError("invalid UTC timestamp") from exc
    if t.tzinfo is None or t.utcoffset() != timedelta(0):
        raise EvidenceError("UTC offset-aware timestamps required")
    return t


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_from_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if len(rows) >= 2_000_000:
                raise EvidenceError("input record count limit")
            rows.append(json.loads(line))
    if not rows:
        raise EvidenceError("empty evidence file")
    return rows


def valid_config(config):
    if not isinstance(config, dict) or config.get("schema") != "forecast-interval-contract-v1":
        raise EvidenceError("unsupported calibration contract")
    h = config.get("model_sha256")
    if not isinstance(h, str) or re.fullmatch(r"[0-9a-f]{64}", h) is None:
        raise EvidenceError("source model SHA256 required")
    times = [iso_utc(config.get(key)) for key in
             ("model_fit_end", "calibration_start", "calibration_end", "evaluation_start")]
    if not times[0] < times[1] < times[2] < times[3]:
        raise EvidenceError("strictly non-overlapping fit, calibration, evaluation boundaries required")
    alpha = config.get("alpha")
    if type(alpha) is not float or not 0 < alpha < 1:
        raise EvidenceError("alpha must be explicit float between zero and one")
    min_n = config.get("min_cell_n")
    if type(min_n) is not int or min_n < 20:
        raise EvidenceError("at least 20 calibration cases per prespecified cell required")
    for key in ("channels", "gap_kinds", "horizons_hours"):
        values = config.get(key)
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise EvidenceError(f"{key} must be nonempty distinct list")
        if key == "horizons_hours":
            if any(type(v) is not int or not 1 <= v <= 2160 for v in values):
                raise EvidenceError("invalid forecast horizons")
        elif any(not isinstance(v, str) or not v or len(v) > 64 for v in values):
            raise EvidenceError(f"invalid {key}")
    if not config.get("source_id") or not isinstance(config["source_id"], str):
        raise EvidenceError("source declaration required")
    return config


def cells(config):
    return [(c, h, g) for c in config["channels"] for h in config["horizons_hours"]
            for g in config["gap_kinds"]]


def cell_key(channel, horizon, gap):
    return canonical([channel, horizon, gap])


BASE = {"row_id", "entity_id", "issued_at", "model_sha256", "channel",
        "horizon_hours", "gap_kind", "point", "lower", "upper"}
LABEL = {"truth", "outcome_available_at"}


def validate_row(row, contract, labeled, phase):
    if not isinstance(row, dict) or set(row) != (BASE | LABEL if labeled else BASE):
        raise EvidenceError("unregistered fields or hidden outcome fields")
    for field in ("row_id", "entity_id"):
        if not isinstance(row[field], str) or not row[field] or len(row[field]) > 256:
            raise EvidenceError(f"invalid {field}")
    if row["model_sha256"] != contract["model_sha256"]:
        raise EvidenceError("wrong frozen source-model hash")
    if (row["channel"] not in contract["channels"]
        or row["horizon_hours"] not in contract["horizons_hours"]
        or row["gap_kind"] not in contract["gap_kinds"]):
        raise EvidenceError("unregistered stratum or horizon")
    for field in ("point", "lower", "upper"):
        if type(row[field]) not in (int, float) or not math.isfinite(row[field]):
            raise EvidenceError("forecast must be finite number")
    if row["lower"] > row["upper"]:
        raise EvidenceError("inverted forecast interval")
    when = iso_utc(row["issued_at"])
    if phase == "calibration":
        if not iso_utc(contract["calibration_start"]) <= when < iso_utc(contract["calibration_end"]):
            raise EvidenceError("calibration row outside frozen time cohort")
    elif when < iso_utc(contract["evaluation_start"]):
        raise EvidenceError("evaluation/inference row precedes frozen start")
    if labeled:
        if type(row["truth"]) not in (int, float) or not math.isfinite(row["truth"]):
            raise EvidenceError("nonfinite outcome")
        outcome_time = iso_utc(row["outcome_available_at"])
        if outcome_time < when + timedelta(hours=row["horizon_hours"]):
            raise EvidenceError("future outcome timestamp predates target horizon")
        if phase == "calibration" and not outcome_time < iso_utc(contract["calibration_end"]):
            raise EvidenceError("label becomes available after frozen calibration cutoff")
    return row


def check_records(rows, contract, labeled, phase):
    seen = set()
    for row in rows:
        validate_row(row, contract, labeled, phase)
        identity = (row["entity_id"], row["issued_at"], row["channel"], row["horizon_hours"])
        if identity in seen:
            raise EvidenceError("duplicate case/target, cannot count as independent")
        seen.add(identity)


def score_interval(lo, hi, y, alpha):
    return (hi - lo) + (2 / alpha) * max(lo - y, 0) + (2 / alpha) * max(y - hi, 0)


def calibrate(config, records, input_sha):
    config = valid_config(config)
    check_records(records, config, True, "calibration")
    by_cell = {cell_key(*c): [] for c in cells(config)}
    for r in records:
        residual = max(r["lower"] - r["truth"], r["truth"] - r["upper"], 0.)
        by_cell[cell_key(r["channel"], r["horizon_hours"], r["gap_kind"])].append(residual)
    result = {}
    for key, residuals in by_cell.items():
        n = len(residuals)
        if n < config["min_cell_n"]:
            raise EvidenceError("insufficient calibration group: " + key)
        k = math.ceil((n + 1) * (1 - config["alpha"]))
        if k > n:
            raise EvidenceError("finite-sample rank unavailable: " + key)
        correction = sorted(residuals)[k - 1]
        result[key] = {"n": n, "correction": correction, "rank": k}
    return {"schema": "forecast-reliability-artifact-v1", "contract": config,
            "calibration_file_sha256": input_sha, "cell_corrections": result,
            "evidence_status": "CALIBRATION_ONLY_UNVERIFIED_SOURCE",
            "coverage_assumptions": "exchangeability within cells; temporal/entity dependence and drift unverified"}


def adjusted(artifact, row, inference=True):
    if artifact.get("schema") != "forecast-reliability-artifact-v1":
        raise EvidenceError("unknown artifact schema")
    contract = valid_config(artifact["contract"])
    validate_row(row, contract, not inference, "inference" if inference else "evaluation")
    key = cell_key(row["channel"], row["horizon_hours"], row["gap_kind"])
    if key not in artifact["cell_corrections"]:
        raise EvidenceError("unqualified calibration stratum")
    adj = artifact["cell_corrections"][key]["correction"]
    if type(adj) not in (int, float) or not math.isfinite(adj) or adj < 0:
        raise EvidenceError("invalid calibration correction")
    return {"point": row["point"], "lower": row["lower"] - adj, "upper": row["upper"] + adj,
            "model_sha256": row["model_sha256"], "channel": row["channel"],
            "horizon_hours": row["horizon_hours"], "gap_kind": row["gap_kind"],
            "issued_at": row["issued_at"], "entity_id": row["entity_id"],
            "qualification": "UNVERIFIED_TEMPORAL_CALIBRATION",
            "decision_authority": "ADVISORY_ONLY"}


def evaluate(artifact, records):
    contract = valid_config(artifact["contract"])
    check_records(records, contract, True, "evaluation")
    aggregates = {cell_key(*c): [] for c in cells(contract)}
    for row in records:
        prediction = adjusted(artifact, row, inference=False)
        y = row["truth"]
        aggregates[cell_key(row["channel"], row["horizon_hours"], row["gap_kind"])].append(
            (row, prediction, y))
    report = {}
    for key, items in aggregates.items():
        if not items:
            report[key] = {"n": 0, "status": "NO_EVALUATION_CASES"}
            continue
        n = len(items)
        report[key] = {
            "n": n, "status": "DESCRIPTIVE_NOT_INDEPENDENTLY_QUALIFIED",
            "raw_coverage": sum(r["lower"] <= y <= r["upper"] for r, _, y in items) / n,
            "adjusted_coverage": sum(p["lower"] <= y <= p["upper"] for _, p, y in items) / n,
            "point_mae": sum(abs(r["point"] - y) for r, _, y in items) / n,
            "raw_interval_score": sum(score_interval(r["lower"], r["upper"], y, contract["alpha"])
                                      for r, _, y in items) / n,
            "adjusted_interval_score": sum(score_interval(p["lower"], p["upper"], y, contract["alpha"])
                                           for _, p, y in items) / n,
            "raw_width": sum(r["upper"] - r["lower"] for r, _, _ in items) / n,
            "adjusted_width": sum(p["upper"] - p["lower"] for _, p, _ in items) / n,
        }
    return {"status": "DESCRIPTIVE_EVALUATION_ONLY", "model_sha256": contract["model_sha256"],
            "source_authenticated": False, "independent_cohort_certified": False,
            "cells": report, "warnings": [
                "no guarantee of temporal/conditional coverage under shift or overlapping horizons",
                "verified source provenance and independent protected TEST release require external gate",
            ]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("calibrate")
    a.add_argument("--contract", type=Path, required=True)
    a.add_argument("--records", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("evaluate")
    b.add_argument("--artifact", type=Path, required=True)
    b.add_argument("--records", type=Path, required=True)
    c = sub.add_parser("infer")
    c.add_argument("--artifact", type=Path, required=True)
    c.add_argument("--request", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        if args.cmd == "calibrate":
            if args.out.exists():
                raise EvidenceError("refusing to overwrite frozen artifact")
            content = calibrate(json.loads(args.contract.read_text()), rows_from_jsonl(args.records),
                                sha_file(args.records))
            with args.out.open("x", encoding="utf-8") as f:
                json.dump(content, f, sort_keys=True, indent=2)
            print(canonical({"status": content["evidence_status"], "artifact": str(args.out)}))
        else:
            artifact = json.loads(args.artifact.read_text())
            if args.cmd == "evaluate":
                print(canonical(evaluate(artifact, rows_from_jsonl(args.records))))
            else:
                print(canonical(adjusted(artifact, json.loads(args.request.read_text()), inference=True)))
        return 0
    except (EvidenceError, OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        print(canonical({"status": "BLOCKED_OR_FAILED", "reason": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
