#!/usr/bin/env python3
"""Offline no-fit audit of EXACT original Kaggle dual-T4 smoke receipts.

This is a deterministic consistency check of Kaggle-produced, self-reported
files; it does not prove independent device ownership, GPU peak memory, or
neural model quality. Never contacts Kaggle, executes workloads or trains.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FILES = ("phone-research-runtime.json", "seed-101.json", "seed-102.json")
EXPECTED_SEEDS = (101, 102)
EXPECTED_SQUARE_SUM = 357389824.0


class SmokeReceiptError(ValueError):
    pass


def _read_json(directory: Path, name: str) -> dict:
    path = directory / name
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32768:
        raise SmokeReceiptError(f"{name}: missing, symlinked or oversized")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise SmokeReceiptError(f"{name}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise SmokeReceiptError(f"{name}: expected JSON object")
    return value


def _must(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeReceiptError(message)


def _int_is(value, expected: int) -> bool:
    return type(value) is int and value == expected


def _t4(name) -> bool:
    return isinstance(name, str) and "T4" in name.upper()


def audit(directory: Path, root: Path = ROOT) -> dict:
    if directory.is_symlink() or not directory.is_dir():
        raise SmokeReceiptError("missing or symlinked receipt directory")
    source = root / "research/kaggle_workloads/dual_t4_smoke.py"
    _must(source.is_file() and not source.is_symlink(), "reviewed smoke workload missing")
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    actual = {name: _read_json(directory, name) for name in FILES}
    runtime = actual[FILES[0]]
    _must(runtime.get("schema") == "tac-osm.kaggle-t4x2-runtime.v1", "wrong runtime schema")
    _must(runtime.get("run_id") == "dual-t4-readiness", "wrong run id")
    _must(runtime.get("requested_accelerator") == "NvidiaTeslaT4", "accelerator mismatch")
    _must(runtime.get("strategy") == "seed_shard", "seed sharding absent")
    _must(runtime.get("seeds") == list(EXPECTED_SEEDS), "frozen seed plan mismatch")
    _must(runtime.get("status") == "COMPLETED_UNQUALIFIED_REQUIRES_INDEPENDENT_EVALUATION",
          "remote runtime not marked completed")
    _must(runtime.get("real_world_claim_authorized") is False,
          "remote runtime unauthorized science claim")
    _must(runtime.get("workload_sha256") == expected_hash, "workload SHA differs")
    observed = runtime.get("observed")
    _must(isinstance(observed, dict), "missing observed device summary")
    _must(_int_is(observed.get("count"), 2), "exactly two GPU devices required")
    _must(observed.get("accelerator") == "NvidiaTeslaT4", "observed accelerator mismatch")
    names = observed.get("names")
    _must(isinstance(names, list) and len(names) == 2 and all(_t4(n) for n in names),
          "not two self-reported T4 device names")
    workers = runtime.get("results")
    _must(isinstance(workers, list) and len(workers) == 2, "missing two worker results")
    _must(all(isinstance(w, dict)
              and type(w.get("worker_index")) is int
              and _int_is(w.get("exit_code"), 0) for w in workers),
          "invalid worker metadata or nonzero exit")
    _must({w["worker_index"] for w in workers} == {0, 1},
          "workers must have distinct indexes 0 and 1")
    for seed, slot in zip(EXPECTED_SEEDS, (0, 1)):
        name = f"seed-{seed}.json"
        row = actual[name]
        _must(row.get("schema") == "tac-osm.no-fit-t4-smoke.v1", f"{name}: schema")
        _must(_int_is(row.get("seed"), seed) and _int_is(row.get("physical_slot"), slot),
              f"{name}: expected isolated seed and slot")
        _must(_t4(row.get("device")), f"{name}: T4 identity missing")
        value = row.get("numeric_diagnostic")
        _must(type(value) in (int, float) and math.isfinite(value)
              and value == EXPECTED_SQUARE_SUM, f"{name}: arithmetic mismatch")
        _must(row.get("source_provenance") == "constructed_tensor_only"
              and row.get("model_training") is False
              and row.get("capability_claim_authorized") is False,
              f"{name}: violates no-fit boundary")
    return {
        "schema": "tac-osm.kaggle-no-fit-receipt-audit.v1",
        "status": "SELF_REPORTED_TWO_WORKER_RECEIPTS_CONSISTENT",
        "source_sha256": expected_hash,
        "source_type": "constructed_tensor_only",
        "seeds": list(EXPECTED_SEEDS),
        "self_reported_two_worker_execution": True,
        "independent_remote_attestation": False,
        "gpu_device_provenance_independently_verified": False,
        "model_training": False,
        "scientific_qualification": "NONE",
        "files_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                         for name in FILES},
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        result = audit(args.output_dir)
    except (SmokeReceiptError, OSError) as exc:
        result = {
            "schema": "tac-osm.kaggle-no-fit-receipt-audit.v1",
            "status": "INCOMPLETE_NOT_VERIFIED",
            "reason": str(exc)[:200],
            "self_reported_two_worker_execution": False,
            "independent_remote_attestation": False,
            "scientific_qualification": "NONE",
        }
        code = 2
    else:
        code = 0
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
