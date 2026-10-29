#!/usr/bin/env python3
"""LANGPRETRAIN075: offline original-byte corpus audit and conditional cost planning.

NO TRAINING, DOWNLOADS, GPU SUBMISSION, TEST ACCESS, OR CLAIMED AUTHORIZATION.
Constructed test fixtures are software checks, never scientific results.
"""
from __future__ import annotations

import argparse
import codecs
import hashlib
import json
from pathlib import Path
import re

SCHEMA = "tac-osm.byte-pretrain-corpus.v1"
CATEGORIES = ("web", "code", "technical", "dialogue")
TARGET_MIX = {"web": 0.50, "code": 0.25, "technical": 0.15, "dialogue": 0.10}
EXPOSURES = (10_000_000, 50_000_000, 100_000_000)
CONTEXT = 128
# Authentic existing 500122p CPU-language pilot, not a GPU benchmark.
CPU_REFERENCE = {"training_targets": 327680, "cpu_process_seconds": 44.17,
                 "source": "LANGCOMPARE031 5-seed 320-step CPU mean"}


class AuditError(ValueError):
    """Malformed, unverified, or unsafe corpus input."""


def _is_hex_hash(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _source_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise AuditError("invalid relative source path")
    candidate = Path(relative)
    if candidate.is_absolute() or any(part in ("", ".", "..") for part in candidate.parts):
        raise AuditError("path traversal or absolute source path")
    current = root
    for part in candidate.parts:
        current = current / part
        if current.is_symlink():
            raise AuditError("symlinked corpus path denied")
    resolved = current.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise AuditError("source file missing or outside approved corpus root")
    return resolved


def _digest_utf8_stream(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    n = 0
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
            decoder.decode(chunk, final=False)
            n += len(chunk)
        decoder.decode(b"", final=True)
    return digest.hexdigest(), n


def audit_corpus(manifest_path: Path, corpus_root: Path) -> dict:
    """Audit locally staged publisher files; never download or perform model fits."""
    root = corpus_root.resolve(strict=True)
    if not root.is_dir():
        raise AuditError("corpus root must be a directory")
    manifest_blob = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_blob)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError("invalid UTF-8 manifest JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != {"schema", "documents"}:
        raise AuditError("manifest requires schema and documents only")
    if manifest["schema"] != SCHEMA or not isinstance(manifest["documents"], list) or not manifest["documents"]:
        raise AuditError("unknown schema or no publisher documents")
    seen_ids: set[str] = set()
    seen_sha: set[str] = set()
    fit_bytes = {category: 0 for category in CATEGORIES}
    split_bytes = {"FIT": 0, "DEV": 0}
    window_count = {"FIT": 0, "DEV": 0}
    audited = []
    for doc in manifest["documents"]:
        required = {"id", "split", "category", "path", "sha256", "license", "publisher", "source_url"}
        if not isinstance(doc, dict) or set(doc) != required:
            raise AuditError("document fields incomplete or unexpected")
        identity = doc["id"]
        if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", identity):
            raise AuditError("unsafe document ID")
        if identity in seen_ids:
            raise AuditError("document ID repeated across partitions")
        seen_ids.add(identity)
        if doc["split"] not in split_bytes:
            raise AuditError("TEST and unknown partitions forbidden in training audit")
        category = doc["category"]
        if category not in CATEGORIES:
            raise AuditError("unknown corpus category")
        if not _is_hex_hash(doc["sha256"]):
            raise AuditError("original source SHA256 required")
        for key in ("license", "publisher", "source_url"):
            if not isinstance(doc[key], str) or not doc[key].strip():
                raise AuditError("publisher, license and source URL required")
        if not doc["source_url"].startswith("https://"):
            raise AuditError("publisher source URL must use https")
        path = _source_path(root, doc["path"])
        sha, size = _digest_utf8_stream(path)
        if sha != doc["sha256"]:
            raise AuditError(f"original byte digest mismatch: {identity}")
        if sha in seen_sha:
            raise AuditError("exact duplicate source across documents/partitions")
        seen_sha.add(sha)
        split_bytes[doc["split"]] += size
        window_count[doc["split"]] += max(0, size - CONTEXT)
        if doc["split"] == "FIT":
            fit_bytes[category] += size
        audited.append({"id": identity, "split": doc["split"], "category": category,
                        "bytes": size, "sha256": sha,
                        "within_document_next_byte_windows": max(0, size - CONTEXT)})
    if not split_bytes["FIT"] or not split_bytes["DEV"] or not window_count["FIT"] or not window_count["DEV"]:
        raise AuditError("nonempty FIT and DEV next-byte windows are mandatory")
    total = split_bytes["FIT"]
    actual_mix = {category: fit_bytes[category] / total for category in CATEGORIES}
    return {
        "schema": "tac-osm.byte-pretrain-audit.v1",
        "status": "SOURCE_DIGEST_AND_UTF8_CHECKED_NOT_AUTHORIZED",
        "manifest_sha256": hashlib.sha256(manifest_blob).hexdigest(),
        "documents": audited,
        "split_bytes": split_bytes,
        "within_document_windows": window_count,
        "fit_mix": actual_mix,
        "target_mix": TARGET_MIX,
        "mix_deviation_pct_points": {
            c: round(100 * (actual_mix[c] - TARGET_MIX[c]), 3) for c in CATEGORIES
        },
        "checks_performed": ["file_containment", "raw_sha256", "strict_utf8",
                             "exact_document_dedup", "no_test", "document_windows"],
        "checks_not_performed": ["publisher_authenticity", "license_legal_review",
                                 "near_dedup", "source_time_integrity", "downstream_eval_leakage",
                                 "independent_prefit_approval", "training_or_checkpoint_replay"],
        "document_state_reset_required": True,
        "approved_to_train": False, "gpu_minutes_consumed": 0,
        "protected_TEST": "NOT_ACCESSED"
    }


def cost_plan(exposure_bytes: int, gpu_aggregate_bytes_per_second: float | None,
              usd_per_gpu_hour: float, gpu_count: int = 2) -> dict:
    if type(exposure_bytes) is not int or exposure_bytes <= 0:
        raise AuditError("positive integer training exposure required")
    if gpu_aggregate_bytes_per_second is not None and not (
        0 < gpu_aggregate_bytes_per_second < float("inf")
    ):
        raise AuditError("finite positive measured aggregate throughput required")
    if not (0 <= usd_per_gpu_hour < float("inf")) or gpu_count != 2:
        raise AuditError("nonnegative finite per-GPU-hour rate and exactly 2 T4 devices required")
    cpu_rate = CPU_REFERENCE["training_targets"] / CPU_REFERENCE["cpu_process_seconds"]
    gpu_seconds = (None if gpu_aggregate_bytes_per_second is None else
                   exposure_bytes / gpu_aggregate_bytes_per_second)
    return {
        "schema": "tac-osm.byte-pretrain-cost.v1",
        "exposure_bytes_per_seed": exposure_bytes,
        "cpu_reference": CPU_REFERENCE,
        "cpu_extrapolated_hours_PER_SEED_NOT_FORECAST": exposure_bytes / cpu_rate / 3600,
        "gpu_aggregate_bytes_per_second": gpu_aggregate_bytes_per_second,
        "gpu_wall_hours_PER_SEED": None if gpu_seconds is None else gpu_seconds / 3600,
        "gpu_count": gpu_count, "gpu_type": "NvidiaTeslaT4",
        "usd_per_gpu_hour_USER_INPUT_NOT_QUOTE": usd_per_gpu_hour,
        "estimated_paid_compute_per_seed_usd": (
            None if gpu_seconds is None else gpu_seconds / 3600 * usd_per_gpu_hour * gpu_count
        ),
        "two_seed_concurrency": "separate frozen seed shards, each T4 independently measured",
        "warning": "GPU rate unknown until measured on actual TAC training; CPU process average "
                   "is exploratory, does not predict long-fit speed or future model sizes",
        "approved_to_spend": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="audit local publisher corpus without training")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--corpus-root", type=Path, required=True)
    plan = sub.add_parser("cost", help="conditional compute estimate; never runs GPUs")
    plan.add_argument("--exposure-bytes", type=int, choices=EXPOSURES, required=True)
    plan.add_argument("--measured-aggregate-bytes-per-second", type=float)
    plan.add_argument("--usd-per-gpu-hour", type=float, default=0.0)
    args = parser.parse_args()
    try:
        report = (audit_corpus(args.manifest, args.corpus_root) if args.command == "audit"
                  else cost_plan(args.exposure_bytes,
                                 args.measured_aggregate_bytes_per_second,
                                 args.usd_per_gpu_hour))
    except (AuditError, OSError, ValueError) as exc:
        parser.exit(2, f"BLOCKED: {exc}\n")
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
