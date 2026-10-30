"""Frozen real-source LANGPERSIST096 episodes; no model execution or fitting.

The identifier strata use scorer-side future source solely to freeze cases.
Actors receive acquisition.npy and queries.npy only. Target names, labels,
source identities, offsets, targets.npy and original files belong to the scorer.
An insufficient cohort is preserved as a failed gate, never silently relaxed.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
import tokenize
from typing import Any, Mapping
import urllib.request

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE_REF = "87d8a3b288242b2db8c4e052f1571ff6005d0e36"
SOURCE_COHORT_SHA256 = "f3d485d9097ed63dee8bec98704f982b6426ffbba099278d411aca8a350fa3bf"
DEFAULT_SOURCE_RECEIPT = ROOT / "research/commercial-value/codevalue003/SOURCE_RECEIPT.json"
SALT = "LANGPERSIST096-96001"
ACQUISITION = 128
QUERY = 64
TARGET = 4
PER_GROUP = 20
GROUPS = ("positive", "negative", "uniform")
PUBLISHERS = ("django/django", "fastapi/fastapi", "Textualize/rich", "pydantic/pydantic", "pytest-dev/pytest")
ASCII_NAME = re.compile(rb"[A-Za-z_][A-Za-z_0-9]*\Z")
# Freeze Python hard and soft keywords, rather than changing eligibility when
# the host's keyword module changes. Soft keywords are conservatively excluded.
KEYWORDS = frozenset(("False None True and as assert async await break class continue def del elif else "
    "except finally for from global if import in is lambda nonlocal not or pass raise return try "
    "while with yield match case type").split())


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def authenticate_bytes(raw: bytes, row: Mapping[str, Any], *, license_only: bool = False) -> dict[str, Any]:
    """Code requires all three frozen identities; license archives pin SHA256."""
    if not isinstance(raw, bytes) or digest(raw) != row.get("sha256"):
        raise ValueError("publisher SHA256 mismatch: " + str(row.get("path")))
    if not license_only and (len(raw) != row.get("size") or git_blob(raw) != row.get("git_blob_sha1")):
        raise ValueError("publisher size/Git-blob mismatch: " + str(row.get("path")))
    return {"size": len(raw), "sha256": digest(raw), "git_blob_sha1": git_blob(raw),
            "all_three_identities_prefrozen": not license_only}


def ascii_name_spans(raw: bytes) -> list[tuple[int, int, bytes]]:
    """Complete ASCII NAME tokens with exact raw-byte offsets, never columns.

    Comments, strings and Unicode identifiers are excluded. Byte columns are
    computed by re-encoding each decoded prefix, including any UTF-8 BOM once.
    """
    if not isinstance(raw, bytes):
        raise TypeError("unmodified original bytes required")
    encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
    bom = 3 if encoding.lower() == "utf-8-sig" else 0
    codec = "utf-8" if bom else encoding
    lines = raw.split(b"\n")
    line_offsets, offset = [], 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line) + 1
    result = []
    for token in tokenize.tokenize(io.BytesIO(raw).readline):
        if token.type != tokenize.NAME or not token.string.isascii():
            continue
        name = token.string.encode("ascii")
        if not ASCII_NAME.fullmatch(name) or token.string in KEYWORDS:
            continue
        row, column = token.start
        end_row, end_column = token.end
        if row != end_row:
            raise ValueError("unexpected multiline NAME token")
        prefix_bom = bom if row == 1 else 0
        line = lines[row - 1][prefix_bom:].decode(codec)
        start = line_offsets[row - 1] + prefix_bom + len(line[:column].encode(codec))
        end = line_offsets[row - 1] + prefix_bom + len(line[:end_column].encode(codec))
        if raw[start:end] != name:
            raise ValueError("token-to-original-byte alignment failed")
        result.append((start, end, name))
    return result


def fit_identifier_inventory(documents: Mapping[str, bytes]) -> set[bytes]:
    return {name for raw in documents.values() for _, _, name in ascii_name_spans(raw)}


def _rank(group: str, publisher: str, file: str, offset: int) -> str:
    return digest(canonical_json([SALT, group, publisher, file, offset]))


def _record(row: Mapping[str, Any], group: str, offset: int, *,
            name: bytes | None = None, historical_offset: int | None = None) -> dict[str, Any]:
    return {"group": group, "publisher": row["repo"], "file": row["file"],
            "target_offset": offset, "acquisition_start": offset - ACQUISITION - QUERY,
            "acquisition_end": offset - QUERY, "query_start": offset - QUERY,
            "query_end": offset, "target_end": offset + TARGET,
            "target_name": name.decode("ascii") if name is not None else None,
            "historical_name_offset": historical_offset,
            "selection_sha256": _rank(group, row["repo"], row["file"], offset)}


def _overlap(record: Mapping[str, Any], occupied: Mapping[str, list[tuple[int, int]]]) -> bool:
    start, end = record["acquisition_start"], record["target_end"]
    return any(start < old_end and old_start < end for old_start, old_end in occupied.get(record["file"], []))


def build_cohort(fit_documents: Mapping[str, bytes], transfer_documents: Mapping[str, bytes],
                 source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fixed 5-publisher x 3-stratum x 20 selection; no tunable thresholds."""
    if {row["repo"] for row in source_rows} != set(PUBLISHERS):
        raise ValueError("exact frozen five-publisher cohort required")
    if any(row.get("partition") != "TRANSFER_DEV" for row in source_rows):
        raise ValueError("TRANSFER_DEV only; no protected TEST")
    if len({row["file"] for row in source_rows}) != len(source_rows):
        raise ValueError("duplicate source file identity")
    if set(transfer_documents) != {row["file"] for row in source_rows}:
        raise ValueError("source/document coverage differs")
    inventory = fit_identifier_inventory(fit_documents)
    inventory_bytes = b"\n".join(sorted(inventory)) + b"\n"
    candidates = {publisher: {group: [] for group in GROUPS} for publisher in PUBLISHERS}
    for row in sorted(source_rows, key=lambda r: r["file"]):
        raw = transfer_documents[row["file"]]
        spans = ascii_name_spans(raw)
        occurrences: dict[bytes, list[tuple[int, int]]] = {}
        for start, end, name in spans:
            occurrences.setdefault(name, []).append((start, end))
        for position, end, name in spans:
            if position < ACQUISITION + QUERY or not 4 <= len(name) <= 24 or name in inventory:
                continue
            acquisition_start, acquisition_end = position - ACQUISITION - QUERY, position - QUERY
            query = raw[acquisition_end:position]
            if name in query:
                continue
            positions = occurrences[name]
            first = bisect_left(positions, (acquisition_start, -1))
            historical = positions[first] if first < len(positions) else None
            if historical is not None and historical[1] <= acquisition_end:
                candidates[row["repo"]]["positive"].append(
                    _record(row, "positive", position, name=name, historical_offset=historical[0]))
            elif name not in raw[acquisition_start:acquisition_end]:
                candidates[row["repo"]]["negative"].append(_record(row, "negative", position, name=name))
    selected, occupied, counts, failures = [], {}, {}, []
    for publisher in sorted(PUBLISHERS):
        counts[publisher] = {}
        for group in GROUPS:  # Scarce positive cases are fixed first.
            if group == "uniform":
                # Compact rank tuples, one publisher at a time: do not retain
                # a full scorer record for every byte in all ten source files.
                pool = sorted((_rank(group, publisher, row["file"], position), row["file"], position)
                    for row in source_rows if row["repo"] == publisher
                    for position in range(ACQUISITION + QUERY, len(transfer_documents[row["file"]]) - TARGET + 1))
            else:
                pool = sorted(candidates[publisher][group], key=lambda r: (r["selection_sha256"], r["file"], r["target_offset"]))
            chosen = []
            by_file = {row["file"]: row for row in source_rows}
            for candidate in pool:
                record = (_record(by_file[candidate[1]], group, candidate[2])
                          if group == "uniform" else candidate)
                if not _overlap(record, occupied):
                    chosen.append(record)
                    occupied.setdefault(record["file"], []).append((record["acquisition_start"], record["target_end"]))
                    if len(chosen) == PER_GROUP:
                        break
            counts[publisher][group] = {"eligible_candidates": len(pool), "selected_nonoverlapping": len(chosen)}
            if len(chosen) != PER_GROUP:
                failures.append({"publisher": publisher, "group": group, "required": PER_GROUP, "selected": len(chosen)})
            selected.extend(chosen)
    proof = {"salt": SALT, "selection_order": list(GROUPS), "per_group_per_publisher": PER_GROUP,
             "candidate_counts": counts, "failures": failures,
             "interval_policy": "same-file disjoint half-open [target_offset-192,target_offset+4)",
             "uniform_policy": "all valid byte positions, independent of NAME and target values before nonoverlap",
             "target_policy": "first four bytes at complete ASCII NAME start; full NAME length4..24; frozen keywords excluded",
             "positive_policy": "full NAME absent all FIT inventory and Q bytes; completed same NAME token entirely inside A",
             "negative_policy": "full NAME absent all FIT inventory and both A/Q byte strings",
             "fit_inventory_names": len(inventory), "fit_inventory_sha256": digest(inventory_bytes),
             "tokenizer": "stdlib tokenize NAME", "python_version": sys.version.split()[0],
             "frozen_keywords": sorted(KEYWORDS)}
    result: dict[str, Any] = {"status": "INSUFFICIENT_COHORT" if failures else "COHORT_READY",
                              "selection_proof": proof, "fit_inventory": inventory_bytes}
    if failures:
        return result
    acquisitions, queries, targets = [], [], []
    for index, record in enumerate(selected):
        record["index"] = index
        raw = transfer_documents[record["file"]]
        acquisitions.append(np.frombuffer(raw[record["acquisition_start"]:record["acquisition_end"]], dtype=np.uint8))
        queries.append(np.frombuffer(raw[record["query_start"]:record["query_end"]], dtype=np.uint8))
        targets.append(np.frombuffer(raw[record["target_offset"]:record["target_end"]], dtype=np.uint8))
    donors = np.empty(len(selected), dtype=np.int64)
    publisher_indices = {publisher: [r["index"] for r in selected if r["publisher"] == publisher] for publisher in sorted(PUBLISHERS)}
    publishers = sorted(PUBLISHERS)
    for publisher_index, publisher in enumerate(publishers):
        for own, donor in zip(publisher_indices[publisher], publisher_indices[publishers[(publisher_index + 1) % len(publishers)]]):
            donors[own] = donor
    proof["donor_policy"] = "fixed lexicographic cyclic other publisher, same within-publisher row index; no target filtering"
    proof["records_sha256"] = digest(canonical_json(selected))
    proof["selected_episodes"] = len(selected)
    result.update(records=selected, arrays={"acquisition.npy": np.stack(acquisitions),
        "queries.npy": np.stack(queries), "targets.npy": np.stack(targets), "donor_indices.npy": donors})
    return result


def _validate_receipt(receipt: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if receipt.get("study") != "CODEVALUE003" or receipt.get("cohort_sha256") != SOURCE_COHORT_SHA256:
        raise ValueError("wrong frozen CODEVALUE003 source cohort")
    original = dict(receipt)
    original.pop("cohort_sha256", None)
    if digest(canonical_json(original)) != SOURCE_COHORT_SHA256:
        raise ValueError("frozen source receipt content differs")
    rows = receipt.get("files", [])
    if any(row.get("partition") not in {"FIT", "DEV", "TRANSFER_DEV"} for row in rows):
        raise ValueError("protected/unknown source partition rejected before payload access")
    fit = [row for row in rows if row["partition"] == "FIT"]
    transfer = [row for row in rows if row["partition"] == "TRANSFER_DEV"]
    if len(fit) != 14 or len(transfer) != 10 or {row["repo"] for row in transfer} != set(PUBLISHERS):
        raise ValueError("complete14FIT/10TRANSFER_DEV cohort required")
    if any(sum(row["repo"] == publisher for row in transfer) != 2 for publisher in PUBLISHERS):
        raise ValueError("two frozen transfer files per publisher required")
    if len({row["file"] for row in fit + transfer}) != 24:
        raise ValueError("unique source files required")
    return fit, transfer


def _raw_url(row: Mapping[str, Any]) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", row["repo"]) or not re.fullmatch(r"[0-9a-f]{40}", row["commit"]):
        raise ValueError("invalid pinned publisher identity")
    parts = Path(row["path"]).parts
    if not parts or Path(row["path"]).is_absolute() or ".." in parts:
        raise ValueError("invalid publisher-relative path")
    return "https://raw.githubusercontent.com/" + row["repo"] + "/" + row["commit"] + "/" + row["path"]


def _cached(directory: Path, row: Mapping[str, Any], *, license_only: bool = False) -> tuple[bytes, dict[str, Any]]:
    filename = row["file"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", filename) or filename in {".", ".."}:
        raise ValueError("unsafe source cache filename")
    path = directory / filename
    if path.is_symlink():
        raise ValueError("symlink source cache rejected")
    cached = path.exists()
    if cached:
        raw = path.read_bytes()
    else:
        with urllib.request.urlopen(_raw_url(row), timeout=45) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError("publisher object exceeds bounded source size")
    measured = authenticate_bytes(raw, row, license_only=license_only)
    if not cached:
        with path.open("xb") as handle:
            handle.write(raw)
    return raw, {**dict(row), **measured, "cache_hit": cached, "url": _raw_url(row)}


def _write_json(path: Path, value: Any) -> None:
    with path.open("xb") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def prepare(data_dir: str | Path, manifest_path: str | Path | None = None) -> dict[str, Any]:
    """Authenticate public sources and freeze data artifacts outside the repo.

    manifest_path is the input SOURCE_RECEIPT.json override, not an output path.
    Existing authenticated raw cache files may be reused but never overwritten.
    A sealed or failed output directory is immutable; use it directly for replay.
    """
    started = time.perf_counter()
    destination = Path(data_dir).resolve()
    if destination.is_relative_to(ROOT.resolve()):
        raise ValueError("original source and episode payloads must stay outside Git repository")
    destination.mkdir(parents=True, exist_ok=True)
    output_manifest = destination / "MANIFEST.json"
    if output_manifest.exists() or output_manifest.is_symlink():
        raise FileExistsError("preserve original cohort manifest; no overwrite")
    source_path = Path(manifest_path) if manifest_path is not None else DEFAULT_SOURCE_RECEIPT
    receipt_raw = source_path.read_bytes()
    receipt = json.loads(receipt_raw)
    fit_rows, transfer_rows = _validate_receipt(receipt)
    needed = {(row["repo"], row["commit"]) for row in fit_rows + transfer_rows}
    license_rows = [row for row in receipt["license_files"] if (row["repo"], row["commit"]) in needed]
    if {(row["repo"], row["commit"]) for row in license_rows} != needed:
        raise ValueError("missing original publisher licenses")
    cache = destination / "sources"
    cache.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as executor:
        code = list(executor.map(lambda row: _cached(cache, row), fit_rows + transfer_rows))
        licenses = list(executor.map(lambda row: _cached(cache, row, license_only=True), license_rows))
    loaded = {row["file"]: raw for raw, row in code}
    fit_documents = {row["file"]: loaded[row["file"]] for row in fit_rows}
    transfer_documents = {row["file"]: loaded[row["file"]] for row in transfer_rows}
    built = build_cohort(fit_documents, transfer_documents, transfer_rows)
    _write_json(destination / "SOURCE_RECEIPT.json", receipt)
    with (destination / "fit_identifier_inventory.txt").open("xb") as handle:
        handle.write(built["fit_inventory"])
    artifacts = {}
    if built["status"] == "COHORT_READY":
        for filename, array in built["arrays"].items():
            with (destination / filename).open("xb") as handle:
                np.save(handle, array, allow_pickle=False)
            raw = (destination / filename).read_bytes()
            artifacts[filename] = {"path": filename, "sha256": digest(raw), "bytes": len(raw),
                                   "shape": list(array.shape), "dtype": str(array.dtype)}
        _write_json(destination / "records.json", built["records"])
        raw = (destination / "records.json").read_bytes()
        artifacts["records.json"] = {"path": "records.json", "sha256": digest(raw), "bytes": len(raw)}
    manifest = {"schema": "tac-osm.langpersist096.corpus.v1", "study": "LANGPERSIST096",
        "status": built["status"], "execution_level": "internal_research",
        "source_ref": SOURCE_REF, "source_cohort_sha256": SOURCE_COHORT_SHA256,
        "original_source_receipt_sha256": digest(receipt_raw), "source_rows": [row for _, row in code],
        "source_rights": [row for _, row in licenses], "partitions_read": {"FIT": 14, "TRANSFER_DEV": 10},
        "excluded_partitions": ["DEV", "protected_TEST"], "protected_TEST": "NOT_ACCESSED",
        "training_steps": 0, "model_executions": 0, "independently_qualified": False,
        "qualification": "internally_authenticated_source_and_frozen_cohort_not_capability_evidence",
        "selection_proof": built["selection_proof"], "artifacts": artifacts,
        "actor_inputs_only": ["acquisition.npy", "queries.npy"],
        "scorer_only": ["targets.npy", "records.json", "fit_identifier_inventory.txt", "sources"],
        "costs": {"source_bytes_authenticated": sum(len(raw) for raw, _ in code),
                  "license_bytes_authenticated": sum(len(raw) for raw, _ in licenses),
                  "preparation_seconds": time.perf_counter() - started},
        "limits": ["Existing TRANSFER_DEV already exposed in historical CODEVALUE003; not fresh blind TEST",
                   "Target-conditioned NAME strata are diagnostic; uniform group reports unconditioned byte outcomes",
                   "Full-source chronology and scorer-side future bytes used only for frozen cohort construction",
                   "License SHA256 was frozen; license byte-size/Git-blob values are newly measured",
                   "No model prediction, optimizer update, GPU use or theory validation"]}
    _write_json(output_manifest, manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path)
    arguments = parser.parse_args()
    result = prepare(arguments.data_dir, arguments.manifest_path)
    print(json.dumps({"status": result["status"], "selection_proof": result["selection_proof"],
                      "artifacts": result["artifacts"], "model_executions": 0}, sort_keys=True))
