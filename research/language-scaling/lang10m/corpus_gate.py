"""LANG10M002: local-only admission gate for future large authentic language FIT.

No downloading, model updates, evaluation, DEV checkpoint selection, protected
TEST access or automatic tokenizer refitting. Publisher/license/hash receipts
alone do not prove complete legal clearance or semantic deduplication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

PARTITIONS = ("FIT", "SELECT_DEV", "TRANSFER_DEV")
KINDS = ("code", "natural_language", "technical_docs")
DIGEST = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
BLOB = re.compile(r"^[0-9a-f]{40}$")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def local_path(manifest: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or "://" in value:
        raise ValueError("locally materialized file paths only; no network")
    p = Path(value)
    return (p if p.is_absolute() else manifest.parent / p).resolve(strict=True)


def validate_manifest(manifest_path: Path, tokenizer_path: Path | None = None) -> dict:
    manifest_path = Path(manifest_path).resolve(strict=True)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("study") != "LANG10M002" or data.get("schema") != "original-source-v1":
        raise ValueError("wrong study/schema")
    rows = data.get("sources")
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty source manifest")
    if "TEST" in data.get("partitions", []):
        raise ValueError("protected TEST cannot enter candidate corpus")
    receipts = []
    owners_by_partition = {p: set() for p in PARTITIONS}
    by_digest = {}
    by_path = set()
    fit_kinds = set()
    group_by_partition = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("source entries must be objects")
        part = row.get("partition")
        kind = row.get("kind")
        group = row.get("publisher_group")
        if part not in PARTITIONS or kind not in KINDS:
            raise ValueError("invalid partition or document kind")
        if not isinstance(group, str) or not group:
            raise ValueError("nonempty publisher_group required")
        prior = group_by_partition.setdefault(group, part)
        if prior != part:
            raise ValueError("publisher group leaks across FIT and heldout splits")
        for key, pattern in (
            ("source_commit", COMMIT), ("sha256", DIGEST),
            ("git_blob_sha1", BLOB), ("license_sha256", DIGEST)
        ):
            if not isinstance(row.get(key), str) or not pattern.fullmatch(row[key]):
                raise ValueError(f"invalid {key}")
        for key in ("repo", "original_path", "license_id"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"missing {key}")
        original = local_path(manifest_path, row.get("local_path"))
        license_file = local_path(manifest_path, row.get("license_path"))
        if original == license_file:
            raise ValueError("source file cannot be its own license evidence")
        if original in by_path:
            raise ValueError("same local file appears twice")
        by_path.add(original)
        if license_file.is_dir() or original.is_dir():
            raise ValueError("source and license must be files")
        license_bytes = license_file.read_bytes()
        if sha256(license_bytes) != row["license_sha256"] or not license_bytes:
            raise ValueError("license notice mismatch or empty")
        source_bytes = original.read_bytes()
        if not source_bytes or len(source_bytes) > 100_000_000:
            raise ValueError("empty or oversized source; use separately reviewed sharding")
        if sha256(source_bytes) != row["sha256"] or git_blob(source_bytes) != row["git_blob_sha1"]:
            raise ValueError("publisher source digest mismatch")
        if row["sha256"] in by_digest:
            raise ValueError("exact duplicate source detected, including across partitions")
        by_digest[row["sha256"]] = part
        try:
            text = source_bytes.decode("utf-8", errors="strict")
        except UnicodeError as e:
            raise ValueError("invalid UTF-8 source") from e
        owners_by_partition[part].add(group)
        if part == "FIT":
            fit_kinds.add(kind)
        receipts.append({
            "partition": part, "kind": kind, "publisher_group": group,
            "repo": row["repo"], "source_commit": row["source_commit"],
            "original_path": row["original_path"],
            "sha256": row["sha256"], "git_blob_sha1": row["git_blob_sha1"],
            "bytes": len(source_bytes), "utf8_characters": len(text),
            "license_id": row["license_id"],
            "license_sha256": row["license_sha256"],
        })
    if not {"code", "natural_language"}.issubset(fit_kinds):
        raise ValueError("FIT requires code and natural-language publisher sources")
    if not owners_by_partition["SELECT_DEV"] or not owners_by_partition["TRANSFER_DEV"]:
        raise ValueError("both DEV selection and transfer cohorts must be publisher held out")

    token_total = None
    tokenizer_digest = None
    if tokenizer_path is not None:
        import sentencepiece as spm
        tokenizer_path = Path(tokenizer_path).resolve(strict=True)
        tokenizer_digest = sha256(tokenizer_path.read_bytes())
        sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        if sp.vocab_size() != 4096:
            raise ValueError("unexpected tokenizer vocabulary")
        token_total = 0
        for row in rows:
            if row["partition"] == "FIT":
                original = local_path(manifest_path, row["local_path"])
                string = original.read_text(encoding="utf-8")
                ids = sp.encode(string, out_type=int)
                if sp.decode(ids) != string:
                    raise ValueError("tokenizer fails lossless source roundtrip")
                token_total += len(ids)
    fit_rows = [r for r in receipts if r["partition"] == "FIT"]
    volume_met = token_total is not None and token_total >= 50_000_000
    groups_met = len(owners_by_partition["FIT"]) >= 10
    return {
        "study": "LANG10M002", "mode": "NO_FIT_DATA_ADMISSION_ONLY",
        "source_manifest_sha256": sha256(manifest_path.read_bytes()),
        "source_count": len(receipts),
        "FIT_sources": len(fit_rows),
        "FIT_bytes": sum(r["bytes"] for r in fit_rows),
        "FIT_by_kind": {kind: sum(r["bytes"] for r in fit_rows if r["kind"] == kind) for kind in KINDS},
        "publisher_groups": {key: len(owners) for key, owners in owners_by_partition.items()},
        "FIT_token_instances": token_total,
        "FIT_tokenizer_sha256": tokenizer_digest,
        "FIT_token_instances_minimum_50M_met": volume_met,
        "FIT_minimum_10_publishers_met": groups_met,
        "200M_token_reference_budget_met": token_total is not None and token_total >= 200_000_000,
        "near_duplicate_and_semantic_contamination": "INCONCLUSIVE_NEEDS_DEDUP_AND_SPLIT_AUDIT",
        "license_authorization": "ARCHIVED_LICENSE_NOTICES_ONLY_HUMAN_REVIEW_REQUIRED",
        "claims": {
            "protected_TEST_accessed": False, "model_trained": False,
            "data_gate_complete": False,
        },
        "sources": receipts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--tokenizer-model", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError("do not overwrite data admission results")
    report = validate_manifest(args.manifest, args.tokenizer_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "sources"}, sort_keys=True))


if __name__ == "__main__":
    main()
