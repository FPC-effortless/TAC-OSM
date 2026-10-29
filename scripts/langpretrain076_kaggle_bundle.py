#!/usr/bin/env python3
"""Build ONE checksum-verified offline Kaggle script with original FIT sources.

No upload, networking, GPU use, model fit or environment access at build time.
Kaggle CLI uploads code_file, so all reviewed supporting source is embedded.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
STUDY = "LANGPRETRAIN076"
SOURCE = "research/mathematical-engineering/language_source023/SOURCE_RECEIPT.json"
SUPPLEMENT = "research/mathematical-engineering/language_source023/UTF8_SUPPLEMENT_RECEIPT.json"
FIT_PREFIX = "research/source-bundles/LANGUAGE-SOURCE023/"
FIXED = {
    "scripts/langpretrain076_t4_measure.py": "workload.py",
    "scripts/langpretrain076_fit_only.py": "scripts/langpretrain076_fit_only.py",
    "scripts/langlayer035_sampling.py": "scripts/langlayer035_sampling.py",
    "scripts/kaggle_dual_t4_entry.py": "launcher.py",
    "research/mathematical-engineering/context026/context_transition.py":
        "research/mathematical-engineering/context026/context_transition.py",
    "research/mathematical-engineering/langlayer038/recurrence_only.py":
        "research/mathematical-engineering/langlayer038/recurrence_only.py",
    "research/mathematical-engineering/langscale074/upper_reallocation.py":
        "research/mathematical-engineering/langscale074/upper_reallocation.py",
    SOURCE: SOURCE,
    SUPPLEMENT: SUPPLEMENT,
}
MANIFEST = "research/mathematical-engineering/langpretrain076/FAIR_COMPARISON.json"
CODE_LIMIT = 2_500_000


class BundleError(ValueError):
    pass


def sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _checked(root: Path, name: str) -> bytes:
    path = PurePosixPath(name)
    if (path.is_absolute() or not name or "\\" in name
        or any(p in ("", ".", "..") for p in path.parts)
        or path.as_posix() != name):
        raise BundleError("unsafe bundle path")
    p = root / name
    if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root):
        raise BundleError("missing or symlinked source file: " + name)
    return p.read_bytes()


def source_members(root: Path) -> dict[str, bytes]:
    # Uses same owner/source/role checks as actual GPU worker.
    sys.path.insert(0, str(root))
    from scripts.langpretrain076_fit_only import fit_sources
    fit_sources(root)  # strict 14 actual publisher FIT SHA and blobs, zero DEV bytes
    receipt = json.loads(_checked(root, SOURCE))
    supplement = json.loads(_checked(root, SUPPLEMENT))
    if len(receipt["files"]) != 17:
        raise BundleError("original source receipt count changed")
    files = dict(FIXED)
    for record in receipt["files"]:
        if record["role"] == "FIT":
            path = FIT_PREFIX + record["path"]
            files[path] = path
    path = FIT_PREFIX + supplement["path"]
    files[path] = path
    if len(files) != len(set(files.values())):
        raise BundleError("payload destination collision")
    return {dest: _checked(root, src) for src, dest in sorted(files.items())}


def bootstrap_source(encoded: str, payload_hash: str, file_hashes: dict[str, str]) -> str:
    # Static loader verifies entire package AND each extracted member before
    # launching the already-audited dual-T4 runtime. No network or subprocess
    # execution at packaging time. All archive members have fixed paths.
    assembled = ('"""LANGPRETRAIN076 offline original-FIT dual-T4 Kaggle runner.\\n'
            'Generated and SHA-verified by scripts/langpretrain076_kaggle_bundle.py.\\n'
            '"""\\n'
            'import base64, hashlib, io, json, pathlib, runpy, zipfile, zlib\\n'
            'PAYLOAD = ' + repr(encoded) + '\\n'
            'PAYLOAD_SHA256 = ' + repr(payload_hash) + '\\n'
            'MEMBERS = ' + repr(file_hashes) + '\\n'
            'def main():\\n'
            '    compressed = base64.b85decode(PAYLOAD.encode("ascii"))\\n'
            '    archive = zlib.decompress(compressed)\\n'
            '    if hashlib.sha256(archive).hexdigest() != PAYLOAD_SHA256:\\n'
            '        raise RuntimeError("whole original source bundle SHA mismatch")\\n'
            '    root = pathlib.Path("/kaggle/working/langpretrain076-original")\\n'
            '    if root.exists():\\n'
            '        raise RuntimeError("refuse to overwrite existing original source run")\\n'
            '    root.mkdir(parents=True, exist_ok=False)\\n'
            '    with zipfile.ZipFile(io.BytesIO(archive)) as z:\\n'
            '        if set(z.namelist()) != set(MEMBERS):\\n'
            '            raise RuntimeError("source bundle member list changed")\\n'
            '        for name in sorted(MEMBERS):\\n'
            '            p = pathlib.PurePosixPath(name)\\n'
            '            if (p.is_absolute() or ".." in p.parts or "\\\\" in name):\\n'
            '                raise RuntimeError("unsafe original source bundle path")\\n'
            '            raw = z.read(name)\\n'
            '            if hashlib.sha256(raw).hexdigest() != MEMBERS[name]:\\n'
            '                raise RuntimeError("original source file SHA mismatch: "+name)\\n'
            '            target = root / name\\n'
            '            target.parent.mkdir(parents=True, exist_ok=True)\\n'
            '            target.write_bytes(raw)\\n'
            '    runpy.run_path(str(root / "launcher.py"), run_name="__main__")\\n'
            'if __name__ == "__main__": main()\\n')

    return assembled.replace(chr(92) + "n", chr(10))

def build(root: Path, destination: Path, username: str) -> dict:
    if (not isinstance(username, str) or not 2 <= len(username) <= 39
        or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in username)):
        raise BundleError("valid existing Kaggle username required")
    if destination.exists():
        raise BundleError("refuse overwriting an existing staged notebook")
    sys.path.insert(0, str(root / "src"))
    from scripts.validate_research_parity import check_manifest
    from scripts.langpretrain076_fit_only import fit_sources
    fit = fit_sources(root)
    if len(fit) != 14:
        raise BundleError("wrong original FIT owner count")
    contract = json.loads(_checked(root, MANIFEST))
    result = check_manifest(contract, "preflight")
    if (result.get("status") != "PASS_PREFIT_SINGLE_MODEL"
        or contract.get("study_id") != STUDY or
        contract.get("parameter_budget_contract") != "owner-2026-10-29-500k-t4-throughput-only"):
        raise BundleError("internal-research 500K source training registration failed")
    from tac_osm.phone_research_ops import decide
    proposal = json.loads(_checked(root, "research/phone_queue/langpretrain076.json"))
    decision = decide(proposal)
    if (decision["state"] != "READY_INTERNAL_RESEARCH_PLAN"
        or decision["backend"] != "kaggle_t4x2"
        or decision["gpu_devices_required"] != 2
        or decision["gpu_budget_minutes"] > 15
        or proposal["seeds"] != [1701, 1702]
        or proposal["execution_level"] != "internal_research"):
        raise BundleError("phone compute/source authorization or bounded two-T4 plan invalid")
    files = source_members(root)
    config = {
        "run_id": proposal["run_id"],
        "gpu_strategy": "seed_shard",
        "seeds": [1701, 1702],
        "accelerator": "NvidiaTeslaT4",
        "workload_sha256": sha(files["workload.py"]),
        "decision_digest": decision["source_contract_fingerprint"],
        "scientific_qualification": "INTERNAL_ONLY_NOT_INDEPENDENT",
    }
    files["config.json"] = (json.dumps(config, sort_keys=True, indent=2) + "\n").encode()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, blob in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, blob)
    raw = stream.getvalue()
    code = bootstrap_source(base64.b85encode(zlib.compress(raw, 9)).decode("ascii"),
                            sha(raw), {name: sha(data) for name, data in files.items()})
    if len(code.encode()) > CODE_LIMIT:
        raise BundleError("oversized single-code-file Kaggle payload")
    destination.mkdir(parents=True, exist_ok=False)
    slug = "tacosm-langpretrain076-h48-t4-frozen"
    handle = username + "/" + slug
    metadata = {
        "id": handle, "title": "TACOSM langpretrain076 h48 t4 frozen",
        "code_file": "entrypoint.py", "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_internet": "false", "enable_gpu": "true",
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    (destination / "entrypoint.py").write_text(code)
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    receipt = {
        "schema": "tac-osm.langpretrain076-stage.v1",
        "status": "STAGED_UNSUBMITTED",
        "code_file_sha256": sha(code.encode()),
        "embedded_archive_sha256": sha(raw),
        "embedded_sources_sha256": {name: sha(data) for name, data in files.items()},
        "source_owner_count": len(fit), "DEV_bytes_embedded": 0,
        "TEST_bytes_embedded": 0,
        "workload": STUDY, "model_parameters": 499623,
        "gpu_count_required": 2, "accelerator": "NvidiaTeslaT4",
        "kernel_id": handle, "free_gpu_quota_status": "UNVERIFIED",
        "approved_to_train": False, "approved_to_spend_paid": False,
    }
    (destination / "staging-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaggle-user", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = build(ROOT, args.out, args.kaggle_user)
    except (BundleError, ValueError, KeyError, OSError, ImportError) as exc:
        parser.exit(2, f"BLOCKED_NO_SUBMISSION: {type(exc).__name__}: {exc}\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
