"""Authenticate and cache the fixed nine final CODEVALUE003 checkpoints.

No model is constructed, scored, trained or unpickled by this downloader.
Historical exposure is matched; training FLOPs were NOT matched. Cryptographic
receipt consistency is internal engineering evidence, not independent qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from urllib.request import urlopen

SOURCE_COMMIT = "87d8a3b288242b2db8c4e052f1571ff6005d0e36"
REPOSITORY = "FPC-effortless/TAC-OSM"
ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = "research/commercial-value/codevalue003"
REPLAY_DIR = "research/replay-bundles/CODEVALUE003"
ARMS = ("tac", "gru", "ssm")
SEEDS = (1701, 1702, 1703)
STEPS = 10240
TARGETS = 10485760
PARAMETERS = {"tac": 499623, "gru": 500165, "ssm": 499782}
COHORT_SHA256 = "f3d485d9097ed63dee8bec98704f982b6426ffbba099278d411aca8a350fa3bf"
MAX_BYTES = 256 * 1024 * 1024
TRAINING_FLOPS = "UNMATCHED_INCOMPLETE_OPERATOR_COVERAGE"
CHECKPOINT_SHA256 = {
    ("tac", 1701): "2e24990a1cbce06b7ca9e7e764b3f6d9106a9ac7706006f502bfdaa8614949a6",
    ("tac", 1702): "6f96f08cba3ace57b37c30d46dfdde43a28464c64c439e745ac49b33c6df41ab",
    ("tac", 1703): "2ca96cb4e3fd12b361d3e43176360e0fb825fd3f853a8b6230626a3a27375b5d",
    ("gru", 1701): "d60e8832da7f339da3f74a6180bd9df6d9306feaf622167904346d1b68e664ec",
    ("gru", 1702): "6ba3b0e55e5cf3d11cfe2e014ba13a2a857e858b6d62bbede8257e461a7f62eb",
    ("gru", 1703): "e3b6f63a76a0d04a9df8eb3517904bc6522eb5a0618d9468214655e41e8236f5",
    ("ssm", 1701): "99b0f8c12a4c75e12d0e1bfd65cd6dc90dcd02c13a4686b5555cbf9aeab91e53",
    ("ssm", 1702): "d97da2a372c6a9e0bd450e349f60cdc9d51b88e83a6fe4360e1612f63390a004",
    ("ssm", 1703): "2006e0185a57aa862784199b6f892141a97af3a919a156fce279e2b8e3a0fe9c",
}
METADATA_SHA256 = {
    f"{SOURCE_DIR}/FINAL_MANIFEST.json": "a47ae8fd1ddb8843c032addbe7f62953a12493e860089a16b80b366f4ab324e3",
    f"{SOURCE_DIR}/SEAL.json": "083326a8399213645350cfcacab21e2d9d30529d8fcbb3fa6b04e93f8310fb43",
    f"{SOURCE_DIR}/SOURCE_RECEIPT.json": "e809d0c5ab42814a7e886732d8d7502106ae80387785ac20a1c30c72be1e15ab",
    f"{SOURCE_DIR}/REGISTER.json": "15a42998a68d61de4e82cca9381eb1e778d66697325a5a6d3ce77997ee02f671",
}
FACTORY_SHA256 = {
    "research/commercial-value/codevalue002/REGISTER.json": "865f692347185cd2a2870438a679444571d88560ea8c916ad73e06848ac847f7",
    "research/commercial-value/codevalue002/train.py": "5aa36c5c0c91d6eb70940f8c052126c3a12dad8b77d232ab1a9b093c5452f560",
    f"{SOURCE_DIR}/core.py": "5072d1ba786aba19cf8f4563b908a791f2be1f641a13d4d02394871e6883bc60",
    f"{SOURCE_DIR}/worker.py": "6b75dff3785f3c28e1bf36b976a2b4c8fd8f050743d6013230bb5b14780345db",
    "research/mathematical-engineering/langscale070/scaled_recurrence.py": "fb51763f1d3f85f2a44bd7bd768b7e34ac1d9ce8f06db90a40ca1abaa76b93a0",
    "research/mathematical-engineering/context026/context_transition.py": "c88f3f7c1b628cb733a4a58dcd168a75411d8bb3d80e382068593713e887510a",
    "research/mathematical-engineering/langlayer038/recurrence_only.py": "e1b52fca3e06b1440d1080924b847aa9649d04cf70ae033287cde37a6dad56f3",
    "research/mathematical-engineering/langhead094/model094.py": "ba7693db65c7a5690954c3e1365134e0ad81f160ea6e00b46844385bc8717acd",
    "research/mathematical-engineering/langhead093/model093.py": "004aabb7ac8e6926d5b038dfdc79d3e4aa1878a2af7b22765c217f6212e81cf4",
    "research/mathematical-engineering/langmem086/compact_tac.py": "7179908fa3672879f3c2046ec4521f0b9f783d5c696aaba82c4c2cd0b6ebc9b1",
}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _digest(value, label):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError(f"invalid SHA256: {label}")
    return value


def _url(relative):
    if not isinstance(relative, str) or relative.startswith("/") or ".." in relative.split("/"):
        raise ValueError("unsafe source-relative path")
    return f"https://raw.githubusercontent.com/{REPOSITORY}/{SOURCE_COMMIT}/{relative}"


def _source_bytes(relative, expected):
    local = ROOT / relative
    if local.exists():
        raw = local.read_bytes()
    else:
        with urlopen(_url(relative), timeout=45) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError("oversized source metadata")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"pinned source bytes changed: {relative}")
    return raw


def validate_original_records(manifest, seal, source_receipt):
    """Return the fixed arm/seed rows after cross-receipt consistency checks."""
    if (manifest.get("study_id") != "CODEVALUE003" or seal.get("study") != "CODEVALUE003"
            or source_receipt.get("study") != "CODEVALUE003"):
        raise ValueError("wrong original study")
    if (seal.get("status") != "SEALED_BEFORE_HELDOUT_SCORING"
            or seal.get("all45_checkpoints_complete") is not True
            or seal.get("independently_qualified") is not False):
        raise ValueError("original seal incomplete or qualification changed")
    cohorts = (manifest.get("real_world_source", {}).get("sha256"),
               manifest.get("split_policy", {}).get("cohort_sha256"),
               seal.get("source_cohort_sha256"), source_receipt.get("cohort_sha256"))
    if any(value != COHORT_SHA256 for value in cohorts):
        raise ValueError("source cohort mismatch")
    budget = manifest.get("budget", {})
    if (budget.get("training_updates"), budget.get("targets_per_model"),
            budget.get("batch_size"), budget.get("context_bytes")) != (STEPS, TARGETS, 8, 128):
        raise ValueError("original exposure budget mismatch")
    if budget.get("training_FLOPs") != TRAINING_FLOPS:
        raise ValueError("original unmatched training FLOPs status changed")
    original = manifest.get("models", [])
    expected = {(a, s) for a in ARMS for s in SEEDS}
    if len(original) != 9 or {(r.get("name"), r.get("seed")) for r in original} != expected:
        raise ValueError("complete fixed nine-model matrix required")
    rows = []
    for row in original:
        arm, seed = row["name"], row["seed"]
        measured = row.get("measured", {})
        count = PARAMETERS[arm]
        if type(seed) is not int or any(row.get(k) != count for k in
                                       ("trainable_parameters", "deployed_parameters")):
            raise ValueError("original parameter count mismatch")
        expected_fields = {"study": "CODEVALUE003", "arm": arm, "seed": seed,
                           "steps": STEPS, "targets": TARGETS, "trainable_parameters": count,
                           "deployed_parameters": count, "source_cohort_sha256": COHORT_SHA256,
                           "protected_TEST": "NOT_OPENED", "training_flops": TRAINING_FLOPS}
        if any(measured.get(k) != v for k, v in expected_fields.items()):
            raise ValueError("original measured metadata mismatch")
        replay = f"{arm}-{seed}/MODEL-{STEPS:06d}.pt"
        checkpoint_sha = CHECKPOINT_SHA256[(arm, seed)]
        if (measured.get("checkpoint_sha256") != checkpoint_sha
                or seal.get("checkpoints", {}).get(replay) != checkpoint_sha):
            raise ValueError("checkpoint seal and final receipt disagree")
        if not math.isfinite(measured.get("cold_replay_max_abs_difference", math.inf)) or not (
                0 <= measured.get("cold_replay_max_abs_difference", math.inf) <= 1e-6):
            raise ValueError("original cold replay failed")
        rows.append({"arm": arm, "seed": seed, "checkpoint_sha256": checkpoint_sha,
                     "original_checkpoint_relative_path": f"{REPLAY_DIR}/{replay}",
                     "trainable_parameters": count, "deployed_parameters": count,
                     "steps": STEPS, "targets": TARGETS, "source_cohort_sha256": COHORT_SHA256,
                     "sample_sha256": _digest(measured.get("sample_sha256"), "training schedule")})
    counts = [r["trainable_parameters"] for r in rows]
    if max(counts) / min(counts) - 1 > .01:
        raise ValueError("original pairwise parameter gap exceeds 1%")
    for seed in SEEDS:
        if len({r["sample_sha256"] for r in rows if r["seed"] == seed}) != 1:
            raise ValueError("arm training schedules do not match within seed")
    return sorted(rows, key=lambda r: (ARMS.index(r["arm"]), r["seed"]))


def verify_checkpoint_metadata(payload, row):
    """Check a safely loaded payload, without loading or executing it here.

    The caller must first authenticate the file hash and use torch.load with
    weights_only=True. Actual instantiated parameter counts are a separate check.
    """
    fields = {"study": "CODEVALUE003", "arm": row["arm"], "seed": row["seed"],
              "steps": STEPS, "next_update": STEPS, "params": row["trainable_parameters"],
              "sample_sha256": row["sample_sha256"], "source_cohort_sha256": COHORT_SHA256}
    if not isinstance(payload, dict) or any(payload.get(k) != v for k, v in fields.items()):
        raise ValueError("loaded original checkpoint metadata mismatch")
    required = ("model", "optimizer", "torch_rng", "cuda_rng", "python_rng",
                "numpy_rng", "sampler_state", "loss_history", "register_sha256")
    if any(k not in payload for k in required):
        raise ValueError("original optimizer/RNG/sampler checkpoint incomplete")
    if not isinstance(payload["model"], dict) or not payload["model"]:
        raise ValueError("empty original learned state")
    if not isinstance(payload["optimizer"], dict) or not payload["optimizer"].get("state"):
        raise ValueError("empty original optimizer state")
    if not isinstance(payload["sampler_state"], dict) or not payload["sampler_state"]:
        raise ValueError("empty original sampler state")
    if len(payload["loss_history"]) != STEPS or not all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in payload["loss_history"]):
        raise ValueError("original training history incomplete or nonfinite")
    _digest(payload["register_sha256"], "embedded training register")
    return {"status": "PASS_SEALED_ORIGINAL_METADATA", "optimizer_updates_performed": 0,
            "training_register_matches_current_final_register": "NOT_ASSUMED"}


def ensure_verified_download(relative, destination, expected, *, byte_limit=MAX_BYTES):
    """Download once, publish atomically without replacement, verify every reuse."""
    _digest(expected, "checkpoint")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if (destination.is_symlink() or not destination.is_file()
                or destination.stat().st_size > byte_limit or sha256_file(destination) != expected):
            raise ValueError(f"existing checkpoint changed; refusing overwrite: {destination}")
        return destination.stat().st_size
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=destination.parent)
    temporary = Path(temporary)
    size = 0
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as target, urlopen(_url(relative), timeout=45) as response:
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > byte_limit:
                raise ValueError("checkpoint content length exceeds byte limit")
            while block := response.read(min(1024 * 1024, byte_limit - size + 1)):
                size += len(block)
                if size > byte_limit:
                    raise ValueError("checkpoint exceeds byte limit")
                digest.update(block)
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
        if digest.hexdigest() != expected:
            raise ValueError("downloaded checkpoint hash mismatch")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or sha256_file(destination) != expected:
                raise ValueError("concurrent checkpoint publication differs") from None
        return size
    finally:
        temporary.unlink(missing_ok=True)


def _immutable_json(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != raw:
            raise ValueError("existing CHECKPOINTS.json differs; refusing overwrite")
        return
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-manifest-", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != raw:
                raise ValueError("concurrent CHECKPOINTS.json publication differs") from None
    finally:
        temporary.unlink(missing_ok=True)


def prepare_checkpoints(cache_dir):
    """Prepare fixed final10240 trained snapshots; no scoring or optimizer use."""
    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    records = {Path(p).name: json.loads(_source_bytes(p, digest))
               for p, digest in METADATA_SHA256.items()}
    rows = validate_original_records(records["FINAL_MANIFEST.json"], records["SEAL.json"],
                                     records["SOURCE_RECEIPT.json"])
    for path, expected in FACTORY_SHA256.items():
        _source_bytes(path, expected)
    total = 0
    for row in rows:
        relative = row["original_checkpoint_relative_path"]
        destination = cache / row["arm"] / str(row["seed"]) / "MODEL-010240.pt"
        row["bytes"] = ensure_verified_download(relative, destination, row["checkpoint_sha256"],
                                                 byte_limit=MAX_BYTES - total)
        total += row["bytes"]
        row["path"] = str(destination)
        row["trained_snapshot"] = True
    report = {"study": "LANGPERSIST096", "status": "PASS_FIXED_ORIGINAL_CHECKPOINT_BYTES",
              "qualification": "INTERNAL_NOT_INDEPENDENTLY_QUALIFIED", "source_commit": SOURCE_COMMIT,
              "source_repository": REPOSITORY, "source_metadata_sha256": METADATA_SHA256,
              "factory_source_sha256": FACTORY_SHA256, "models": rows,
              "verified_checkpoint_count": len(rows), "total_checkpoint_bytes": total,
              "maximum_total_checkpoint_bytes": MAX_BYTES, "original_training_flops": TRAINING_FLOPS,
              "comparison_scope": "EXPOSURE_MATCHED_NOT_TRAINING_FLOP_MATCHED",
              "optimizer_updates_performed": 0, "checkpoint_scoring_performed": False,
              "protected_TEST": "NOT_OPENED", "full_composed_source_math": "OPEN",
              "metadata_payload_validation": "REQUIRED_AFTER_SAFE_LOAD_BY_SCORING_WORKER"}
    _immutable_json(cache / "CHECKPOINTS.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    args = parser.parse_args()
    result = prepare_checkpoints(args.cache_dir)
    print(json.dumps({k: result[k] for k in ("status", "verified_checkpoint_count",
                                           "total_checkpoint_bytes", "qualification")}, indent=2))
