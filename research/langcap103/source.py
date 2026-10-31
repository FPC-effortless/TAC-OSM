"""Publisher-pinned, raw-byte CPython corpus audit. Never executes source.

Source checks are reproducibility evidence, not an independent signed audit.
Public DEV is exposed; this program does not create or open protected TEST.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def git_blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def source_group(name):
    parts = PurePosixPath(name).parts
    if len(parts) < 2 or parts[0] != "Lib" or ".." in parts:
        raise ValueError("Expected original Lib-relative publisher path")
    return parts[1].removesuffix(".py") if len(parts) == 2 else parts[1]


def split_for(group, plan):
    s = plan["split"]
    value = int(digest((s["salt"] + ":" + group).encode()), 16)
    return "DEV" if value % s["dev_modulus"] == s["dev_residue"] else "FIT"


def candidate(name, data, plan):
    parts = PurePosixPath(name).parts
    p = plan["inclusion"]
    return (len(parts) >= 2 and parts[0] == p["root"] and name.endswith(p["suffix"])
            and not set(parts).intersection(p["excluded_components"])
            and p["minimum_file_bytes"] <= len(data) <= p["maximum_file_bytes"])


def select_records(records, plan):
    multiplicity = Counter(row["sha256"] for row in records)
    accepted, excluded = [], []
    for row in sorted(records, key=lambda r: r["path"]):
        if multiplicity[row["sha256"]] != 1:
            excluded.append({"path": row["path"], "reason": "duplicate_content_hash"})
            continue
        group = source_group(row["path"])
        accepted.append({**row, "group": group, "split": split_for(group, plan)})
    if len({row["path"] for row in accepted}) != len(accepted):
        raise ValueError("Duplicate publisher path")
    stats = {}
    for split in ("FIT", "DEV"):
        part = [r for r in accepted if r["split"] == split]
        stats[split] = {"files": len(part), "bytes": sum(r["bytes"] for r in part),
                        "groups": sorted({r["group"] for r in part}),
                        "manifest_sha256": digest(canonical(part))}
    if set(stats["FIT"]["groups"]) & set(stats["DEV"]["groups"]):
        raise ValueError("Group leakage")
    return accepted, excluded, stats


def audit(source_dir, out, plan):
    source_dir, out = Path(source_dir).resolve(), Path(out).resolve()
    if out == source_dir or out.is_relative_to(source_dir):
        raise ValueError("Audit output must be outside original publisher checkout")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source_dir), *args])
    if git("rev-parse", "HEAD").decode().strip() != plan["commit"]:
        raise ValueError("Publisher commit differs from pinned release")
    if git("rev-parse", "HEAD^{tree}").decode().strip() != plan["tree"]:
        raise ValueError("Publisher tree mismatch")
    if git("status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("Publisher checkout is dirty")
    records, content, licenses = [], {}, []
    for item in git("ls-tree", "-rz", "HEAD").split(b"\0"):
        if not item:
            continue
        metadata, encoded_name = item.split(b"\t", 1)
        mode, kind, blob = metadata.decode().split()
        name = encoded_name.decode("utf-8")
        parts = PurePosixPath(name).parts
        is_code = len(parts) >= 2 and parts[0] == "Lib" and name.endswith(".py")
        is_license = (PurePosixPath(name).name.upper().startswith(("LICENSE", "COPYING", "COPYRIGHT"))
                      or name in ("Misc/ACKS", "Doc/license.rst"))
        if not (is_code or is_license):
            continue
        path = source_dir / name
        if mode not in ("100644", "100755") or kind != "blob" or path.is_symlink():
            raise ValueError("Non-regular candidate source: " + name)
        if not path.resolve().is_relative_to(source_dir):
            raise ValueError("Source path escaped publisher checkout")
        data = path.read_bytes()
        if git_blob(data) != blob:
            raise ValueError("Publisher blob mismatch: " + name)
        row = {"path": name, "bytes": len(data), "git_blob": blob, "sha256": digest(data)}
        if is_license:
            licenses.append(row); content[name] = data
        if is_code and candidate(name, data, plan):
            records.append(row); content[name] = data
    license_row = next((r for r in licenses if r["path"] == plan["license"]["path"]), None)
    if license_row is None or license_row["git_blob"] != plan["license"]["git_blob"]:
        raise ValueError("Pinned publisher license absent or changed")
    rows, excluded, stats = select_records(records, plan)
    for split in ("FIT", "DEV"):
        if stats[split]["bytes"] < plan["split"]["minimum_" + split.lower() + "_bytes"]:
            raise ValueError("Insufficient unique source bytes: " + split)
        if len(stats[split]["groups"]) < plan["split"]["minimum_groups_each"]:
            raise ValueError("Insufficient original module/package groups: " + split)
    identity = {"publisher_commit": plan["commit"], "source_plan_sha256": digest(canonical(plan)),
                "files": rows, "licenses": sorted(licenses, key=lambda r: r["path"])}
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "corpus.tar"
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as tar:
        for name in sorted({r["path"] for r in rows + licenses}):
            data = content[name]
            entry = tarfile.TarInfo(name); entry.size = len(data); entry.mtime = 0; entry.mode = 0o644
            tar.addfile(entry, io.BytesIO(data))
    receipt = {"schema": "langcap103.publisher-corpus.v1",
        "status": "PUBLISHER_BYTES_VERIFIED_NOT_SIGNED_RESEARCH_QUALIFICATION",
        "identity": identity, "source_sha256": digest(canonical(identity)),
        "archive_sha256": digest(archive.read_bytes()), "stats": stats,
        "exclusions": excluded, "source_files_executed": False,
        "optimizer_updates": 0, "dev_scoring": False, "protected_test_access": False,
        "limitations": ["same-project public exposed DEV; no independent TEST",
            "exact-content duplicate removal only; near-duplicate audit remains OPEN",
            "publisher and content integrity do not replace independent source/reward/ACL evidence",
            "path-specific license notices retained; final permission review remains separate"]}
    (out / "SOURCE_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def verify_package(receipt_path, archive_path):
    receipt = json.loads(Path(receipt_path).read_text())
    if receipt.get("schema") != "langcap103.publisher-corpus.v1":
        raise ValueError("Unknown corpus receipt")
    identity = receipt["identity"]
    if digest(canonical(identity)) != receipt["source_sha256"]:
        raise ValueError("Source identity checksum mismatch")
    archive_bytes = Path(archive_path).read_bytes()
    if digest(archive_bytes) != receipt["archive_sha256"]:
        raise ValueError("Corpus archive checksum mismatch")
    expected = {r["path"]: r for r in identity["files"] + identity["licenses"]}
    data = {}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as tar:
        for member in tar.getmembers():
            if (not member.isfile() or member.name not in expected or member.name in data
                    or ".." in PurePosixPath(member.name).parts or member.name.startswith("/")):
                raise ValueError("Unexpected or unsafe corpus archive member")
            row = expected[member.name]
            value = tar.extractfile(member).read()
            if (len(value) != row["bytes"] or digest(value) != row["sha256"]
                    or git_blob(value) != row["git_blob"]):
                raise ValueError("Corpus member mismatch: " + member.name)
            data[member.name] = value
    if data.keys() != expected.keys():
        raise ValueError("Corpus package missing files")
    fit = [r for r in identity["files"] if r["split"] == "FIT"]
    dev = [r for r in identity["files"] if r["split"] == "DEV"]
    if {r["group"] for r in fit} & {r["group"] for r in dev}:
        raise ValueError("Cross-split module/package leakage")
    if {r["sha256"] for r in fit} & {r["sha256"] for r in dev}:
        raise ValueError("Cross-split exact content leakage")
    return receipt, data


def next_byte_batch(receipt, data, *, length=256, batch=2):
    """Deterministic FIT-only resource probe; never crosses original file edges."""
    rows = [r for r in receipt["identity"]["files"]
            if r["split"] == "FIT" and r["bytes"] >= length + 1]
    if length < 1 or batch < 1 or len(rows) < batch:
        raise ValueError("Insufficient FIT-only complete sequences")
    chosen = sorted(rows, key=lambda r: r["path"])[:batch]
    inputs, targets, identities = [], [], []
    for r in chosen:
        b = data[r["path"]][:length+1]
        inputs.append(list(b[:-1])); targets.append(list(b[1:]))
        identities.append({"path": r["path"], "offset": 0, "source_sha256": r["sha256"]})
    return inputs, targets, identities


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    plan = json.loads((HERE / "SOURCE_PLAN.json").read_text())
    receipt = audit(args.source_dir, args.out, plan)
    verify_package(args.out / "SOURCE_RECEIPT.json", args.out / "corpus.tar")
    print(json.dumps({"status": receipt["status"], "source_sha256": receipt["source_sha256"],
        "archive_sha256": receipt["archive_sha256"],
        "stats": {k: {**v, "groups": len(v["groups"])} for k, v in receipt["stats"].items()},
        "exact_duplicate_paths_excluded": len(receipt["exclusions"]), "optimizer_updates": 0}))
