"""Only original publisher FIT bytes for LANGPRETRAIN076.

Unlike older original_source(), this loader never opens DEV or protected TEST
document bytes. It verifies frozen publisher metadata and 14 original FIT
files. No network access, model calls or dataset fabrication.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PUBLISHER_COMMIT = "60403a5409ff2c3f3b07dd2ca91a7a3e096839c7"
SOURCE_SHA = "dd5aabf90646be409e3f18ecf763dbcefc84b6575e12ff54f4a7195f92ff150e"
BUNDLE = Path("research/source-bundles/LANGUAGE-SOURCE023")
RECEIPT = Path("research/mathematical-engineering/language_source023/SOURCE_RECEIPT.json")
SUPPLEMENT = Path("research/mathematical-engineering/language_source023/UTF8_SUPPLEMENT_RECEIPT.json")
WINDOW = 129

def raw_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _read_checked(root: Path, path: str, record: dict) -> bytes:
    if (not isinstance(path, str) or
        not re.fullmatch(r"[A-Za-z0-9_./-]+", path) or
        path.startswith("/") or
        any(part in ("", ".", "..") for part in path.split("/"))):
        raise ValueError("invalid publisher source file path")
    absolute = root / BUNDLE / path
    if absolute.is_symlink() or not absolute.is_file() or not absolute.resolve().is_relative_to(root):
        raise ValueError("missing/escaped original FIT source")
    raw = absolute.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if (raw_sha(raw) != record["sha256"] or len(raw) != record["bytes"] or
        blob != record["git_blob"] or len(raw) < WINDOW):
        raise ValueError("original FIT source SHA/blob/byte count mismatch")
    return raw

def fit_sources(root: Path = ROOT) -> dict[str, bytes]:
    root = root.resolve(strict=True)
    source = json.loads((root / RECEIPT).read_text())
    supplement = json.loads((root / SUPPLEMENT).read_text())
    if (source.get("commit") != PUBLISHER_COMMIT or source.get("release") != "v3.13.0"
        or source.get("source_sha256") != SOURCE_SHA
        or source.get("license") != "PSF License Version 2; documentation code examples additionally 0BSD"
        or len(source.get("files", [])) != 17):
        raise ValueError("publisher receipt/known source mismatch")
    records = source["files"]
    names = [rec["path"] for rec in records]
    if sorted(names) != names or len(names) != len(set(names)):
        raise ValueError("source ordering/uniqueness changed")
    out = {}
    dev_names = []
    for i, rec in enumerate(records):
        expected = "DEV" if i % 4 == 3 else "FIT"
        if rec.get("role") != expected:
            raise ValueError("original frozen source split mismatch")
        if expected == "FIT":
            out[rec["path"]] = _read_checked(root, rec["path"], rec)
        else:
            dev_names.append(rec["path"])  # metadata only: NO DEV BYTES
    if len(out) != 13 or len(dev_names) != 4:
        raise ValueError("original FIT/DEV owner counts changed")
    if (supplement.get("path") != "Doc/howto/unicode.rst"
        or supplement.get("role") != "FIT_SUPPLEMENT"
        or supplement.get("publisher_commit") != PUBLISHER_COMMIT
        or supplement["path"] in out or supplement["path"] in dev_names):
        raise ValueError("publisher UTF8 FIT supplement has changed")
    out[supplement["path"]] = _read_checked(root, supplement["path"], supplement)
    if len(out) != 14:
        raise ValueError("expected 14 publisher FIT owners")
    return out
