#!/usr/bin/env python3
"""Download the official SCANIA v3 ZIP; inspect ONLY allow-listed TRAIN/VALIDATION CSVs.

The ZIP contains unsealed bytes including publisher TEST labels; do not
extract/read/parse these entries. We rely on name-only central directory
inspection to find allowlisted small files. This is acquisition/audit ONLY.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ARCHIVE_URL = "https://api.researchdata.se/dataset/2024-34/3/file/zip"
ARCHIVE_LIMIT = 2_600_000_000
ALLOW_FILES = frozenset({
    "train_tte.csv",
    "train_specifications.csv",
    "validation_labels.csv",
})
LIMITS = {
    "train_tte.csv": 3_000_000,
    "train_specifications.csv": 4_000_000,
    "validation_labels.csv": 1_000_000,
}


def extract_approved(archive_path, output):
    output = Path(output)
    retrieved = {}
    with zipfile.ZipFile(archive_path) as archive:
        by_name = {}
        # Listing ZIP names is not opening publisher TEST *content*.
        for entry in archive.infolist():
            leaf = Path(entry.filename).name
            if leaf not in ALLOW_FILES:
                continue
            if leaf in by_name:
                raise ValueError(f"Multiple archive entries for {leaf}")
            if entry.is_dir() or entry.file_size <= 0 or entry.file_size > LIMITS[leaf]:
                raise ValueError(f"Unexpected size/type for {leaf}")
            by_name[leaf] = entry
        if set(by_name) != set(ALLOW_FILES):
            raise ValueError("Archive is missing registered source files: " +
                             ", ".join(sorted(ALLOW_FILES - set(by_name))))
        for name in sorted(ALLOW_FILES):
            info = by_name[name]
            # Read only registered file bodies, never unselected archive entries.
            with archive.open(info) as stream:
                raw = stream.read(LIMITS[name] + 1)
            if len(raw) > LIMITS[name] or len(raw) != info.file_size:
                raise ValueError(f"Short/oversized source file {name}")
            if not raw or raw.lstrip().lower().startswith((b"<html", b"<!doctype", b"{")):
                raise ValueError("Invalid CSV source body " + name)
            raw.decode("utf-8-sig")
            output.mkdir(parents=True, exist_ok=True)
            (output / name).write_bytes(raw)
            retrieved[name] = {
                "filename": name, "archive_member_name": info.filename,
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                "source_archive": ARCHIVE_URL,
            }
    return retrieved


def acquire(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "original-v3.zip"
    digest = hashlib.sha256()
    size = 0
    req = urllib.request.Request(
        ARCHIVE_URL, headers={"User-Agent": "TAC-OSM-SEQ009-research/1.0",
                              "Accept": "application/zip"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as stream, archive_path.open("wb") as dst:
            if getattr(stream, "status", None) != 200:
                raise ValueError("Publisher returned non-success status")
            while True:
                chunk = stream.read(2 * 1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > ARCHIVE_LIMIT:
                    raise ValueError("Unexpected archive size above limit")
                digest.update(chunk)
                dst.write(chunk)
        files = extract_approved(archive_path, destination / "raw")
        return {
            "publisher_zip_url": ARCHIVE_URL,
            "zip_bytes": size, "zip_sha256": digest.hexdigest(),
            "opened_nonapproved_member_bodies": False,
            "extracted_files": files,
        }
    finally:
        archive_path.unlink(missing_ok=True)
