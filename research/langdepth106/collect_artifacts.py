"""Collect exact original GitHub artifact ZIPs with bounded safe extraction.

The caller supplies publisher artifact metadata and already authorized ephemeral
HTTPS download URLs. This helper handles no API credentials or GitHub mutations.
It verifies original archive size/SHA-256 and member bytes, never loads a model or
scores held-out data. Invoke only after the root's final-evaluation access boundary
allows the selected artifacts. This is transport/integrity evidence, not proof of
the workflow's scientific validity or independent execution authenticity.

Manifest: {"artifacts": [{"id": integer, "name": safe_basename,
"size_in_bytes": integer, "digest": "sha256:<64hex>",
"workflow_run": {"id": 41232595986,
"head_sha": "71cbe65ccedac8b59fbc3f496f4078f261741f6a"},
"download_url": "https://..."}]}. download_url is optional for existing archives.
--verify-existing performs no network access. Receipts exclude all download URLs.
An existing receipt is immutable; a new selection requires a fresh output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zipfile


EXPECTED_RUN_ID = 41232595986
EXPECTED_HEAD_SHA = "71cbe65ccedac8b59fbc3f496f4078f261741f6a"
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_ARTIFACT_BYTES = 200 * 1024 * 1024
MAX_MEMBERS = 10_000
CHUNK_BYTES = 1024 * 1024
RECEIPT_NAME = "ARTIFACT_VERIFICATION.json"


class ArtifactError(ValueError):
    """Fail-closed collection error whose message never includes a signed URL."""


def _safe_name(name: Any) -> str:
    if (type(name) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name)
            or ".." in name or name.endswith(".")):
        raise ArtifactError("unsafe artifact name")
    stem = name.split(".", 1)[0].upper()
    if stem in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        raise ArtifactError("reserved artifact name")
    return name


def _https_url(value: Any) -> str:
    if type(value) is not str:
        raise ArtifactError("download URL must be HTTPS")
    try:
        parsed = urlsplit(value)
        valid = (parsed.scheme == "https" and bool(parsed.hostname)
                 and parsed.username is None and parsed.password is None
                 and not parsed.fragment and not any(ord(c) < 32 for c in value))
        _ = parsed.port
    except (ValueError, UnicodeError):
        valid = False
    if not valid:
        raise ArtifactError("download URL must be HTTPS without user credentials or fragments")
    return value


def _metadata(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ArtifactError("artifact metadata must be an object")
    artifact_id = raw.get("id")
    size = raw.get("size_in_bytes")
    if type(artifact_id) is not int or artifact_id <= 0 or type(size) is not int or size < 1:
        raise ArtifactError("invalid artifact ID or original archive size")
    name = _safe_name(raw.get("name"))
    digest = raw.get("digest")
    if type(digest) is not str or not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest):
        raise ArtifactError("original artifact SHA-256 digest is missing or malformed")
    run = raw.get("workflow_run")
    if (not isinstance(run, dict) or type(run.get("id")) is not int
            or run.get("id") != EXPECTED_RUN_ID or run.get("head_sha") != EXPECTED_HEAD_SHA):
        raise ArtifactError("artifact workflow run or source head differs from the frozen cohort")
    if "download_url" in raw:
        _https_url(raw["download_url"])
    return {
        "id": artifact_id, "name": name, "size_in_bytes": size,
        "digest": digest.lower(),
        "workflow_run": {"id": EXPECTED_RUN_ID, "head_sha": EXPECTED_HEAD_SHA},
    }


def _ensure_directory(path: Path) -> None:
    if path.is_symlink():
        raise ArtifactError("collection directory cannot be a symlink")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ArtifactError("collection destination is not a directory")


def _hash_file(path: Path) -> tuple[int, str]:
    if path.is_symlink() or not path.is_file():
        raise ArtifactError("expected a regular file, not a link or non-file")
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            total += len(chunk)
            digest.update(chunk)
    return total, digest.hexdigest()


def _verify_archive(path: Path, metadata: dict[str, Any]) -> None:
    size, digest = _hash_file(path)
    if size != metadata["size_in_bytes"] or "sha256:" + digest != metadata["digest"]:
        raise ArtifactError("original archive size or SHA-256 differs from publisher metadata")


class _HTTPSRedirectsOnly(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download(url: str, destination: Path, metadata: dict[str, Any]) -> None:
    """Stream to a temporary file; publish only a verified exact original ZIP."""
    _https_url(url)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".archive-", suffix=".partial", dir=destination.parent, delete=False) as temp:
            temp_path = Path(temp.name)
            opener = build_opener(_HTTPSRedirectsOnly())
            request = Request(url, headers={"User-Agent": "LANGDEPTH106-artifact-integrity/1"})
            with opener.open(request, timeout=30) as response:
                _https_url(response.geturl())
                total = 0
                while chunk := response.read(CHUNK_BYTES):
                    total += len(chunk)
                    if total > metadata["size_in_bytes"]:
                        raise ArtifactError("download exceeds original artifact size")
                    temp.write(chunk)
            temp.flush()
            os.fsync(temp.fileno())
        _verify_archive(temp_path, metadata)
        # Exclusive link preserves existing originals even if another process
        # races this collector. Staging and destination share a filesystem.
        try:
            os.link(temp_path, destination)
        except FileExistsError:
            _verify_archive(destination, metadata)
    except ArtifactError:
        raise
    except Exception:
        # HTTPError/URLError may include signed URLs; never expose their text.
        raise ArtifactError("artifact download or local publication failed") from None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _member_parts(info: zipfile.ZipInfo) -> tuple[str, ...]:
    name = info.orig_filename
    if (name != info.filename or not name or "\\" in name or ":" in name
            or any(ord(c) < 32 for c in name) or name.startswith("/")):
        raise ArtifactError("ZIP member has an unsafe path")
    body = name[:-1] if name.endswith("/") else name
    parts = tuple(body.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise ArtifactError("ZIP member has a traversal or ambiguous path")
    file_type = stat.S_IFMT(info.external_attr >> 16)
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR} or stat.S_ISLNK(info.external_attr >> 16):
        raise ArtifactError("ZIP member is a symlink or unsupported special file")
    if file_type == stat.S_IFDIR and not info.is_dir():
        raise ArtifactError("ZIP directory type and path disagree")
    if info.flag_bits & 1:
        raise ArtifactError("encrypted ZIP members are not accepted")
    return parts


def _inspect(archive: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, tuple[str, ...]]]:
    infos = archive.infolist()
    if len(infos) > MAX_MEMBERS:
        raise ArtifactError("ZIP member count exceeds the collector bound")
    members = []
    paths: dict[tuple[str, ...], bool] = {}
    total = 0
    for info in infos:
        parts = _member_parts(info)
        if parts in paths:
            raise ArtifactError("ZIP contains duplicate member paths")
        paths[parts] = info.is_dir()
        if info.file_size < 0 or info.file_size > MAX_FILE_BYTES:
            raise ArtifactError("ZIP member exceeds the 64 MiB uncompressed bound")
        if info.is_dir() and info.file_size:
            raise ArtifactError("ZIP directory unexpectedly contains payload bytes")
        total += info.file_size
        if total > MAX_ARTIFACT_BYTES:
            raise ArtifactError("ZIP exceeds the 200 MiB uncompressed artifact bound")
        members.append((info, parts))
    for parts in paths:
        if any(parent in paths and not paths[parent] for i in range(1, len(parts))
               for parent in (parts[:i],)):
            raise ArtifactError("ZIP file path is also used as a parent directory")
    return members


def _extract(archive_path: Path, destination: Path) -> dict[str, Any]:
    _ensure_directory(destination)
    members_evidence = []
    actual_total = 0
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = _inspect(archive)  # Reject every unsafe path before writing.
            expected_files = {parts for info, parts in members if not info.is_dir()}
            expected_dirs = {parts for info, parts in members if info.is_dir()}
            for _, parts in members:
                expected_dirs.update(parts[:i] for i in range(1, len(parts)))
            for existing in destination.rglob("*"):
                relative = existing.relative_to(destination).parts
                if existing.is_symlink():
                    raise ArtifactError("existing extraction tree contains a symlink")
                if ((existing.is_dir() and relative not in expected_dirs)
                        or (existing.is_file() and relative not in expected_files)
                        or (not existing.is_dir() and not existing.is_file())):
                    raise ArtifactError("existing extraction tree contains undeclared entries")
            for info, parts in members:
                target = destination.joinpath(*parts)
                cursor = destination
                for part in parts[:-1]:
                    cursor = cursor / part
                    _ensure_directory(cursor)
                if info.is_dir():
                    _ensure_directory(target)
                    continue
                if target.exists() and (target.is_symlink() or not target.is_file()):
                    raise ArtifactError("extraction target is not a regular file")
                if target.is_symlink():
                    raise ArtifactError("extraction target cannot be a dangling symlink")
                temp_path: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(prefix=".member-", suffix=".partial", dir=target.parent, delete=False) as temp:
                        temp_path = Path(temp.name)
                        digest = hashlib.sha256()
                        size = 0
                        with archive.open(info) as source:
                            while chunk := source.read(CHUNK_BYTES):
                                size += len(chunk)
                                actual_total += len(chunk)
                                if size > MAX_FILE_BYTES or actual_total > MAX_ARTIFACT_BYTES:
                                    raise ArtifactError("decompressed bytes exceed collector bounds")
                                digest.update(chunk)
                                temp.write(chunk)
                        temp.flush()
                    if size != info.file_size:
                        raise ArtifactError("ZIP member uncompressed size is inconsistent")
                    hexdigest = digest.hexdigest()
                    if target.exists():
                        if _hash_file(target) != (size, hexdigest):
                            raise ArtifactError("existing extracted member differs; refusing to overwrite")
                    else:
                        try:
                            os.link(temp_path, target)
                        except FileExistsError:
                            if _hash_file(target) != (size, hexdigest):
                                raise ArtifactError("racing extraction target differs")
                    members_evidence.append({"path": "/".join(parts), "uncompressed_bytes": size, "sha256": hexdigest})
                finally:
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)
    except ArtifactError:
        raise
    except Exception:
        raise ArtifactError("ZIP inspection, decompression, CRC verification, or extraction failed") from None
    return {"total_uncompressed_bytes": actual_total, "files": members_evidence}


def collect_artifacts(manifest: dict[str, Any], output_dir: Path, *, verify_existing: bool = False) -> dict[str, Any]:
    """Verify selected originals, reuse identical files, and preserve public evidence."""
    if not isinstance(manifest, dict) or type(manifest.get("artifacts")) is not list or not manifest["artifacts"]:
        raise ArtifactError("manifest requires a nonempty artifacts list")
    originals = manifest["artifacts"]
    metadata = [_metadata(item) for item in originals]
    if len({item["id"] for item in metadata}) != len(metadata) or len({item["name"].casefold() for item in metadata}) != len(metadata):
        raise ArtifactError("manifest contains duplicate artifact IDs or names")
    output = Path(output_dir)
    _ensure_directory(output)
    archive_dir = output / "original_artifacts"
    extracted_dir = output / "extracted"
    _ensure_directory(archive_dir)
    _ensure_directory(extracted_dir)
    results = []
    for original, public in zip(originals, metadata):
        archive_path = archive_dir / (public["name"] + ".zip")
        if archive_path.exists() or archive_path.is_symlink():
            _verify_archive(archive_path, public)
        elif verify_existing:
            raise ArtifactError("required original archive is missing in no-network verification mode")
        elif "download_url" not in original:
            raise ArtifactError("missing download URL for an unavailable original archive")
        else:
            _download(original["download_url"], archive_path, public)
        extracted = _extract(archive_path, extracted_dir / public["name"])
        results.append({
            "publisher_metadata": public,
            "original_zip": "original_artifacts/" + public["name"] + ".zip",
            "extraction_directory": "extracted/" + public["name"],
            **extracted,
        })
    receipt = {
        "format": "LANGDEPTH106-original-artifact-verification-v1",
        "workflow_run_id": EXPECTED_RUN_ID,
        "head_sha": EXPECTED_HEAD_SHA,
        "scope": "exact original archive and member integrity; no model loading or held-out scoring",
        "artifacts": results,
    }
    encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    evidence_path = output / RECEIPT_NAME
    if evidence_path.exists() or evidence_path.is_symlink():
        if evidence_path.is_symlink() or not evidence_path.is_file() or evidence_path.read_bytes() != encoded:
            raise ArtifactError("existing verification evidence differs; use a fresh output directory")
    else:
        try:
            with evidence_path.open("xb") as handle:
                handle.write(encoded)
        except FileExistsError:
            raise ArtifactError("verification evidence was created concurrently; rerun to verify") from None
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true", help="Never download; require already saved exact original ZIPs")
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        receipt = collect_artifacts(manifest, args.output_dir, verify_existing=args.verify_existing)
    except Exception as exc:
        # Malformed JSON and transport errors may embed manifest URL fragments.
        message = str(exc) if isinstance(exc, ArtifactError) else "manifest parsing or collection failed"
        parser.exit(1, "Artifact collection failed: " + message + "\n")
    print(json.dumps({"verified_artifacts": len(receipt["artifacts"]), "evidence_file": RECEIPT_NAME}))


if __name__ == "__main__":
    main()
