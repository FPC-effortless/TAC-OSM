#!/usr/bin/env python3
"""Fail-closed identity preflight for *candidate* mobile model packages.

A passing result verifies local artifact/hash/schema, NOT scientific accuracy,
on-device execution, source legality, model training or signed evaluator trust.
No network, model execution, training, hidden data or external dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_FORMATS = {"onnx", "executorch", "coreml"}
_ALLOWED_DTYPE = {"float32", "float16", "int8", "int32", "int64"}


class PackageError(ValueError):
    """Malformed, unverifiable or unsafe artifact contract."""


def _record(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise PackageError(f"{label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PackageError(f"{label} must be nonempty text")
    return value


def _positive_int(value: object, label: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise PackageError(f"{label} must be an integer >= {minimum}")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise PackageError(f"{label} must be a lowercase SHA256 hex digest")
    return value


def _relative_file(root: Path, filename: object, label: str) -> Path:
    text = _string(filename, label)
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts or text.startswith("~"):
        raise PackageError(f"{label} must stay inside package root")
    file_path = root / relative
    if any(part.is_symlink() for part in [root, *list(file_path.parents)[: len(relative.parts)], file_path]):
        raise PackageError(f"{label} must not follow symlinks")
    resolved = file_path.resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise PackageError(f"{label} must name an existing regular file in package")
    return resolved


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _tensor_defs(value: object, label: str) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise PackageError(f"{label} must be a nonempty tensor list")
    names = set()
    for i, item in enumerate(value):
        item = _record(item, f"{label}[{i}]")
        name = _string(item.get("name"), f"{label}[{i}].name")
        if name in names:
            raise PackageError(f"{label} contains duplicate tensor name")
        names.add(name)
        if item.get("dtype") not in _ALLOWED_DTYPE:
            raise PackageError(f"{label}[{i}].dtype unsupported")
        shape = item.get("shape")
        if not isinstance(shape, list) or not shape or any(type(n) is not int or n <= 0 for n in shape):
            raise PackageError(f"{label}[{i}].shape must contain positive fixed dimensions")
    return value


def validate_package(root: Path, require_device: bool = False) -> dict:
    root = root.resolve()
    manifest_path = root / "mobile-manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise PackageError("missing regular mobile-manifest.json")
    manifest = _record(json.loads(manifest_path.read_text(encoding="utf-8")), "manifest")
    if manifest.get("schema_version") != 1:
        raise PackageError("unsupported schema_version")
    model = _record(manifest.get("model"), "model")
    fmt = model.get("format")
    if fmt not in _ALLOWED_FORMATS:
        raise PackageError("model.format must be onnx, executorch or coreml")
    artifact = _relative_file(root, model.get("path"), "model.path")
    expected_sha = _sha(model.get("sha256"), "model.sha256")
    if _digest(artifact) != expected_sha:
        raise PackageError("model bytes do not match pinned SHA256")
    if artifact.stat().st_size != _positive_int(model.get("bytes"), "model.bytes"):
        raise PackageError("model size mismatch")
    _sha(model.get("original_checkpoint_sha256"), "model.original_checkpoint_sha256")
    _string(model.get("source_commit"), "model.source_commit")
    trainable = _positive_int(model.get("trainable_parameters"), "model.trainable_parameters")
    deployed = _positive_int(model.get("deployed_parameters"), "model.deployed_parameters")
    if not (95_000 <= trainable <= 105_000 and 95_000 <= deployed <= 105_000):
        raise PackageError("prospective TAC research parameter policy (95k–105k) not met")
    if deployed < trainable:
        raise PackageError("deployed parameters cannot be fewer than trainable")
    interface = _record(manifest.get("interface"), "interface")
    inp = _tensor_defs(interface.get("inputs"), "interface.inputs")
    out = _tensor_defs(interface.get("outputs"), "interface.outputs")
    input_names = {x["name"] for x in inp}
    output_names = {x["name"] for x in out}
    state = interface.get("state_pairs")
    if not isinstance(state, list) or not state:
        raise PackageError("state_pairs must specify explicit recurrent state input and output")
    used_in, used_out = set(), set()
    for pair in state:
        pair = _record(pair, "state pair")
        a = _string(pair.get("input"), "state input")
        b = _string(pair.get("output"), "state output")
        if a not in input_names or b not in output_names or a in used_in or b in used_out:
            raise PackageError("invalid or duplicate state input/output pair")
        used_in.add(a)
        used_out.add(b)
    _string(interface.get("reset_semantics"), "interface.reset_semantics")
    scientific = _record(manifest.get("scientific_evidence"), "scientific_evidence")
    if scientific.get("status") not in {"FAILED", "UNKNOWN", "QUALIFIED"}:
        raise PackageError("scientific_evidence.status must explicitly describe qualification")
    _string(scientific.get("evidence_ref"), "scientific_evidence.evidence_ref")
    report = {
        "package_integrity": "PASS",
        "artifact_sha256": expected_sha,
        "artifact_bytes": artifact.stat().st_size,
        "hardware_evidence": "BLOCKED_NO_DEVICE_REPORT",
        "numerical_export_equivalence": "NOT_VERIFIED_BY_THIS_TOOL",
        "scientific_qualification": scientific["status"],
        "deployment_qualification": "NOT_ESTABLISHED",
    }
    measurement = manifest.get("device_measurement")
    if measurement is not None:
        measurement = _record(measurement, "device_measurement")
        receipt_path = _relative_file(root, measurement.get("path"), "device_measurement.path")
        if _digest(receipt_path) != _sha(measurement.get("sha256"), "device_measurement.sha256"):
            raise PackageError("device receipt SHA256 mismatch")
        receipt = _record(json.loads(receipt_path.read_text(encoding="utf-8")), "device receipt")
        if receipt.get("model_sha256") != expected_sha:
            raise PackageError("device receipt references different model")
        if receipt.get("source") != "physical-device":
            raise PackageError("device receipt must identify physical-device measurements")
        for name in ("device_model", "android_version", "execution_backend", "benchmark_script"):
            _string(receipt.get(name), f"device receipt {name}")
        latencies = receipt.get("latency_ms")
        if not isinstance(latencies, list) or len(latencies) < 30:
            raise PackageError("device receipt requires >=30 latency observations")
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in latencies):
            raise PackageError("device latency observations must be finite and positive")
        _positive_int(receipt.get("peak_rss_bytes"), "device receipt peak_rss_bytes")
        ordered = sorted(latencies)
        report.update({
            "hardware_evidence": "RECEIPT_FORMAT_PASS_INDEPENDENT_AUTHENTICATION_OPEN",
            "latency_p50_ms": ordered[len(ordered) // 2],
            "latency_p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        })
    if require_device and measurement is None:
        raise PackageError("physical-device evidence required but absent")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_root", type=Path)
    parser.add_argument("--require-device", action="store_true", help="require a hash-checked physical-device receipt")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(validate_package(args.package_root, args.require_device), sort_keys=True))
        return 0
    except (PackageError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"package_integrity": "FAIL_OR_BLOCKED", "reason": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
