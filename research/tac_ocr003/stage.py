"""Build a credential-free, immutable, offline Kaggle input + notebook.

This command never calls Kaggle, starts a GPU, reads credentials or fits.
Only the explicit files from the supplied Git commit enter the source ZIP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

from .core import PROTOCOL, ROOT, sha256, write_json


ENCODER_FILES = ("config.json", "generation_config.json", "preprocessor_config.json",
                 "model.safetensors", "README.md")
WEIGHT_SHA256 = "49350a39968df83e5a1adc90fc0ede02ff247671aed70b842af350fd4a7103f3"


def source_paths(commit):
    result = subprocess.run(["git", "ls-tree", "-r", "--name-only", commit],
                            cwd=ROOT, check=True, capture_output=True, text=True)
    prefixes = ("research/tac_ocr001/", "research/tac_ocr003/", "src/tac_osm/")
    exact = {"research/langcache104/model.py", "research/langcache104/kernels.py",
             "research/langcache104/__init__.py", "scripts/validate_research_parity.py",
             "research/mathematical-engineering/tac_ocr003/FAIR_COMPARISON.json"}
    return [p for p in result.stdout.splitlines() if p in exact or (
        p.startswith(prefixes) and (p.endswith(".py") or
                                  p == "research/tac_ocr003/PROTOCOL.json"))]


def source_zip(commit, output):
    files = source_paths(commit)
    if "research/tac_ocr003/train.py" not in files:
        raise ValueError("supplied immutable commit has no reviewed OCR003 trainer")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            if any(x in name.lower() for x in ("kaggle.json", ".env", "credential", "private_key")):
                raise ValueError("credential path forbidden in source bundle")
            raw = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=ROOT,
                                 check=True, capture_output=True).stdout
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, raw)
    return files


def notebook_driver(assets_sha256):
    return '''import hashlib, json, os, pathlib, stat, subprocess, sys, zipfile
if sys.version_info < (3, 11):
    raise RuntimeError('pinned offline wheel bundle requires Python 3.11 or newer')
P = pathlib.Path
def digest(path):
    h = hashlib.sha256()
    with P(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
def extract(path, target):
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        if sum(x.file_size for x in archive.infolist()) > 8 * 1024**3:
            raise RuntimeError('oversized archive')
        for member in archive.infolist():
            destination = (target / member.filename).resolve()
            if not destination.is_relative_to(target.resolve()) or stat.S_ISLNK(member.external_attr >> 16):
                raise RuntimeError('unsafe archive member')
        archive.extractall(target)
matches = [p for p in P('/kaggle/input').rglob('TAC_OCR003_ASSETS.json')
           if digest(p) == 'ASSETS_SHA_PLACEHOLDER']
if len(matches) != 1:
    raise RuntimeError('immutable input manifest missing or ambiguous')
assets = json.loads(matches[0].read_text()); inputs = matches[0].parent
work = P('/kaggle/working/tac-ocr003')
work.mkdir(exist_ok=True)
for name, expected in assets['files'].items():
    if P(name).name != name or digest(inputs / name) != expected['sha256']:
        raise RuntimeError('input asset changed: ' + name)
extract(inputs / 'source.zip', work / 'source')
extract(inputs / 'wheels.zip', work / 'wheels')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-index', '--no-deps',
    '--find-links', str(work / 'wheels'), '--target', str(work / 'vendor'),
    'transformers==4.52.1', 'huggingface_hub==0.36.0', 'tokenizers==0.21.4',
    'safetensors==0.6.2', 'cryptography==45.0.7'], check=True)
env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(work / 'source'),
    str(work / 'source/src'), str(work / 'vendor')]),
    HF_HOME=str(work / 'hf'), HF_HUB_CACHE=str(work / 'hf/hub'),
    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
    CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false')
# Public verification key and signed receipt, never a private signing key.
# These must come from an operator-controlled assessor, outside this bundle.
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    for label in ('TAC_OSM_REWARD_VERIFIER_PUBLIC_KEY_HEX', 'TAC_OSM_PREFIT_ATTESTATION_JSON'):
        if not env.get(label):
            env[label] = secrets.get_secret(label)
except Exception:
    pass
# Fail before extracting image payloads or allocating CUDA if unsigned.
subprocess.run([sys.executable, '-c',
    'from research.tac_ocr003.core import require_prefit; require_prefit()'],
    cwd=work / 'source', env=env, check=True)
extract(inputs / 'data.zip', work / 'data')
extract(inputs / 'encoder.zip', work / 'hf/hub')
command = [sys.executable, '-m', 'research.tac_ocr003.train', '--data', str(work / 'data'),
           '--output', str(work / 'outputs')]
subprocess.run(command, cwd=work / 'source', env=env, check=True)
try:
    env['TAC_OSM_POSTFIT_ATTESTATION_JSON'] = secrets.get_secret('TAC_OSM_POSTFIT_ATTESTATION_JSON')
except Exception:
    pass
final = subprocess.run([sys.executable, '-m', 'research.tac_ocr003.assess',
    '--output', str(work / 'outputs')], cwd=work / 'source', env=env)
print(json.dumps({'fits_completed': True, 'independently_qualified': final.returncode == 0,
                  'outputs': str(work / 'outputs')}))
'''.replace("ASSETS_SHA_PLACEHOLDER", assets_sha256)


def package(*, commit, data_zip, encoder, wheels, destination, kaggle_user):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("immutable source commit required")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,38}", kaggle_user):
        raise ValueError("invalid Kaggle account slug")
    protocol = json.loads(subprocess.run(["git", "show", f"{commit}:research/tac_ocr003/PROTOCOL.json"],
        cwd=ROOT, check=True, capture_output=True, text=True).stdout)
    for path in ("research/langcache104/model.py", "research/langcache104/kernels.py",
                 "research/tac_ocr001/model.py", "research/tac_ocr001/encoder.py",
                 "research/tac_ocr001/presets.py", "research/tac_ocr001/tokenizer.py"):
        original, staged = [subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT,
                             check=True, capture_output=True).stdout
                            for ref in (protocol["neural_source_commit"], commit)]
        if original != staged:
            raise ValueError("neural source changed without a new reciprocal mathematical record: " + path)
    if sha256(data_zip) != protocol["data_zip_sha256"]:
        raise ValueError("source dataset ZIP changed")
    if sha256(encoder / "model.safetensors") != WEIGHT_SHA256:
        raise ValueError("pretrained encoder distribution differs from publisher hash")
    if destination.exists():
        raise ValueError("preserve prior stage; use a fresh destination")
    dataset, notebook = destination / "dataset", destination / "notebook"
    dataset.mkdir(parents=True)
    notebook.mkdir()
    shutil.copyfile(data_zip, dataset / "data.zip")
    files = source_zip(commit, dataset / "source.zip")
    hf_prefix = f"models--microsoft--trocr-small-printed/snapshots/{protocol['encoder_revision']}"
    with zipfile.ZipFile(dataset / "encoder.zip", "w", compression=zipfile.ZIP_STORED) as archive:
        for name in ENCODER_FILES:
            archive.write(encoder / name, hf_prefix + "/" + name)
    with zipfile.ZipFile(dataset / "wheels.zip", "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(wheels.glob("*.whl")):
            archive.write(path, path.name)
    asset_files = {name: {"sha256": sha256(dataset / name), "bytes": (dataset / name).stat().st_size}
                   for name in ("data.zip", "source.zip", "encoder.zip", "wheels.zip")}
    assets = {"schema": "tac-ocr003.offline-assets.v1", "source_commit": commit,
              "files": asset_files, "source_file_count": len(files),
              "contains_kaggle_credentials": False, "training_started": False,
              "runtime_network": "disabled", "test_payload": "not_included"}
    write_json(dataset / "TAC_OCR003_ASSETS.json", assets)
    dataset_id = kaggle_user + "/tac-ocr003-cord-grounding"
    write_json(dataset / "dataset-metadata.json", {"id": dataset_id,
        "title": "TAC OCR003 Private CORD Grounding",
        "licenses": [{"name": "other"}],
        "description": "Private research bundle. CORD-v2 original region crops CC-BY-4.0 (Clova AI); Microsoft TrOCR publisher checkpoint and model card; source-pinned TAC-OSM pilot code; offline dependency wheels. See ATTRIBUTION.txt. No API credentials or TEST payload. No fits performed; independently signed preflight required."})
    (dataset / "ATTRIBUTION.txt").write_text(
        "CORD v2: https://github.com/clovaai/cord; CC-BY-4.0; original license is inside data.zip. "
        "Region preparation preserves original images, text and source receipts.\n"
        "Pretrained Microsoft TrOCR: https://huggingface.co/microsoft/trocr-small-printed/tree/"
        + protocol["encoder_revision"] + "; publisher card inside encoder.zip. "
        "The pinned card has no explicit license tag; upstream Microsoft UniLM repository has an MIT license. "
        "No separate licensing determination or redistribution grant asserted for the combined bundle.\n"
        "TAC-OSM source: https://github.com/FPC-effortless/TAC-OSM/tree/" + commit
        + "; owner-authorized private research copy.\n"
        "Dependency wheels retain their original embedded license notices.\n")
    handle = kaggle_user + "/tac-ocr003-cord-grounding-pilot"
    write_json(notebook / "kernel-metadata.json", {"id": handle,
        "title": "TAC OCR003 CORD Grounding Pilot", "code_file": "train.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [dataset_id], "competition_sources": [], "kernel_sources": [], "model_sources": []})
    write_json(notebook / "train.ipynb", {"nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [{"cell_type": "markdown", "id": "protocol", "metadata": {}, "source":
            "# TAC-OCR-003: real receipt-region grounding pilot\n\n"
            "**Saved without execution. Training is blocked until independent signed preflight is supplied.**\n\n"
            "Two T4s, two TAC seeds; frozen visual features computed once; FP32; 15 epochs; "
            "11,199 training and 1,244 validation regions. The data covers Indonesian receipts. "
            "No Pidgin, whole-page, long-document or architecture-superiority result is established.\n\n"
            "Source commit: `" + commit + "`. Session timeout: 7,200 seconds, at most 240 allocated GPU-minutes. "
            "Configure `TAC_OSM_REWARD_VERIFIER_PUBLIC_KEY_HEX` and `TAC_OSM_PREFIT_ATTESTATION_JSON` "
            "from an access-separated assessor through Kaggle Secrets. Never add a signing private key. "
            "Source near-duplicate and ambiguous-label reviews remain open.\n"},
          {"cell_type": "code", "id": "guarded-run", "metadata": {}, "execution_count": None,
           "outputs": [], "source": notebook_driver(sha256(dataset / "TAC_OCR003_ASSETS.json"))}]})
    report = {"status": "STAGED_UNSUBMITTED_NO_FIT", "dataset": dataset_id,
              "notebook": handle, "source_commit": commit,
              "assets_sha256": sha256(dataset / "TAC_OCR003_ASSETS.json"),
              "total_upload_bytes": sum(p.stat().st_size for p in dataset.iterdir() if p.is_file()),
              "credential_files_included": False, "allocated_gpu_minutes": 0}
    write_json(destination / "staging-receipt.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--encoder-snapshot", type=Path, required=True)
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kaggle-user", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(package(commit=args.source_commit, data_zip=args.data_zip,
        encoder=args.encoder_snapshot, wheels=args.wheels, destination=args.out,
        kaggle_user=args.kaggle_user)))


if __name__ == "__main__":
    main()
