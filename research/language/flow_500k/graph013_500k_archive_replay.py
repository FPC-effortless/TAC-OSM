"""GRAPH013: read-only replay of original five 500K TAC on exposed Vim DEV.

Requires original researcher-owned archive and publisher Vim files. This script
NEVER trains, opens sealed TEST, or declares a matched SSM comparison.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import torch
from torch.nn import functional as F
from numpy._core.multiarray import _reconstruct, scalar
from tac_wave_ced_500k import WaveCED500K

ROOT = "tac_language_500k/"
SEEDS = tuple(range(1701, 1706))
EXPECTED_ARCHIVE_SHA256 = "cde9a91be40a4e613975c0bd8caa4c19ac24e3d1d0e993db5cacfbd002fecc32"
EXPECTED_MODEL_SHA256 = "acf80cb9051b7f2d2c3900d1676eed212890208cd5213088c6751f9cfd24392c"
REQUIRED_KEYS = {"model", "optimizer", "torch_rng", "numpy_rng", "python_rng",
                 "sampler_rng", "seed", "samples_sha256", "dev", "updates"}

def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def _safe_checkpoint(raw: bytes):
    """Only allowlist the NumPy RNG/scalar types in the historical checkpoint."""
    with torch.serialization.safe_globals([_reconstruct, scalar, np.ndarray,
                                            np.dtype, np.dtypes.UInt32DType,
                                            np.dtypes.Float64DType]):
        return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)

def replay(archive: Path, source_root: Path):
    if not archive.is_file():
        raise ValueError("original historical 500K archive is absent")
    if _sha(archive.read_bytes()) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("historical archive fails independently pinned SHA256")
    torch.set_num_threads(2)
    model_source = Path(__file__).with_name("tac_wave_ced_500k.py")
    if not model_source.is_file():
        raise ValueError("run alongside original WaveCED500K source")
    with ZipFile(archive) as z:
        if len(set(z.namelist())) != len(z.namelist()):
            raise ValueError("duplicate ZIP member names")
        hashes = json.loads(z.read(ROOT + "history_long_003/BUNDLE_FILE_HASHES.json"))["paths"]
        def get(name):
            if name not in hashes:
                raise ValueError("member absent from original hash manifest: " + name)
            record = hashes[name]
            raw = z.read(name)
            if len(raw) != record["bytes"] or _sha(raw) != record["sha256"]:
                raise ValueError("source/checkpoint byte hash mismatch: " + name)
            return raw
        model_bytes = get(ROOT + "tac_wave_ced_500k.py")
        if _sha(model_bytes) != EXPECTED_MODEL_SHA256 or model_source.read_bytes() != model_bytes:
            raise ValueError("local model differs from original checkpoint source")
        proto = json.loads(get(ROOT + "SOURCE_PARTITION_MANIFEST.json"))["source"]
        parts = proto["doc_partitions"]
        if len(parts["fit"]) != 30 or len(parts["dev"]) != 6 or set(parts["fit"]) & set(parts["dev"]):
            raise ValueError("original source partitions invalid")
        texts = {}
        for name in parts["fit"] + parts["dev"]:
            raw = (source_root / name).read_bytes()
            if _sha(raw) != proto["byte_sha256"][name]:
                raise ValueError("publisher original source SHA mismatch: " + name)
            arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int64).copy()
            arr[arr >= 128] = 63
            texts[name] = arr
        windows = []
        for name in parts["dev"]:
            arr = texts[name]
            for offset in np.linspace(0, len(arr)-129, 8, dtype=np.int64):
                windows.append(torch.tensor(arr[offset:offset+129], dtype=torch.long))
        if len(windows) != 48:
            raise ValueError("original DEV windows missing")
        results = []
        for seed in SEEDS:
            raw = get(ROOT + f"seed_{seed}.pt")
            ck = _safe_checkpoint(raw)
            if set(ck) != REQUIRED_KEYS or ck["seed"] != seed or ck["updates"] != 120:
                raise ValueError("checkpoint identity/optimizer updates mismatch")
            if not ck["optimizer"].get("state") or ck["torch_rng"].numel() == 0:
                raise ValueError("optimizer/RNG state absent")
            rng = np.random.default_rng(seed)
            schedule = hashlib.sha256()
            for _ in range(120):
                for _ in range(8):
                    src = parts["fit"][int(rng.integers(30))]
                    arr = texts[src]
                    position = int(rng.integers(len(arr)-129))
                    schedule.update(f"{src}:{position};".encode())
            if schedule.hexdigest() != ck["samples_sha256"]:
                raise ValueError("original FIT training sample schedule mismatch")
            m = WaveCED500K()
            if m.parameter_count() != 499290:
                raise ValueError("original model parameter count mismatch")
            m.load_state_dict(ck["model"], strict=True)
            m.eval()
            nll_sum = 0.0
            correct = 0
            count = 0
            with torch.no_grad():
                for index in range(0, 48, 8):
                    batch = torch.stack(windows[index:index+8])
                    labels = batch[:, 1:]
                    pred = m(batch[:, :-1])
                    nll_sum += float(F.cross_entropy(pred.reshape(-1, 128),
                                                      labels.reshape(-1), reduction="sum"))
                    correct += int((pred.argmax(-1) == labels).sum())
                    count += labels.numel()
            value = nll_sum/count
            original = float(ck["dev"]["nll"])
            if count != 6144 or abs(value-original) > 1e-6:
                raise ValueError(f"frozen score mismatch seed {seed}: {value}, {original}")
            results.append(dict(seed=seed, checkpoint_sha256=_sha(raw),
                                sample_schedule_sha256=schedule.hexdigest(),
                                original_dev_nll=original, replayed_dev_nll=value,
                                accuracy=correct/count, targets=count,
                                score_abs_error=abs(value-original)))
    return dict(schema="graph013-original-500k-tac-replay-v1",
                evidence_status="LOCAL_FROZEN_REPLAY_EXPOSED_DEV_ONLY",
                archive_sha256=_sha(archive.read_bytes()),
                original_model_sha256=_sha(model_bytes),
                original_model_active_parameters=499290,
                authenticated_publisher_document_hashes=36,
                checkpoint_count=len(results),
                mean_dev_nll=math.fsum(r["replayed_dev_nll"] for r in results)/len(results),
                seeds=results,
                gates={"REPO_REPLAY":"BLOCKED: original checkpoint bytes not in repository",
                       "MATCHED_500K_SSM":"NOT_RUN: original SSM binary unavailable",
                       "FAIR_TRAINING_COMPUTE":"NOT_ESTABLISHED",
                       "INDEPENDENT_TEST":"NOT_RUN",
                       "ARCHITECTURE_SUPERIORITY":"NOT_ESTABLISHED"})

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--source-root", type=Path, default=Path("/usr/share/vim/vim91/doc"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = replay(args.archive, args.source_root)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    print(f"GRAPH013 original TAC frozen DEV replay passed {result['checkpoint_count']} seeds; no SSM comparison")

if __name__ == "__main__":
    main()
