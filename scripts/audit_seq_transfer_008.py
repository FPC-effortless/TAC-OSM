#!/usr/bin/env python3
"""SEQ008 independent RESTORE/REPLAY of original pilot artifacts, never refit.

Restores six original training checkpoints and regenerates registered validation
inputs only. This is independent artifact verification, NOT an independent
physical-world simulator or a new held-out evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_seq_transfer_006 import ARMS, StreamingAgent, evaluate
from run_seq_transfer_008 import (
    PILOT_SEEDS, TRAIN_SEED, VALIDATION_SEEDS, generate,
)

ORIGINAL_RUN = 39309620973
ORIGINAL_SOURCE_COMMIT = "7ffa779a2c0f2e853d84ca62a3a97f59655434f8"
ORIGINAL_FILE_BLOBS = {
    "scripts/run_seq_transfer_006.py": "bae4a61e778c290472211cbd6772e2e184cc0b89",
    "scripts/run_seq_transfer_008.py": "7a775e986e5d9c7562db25f8c912e6f4d1f2ed7e",
}


def git_blob_sha1(path):
    payload = Path(path).read_bytes()
    header = b"blob " + str(len(payload)).encode() + b"\x00"
    return hashlib.sha1(header + payload).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore_replay(root, out):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    repo_root = Path(__file__).resolve().parent.parent
    for relpath, sha in ORIGINAL_FILE_BLOBS.items():
        assert git_blob_sha1(repo_root / relpath) == sha, ("Source changed: " + relpath)
    original = json.loads((root / "pilot.json").read_text())
    assert original["study"] == "SEQ-TRANSFER-008"
    assert original["status"] == "EXPLORATORY_VALIDATION_ONLY"
    assert not original["test_opened"] and original["test_evaluations"] == 0
    assert original["teacher_intermediate_states_consumed"] is False
    expected_grid = {(a, s) for a in ARMS for s in PILOT_SEEDS}
    rows = original["fits"]
    assert len(rows) == 6
    assert {(row["arm"], row["seed"]) for row in rows} == expected_grid

    # Validate original train/validation source manifests, not train anything.
    train, mt = generate(TRAIN_SEED, 3000, (32, 64), 4, purpose="train")
    v64, m64 = generate(VALIDATION_SEEDS[0], 600, (64,), 4,
                        purpose="validation")
    v128, m128 = generate(VALIDATION_SEEDS[1], 600, (128,), 4,
                          purpose="validation")
    original_sources = original["data"]
    for key, produced in (
        ("train", mt), ("L64_E4", m64), ("L128_E4", m128)
    ):
        assert produced["sha256_inputs_labels"] == original_sources[key][
            "sha256_inputs_labels"], "Source manifest mismatch " + key

    output = {
        "study": "SEQ-TRANSFER-008-RECORDED-REPLAY",
        "run_id": ORIGINAL_RUN,
        "original_source_head": ORIGINAL_SOURCE_COMMIT,
        "status": "PASSED_ORIGINAL_CHECKPOINTS_REPLAYED",
        "test_opened": False,
        "fits_retrained": 0,
        "original_checkpoints_verified": 0,
        "per_fit": [],
    }
    for row in rows:
        # No fallback checkpoints and no model choice based on replay results.
        path = Path(row["checkpoint"])
        assert not path.is_absolute() and ".." not in path.parts
        file = root / path
        digest = sha256(file)
        assert digest == row["sha256"], "Checkpoint checksum mismatch " + str(path)
        saved = torch.load(file, map_location="cpu", weights_only=False)
        assert saved["arm"] == row["arm"] and saved["seed"] == row["seed"]
        assert "optimizer" in saved and "weights" in saved
        model = StreamingAgent(row["arm"])
        assert sum(p.numel() for p in model.parameters()) == row["parameters"]
        model.load_state_dict(saved["weights"], strict=True)
        scores = {
            "L64_E4": evaluate(model, v64),
            "L128_E4": evaluate(model, v128),
        }
        for partition, tasks in scores.items():
            for task, value in tasks.items():
                reported = row["validation"][partition][task]
                assert abs(value - reported) <= 1e-8, (
                    f"Original score mismatch {path}/{partition}/{task}: "
                    f"{value} vs {reported}"
                )
        output["original_checkpoints_verified"] += 1
        output["per_fit"].append({
            "arm": row["arm"], "seed": row["seed"],
            "sha256": digest, "validation": scores,
            "scores_equal_original": True,
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": output["status"],
        "fits_retrained": output["fits_retrained"],
        "verified": output["original_checkpoints_verified"],
        "test_opened": output["test_opened"],
        "output": str(out),
    }, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", default="results/seq008-independent/replay.json")
    args = parser.parse_args()
    restore_replay(Path(args.artifact_root), Path(args.output))


if __name__ == "__main__":
    main()
