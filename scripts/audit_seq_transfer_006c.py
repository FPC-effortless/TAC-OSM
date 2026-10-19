#!/usr/bin/env python3
"""Independent recorded-only SEQ006C replay; NO training or test selection.

Restore the original GitHub Actions archive, verify original checkpoint SHA256
and replay every registered TEST task/length result with the frozen model source.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_seq_transfer_006 import ARMS, StreamingAgent, evaluate, TASKS
from confirm_seq_transfer_006c import TEST_SPECS, test_data_after_exposure


def audit(root, output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    original = json.loads((root / "test-results.json").read_text())
    ledger = json.loads((root / "all-trained.json").read_text())
    marker = json.loads((root / "TEST-EXPOSURE.json").read_text())
    assert len(original["models"]) == 60 and len(ledger["fits"]) == 60
    assert marker["status"] == "TEST_EXPOSED_AFTER_60_FITS"
    assert marker["training_sha256"] == hashlib.sha256(
        (root / "all-trained.json").read_bytes()).hexdigest()
    assert original["training_sha256"] == marker["training_sha256"]
    assert original["source_sha"] == ledger["source_sha"] == marker["source_sha"]
    assert sorted(TEST_SPECS) == sorted(original["test_sources"])
    testing = {}
    for name in TEST_SPECS:
        data, source = test_data_after_exposure(root, name)
        assert source == original["test_sources"][name]
        testing[name] = data
    checkpoint_ledger = {
        (row["arm"], row["seed"]): row for row in ledger["fits"]
    }
    assert len(checkpoint_ledger) == 60
    differences = []
    count = 0
    restored = 0
    start = time.monotonic()
    for record in original["models"]:
        key = (record["arm"], record["seed"])
        row = checkpoint_ledger[key]
        archive_path = root / row["checkpoint"]
        assert archive_path.is_file()
        sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        assert sha == row["sha256"] == record["original_checkpoint_sha256"]
        payload = torch.load(archive_path, weights_only=True, map_location="cpu")
        assert payload["seed"] == record["seed"]
        assert payload["arm"] == record["arm"]
        model = StreamingAgent(record["arm"])
        model.load_state_dict(payload["weights"], strict=True)
        model.eval()
        restored += 1
        for name, dataset in testing.items():
            actual = evaluate(model, dataset)
            for task in TASKS:
                expected = record["test"][name][task]
                diff = abs(float(actual[task]) - float(expected))
                differences.append(diff)
                count += 1
                if diff > 1e-8:
                    raise AssertionError(
                        f"Replay mismatch {key} {name}/{task}: expected={expected}, got={actual[task]}"
                    )
    report = {
        "study": "SEQ006C-INDEPENDENT-RECORDED-REPLAY",
        "status": "COMPLETE",
        "no_fitting": True,
        "test_selection_changed": False,
        "original_training_sha256": marker["training_sha256"],
        "original_source_sha": ledger["source_sha"],
        "checkpoint_hashes_verified": restored,
        "heldout_source_hashes_verified": len(testing),
        "compared_accuracy_cells": count,
        "max_abs_difference": max(differences, default=None),
        "elapsed_wall_seconds": time.monotonic() - start,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive-directory", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = audit(Path(args.archive_directory), Path(args.output))
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
