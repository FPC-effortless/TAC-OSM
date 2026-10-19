#!/usr/bin/env python3
"""SEQ006C registered 20-seed confirmation with write-before-TEST exposure.

The test generator is callable ONLY after all 60 fixed fits finish and
TEST-EXPOSURE.json has been written. Never use outputs for model selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_seq_transfer_006 import (
    ARMS, MAX_ENTITIES, TEST_SEED, TASKS, TRAIN_SEED, VALID_SEED,
    PILOT_SEEDS, STEPS, StreamingAgent, _build_one, evaluate, fit, generate,
)

FRESH_SEEDS = tuple(range(690701, 690721))
TEST_SPECS = {
    "L256_E4": (690603, 256, 4),
    "L512_E4": (690613, 512, 4),
    "L512_E6": (690623, 512, 6),
}
EXPECTED_HASHES = {
    "train": "d7df850483e85fdf3380c4d80c87f3ad3e62e0ee2ea5eca98a01442b65168baf",
    "L64_E4": "743f20cc77d07ed5f8baededcb8d18899a8a8bfd49c2ce69ffbe6929e7b419f8",
    "L128_E4": "77851fe3b5017ada28716d1dbe0121da630c34d0a9e780024b40aadaaa537e3b",
}


def json_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    temp.replace(path)


def read_data():
    train, mtrain = generate(TRAIN_SEED, 3000, (32, 64), 4, purpose="train")
    v64, mv64 = generate(VALID_SEED, 600, (64,), 4, purpose="validation")
    v128, mv128 = generate(VALID_SEED + 10, 600, (128,), 4, purpose="validation")
    for name, report in (("train", mtrain), ("L64_E4", mv64), ("L128_E4", mv128)):
        if report["sha256_public_inputs_and_labels"] != EXPECTED_HASHES[name]:
            raise RuntimeError("Original pilot source changed: " + name)
    return train, {"L64_E4": v64, "L128_E4": v128}, {
        "train": mtrain, "validation": {"L64_E4": mv64, "L128_E4": mv128},
    }


def test_data_after_exposure(root, name):
    """The TEST namespace cannot be reached without a completed fit ledger."""
    marker = root / "TEST-EXPOSURE.json"
    full_manifest = root / "all-trained.json"
    if not marker.is_file() or not full_manifest.is_file():
        raise PermissionError("Confirmatory TEST remains closed until 60 fits and marker")
    declaration = json.loads(marker.read_text())
    fitted = json.loads(full_manifest.read_text())
    if declaration.get("status") != "TEST_EXPOSED_AFTER_60_FITS":
        raise PermissionError("TEST marker incomplete")
    if declaration.get("training_sha256") != hashlib.sha256(full_manifest.read_bytes()).hexdigest():
        raise PermissionError("TEST marker does not bind the frozen fit ledger")
    if len(fitted["fits"]) != 60 or any(not x.get("sha256") for x in fitted["fits"]):
        raise PermissionError("All 60 immutable fit hashes required")
    if name not in TEST_SPECS:
        raise ValueError(name)
    seed, L, n_entities = TEST_SPECS[name]
    rng = np.random.default_rng(seed)
    n = 600
    ids = np.zeros((n, L), dtype=np.int64)
    ops = np.zeros((n, L), dtype=np.int64)
    states = np.zeros((n, L, MAX_ENTITIES), dtype=np.float32)
    queries = np.zeros((n, 2), dtype=np.int64)
    tasks = np.zeros(n, dtype=np.int64)
    labels = np.zeros(n, dtype=np.float32)
    lags = []
    for i in range(n):
        task, desired = i % 3, (i // 3) % 2
        e, o, truth, qa, qb, y, age = _build_one(rng, L, n_entities, task, desired)
        if age < L // 3:
            raise AssertionError(f"Insufficient factual age {name} item={i}: {age}")
        ids[i], ops[i], states[i] = e, o, truth
        queries[i] = qa, qb
        tasks[i], labels[i] = task, y
        lags.append(age)
        # A second independent symbolic event replay checks every time step.
        reconstructed = np.zeros(MAX_ENTITIES, dtype=np.float32)
        for t in range(L):
            k, symbol = int(e[t]), int(o[t])
            if symbol == 0:
                reconstructed[k] = 0
            elif symbol == 1:
                reconstructed[k] = 1
            elif symbol == 2:
                reconstructed[k] = 1 - reconstructed[k]
            if not np.array_equal(reconstructed, truth[t]):
                raise AssertionError(f"Independent state replay mismatch {name}/{i}/{t}")
        answer = int(reconstructed[qa])
        if task == 2:
            answer ^= int(reconstructed[qb])
        if answer != int(y):
            raise AssertionError("Independent target replay mismatch")
    report = {
        "name": name, "seed": seed, "sequence_length": L, "entity_count": n_entities,
        "count": n, "min_target_lag": min(lags), "median_target_lag": float(np.median(lags)),
        "inputs_sha256": hashlib.sha256(b"".join([ids.tobytes(), ops.tobytes(),
                                                  queries.tobytes(), labels.tobytes()])).hexdigest(),
        "label_balance": {
            task: {"n": int(np.count_nonzero(tasks == k)),
                   "positive": int(np.count_nonzero((tasks == k) & (labels == 1)))}
            for k, task in enumerate(TASKS)
        },
    }
    for task in TASKS:
        if report["label_balance"][task] != {"n": 200, "positive": 100}:
            raise AssertionError("Unbalanced test set: " + name)
    tensors = tuple(torch.from_numpy(a) for a in (
        ids, ops, np.ones_like(ids, np.float32), states, queries, tasks, labels))
    return tensors, report


def run(output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if (output / "TEST-EXPOSURE.json").exists():
        raise RuntimeError("This run already exposed TEST: replay archived weights instead")
    start = time.monotonic()
    train, validations, sources = read_data()
    ledger = {
        "study": "SEQ-TRANSFER-006C",
        "status": "FITTING_IN_PROGRESS_TEST_CLOSED",
        "source_sha": os.environ.get("GITHUB_SHA", "local-uncommitted"),
        "test_opened": False, "test_evaluations": 0,
        "source_manifests": sources,
        "fresh_seeds": list(FRESH_SEEDS), "arms": list(ARMS),
        "train_updates": STEPS, "fits": [],
    }
    json_write(output / "training-progress.json", ledger)
    for seed in FRESH_SEEDS:
        for arm in ARMS:
            record = fit(arm, seed, train, validations, STEPS, output, save=True)
            checkpoint = output / record["checkpoint"]
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != record["sha256"]:
                raise ArithmeticError("Checkpoint digest mismatch")
            ledger["fits"].append(record)
            json_write(output / "training-progress.json", ledger)
            print(json.dumps({"arm": arm, "seed": seed, "validation": record["validation"],
                              "sha256": record["sha256"]}, sort_keys=True), flush=True)
    if len(ledger["fits"]) != 60:
        raise RuntimeError("Incomplete fresh fit matrix; leave TEST unopened")
    ledger["status"] = "ALL_60_FITS_COMPLETED_TEST_CLOSED"
    json_write(output / "all-trained.json", ledger)
    ledger_sha = hashlib.sha256((output / "all-trained.json").read_bytes()).hexdigest()
    marker = {
        "study": "SEQ-TRANSFER-006C",
        "status": "TEST_EXPOSED_AFTER_60_FITS",
        "training_sha256": ledger_sha,
        "source_sha": ledger["source_sha"],
        "test_sources": {key: list(val) for key, val in TEST_SPECS.items()},
        "selection_locked": "final_step_all_seeds_no_test_retuning",
    }
    # The test exposure marker is permanently emitted BEFORE any TEST records.
    json_write(output / "TEST-EXPOSURE.json", marker)
    print("TEST_EXPOSURE_MARKER_WRITTEN; start independent test generation", flush=True)
    fresh = {}
    source_tests = {}
    for name in TEST_SPECS:
        data, manifest = test_data_after_exposure(output, name)
        fresh[name], source_tests[name] = data, manifest
    results = []
    for row in ledger["fits"]:
        checkpoint = output / row["checkpoint"]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError("Fitting weights changed between training and test")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload["arm"] != row["arm"] or payload["seed"] != row["seed"]:
            raise RuntimeError("Wrong checkpoint identity")
        agent = StreamingAgent(row["arm"])
        agent.load_state_dict(payload["weights"], strict=True)
        agent.eval()
        scores = {name: evaluate(agent, data) for name, data in fresh.items()}
        record = {"arm": row["arm"], "seed": row["seed"], "test": scores,
                  "original_checkpoint_sha256": row["sha256"]}
        results.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    per_arm = {}
    for arm in ARMS:
        chosen = [x for x in results if x["arm"] == arm]
        per_cell = {
            name: {task: {
                "mean_accuracy": float(np.mean([r["test"][name][task] for r in chosen])),
                "seeds_at_or_above_95": int(sum(r["test"][name][task] >= .95 for r in chosen)),
            } for task in TASKS} for name in TEST_SPECS
        }
        passed = all(per_cell[name][task]["seeds_at_or_above_95"] >= 18
                     for name in TEST_SPECS for task in TASKS)
        per_arm[arm] = {"passed_individual_reliability_gate": bool(passed),
                        "per_cell": per_cell}
    results_manifest = {
        "study": "SEQ-TRANSFER-006C",
        "status": "CONFIRMATORY_EVALUATED_SYNTHETIC_ONLY",
        "test_opened": True, "test_evaluations": 1,
        "source_sha": ledger["source_sha"], "training_sha256": ledger_sha,
        "test_sources": source_tests, "models": results,
        "summary": per_arm,
        "elapsed_cpu_wall_seconds_single_thread": float(time.monotonic() - start),
        "scientific_limits": [
            "Scratch TAC-inspired agent, not DEV061.5 pretrained weight attribution",
            "Exact intermediate state teacher supplied during training",
            "Public entity identity routing remains a structural prior",
            "No CASM-specific superiority if matched routed GRU succeeds",
            "No real-world economic or cross-domain validation",
        ],
    }
    json_write(output / "test-results.json", results_manifest)
    print("RESULTS_WRITTEN -- NO POSTTEST SELECTION", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/seq006c")
    a = p.parse_args()
    run(Path(a.output))


if __name__ == "__main__":
    main()
