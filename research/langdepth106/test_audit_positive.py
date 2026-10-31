"""Constructed six-member trainer/evaluator interoperability fixture.

This is NOT the frozen 1M/4,096-update cohort: each tiny fixture performs TWO
actual optimizer updates, and its explicitly fixture-only in-memory protocol
declares exactly those two updates. Counters are never relabeled as completed
real fits. The frozen protocol loader rejects this fixture regime. No real
TRAIN, DEV, TEST or final-evaluation entrypoint is opened by this test.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import torch

from research.langcache104.model import Config104, DeepCacheTAC104
from research.langdepth106 import evaluate
from research.langdepth106.source import SOURCE_SHA256
from research.langdepth106.train import (
    SCIENTIFIC_STATUS, OffsetSampler, checkpoint_payload, make_batch,
    make_optimizer, optimization_step, parameter_digest, save_checkpoint,
)


def test_complete_constructed_six_member_artifact_tree_passes_actual_cohort_auditor(tmp_path):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config = Config104(d_model=8, n_layers=1, n_heads=1, key_dim=2, value_dim=2,
                       write_rank=1, ffn_dim=8, window=4, chunk_size=2)
    fit = {"steps": 2, "batch_size": 2, "sequence_length": 5,
           "optimizer": "AdamW", "lr": .0003, "min_lr": .00003, "warmup_steps": 1,
           "betas": [.9, .95], "eps": 1e-8, "weight_decay": .1, "clip_grad_norm": 1.}
    fixture_train = torch.arange(64, dtype=torch.long)
    fixture_train_sha = hashlib.sha256(bytes(fixture_train.tolist())).hexdigest()
    count = DeepCacheTAC104(config).counts()["trainable"]
    protocol = {
        "fixture_only": True,
        "seeds": [10601, 10602, 10603], "fit": fit,
        "models": {arm: {"config": asdict(config), "parameters": count}
                   for arm in ("four", "eight")},
        "source": {"splits": {"TRAIN": {"sha256": fixture_train_sha}}},
    }
    protocol_sha = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    optimizer_meta = {key: fit[key] for key in
        ("optimizer", "lr", "betas", "eps", "weight_decay", "clip_grad_norm")}
    scheduler_meta = {key: fit[key] for key in ("steps", "lr", "min_lr", "warmup_steps")}
    # actions/download-artifact with merge-multiple:false recreates one directory
    # per artifact. Receipt discovery must work in this exact nested layout.
    root = tmp_path/"ci-input"/"fits"
    expected = {}
    for arm in ("four", "eight"):
        for seed in protocol["seeds"]:
            torch.manual_seed(seed)
            random.seed(seed)
            model = DeepCacheTAC104(config).float()
            initialized = parameter_digest(model)
            optimizer = make_optimizer(model, fit)
            sampler = OffsetSampler(len(fixture_train), fit["sequence_length"], seed)
            for step in range(1, fit["steps"]+1):
                x, y = make_batch(fixture_train, sampler.draw(fit["batch_size"]), fit["sequence_length"])
                optimization_step(model, optimizer, x, y, step=step, fit=fit,
                                  owners=("constructed-A", "constructed-B"))
            assert parameter_digest(model) != initialized
            metadata = {
                "fixture_only": True, "arm": arm, "seed": seed, "config": asdict(config),
                "parameter_count": count, "protocol_sha256": protocol_sha,
                # This is an identity-field compatibility fixture. It does not
                # supply corpus bytes or pass the frozen source preparation gate.
                "source_sha256": SOURCE_SHA256, "train_sha256": fixture_train_sha,
                "scientific_status": SCIENTIFIC_STATUS, "smoke": False,
                "initialized_parameters_sha256": initialized,
                "optimizer_metadata": optimizer_meta, "scheduler_metadata": scheduler_meta,
                "status": "FIT_COMPLETED_EXPLORATORY",
            }
            payload = checkpoint_payload(model, optimizer, sampler, step=fit["steps"], metadata=metadata)
            directory = root/f"LANGDEPTH106-fit-{arm}-{seed}-attempt-1"
            directory.mkdir(parents=True)
            checkpoint_name = "checkpoint-2.pt"
            checksum = save_checkpoint(directory/checkpoint_name, payload)
            receipt = {
                **metadata, "actual_steps": fit["steps"], "expected_steps": fit["steps"],
                "dev_scored": False, "test_accessed": False,
                "training_byte_presentations": fit["steps"]*fit["batch_size"]*fit["sequence_length"],
                "checkpoint_file": checkpoint_name, "checkpoint_sha256": checksum,
                "original_parameters_sha256": initialized,
                "final_parameters_sha256": parameter_digest(model),
                "sample_offset_sha256": sampler.fingerprint(), "sample_draws": sampler.draws,
            }
            (directory/"FIT_RECEIPT.json").write_text(json.dumps(receipt), encoding="utf-8")
            expected[(arm, seed)] = parameter_digest(model)
    rows = evaluate.audit_final_checkpoints(root, protocol, protocol_sha)
    assert len(rows) == 6
    assert {(row["arm"], row["seed"]) for row in rows} == set(expected)
    for row in rows:
        assert parameter_digest(row["model"]) == expected[(row["arm"], row["seed"])]
        assert row["checkpoint_integrity"]["actual_steps"] == 2
        assert row["checkpoint_integrity"]["sample_draws"] == 4
        assert row["checkpoint_integrity"]["status"] == "PASS_REGISTERED_FINAL_CHECKPOINT_INTEGRITY"
