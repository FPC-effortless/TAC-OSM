"""Offline Kaggle T4x2 pilot; all optimizer paths require signed preflight.

Two GPUs first build disjoint image-only feature shards, then run separate
registered TAC seeds. The pretrained text decoder is never used as the TAC.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from research.tac_ocr001.encoder import FrozenTrOCREncoder
from research.tac_ocr001.model import GroundedOCRDecoder, compress_references
from research.tac_ocr001.presets import matched_presets
from .core import (MANIFEST, PROTOCOL, ROOT, atomic_checkpoint, epoch_batches,
                   learning_rate, load_rows, predict, require_prefit,
                   restore_checkpoint, seed_all, sha256, targets,
                   transcription_metrics, write_json)

STOP_REQUESTED = False


def request_stop(*_):
    # Stop between updates so the saved optimizer/cursor remain consistent.
    global STOP_REQUESTED
    STOP_REQUESTED = True


def verify_two_t4s():
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(names) != 2 or any("T4" not in x.upper() for x in names):
        raise RuntimeError(f"exactly two physical NVIDIA T4 GPUs required; found {names}")
    for i in range(2):
        if torch.ones(1, device=f"cuda:{i}").item() != 1:
            raise RuntimeError("T4 device arithmetic unavailable")
    return names


def corpus(data, protocol):
    train = load_rows(data, "train", protocol["train_sha256"])
    validation = load_rows(data, "validation", protocol["validation_sha256"])
    if (len(train), len(validation)) != (11199, 1244):
        raise ValueError("registered source cohort size differs")
    # IDs organize cache lookups outside the neural graph, never embeddings.
    images = {}
    for row in train + validation:
        images[row["image_sha256"]] = row["image_path"]
    items = sorted(images.items())
    return train, validation, items


def feature_worker(args, protocol):
    require_prefit()
    seed_all(protocol["seeds"][args.slot])
    _, _, items = corpus(args.data, protocol)
    device = torch.device("cuda:0")
    encoder = FrozenTrOCREncoder(protocol["encoder_id"], protocol["encoder_revision"]).to(device)
    if encoder.learned_parameters != protocol["frozen_encoder_parameters"]:
        raise RuntimeError("actual encoder parameter count changed")
    subset = items[args.slot::2]
    output = args.output / "features"
    output.mkdir(exist_ok=True, parents=True)
    cache = np.lib.format.open_memmap(output / f"shard-{args.slot}.npy", mode="w+",
                dtype=np.float32, shape=(len(subset), 128, 384))
    batch_size = protocol["encoder_batch_size"]
    started = time.monotonic()
    for offset in range(0, len(subset), batch_size):
        if STOP_REQUESTED or time.time() >= args.deadline:
            raise TimeoutError("feature construction deadline; partial cache cannot train")
        images = []
        for digest, relative in subset[offset:offset + batch_size]:
            path = (args.data / relative).resolve()
            if not path.is_relative_to(args.data.resolve()) or sha256(path) != digest:
                raise ValueError("crop bytes/path differ from registered source")
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        features, boxes = encoder(images)  # strict FP32, no AMP
        features, boxes = compress_references(features, boxes, 128)
        cache[offset:offset + len(images)] = features.cpu().numpy()
        if offset == 0:
            torch.save(boxes[:1].cpu().clone(), output / f"boxes-{args.slot}.pt")
        if offset % (batch_size * 20) == 0:
            print(json.dumps({"phase": "frozen_encoder", "slot": args.slot,
                              "images": offset + len(images), "total": len(subset)}), flush=True)
    cache.flush()
    write_json(output / f"shard-{args.slot}.json", {
        "image_sha256": [x[0] for x in subset],
        "cache_sha256": sha256(output / f"shard-{args.slot}.npy"),
        "encoder_parameters": encoder.learned_parameters,
        "elapsed_seconds": time.monotonic() - started,
        "precision": "FP32", "label_inputs": [], "slot": args.slot})


class CachedFeatures:
    def __init__(self, root):
        root = Path(root)
        self.arrays, self.index = [], {}
        for slot in range(2):
            receipt = json.loads((root / f"shard-{slot}.json").read_text())
            if sha256(root / f"shard-{slot}.npy") != receipt["cache_sha256"]:
                raise ValueError("frozen feature cache changed")
            array = np.load(root / f"shard-{slot}.npy", mmap_mode="r")
            self.arrays.append(array)
            for i, digest in enumerate(receipt["image_sha256"]):
                self.index[digest] = (slot, i)
        self.boxes = torch.load(root / "boxes-0.pt", weights_only=True)
        if not torch.equal(self.boxes, torch.load(root / "boxes-1.pt", weights_only=True)):
            raise ValueError("image-only spatial grids disagree")

    def batch(self, rows, device):
        arrays = [self.arrays[slot][index] for slot, index in
                  (self.index[r["image_sha256"]] for r in rows)]
        features = torch.from_numpy(np.stack(arrays)).to(device)
        return features, self.boxes.expand(len(rows), -1, -1).to(device)


def evaluate(model, rows, cache, device, output, max_bytes):
    predictions, terminated = [], []
    for offset in range(0, len(rows), 64):
        batch = rows[offset:offset + 64]
        features, boxes = cache.batch(batch, device)
        texts, eos = predict(model, features, boxes,
                             tuple(r["sample_id"] for r in batch), max_bytes)
        predictions.extend(texts)
        terminated.extend(eos)
    with output.open("w") as stream:
        for row, text, eos in zip(rows, predictions, terminated):
            stream.write(json.dumps({"sample_id": row["sample_id"], "gold": row["text"],
                "prediction": text, "eos": eos, "source_flags": row["flags"]}, ensure_ascii=False) + "\n")
    report = transcription_metrics([r["text"] for r in rows], predictions)
    report["unterminated_regions"] = sum(not x for x in terminated)
    report["prediction_file_sha256"] = sha256(output)
    report["qualification"] = "development_only_not_independent_or_sealed_test"
    return report


def train_worker(args, protocol):
    # This call occurs before optimizer construction, feature reads or updates.
    prefit = require_prefit()
    seed = protocol["seeds"][args.slot]
    seed_all(seed)
    train, validation, _ = corpus(args.data, protocol)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    model = GroundedOCRDecoder(matched_presets()["tac"]).to(device)
    counts = model.counts(frozen_encoder_parameters=protocol["frozen_encoder_parameters"])
    if (counts["trainable"], counts["total_deployed"]) != (1309376, 23121728):
        raise RuntimeError("instantiated parameter counts differ from registration")
    cache = CachedFeatures(args.output / "features")
    output = args.output / f"seed-{seed}"
    output.mkdir(exist_ok=True)
    if (output / "training.jsonl").exists():
        raise ValueError("preserve original fit outputs; resume into a fresh output directory")
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"],
                                 weight_decay=protocol["weight_decay"], foreach=False)
    identity = {"protocol_sha256": sha256(PROTOCOL), "manifest_sha256": sha256(MANIFEST),
                "train_sha256": protocol["train_sha256"], "seed": seed,
                "neural_source_commit": protocol["neural_source_commit"]}
    cursor = {"epoch": 0, "batch": 0}
    counters = {"steps": 0, "examples": 0, "target_bytes_including_eos": 0}
    if args.resume:
        cursor, counters = restore_checkpoint(args.resume / f"seed-{seed}" / "last.pt",
                                               model, optimizer, identity)
    all_batches = [epoch_batches(train, seed, epoch, protocol["batch_size"])
                   for epoch in range(protocol["epochs"])]
    total_steps = sum(map(len, all_batches))
    order = hashlib.sha256()
    # Full schedule is replayable, including the prefix already in a resumed fit.
    for batches in all_batches:
        for batch in batches:
            for i in batch:
                order.update((train[i]["sample_id"] + "\n").encode())
    write_json(output / "registration.json", {"identity": identity, "counts": counts,
        "config": asdict(model.config), "prefit": prefit,
        "schedule_sha256": order.hexdigest(), "scheduled_updates": total_steps})
    started = time.monotonic()
    epoch_start = cursor["epoch"]
    try:
        with (output / "training.jsonl").open("a", buffering=1) as log:
            for epoch in range(epoch_start, protocol["epochs"]):
                model.train()
                for batch_index, indices in enumerate(all_batches[epoch]):
                    if epoch == epoch_start and batch_index < cursor["batch"]:
                        continue
                    if STOP_REQUESTED or time.time() >= args.deadline:
                        raise TimeoutError("registered allocation deadline; no extra quota or fallback")
                    batch = [train[i] for i in indices]
                    features, boxes = cache.batch(batch, device)
                    inputs, labels = targets(batch, device)
                    owners = tuple(r["sample_id"] for r in batch)
                    refs = model.prepare_visual(features, boxes, owners=owners,
                                                page_ids=tuple("region" for _ in owners))
                    optimizer.zero_grad(set_to_none=True)
                    logits, _ = model(inputs, refs, owners=owners)
                    loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("nonfinite training loss")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    if counters["steps"] == 0:
                        write_json(output / "first_backward.json", {
                            "parameter_tensors": len(list(model.parameters())),
                            "missing_gradient_names": [name for name, p in model.named_parameters() if p.grad is None],
                            "zero_gradient_names": [name for name, p in model.named_parameters() if p.grad is not None and not bool(p.grad.any())],
                            "gradient_norm": float(norm), "not_a_capability_measurement": True})
                    lr = learning_rate(counters["steps"], total_steps, len(all_batches[0]), protocol["learning_rate"])
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.step()
                    counters["steps"] += 1
                    counters["examples"] += len(batch)
                    counters["target_bytes_including_eos"] += int((labels != -100).sum())
                    cursor = {"epoch": epoch, "batch": batch_index + 1}
                    log.write(json.dumps({**counters, "epoch": epoch + 1, "loss": float(loss),
                        "learning_rate": lr, "gradient_norm": float(norm)}) + "\n")
                cursor = {"epoch": epoch + 1, "batch": 0}
                atomic_checkpoint(output / "last.pt", model, optimizer, cursor=cursor,
                                  identity=identity, counters=counters)
                # Retain every epoch, including bad checkpoints and failures.
                import shutil
                shutil.copyfile(output / "last.pt", output / f"epoch-{epoch + 1:02d}.pt")
                if epoch + 1 in protocol["development_evaluation_epochs"]:
                    report = evaluate(model, validation, cache, device,
                        output / f"validation-{epoch + 1:02d}.jsonl", protocol["maximum_generated_bytes"])
                    write_json(output / f"validation-{epoch + 1:02d}.json", report)
                    print(json.dumps({"seed": seed, "epoch": epoch + 1, **report}), flush=True)
        write_json(output / "completion.json", {"status": "FIT_COMPLETED_UNQUALIFIED",
            "counters": counters, "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "last_checkpoint_sha256": sha256(output / "last.pt"), "counts": counts,
            "independent_postfit_qualification": "OPEN", "sealed_test_opened": False,
            "training_flops": None, "flops_status": "UNMEASURED_NO_COMPARATIVE_COMPUTE_CLAIM"})
    except BaseException as exc:
        atomic_checkpoint(output / "interrupted.pt", model, optimizer, cursor=cursor,
                          identity=identity, counters=counters)
        write_json(output / "failure.json", {"status": "FAILED_PRESERVED",
            "exception_type": type(exc).__name__, "message": str(exc)[:300],
            "cursor": cursor, "counters": counters})
        raise


def pair(args, phase, deadline):
    children, streams = [], []
    try:
        for slot in range(2):
            stream = (args.output / f"{phase}-{slot}.log").open("w")
            streams.append(stream)
            command = [sys.executable, "-m", "research.tac_ocr003.train", "--phase", phase,
                       "--data", str(args.data), "--output", str(args.output),
                       "--slot", str(slot), "--deadline", str(deadline)]
            if args.resume:
                command += ["--resume", str(args.resume)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot),
                       TAC_OCR003_PHYSICAL_SLOT=str(slot), CUBLAS_WORKSPACE_CONFIG=":4096:8")
            children.append(subprocess.Popen(command, cwd=ROOT, env=env,
                                               stdout=stream, stderr=subprocess.STDOUT))
        while any(child.poll() is None for child in children):
            if time.time() >= deadline or any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError("worker failure/allocation deadline; inspect preserved logs")
            time.sleep(1)
        if any(child.returncode for child in children):
            raise RuntimeError("dual-T4 worker failed")
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
        for stream in streams:
            stream.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("launch", "features", "train"), default="launch")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slot", type=int, choices=(0, 1))
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    if args.phase != "launch" and (args.slot is None or args.deadline is None):
        parser.error("workers require an explicit physical slot and allocation deadline")
    signal.signal(signal.SIGTERM, request_stop)
    args.output.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(2)
    protocol = json.loads(PROTOCOL.read_text())
    # Must fail before CUDA/encoder/optimizer work when external evidence is absent.
    try:
        require_prefit()
    except Exception as exc:
        write_json(args.output / "prefit-blocked.json", {"status": "BLOCKED_BEFORE_FIT",
            "exception_type": type(exc).__name__, "reason": str(exc)[:600],
            "optimizer_updates": 0, "test_opened": False})
        raise
    if args.phase == "features":
        feature_worker(args, protocol)
    elif args.phase == "train":
        train_worker(args, protocol)
    else:
        if (args.output / "runtime.json").exists():
            raise ValueError("preserve original allocation outputs; use a fresh output directory")
        names = verify_two_t4s()
        deadline = time.time() + protocol["workload_wall_seconds_cap"]
        write_json(args.output / "runtime.json", {"physical_gpus": names,
            "strategy": "two_disjoint_feature_shards_then_two_independent_seed_workers",
            "seeds": protocol["seeds"], "internet": "disabled_in_provider_metadata",
            "precision": "FP32", "deadline_unix": deadline})
        pair(args, "features", deadline)
        pair(args, "train", deadline)
        # A later independent assessor must supply checkpoint-bound final evidence.
        # No local DEV score promotes a checkpoint or bypasses --phase final.
        write_json(args.output / "status.json", {"status": "FITS_COMPLETED_POSTFIT_GATE_OPEN",
            "optimizer_runs": 2, "independent_qualification": False})


if __name__ == "__main__":
    main()
