"""Owner-authorized exploratory Transformer fitting on TRAIN bytes only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.langbase107.model import CausalTransformer107, TransformerConfig107, parameter_count
from research.langdepth106.source import verify_training_stage
from research.langdepth106.train import OffsetSampler, make_batch


HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_digest(model) -> str:
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def learning_rate(step: int, fit: dict) -> float:
    if step <= fit["warmup_steps"]:
        return fit["lr"] * step / fit["warmup_steps"]
    progress = (step - fit["warmup_steps"]) / (fit["steps"] - fit["warmup_steps"])
    return fit["min_lr"] + 0.5 * (fit["lr"] - fit["min_lr"]) * (1 + math.cos(math.pi * progress))


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_protocol() -> tuple[dict, str]:
    path = HERE / "OWNER_PROTOCOL.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    require(protocol.get("schema") == "langbase107.owner-authorized-transformer-fit.v1", "unexpected protocol")
    require(protocol.get("owner_authorized") is True and protocol.get("independent_attestation") is False,
            "owner-authorized exploration identity changed")
    require(protocol.get("architecture_superiority_eligible") is False and
            protocol.get("protected_test_newly_sealed") is False,
            "post-hoc benchmark may not become qualified superiority evidence")
    return protocol, sha256_file(path)


def checkpoint(path: Path, model, optimizer, sampler, metadata: dict, step: int) -> str:
    payload = {**metadata, "schema": "langbase107.owner-checkpoint.v1", "step": step,
               "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
               "sampler_state": sampler.state_dict(), "parameter_sha256": parameter_digest(model)}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return sha256_file(path)


def fit(args) -> dict:
    protocol, protocol_sha = load_protocol()
    require(args.seed in protocol["seeds"], "unregistered seed")
    require(args.steps == 8 if args.smoke else args.steps is None, "only an eight-update smoke override is allowed")
    require(not args.output.exists(), "refuse to overwrite training evidence")
    receipt = verify_training_stage(args.train_bin.parent)
    require(args.receipt.resolve() == args.train_bin.resolve().parent / "source_receipt.json", "TRAIN stage receipt mismatch")
    raw = args.train_bin.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == protocol["source"]["train_sha256"], "TRAIN bytes changed")
    require(receipt["source"]["sha256"] == protocol["source"]["canonical_sha256"], "source identity changed")
    spec = protocol["fit"]
    torch.set_num_threads(spec["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = CausalTransformer107(TransformerConfig107(**protocol["model"]["config"])).float().train()
    require(parameter_count(model) == protocol["model"]["trainable_parameters"], "model parameter count changed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], betas=tuple(spec["betas"]),
                                  eps=spec["eps"], weight_decay=spec["weight_decay"], foreach=False)
    sampler = OffsetSampler(len(raw), spec["sequence_length"], args.seed)
    train = torch.tensor(list(raw), dtype=torch.long)
    original = parameter_digest(model)
    goal = 8 if args.smoke else spec["steps"]
    args.output.mkdir(parents=True)
    started_wall, started_cpu = time.monotonic(), time.process_time()
    log_path = args.output / "TRAIN_LOG.jsonl"
    checkpoints = []
    first_loss = last_loss = None
    common = {"arm": "causal_transformer", "seed": args.seed,
              "config": protocol["model"]["config"], "parameter_count": parameter_count(model),
              "source_sha256": protocol["source"]["canonical_sha256"],
              "train_sha256": protocol["source"]["train_sha256"], "protocol_sha256": protocol_sha,
              "scientific_status": protocol["scientific_status"], "smoke": args.smoke,
              "initialized_parameters_sha256": original}

    def save(step: int) -> tuple[str, str]:
        name = f"checkpoint-{step}.pt"
        checksum = checkpoint(args.output / name, model, optimizer, sampler, common, step)
        checkpoints.append({"step": step, "file": name, "sha256": checksum})
        return name, checksum

    last_name, last_sha = save(0)
    status = "IN_PROGRESS"
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            for step in range(1, goal + 1):
                require(time.monotonic() - started_wall < protocol["resource_bounds"]["per_fit_max_seconds"],
                        "predeclared fit wall bound exceeded")
                offsets = sampler.draw(spec["batch_size"])
                x, y = make_batch(train, offsets, spec["sequence_length"])
                lr = learning_rate(step, spec)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                require(bool(torch.isfinite(loss)), "nonfinite loss")
                loss.backward()
                require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                        "missing or nonfinite parameter gradient")
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec["clip_grad_norm"], error_if_nonfinite=True)
                optimizer.step()
                first_loss = float(loss.detach()) if first_loss is None else first_loss
                last_loss = float(loss.detach())
                row = {"step": step, "loss": last_loss, "lr": lr, "offsets": offsets,
                       "unclipped_gradient_norm": float(gradient_norm),
                       "wall_seconds": time.monotonic() - started_wall}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                if step in (1, 8, 128, 256, 1024, 2048, 4096):
                    print(json.dumps({"status": "FIT_PROGRESS", "seed": args.seed, **row}), flush=True)
                if step in spec["save_checkpoints_at"] or step == goal:
                    last_name, last_sha = save(step)
        require(parameter_digest(model) != original, "optimizer left initialization unchanged")
        status = "SMOKE_COMPLETED_NOT_COHORT" if args.smoke else "FIT_COMPLETED_EXPLORATORY"
    except Exception:
        status = "FIT_FAILED_PRESERVED"
        raise
    finally:
        wall, cpu = time.monotonic() - started_wall, time.process_time() - started_cpu
        result = {**common, "schema": "langbase107.owner-fit-receipt.v1", "status": status,
                  "actual_steps": sampler.draws // spec["batch_size"], "expected_steps": spec["steps"],
                  "sample_draws": sampler.draws, "sample_offset_sha256": sampler.fingerprint(),
                  "training_byte_presentations": sampler.draws * spec["sequence_length"],
                  "first_fit_loss": first_loss, "last_fit_loss": last_loss,
                  "final_parameters_sha256": parameter_digest(model), "parameters_changed": parameter_digest(model) != original,
                  "checkpoint_file": last_name, "checkpoint_sha256": last_sha, "checkpoints": checkpoints,
                  "wall_seconds": wall, "process_cpu_seconds": cpu,
                  "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                  "hardware": "CPU", "cpu_threads": torch.get_num_threads(), "precision": "float32",
                  "dev_scored": False, "test_accessed": False, "independently_qualified": False,
                  "architecture_superiority": False, "full_training_flops": "NOT_CERTIFIED"}
        atomic_json(args.output / "FIT_RECEIPT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-bin", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    result = fit(args)
    print(json.dumps({key: result[key] for key in ("status", "seed", "actual_steps", "checkpoint_sha256")}))


if __name__ == "__main__":
    main()

