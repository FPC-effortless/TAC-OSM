"""Owner-authorized exploratory Tiny Shakespeare FIT; never opens DEV or TEST.

The qualified-research attestation gate is unchanged. This runner implements
the separately frozen owner-authorized CPU experiment in PROTOCOL.json.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from research.langcache104.model import Config104, DeepCacheTAC104
from research.langdepth106.source import SOURCE_SHA256, verify_training_stage

CHECKPOINT_SCHEMA = "langdepth106.checkpoint.v1"
SCIENTIFIC_STATUS = "OWNER_AUTHORIZED_EXPLORATORY_DEPTH_WIDTH_ALLOCATION_STUDY"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_digest(model):
    """Hash named learned tensors, counting a tied parameter only once."""
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def peak_rss_bytes():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw*1024)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def learning_rate(step, fit):
    """One-based updates; warmup reaches max LR, final update reaches min LR."""
    require(type(step) is int and 1 <= step <= fit["steps"], "invalid scheduler update")
    warmup = fit["warmup_steps"]
    if step <= warmup:
        return fit["lr"]*step/warmup
    progress = (step-warmup)/(fit["steps"]-warmup)
    return fit["min_lr"] + .5*(fit["lr"]-fit["min_lr"])*(1+math.cos(math.pi*progress))


class OffsetSampler:
    """Independent CPU RNG with a complete replayable legal-offset history."""
    def __init__(self, train_bytes, sequence_length, seed):
        require(type(train_bytes) is int and type(sequence_length) is int and
                sequence_length >= 1 and train_bytes > sequence_length,
                "training partition must contain at least one full source/target window")
        require(type(seed) is int and seed >= 0, "nonnegative sampler seed required")
        self.train_bytes, self.sequence_length, self.seed = train_bytes, sequence_length, seed
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.offsets = []

    def draw(self, batch_size):
        require(type(batch_size) is int and batch_size > 0, "positive batch size required")
        # Exclusive upper bound ensures the last target is still within TRAIN.
        offsets = torch.randint(self.train_bytes-self.sequence_length,
                                (batch_size,), generator=self.generator).tolist()
        self.offsets.extend(offsets)
        return offsets

    @property
    def draws(self):
        return len(self.offsets)

    def fingerprint(self):
        return hashlib.sha256(torch.tensor(self.offsets, dtype=torch.int64).numpy().tobytes()).hexdigest()

    def state_dict(self):
        return {"train_bytes": self.train_bytes, "sequence_length": self.sequence_length,
                "seed": self.seed, "rng": self.generator.get_state(),
                "offsets": torch.tensor(self.offsets, dtype=torch.int64)}

    def load_state_dict(self, state):
        require(isinstance(state, dict) and set(state) == {
            "train_bytes", "sequence_length", "seed", "rng", "offsets"},
            "complete sampler state required")
        for name in ("train_bytes", "sequence_length", "seed"):
            require(state[name] == getattr(self, name), "sampler configuration differs: "+name)
        offsets = state["offsets"]
        require(isinstance(offsets, torch.Tensor) and offsets.dtype == torch.int64 and
                offsets.device.type == "cpu" and offsets.ndim == 1,
                "sampler history must contain CPU int64 offsets")
        require(not offsets.numel() or bool(((offsets >= 0) &
                (offsets < self.train_bytes-self.sequence_length)).all()),
                "sampler history crosses training partition")
        self.generator.set_state(state["rng"])
        self.offsets = offsets.tolist()


def make_batch(train, offsets, sequence_length):
    require(isinstance(train, torch.Tensor) and train.dtype == torch.long and train.ndim == 1,
            "TRAIN byte tensor must be one-dimensional int64")
    rows = []
    for offset in offsets:
        require(type(offset) is int and 0 <= offset and offset+sequence_length < train.numel(),
                "source/target window leaves TRAIN")
        rows.append(train[offset:offset+sequence_length+1])
    joined = torch.stack(rows)
    return joined[:, :-1].contiguous(), joined[:, 1:].contiguous()


def make_optimizer(model, fit):
    return torch.optim.AdamW(model.parameters(), lr=fit["lr"], betas=tuple(fit["betas"]),
                             eps=fit["eps"], weight_decay=fit["weight_decay"], foreach=False)


def optimization_step(model, optimizer, x, y, *, step, fit, owners):
    """One actual next-byte optimizer update with no carried state across draws."""
    model.train()
    lr = learning_rate(step, fit)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(x, owners=owners, mode="chunk")
    require(logits.shape == (*x.shape, 256), "unexpected next-byte output geometry")
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
    require(bool(torch.isfinite(loss)), "nonfinite training loss")
    loss.backward()
    require(all(value.grad is not None and bool(torch.isfinite(value.grad).all())
                for value in model.parameters() if value.requires_grad),
            "disconnected or nonfinite trained parameter gradient")
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), fit["clip_grad_norm"],
                                                  error_if_nonfinite=True)
    optimizer.step()
    require(all(bool(torch.isfinite(value).all()) for value in model.parameters()),
            "optimizer produced nonfinite parameters")
    return {"step": step, "loss": float(loss.detach()), "lr": lr,
            "unclipped_gradient_norm": float(gradient_norm),
            "gradient_clipped": bool(gradient_norm > fit["clip_grad_norm"])}


def load_protocol(path=None):
    path = Path(path) if path is not None else HERE/"PROTOCOL.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    require(protocol.get("schema") == "langdepth106.owner-authorized-depth-comparison.v1",
            "unexpected exploratory protocol schema")
    require(protocol.get("owner_authorized") is True and
            protocol.get("independent_attestation") is False and
            protocol.get("architecture_superiority_eligible") is False and
            protocol.get("scientific_status") == SCIENTIFIC_STATUS,
            "owner exploration must not impersonate independently qualified research")
    require(protocol["source"]["canonical_sha256"] == SOURCE_SHA256,
            "source is not the frozen Tiny Shakespeare corpus")
    fit = protocol["fit"]
    required = {"steps": 4096, "batch_size": 8, "sequence_length": 128,
                "lr": .0003, "min_lr": .00003, "warmup_steps": 128,
                "eps": 1e-8, "weight_decay": .1, "clip_grad_norm": 1.,
                "cpu_threads": 2, "precision": "float32"}
    require(all(fit[key] == expected for key, expected in required.items()),
            "frozen training regime changed")
    require(fit["betas"] == [.9, .95] and fit["deterministic_algorithms"] is True and
            fit["dev_scoring_in_training"] is False and fit["test_access_in_training"] is False,
            "frozen optimizer or TRAIN-only boundary changed")
    require(protocol["seeds"] == [10601, 10602, 10603], "frozen seed matrix changed")
    for relative, expected in protocol["implementation"]["source_files_sha256"].items():
        require(sha256_file(ROOT/relative) == expected, "frozen neural graph changed: "+relative)
    require(set(protocol["models"]) == {"four", "eight"}, "depth matrix changed")
    counts = [entry["parameters"] for entry in protocol["models"].values()]
    require(max(counts)/min(counts)-1 <= .01, "declared parameter parity exceeds one percent")
    return protocol, sha256_file(path)


def load_checkpoint(path, *, protocol_sha256=None, source_sha256=None,
                    arm=None, seed=None, config=None):
    """Safe tensor/primitive deserialization plus explicit identity validation."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    needed = {"schema", "model", "optimizer", "rng_torch", "rng_python", "sampler_rng", "smoke",
              "sampler_state", "step", "arm", "seed", "config", "parameter_count",
              "protocol_sha256", "source_sha256", "train_sha256", "scientific_status",
              "initialized_parameters_sha256", "trained_parameters_sha256", "sample_draws",
              "sample_offset_sha256", "resume_safe", "optimizer_metadata", "scheduler_metadata"}
    require(isinstance(payload, dict) and needed <= set(payload), "incomplete checkpoint")
    require(type(payload["smoke"]) is bool, "explicit checkpoint smoke identity required")
    require(payload["schema"] == CHECKPOINT_SCHEMA and
            payload["scientific_status"] == SCIENTIFIC_STATUS,
            "checkpoint scientific or schema identity differs")
    for key, expected in (("protocol_sha256", protocol_sha256), ("source_sha256", source_sha256),
                          ("arm", arm), ("seed", seed), ("config", config)):
        require(expected is None or payload[key] == expected, "checkpoint identity differs: "+key)
    require(type(payload["step"]) is int and 0 <= payload["step"] <= 4096,
            "checkpoint update index invalid")
    require(isinstance(payload["model"], dict) and all(isinstance(name, str) and
            isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
            for name, value in payload["model"].items()), "invalid or nonfinite model tensors")
    require(isinstance(payload["optimizer"], dict) and
            {"state", "param_groups"} <= set(payload["optimizer"]), "missing optimizer state")
    require(torch.equal(payload["sampler_rng"], payload["sampler_state"]["rng"]),
            "sampler RNG aliases disagree")
    sampler = OffsetSampler(payload["sampler_state"]["train_bytes"],
                            payload["sampler_state"]["sequence_length"], payload["seed"])
    sampler.load_state_dict(payload["sampler_state"])
    require(sampler.draws == payload["sample_draws"] and
            sampler.fingerprint() == payload["sample_offset_sha256"], "sample replay identity differs")
    torch.Generator().set_state(payload["rng_torch"])
    random.Random().setstate(payload["rng_python"])
    return payload


def checkpoint_payload(model, optimizer, sampler, *, step, metadata, resume_safe=True):
    sampler_state = sampler.state_dict()
    return {**metadata, "schema": CHECKPOINT_SCHEMA, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "rng_torch": torch.get_rng_state(),
            "rng_python": random.getstate(), "sampler_rng": sampler_state["rng"],
            "sampler_state": sampler_state, "step": step,
            "trained_parameters_sha256": parameter_digest(model), "sample_draws": sampler.draws,
            "sample_offset_sha256": sampler.fingerprint(), "resume_safe": resume_safe}


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return sha256_file(path)


def restore_runtime(payload, model, optimizer, sampler):
    require(payload["resume_safe"] is True, "failure occurred inside an update; unsafe resume")
    require(payload["config"] == asdict(model.config), "resume model geometry differs")
    model.load_state_dict(payload["model"], strict=True)
    require(parameter_digest(model) == payload["trained_parameters_sha256"],
            "loaded learned tensors differ from saved digest")
    optimizer.load_state_dict(payload["optimizer"])
    sampler.load_state_dict(payload["sampler_state"])
    torch.set_rng_state(payload["rng_torch"])
    random.setstate(payload["rng_python"])
    for value in optimizer.state.values():
        require(int(value["step"].item()) == payload["step"] and
                bool(torch.isfinite(value["exp_avg"]).all()) and
                bool(torch.isfinite(value["exp_avg_sq"]).all()), "invalid optimizer replay state")


def verify_readback(path, model, fit, probe):
    payload = load_checkpoint(path, config=asdict(model.config))
    # Building a replay model must not perturb the live training RNG.
    with torch.random.fork_rng(devices=[]):
        replica = DeepCacheTAC104(model.config).float().eval()
        optimizer = make_optimizer(replica, fit)
        sampler = OffsetSampler(payload["sampler_state"]["train_bytes"],
                                payload["sampler_state"]["sequence_length"], payload["seed"])
        prior_python = random.getstate()
        try:
            restore_runtime(payload, replica, optimizer, sampler)
            mode = model.training
            model.eval()
            try:
                with torch.no_grad():
                    a = model(probe, owners=("checkpoint-replay",))[0]
                    b = replica(probe, owners=("checkpoint-replay",))[0]
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            finally:
                model.train(mode)
        finally:
            random.setstate(prior_python)
    return {"checkpoint_tensors_reloaded": True, "optimizer_reloaded": True,
            "sampler_rng_reloaded": True, "fixed_TRAIN_probe_logits_exact": True}


def fit(args):
    protocol, protocol_sha = load_protocol()
    require(args.arm in protocol["models"] and args.seed in protocol["seeds"],
            "unregistered depth or seed")
    require(args.steps == 8 if args.smoke else args.steps is None,
            "only explicit standalone eight-update smoke may override full duration")
    require(not args.smoke or args.resume is None, "smoke is independent, not a continuation")
    maximum = protocol["resource_bounds"]["per_fit_max_seconds"]
    require(0 < args.max_seconds <= maximum, "wall-time bound exceeds frozen resource policy")
    require(not args.output.exists(), "refuse overwrite of existing training evidence")
    train_path, receipt_path = args.train_bin.resolve(), args.receipt.resolve()
    require(train_path.name == "train.bin" and receipt_path == train_path.parent/"source_receipt.json",
            "TRAIN bytes and metadata must occupy the isolated source stage")
    receipt = verify_training_stage(train_path.parent)
    require(receipt["source"]["sha256"] == protocol["source"]["canonical_sha256"],
            "canonical corpus identity differs")
    raw = train_path.read_bytes()  # The only corpus payload this runner opens.
    train_sha = hashlib.sha256(raw).hexdigest()
    require(train_sha == protocol["source"]["splits"]["TRAIN"]["sha256"],
            "TRAIN payload differs from frozen protocol")
    fit_spec = protocol["fit"]
    torch.set_num_threads(fit_spec["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    config = Config104(**protocol["models"][args.arm]["config"])
    model = DeepCacheTAC104(config).float()
    counts = model.counts()
    require(counts["trainable"] == protocol["models"][args.arm]["parameters"] and
            counts["deployed"] == counts["trainable"], "actual learned parameter count differs")
    optimizer = make_optimizer(model, fit_spec)
    sampler = OffsetSampler(len(raw), fit_spec["sequence_length"], args.seed)
    train = torch.tensor(list(raw), dtype=torch.long)
    original_digest = parameter_digest(model)
    step, previous_wall, previous_cpu = 0, 0., 0.
    resumed = None
    if args.resume is not None:
        resumed = load_checkpoint(args.resume, protocol_sha256=protocol_sha,
            source_sha256=SOURCE_SHA256, arm=args.arm, seed=args.seed, config=asdict(config))
        require(resumed["train_sha256"] == train_sha, "resume TRAIN partition differs")
        require(resumed["smoke"] is False, "standalone smoke checkpoint cannot become a cohort continuation")
        require(resumed["initialized_parameters_sha256"] == original_digest,
                "seeded original initialization differs from checkpoint lineage")
        restore_runtime(resumed, model, optimizer, sampler)
        step = resumed["step"]
        require(sampler.draws == step*fit_spec["batch_size"], "resume sampler/update exposure mismatch")
        previous_wall = resumed.get("training_wall_seconds", 0.)
        previous_cpu = resumed.get("training_process_cpu_seconds", 0.)
    goal = 8 if args.smoke else fit_spec["steps"]
    require(step < goal, "checkpoint already reached declared duration")
    args.output.mkdir(parents=True, exist_ok=False)
    started_wall, started_cpu = time.monotonic(), time.process_time()
    owners = tuple(f"train-seed-{args.seed}-slot-{i}" for i in range(fit_spec["batch_size"]))
    common = {"arm": args.arm, "seed": args.seed, "config": asdict(config), "smoke": args.smoke,
              "parameter_count": counts["trainable"], "protocol_sha256": protocol_sha,
              "source_sha256": SOURCE_SHA256, "train_sha256": train_sha,
              "source_receipt_sha256": sha256_file(receipt_path),
              "scientific_status": SCIENTIFIC_STATUS,
              "initialized_parameters_sha256": original_digest,
              "implementation_sha256": protocol["implementation"]["source_files_sha256"],
              "optimizer_metadata": {key: fit_spec[key] for key in
                  ("optimizer", "lr", "betas", "eps", "weight_decay", "clip_grad_norm")},
              "scheduler_metadata": {key: fit_spec[key] for key in
                  ("steps", "lr", "min_lr", "warmup_steps")}}
    checkpoints = []
    last_path, last_sha, replay = None, None, None
    last_loss, first_loss = None, None
    status, in_update = "IN_PROGRESS", False
    log_path = args.output/"TRAIN_LOG.jsonl"

    def elapsed():
        return (previous_wall+time.monotonic()-started_wall,
                previous_cpu+time.process_time()-started_cpu)

    def preserve(name, *, safe=True, readback=True):
        wall, cpu = elapsed()
        payload = checkpoint_payload(model, optimizer, sampler, step=step,
            metadata={**common, "training_wall_seconds": wall,
                      "training_process_cpu_seconds": cpu, "status": status}, resume_safe=safe)
        path = args.output/name
        checksum = save_checkpoint(path, payload)
        result = verify_readback(path, model, fit_spec, train[:fit_spec["sequence_length"]][None]) if readback else None
        checkpoints.append({"step": step, "file": name, "sha256": checksum, "readback": result})
        return path, checksum, result

    def write_receipt(error=None):
        wall, cpu = elapsed()
        result = {**common, "schema": "langdepth106.fit-receipt.v1", "status": status,
                  "actual_steps": step, "expected_steps": fit_spec["steps"], "smoke": args.smoke,
                  "sample_draws": sampler.draws, "sample_offset_sha256": sampler.fingerprint(),
                  "training_byte_presentations": step*fit_spec["batch_size"]*fit_spec["sequence_length"],
                  "original_parameters_sha256": original_digest,
                  "final_parameters_sha256": parameter_digest(model),
                  "parameters_changed": parameter_digest(model) != original_digest,
                  "checkpoint_file": last_path.name if last_path else None,
                  "checkpoint_sha256": last_sha, "checkpoint_readback": replay,
                  "checkpoints": checkpoints, "first_fit_loss": first_loss, "last_fit_loss": last_loss,
                  "wall_seconds": wall, "process_cpu_seconds": cpu, "peak_rss_bytes": peak_rss_bytes(),
                  "torch_version": str(torch.__version__), "python_version": sys.version,
                  "hardware": "CPU", "precision": "float32", "cpu_threads": torch.get_num_threads(),
                  "optimizer_updates": step, "state_policy": "RESET_EACH_INDEPENDENT_TRAIN_WINDOW",
                  "dev_scored": False, "test_accessed": False, "independently_qualified": False,
                  "architecture_superiority": False, "full_training_flops": "NOT_MEASURED",
                  "resumed_from_checkpoint_sha256": sha256_file(args.resume) if args.resume else None,
                  "error": error}
        atomic_json(args.output/"FIT_RECEIPT.json", result)
        return result

    try:
        last_path, last_sha, replay = preserve(f"checkpoint-{step}.pt")
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            while step < goal:
                require(elapsed()[0] < args.max_seconds, "predeclared cumulative wall budget exhausted")
                in_update = True
                offsets = sampler.draw(fit_spec["batch_size"])
                x, y = make_batch(train, offsets, fit_spec["sequence_length"])
                row = optimization_step(model, optimizer, x, y, step=step+1, fit=fit_spec, owners=owners)
                step += 1
                in_update = False
                last_loss = row["loss"]
                if first_loss is None:
                    first_loss = last_loss
                row.update({"offsets": offsets, "sample_draws": sampler.draws,
                            "training_byte_presentations": step*x.numel(),
                            "wall_seconds": elapsed()[0], "process_cpu_seconds": elapsed()[1]})
                log.write(json.dumps(row, allow_nan=False)+"\n")
                if step in (1, 8, 64, 128, 256, 512, 1024, 2048, 4096):
                    print(json.dumps({"status": "SMOKE_PROGRESS" if args.smoke else "FIT_PROGRESS",
                        "arm": args.arm, "seed": args.seed, "step": step,
                        "loss": last_loss, "lr": row["lr"], "wall_seconds": elapsed()[0]}), flush=True)
                if step == goal:
                    require(sampler.draws == goal*fit_spec["batch_size"], "actual exposure differs from declared fit")
                    require(parameter_digest(model) != original_digest, "optimizer left initialization unchanged")
                    status = "SMOKE_COMPLETED_NOT_COHORT" if args.smoke else "FIT_COMPLETED_EXPLORATORY"
                if step in fit_spec["save_checkpoints_at"] or step == goal:
                    last_path, last_sha, replay = preserve(f"checkpoint-{step}.pt")
        require(sampler.draws == goal*fit_spec["batch_size"], "actual exposure differs from declared fit")
        require(parameter_digest(model) != original_digest, "optimizer left initialization unchanged")
        status = "SMOKE_COMPLETED_NOT_COHORT" if args.smoke else "FIT_COMPLETED_EXPLORATORY"
        result = write_receipt()
    except Exception as error:
        status = "FIT_FAILED_PRESERVED"
        try:
            last_path, last_sha, replay = preserve("checkpoint-failed.pt", safe=not in_update, readback=False)
        finally:
            write_receipt({"type": type(error).__name__, "message": str(error),
                           "resume_safe": not in_update})
        raise
    print(json.dumps({key: result[key] for key in
        ("status", "arm", "seed", "actual_steps", "checkpoint_sha256")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-bin", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--arm", choices=("four", "eight"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-seconds", type=float, default=2700.)
    fit(parser.parse_args())


if __name__ == "__main__":
    main()
