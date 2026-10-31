"""Frozen-checkpoint LANGDEPTH106 evaluation, never model fitting or selection.

TEST is opened only after all six registered final fits pass integrity checks,
with an explicit final-test flag and an exclusive once-only access journal.
Reset-window and continuous-carry scores have separate accounting. Every legal
next-byte target in a partition is counted exactly once, including its tail.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import resource
import statistics
import time
from typing import Any

import torch

from research.langcache104.model import Config104, DeepCacheTAC104, State104
from research.langdepth106.source import SOURCE_BYTES, SOURCE_SHA256, partition_ranges

ROOT = Path(__file__).resolve().parents[2]
STATUS = "OWNER_AUTHORIZED_EXPLORATORY_DEPTH_WIDTH_ALLOCATION_STUDY"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


class ByteMetrics:
    """Sum losses/counts first; never average differently sized chunk means."""
    def __init__(self):
        self.nll_sum_nats = 0.0
        self.correct = 0
        self.targets = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        if logits.ndim != 3 or logits.shape[:2] != targets.shape or logits.shape[-1] != 256:
            raise ValueError("aligned [B,T,256] logits and [B,T] targets required")
        if targets.dtype != torch.long or bool((targets < 0).any()) or bool((targets > 255).any()):
            raise ValueError("raw byte targets required")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("nonfinite prediction logits")
        scores = logits.detach().to(torch.float64)
        losses = torch.logsumexp(scores, dim=-1) - scores.gather(-1, targets[..., None]).squeeze(-1)
        self.nll_sum_nats += float(losses.sum())
        self.correct += int((logits.argmax(-1) == targets).sum())
        self.targets += targets.numel()

    def result(self) -> dict[str, Any]:
        if self.targets < 1:
            raise ValueError("at least one prediction target required")
        nll = self.nll_sum_nats/self.targets
        overflow = nll > math.log(float.fromhex("0x1.fffffffffffffp+1023"))
        return {"nll_sum_nats": self.nll_sum_nats, "target_count": self.targets,
                "correct_count": self.correct, "next_byte_nll_nats": nll,
                "bits_per_byte": nll/math.log(2),
                "perplexity": None if overflow else math.exp(nll),
                "perplexity_overflow": overflow,
                "next_byte_accuracy": self.correct/self.targets}


def windows(partition: bytes, sequence_length: int):
    """Inputs are disjoint; their source+target slices overlap by one byte."""
    if type(partition) is not bytes or len(partition) < 2:
        raise ValueError("a partition must contain at least two raw bytes")
    if type(sequence_length) is not int or sequence_length < 1:
        raise ValueError("positive window length required")
    for start in range(0, len(partition)-1, sequence_length):
        stop = min(start+sequence_length, len(partition)-1)
        inputs = torch.tensor(list(partition[start:stop]), dtype=torch.long)[None]
        targets = torch.tensor(list(partition[start+1:stop+1]), dtype=torch.long)[None]
        yield start, inputs, targets


@torch.inference_mode()
def partition_metrics(model, partition: bytes, *, sequence_length: int = 128,
                      streaming: bool = False, owner: str = "evaluation-partition") -> dict[str, Any]:
    model.eval()
    accumulator = ByteMetrics()
    state = None
    chunk_count = 0
    for _, inputs, targets in windows(partition, sequence_length):
        logits, next_state = model(inputs, state=state if streaming else None,
                                   owners=(owner,), mode="chunk")
        accumulator.update(logits, targets)
        state = next_state if streaming else None
        chunk_count += 1
    result = accumulator.result()
    if result["target_count"] != len(partition)-1:
        raise ValueError("partition target accounting does not cover every legal pair exactly once")
    result.update({"mode": "continuous_carry" if streaming else "reset_windows",
                   "partition_bytes": len(partition), "sequence_length": sequence_length,
                   "chunks": chunk_count, "final_carried_clock": state.clock if streaming else None})
    return result


def physical_state_bytes(state: State104) -> int:
    storages = {}
    for layer in state.layers:
        for tensor in layer.tensors().values():
            storage = tensor.untyped_storage()
            identity = (str(tensor.device), storage.data_ptr())
            storages[identity] = storage.nbytes()
    return sum(storages.values())


@torch.inference_mode()
def serialization_replay(model, state: State104, observed_byte: int = 10) -> dict[str, Any]:
    buffer = io.BytesIO()
    torch.save(state.payload(), buffer)
    buffer.seek(0)
    restored = State104.from_payload(torch.load(buffer, map_location="cpu", weights_only=True))
    token = torch.tensor([[observed_byte]], dtype=torch.long)
    expected, expected_state = model.step(token, state=state, owners=state.owners)
    actual, actual_state = model.step(token, state=restored, owners=restored.owners)
    if not torch.equal(expected, actual):
        raise ValueError("serialized-state next-byte replay differs")
    for left, right in zip(expected_state.layers, actual_state.layers):
        for key in left.tensors():
            if not torch.equal(left.tensors()[key], right.tensors()[key]):
                raise ValueError("serialized-state update replay differs: "+key)
    return {"status": "PASS_EXACT_CPU_REPLAY", "logits_max_abs_difference": 0.0,
            "state_config_signature": state.config_signature,
            "scope": "same loaded model and config; not independent checkpoint authenticity"}


@torch.inference_mode()
def generate_bytes(model, *, prompt: bytes = b"ROMEO:\n", length: int = 512,
                   temperature: float = .8, generator_seed: int = 10699) -> tuple[bytes, dict[str, Any]]:
    if type(prompt) is not bytes or not prompt or type(length) is not int or length < 1:
        raise ValueError("nonempty byte prompt and positive generation length required")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("finite positive sampling temperature required")
    model.eval()
    owner = ("frozen-generation",)
    logits, state = model(torch.tensor(list(prompt), dtype=torch.long)[None], owners=owner)
    generator = torch.Generator(device="cpu").manual_seed(generator_seed)
    generated = bytearray()
    for _ in range(length):
        probabilities = (logits[:, -1]/temperature).softmax(-1)
        token = torch.multinomial(probabilities, 1, generator=generator)
        generated.append(int(token[0, 0]))
        # The sampled byte becomes an observation exactly once. Prompt bytes are
        # never re-fed; the final sampled byte is consumed as well.
        logits, state = model.step(token, state=state, owners=owner)
    if state.clock != len(prompt)+length:
        raise ValueError("generation state-clock accounting mismatch")
    replay = serialization_replay(model, state)
    data = prompt+bytes(generated)
    return data, {"prompt_hex": prompt.hex(), "generated_bytes": length,
                  "total_bytes_including_prompt": len(data), "temperature": temperature,
                  "generator_seed": generator_seed, "vocabulary": "all 256 byte IDs",
                  "sha256": hashlib.sha256(data).hexdigest(),
                  "final_clock": state.clock, "serialization_replay": replay}


@torch.inference_mode()
def timing_profile(model, *, prefill_length: int = 128, decode_steps: int = 64,
                   warmups: int = 1, repetitions: int = 3) -> dict[str, Any]:
    if min(prefill_length, decode_steps, repetitions) < 1 or warmups < 0:
        raise ValueError("positive measured workload dimensions required")
    model.eval()
    owner = ("resource-fixture",)
    fixture = (torch.arange(prefill_length, dtype=torch.long)*17 % 256)[None]
    prefill_times, decode_times = [], []
    final_state = None
    for iteration in range(warmups+repetitions):
        begin = time.perf_counter()
        _, state = model(fixture, owners=owner, mode="chunk")
        prefill_done = time.perf_counter()
        for step in range(decode_steps):
            token = torch.tensor([[(step*29+11) % 256]], dtype=torch.long)
            _, state = model.step(token, state=state, owners=owner)
        decode_done = time.perf_counter()
        if iteration >= warmups:
            prefill_times.append(prefill_done-begin)
            decode_times.append(decode_done-prefill_done)
        final_state = state
    return {"hardware": "CPU", "torch_version": torch.__version__,
            "threads": torch.get_num_threads(), "batch_size": 1,
            "prefill_bytes": prefill_length, "decode_steps": decode_steps,
            "warmups": warmups, "repetitions": repetitions,
            "prefill_seconds": prefill_times, "decode_seconds": decode_times,
            "median_prefill_seconds": statistics.median(prefill_times),
            "median_decode_seconds_per_byte": statistics.median(decode_times)/decode_steps,
            "median_prefill_bytes_per_second": prefill_length/statistics.median(prefill_times),
            "median_decode_bytes_per_second": decode_steps/statistics.median(decode_times),
            "persistent_logical_tensor_bytes": final_state.state_bytes(),
            "persistent_physical_storage_bytes": physical_state_bytes(final_state),
            "process_peak_rss_kib_cumulative": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "rss_scope": "Linux process cumulative high-water mark; not isolated per-model peak",
            "gpu_measured": False, "full_flops_measured": False,
            "workload_scope": "fixed synthetic resource fixture, no capability metric"}


def paired_summary(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    result = {}
    for split in ("dev", "test"):
        for mode in ("reset_windows", "continuous_carry"):
            available = [r for r in rows if split in r["metrics"]]
            if not available:
                continue
            by_key = {(r["arm"], r["seed"]): r["metrics"][split][mode] for r in available}
            paired = []
            for seed in seeds:
                if ("four", seed) not in by_key or ("eight", seed) not in by_key:
                    raise ValueError("incomplete paired seed coverage")
                four, eight = by_key[("four", seed)], by_key[("eight", seed)]
                paired.append({"seed": seed,
                    "nll_eight_minus_four": eight["next_byte_nll_nats"]-four["next_byte_nll_nats"],
                    "accuracy_eight_minus_four": eight["next_byte_accuracy"]-four["next_byte_accuracy"]})
            values = [r["nll_eight_minus_four"] for r in paired]
            result[split+"/"+mode] = {"paired_runs": paired, "run_count": len(paired),
                "mean_nll_eight_minus_four": statistics.mean(values),
                "minimum_nll_difference": min(values), "maximum_nll_difference": max(values),
                "uncertainty_scope": "three initialization/sampling runs on one fixed text split; no byte-level bootstrap or architecture-superiority claim"}
    return result


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    protocol = json.loads(raw)
    if (protocol["study_id"] != "LANGDEPTH106" or protocol["owner_authorized"] is not True or
            protocol["scientific_status"] != STATUS or protocol["seeds"] != [10601, 10602, 10603] or
            protocol["fit"]["steps"] != 4096 or
            protocol["evaluation"]["require_all_six_final_fits"] is not True or
            set(protocol["models"]) != {"four", "eight"} or
            protocol["source"]["canonical_sha256"] != SOURCE_SHA256):
        raise ValueError("not the frozen six-fit LANGDEPTH106 protocol")
    config_fields = set(asdict(Config104()))
    for arm in ("four", "eight"):
        config = protocol["models"][arm]["config"]
        if set(config) != config_fields:
            raise ValueError("complete model configuration required")
        Config104(**config)
    for relative, digest in protocol["implementation"]["source_files_sha256"].items():
        source = (ROOT/relative).resolve()
        if not source.is_relative_to(ROOT) or file_sha256(source) != digest:
            raise ValueError("model implementation differs from frozen protocol: "+relative)
    return protocol, hashlib.sha256(raw).hexdigest()


def audit_checkpoint(path: Path, *, config: dict[str, Any], parameter_count: int,
                     expected_step: int, protocol_sha256: str, source_sha256: str,
                     train_sha256: str, arm: str, seed: int,
                     expected_sample_draws: int) -> tuple[DeepCacheTAC104, dict[str, Any]]:
    """Load only a matching final checkpoint; reject malformed partial evidence."""
    from research.langdepth106.train import load_checkpoint, parameter_digest
    payload = load_checkpoint(path, protocol_sha256=protocol_sha256,
                              source_sha256=source_sha256, arm=arm, seed=seed, config=config)
    required = {"step", "model", "optimizer", "rng_torch", "rng_python", "sampler_rng",
                "sampler_state", "sample_draws", "sample_offset_sha256", "parameter_count",
                "train_sha256", "initialized_parameters_sha256", "trained_parameters_sha256",
                "optimizer_metadata", "scheduler_metadata", "scientific_status", "resume_safe", "status", "smoke"}
    if not required.issubset(payload):
        raise ValueError("checkpoint lacks required training/replay evidence")
    if (type(payload["step"]) is not int or payload["step"] != expected_step or
            payload["parameter_count"] != parameter_count or payload["train_sha256"] != train_sha256 or
            payload["sample_draws"] != expected_sample_draws or payload["scientific_status"] != STATUS or
            payload["resume_safe"] is not True or payload["status"] != "FIT_COMPLETED_EXPLORATORY" or
            payload["smoke"] is not False):
        raise ValueError("checkpoint final-step/count/source/exposure mismatch")
    for name in ("rng_torch", "sampler_rng"):
        value = payload[name]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1 or value.numel() < 1:
            raise ValueError("checkpoint lacks replayable RNG state: "+name)
    if not isinstance(payload["rng_python"], tuple) or not isinstance(payload["sampler_state"], dict):
        raise ValueError("checkpoint lacks Python/sampler replay state")
    optimizer = payload["optimizer"]
    if (not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict) or
            not optimizer["state"] or not isinstance(optimizer.get("param_groups"), list) or
            not optimizer["param_groups"] or not isinstance(payload["optimizer_metadata"], dict) or
            not isinstance(payload["scheduler_metadata"], dict)):
        raise ValueError("checkpoint lacks optimizer/scheduler replay metadata")
    for name in ("sample_offset_sha256", "initialized_parameters_sha256", "trained_parameters_sha256"):
        digest = payload[name]
        if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid training evidence digest: "+name)
    if payload["initialized_parameters_sha256"] == payload["trained_parameters_sha256"]:
        raise ValueError("final checkpoint has unchanged initialization")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = DeepCacheTAC104(Config104(**config)).cpu().eval()
    if parameter_digest(model) != payload["initialized_parameters_sha256"]:
        raise ValueError("recorded initialization differs from declared config/seed")
    parameter_ids = [identifier for group in optimizer["param_groups"] for identifier in group["params"]]
    parameters = list(model.parameters())
    if len(parameter_ids) != len(parameters) or len(set(parameter_ids)) != len(parameter_ids) or set(optimizer["state"]) != set(parameter_ids):
        raise ValueError("optimizer does not cover exactly the actual learned parameters")
    for identifier, parameter in zip(parameter_ids, parameters):
        entry = optimizer["state"][identifier]
        if (not isinstance(entry, dict) or not {"step", "exp_avg", "exp_avg_sq"}.issubset(entry) or
                not isinstance(entry["step"], torch.Tensor) or entry["step"].numel() != 1 or
                float(entry["step"]) != expected_step):
            raise ValueError("optimizer update counter differs from final checkpoint")
        for field in ("exp_avg", "exp_avg_sq"):
            value = entry[field]
            if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape or
                    value.dtype != parameter.dtype or not bool(torch.isfinite(value).all())):
                raise ValueError("optimizer moment shape/precision/finite mismatch")
    model.load_state_dict(payload["model"], strict=True)
    if model.counts()["trainable"] != parameter_count or model.counts()["deployed"] != parameter_count:
        raise ValueError("actual checkpoint model count differs from frozen count")
    if parameter_digest(model) != payload["trained_parameters_sha256"]:
        raise ValueError("checkpoint learned weights do not match recorded digest")
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError("checkpoint contains nonfinite model tensors")
    evidence = {"arm": arm, "seed": seed, "actual_steps": expected_step,
                "parameter_count": parameter_count, "checkpoint_file": Path(path).name,
                "checkpoint_sha256": file_sha256(path),
                "original_parameters_sha256": payload["initialized_parameters_sha256"],
                "final_parameters_sha256": payload["trained_parameters_sha256"],
                "sample_offset_sha256": payload["sample_offset_sha256"],
                "sample_draws": payload["sample_draws"],
                "optimizer_metadata": payload["optimizer_metadata"],
                "scheduler_metadata": payload["scheduler_metadata"],
                "status": "PASS_REGISTERED_FINAL_CHECKPOINT_INTEGRITY",
                "independent_attestation": False}
    return model, evidence


def audit_final_checkpoints(run_dir: Path, protocol: dict[str, Any],
                            protocol_sha256: str) -> list[dict[str, Any]]:
    expected = {(arm, seed) for arm in ("four", "eight") for seed in protocol["seeds"]}
    receipts = {}
    for path in sorted(Path(run_dir).rglob("FIT_RECEIPT.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        key = (receipt.get("arm"), receipt.get("seed"))
        if key not in expected:
            raise ValueError("unexpected fit receipt in checkpoint root: "+str(path))
        if key in receipts:
            raise ValueError("duplicate arm/seed final fit receipts")
        receipts[key] = (path, receipt)
    if set(receipts) != expected:
        raise ValueError("all six final fit receipts must be present before any held-out evaluation")
    models = []
    for arm in ("four", "eight"):
        for seed in protocol["seeds"]:
            receipt_path, receipt = receipts[(arm, seed)]
            config, count = protocol["models"][arm]["config"], protocol["models"][arm]["parameters"]
            steps = protocol["fit"]["steps"]
            checks = {"arm": arm, "seed": seed, "actual_steps": steps, "expected_steps": steps,
                      "protocol_sha256": protocol_sha256, "source_sha256": SOURCE_SHA256,
                      "train_sha256": protocol["source"]["splits"]["TRAIN"]["sha256"],
                      "config": config, "parameter_count": count, "smoke": False,
                      "status": "FIT_COMPLETED_EXPLORATORY", "dev_scored": False, "test_accessed": False,
                      "training_byte_presentations": steps*protocol["fit"]["batch_size"]*protocol["fit"]["sequence_length"]}
            if any(name not in receipt or receipt[name] != value for name, value in checks.items()):
                raise ValueError("final fit receipt differs from frozen contract: "+str(receipt_path))
            filename = receipt["checkpoint_file"]
            if type(filename) is not str or Path(filename).name != filename or filename != f"checkpoint-{steps}.pt":
                raise ValueError("unsafe or nonfinal receipt checkpoint path")
            checkpoint = receipt_path.parent/filename
            if checkpoint.is_symlink() or not checkpoint.is_file() or file_sha256(checkpoint) != receipt["checkpoint_sha256"]:
                raise ValueError("final checkpoint file differs from original fit receipt")
            model, evidence = audit_checkpoint(checkpoint, config=config, parameter_count=count,
                expected_step=steps, protocol_sha256=protocol_sha256, source_sha256=SOURCE_SHA256,
                train_sha256=checks["train_sha256"], arm=arm, seed=seed,
                expected_sample_draws=steps*protocol["fit"]["batch_size"])
            expected_optimizer = {key: protocol["fit"][key] for key in
                ("optimizer", "lr", "betas", "eps", "weight_decay", "clip_grad_norm")}
            expected_scheduler = {key: protocol["fit"][key] for key in
                ("steps", "lr", "min_lr", "warmup_steps")}
            if (evidence["optimizer_metadata"] != expected_optimizer or
                    evidence["scheduler_metadata"] != expected_scheduler):
                raise ValueError("optimizer/scheduler contract differs from frozen protocol")
            for name in ("original_parameters_sha256", "final_parameters_sha256", "sample_offset_sha256", "sample_draws"):
                if receipt[name] != evidence[name]:
                    raise ValueError("checkpoint replay evidence differs from original fit receipt: "+name)
            models.append({"arm": arm, "seed": seed, "model": model,
                           "checkpoint_integrity": evidence, "training_receipt": receipt})
    by_key = {(row["arm"], row["seed"]): row for row in models}
    for seed in protocol["seeds"]:
        if (by_key[("four", seed)]["checkpoint_integrity"]["sample_offset_sha256"] !=
                by_key[("eight", seed)]["checkpoint_integrity"]["sample_offset_sha256"]):
            raise ValueError("paired arms did not receive the same sampled training offsets")
    return models


def verify_evaluation_partition(source_dir: Path, protocol: dict[str, Any], name: str) -> bytes:
    if name not in ("dev", "test"):
        raise ValueError("only declared held-out partitions may be opened")
    receipt = json.loads((Path(source_dir)/"source_receipt.json").read_text(encoding="utf-8"))
    if receipt["source"]["sha256"] != SOURCE_SHA256:
        raise ValueError("source receipt is not pinned Tiny Shakespeare")
    metadata = receipt["partitions"][name]
    expected = protocol["source"]["splits"][name.upper()]
    start, stop = partition_ranges(SOURCE_BYTES)[name]
    if (metadata["sha256"] != expected["sha256"] or metadata["bytes"] != expected["bytes"] or
            metadata["source_byte_start_inclusive"] != start or
            metadata["source_byte_end_exclusive"] != stop or
            metadata["adjacent_source_target_pairs"] != stop-start-1):
        raise ValueError("partition receipt differs from frozen split")
    data = (Path(source_dir)/"evaluation"/(name+".bin")).read_bytes()
    if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
        raise ValueError("held-out partition bytes differ from frozen source")
    return data


def markdown_report(result: dict[str, Any]) -> str:
    lines = ["# LANGDEPTH106 exploratory depth/width results", "",
        "All six fixed-step checkpoints passed source/configuration/weight and replay-evidence checks before scoring.",
        "This is an owner-authorized Tiny Shakespeare experiment; independent research qualification and architectural superiority are not established.", "",
        "| Arm | Seed | Split | Mode | NLL (nats) | Bits/byte | Perplexity | Accuracy | Targets |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in result["runs"]:
        for split, modes in row["metrics"].items():
            for mode, score in modes.items():
                perplexity = "overflow" if score["perplexity"] is None else f'{score["perplexity"]:.4f}'
                lines.append(f'| {row["arm"]} | {row["seed"]} | {split} | {mode} | '
                    f'{score["next_byte_nll_nats"]:.6f} | {score["bits_per_byte"]:.6f} | '
                    f'{perplexity} | {100*score["next_byte_accuracy"]:.3f}% | {score["target_count"]} |')
    lines.extend(["", "Paired differences are eight minus four; negative NLL differences favor the eight-layer allocation on this split.", "",
                  "| Split / mode | Paired runs | Mean NLL difference | Run range |", "| --- | ---: | ---: | --- |"])
    for name, summary in result["paired_summary"].items():
        lines.append(f'| {name} | {summary["run_count"]} | {summary["mean_nll_eight_minus_four"]:.6f} | '
            f'{summary["minimum_nll_difference"]:.6f} to {summary["maximum_nll_difference"]:.6f} |')
    lines.extend(["", "Each target is counted once per score; uncertainty is across three runs on one fixed corpus, not individual bytes.",
        "Widths differ to match weights. Eight layers hold twice as many per-layer cache states. CPU timings, state bytes, training resource receipts and raw generated samples are in RESULT.json and samples/.",
        "GPU speed and complete FLOPs were not measured. Generated text has no alphabet filter; raw .bin files are authoritative and .txt files replace invalid UTF-8 for display.",
        "TEST access: "+("one declared final scoring pass" if result["test_access"]["opened"] else "not opened"), ""])
    return "\n".join(lines)


def run_evaluation(*, protocol_path: Path, source_dir: Path, run_dir: Path,
                   output_dir: Path, authorize_final_test: bool = False) -> dict[str, Any]:
    protocol, protocol_sha256 = load_protocol(protocol_path)
    torch.set_num_threads(protocol["fit"]["cpu_threads"])
    torch.use_deterministic_algorithms(protocol["fit"]["deterministic_algorithms"])
    models = audit_final_checkpoints(run_dir, protocol, protocol_sha256)
    # Nothing below this boundary runs while any registered final fit is absent,
    # partial, mismatched, nonfinite, or missing optimizer/RNG replay evidence.
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"RESULT.json").exists():
        raise FileExistsError("evaluation result already exists; do not silently rescore/select")
    final_test = authorize_final_test is True
    journal = output/"FINAL_TEST_STARTED.json"
    if final_test:
        if protocol["evaluation"]["allow_final_test"] is not True:
            raise ValueError("protocol does not permit final TEST")
        with journal.open("x", encoding="utf-8") as stream:
            json.dump({"protocol_sha256": protocol_sha256, "state": "STARTED",
                "checkpoint_sha256": [r["checkpoint_integrity"]["checkpoint_sha256"] for r in models],
                "policy": "one declared final TEST comparison; no adaptive selection"}, stream, sort_keys=True)
    partitions = {"dev": verify_evaluation_partition(source_dir, protocol, "dev")}
    if final_test:
        partitions["test"] = verify_evaluation_partition(source_dir, protocol, "test")
    samples = output/"samples"
    samples.mkdir(exist_ok=True)
    rows = []
    generation = protocol["evaluation"]["generation"]
    for item in models:
        model, arm, seed = item["model"], item["arm"], item["seed"]
        metrics = {}
        for split, data in partitions.items():
            metrics[split] = {}
            for streaming, mode in ((False, "reset_windows"), (True, "continuous_carry")):
                metrics[split][mode] = partition_metrics(model, data,
                    sequence_length=protocol["fit"]["sequence_length"], streaming=streaming,
                    owner=f"held-out-{split}")
        sample, sample_meta = generate_bytes(model, prompt=generation["prompt"].encode("utf-8"),
            length=generation["bytes"], temperature=generation["temperature"],
            generator_seed=generation["seed"])
        stem = f"{arm}_seed{seed}"
        (samples/(stem+".bin")).write_bytes(sample)
        (samples/(stem+".txt")).write_text(sample.decode("utf-8", errors="replace"), encoding="utf-8")
        sample_meta.update({"raw_file": "samples/"+stem+".bin", "display_file": "samples/"+stem+".txt"})
        rows.append({"arm": arm, "seed": seed, "metrics": metrics,
            "checkpoint_integrity": item["checkpoint_integrity"],
            "training_receipt": item["training_receipt"], "generation": sample_meta,
            "cpu_inference_resource_fixture": timing_profile(model)})
    result = {"schema": "langdepth106.frozen-final-evaluation.v1", "study_id": "LANGDEPTH106",
        "scientific_status": STATUS, "independent_attestation": False,
        "architecture_superiority_eligible": False, "protocol_sha256": protocol_sha256,
        "source_sha256": SOURCE_SHA256, "all_six_final_checkpoint_audit": "PASS",
        "test_access": {"opened": final_test, "explicit_final_flag": final_test,
                        "once_only_journal": journal.name if final_test else None},
        "runs": rows, "paired_summary": paired_summary(rows, protocol["seeds"]),
        "interpretation_limits": protocol["interpretation_limits"]}
    write_json(output/"RESULT.json", result)
    (output/"RESULTS.md").write_text(markdown_report(result), encoding="utf-8")
    if final_test:
        with journal.open("r", encoding="utf-8") as stream:
            record = json.load(stream)
        record.update({"state": "COMPLETED", "result_sha256": file_sha256(output/"RESULT.json")})
        write_json(journal, record)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--source-dir", "--corpus-dir", dest="source_dir", required=True, type=Path)
    parser.add_argument("--run-dir", "--checkpoint-root", dest="run_dir", required=True, type=Path)
    parser.add_argument("--output-dir", "--output", dest="output_dir", required=True, type=Path)
    parser.add_argument("--authorize-final-test", action="store_true")
    args = parser.parse_args()
    result = run_evaluation(protocol_path=args.protocol, source_dir=args.source_dir,
        run_dir=args.run_dir, output_dir=args.output_dir, authorize_final_test=args.authorize_final_test)
    print(json.dumps({"study_id": result["study_id"], "runs": len(result["runs"]),
                      "test_opened": result["test_access"]["opened"],
                      "result_file": str(args.output_dir/"RESULT.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
