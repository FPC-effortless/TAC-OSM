"""Frozen post-hoc evaluation of three owner-authorized Transformer fits."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.langbase107.model import CausalTransformer107, TransformerConfig107, parameter_count
from research.langdepth106.source import verify_partition


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cohort(protocol: dict, run_dir: Path) -> list[tuple[int, CausalTransformer107, dict]]:
    rows = []
    receipts = list(run_dir.rglob("FIT_RECEIPT.json"))
    require(len(receipts) == len(protocol["seeds"]), "complete three-fit cohort required")
    for receipt_path in receipts:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        require(receipt["status"] == "FIT_COMPLETED_EXPLORATORY" and receipt["actual_steps"] == 4096,
                "only complete final fits may be evaluated")
        require(receipt["seed"] in protocol["seeds"] and receipt["test_accessed"] is False,
                "fit identity or TRAIN-only boundary differs")
        checkpoint_path = receipt_path.parent / receipt["checkpoint_file"]
        require(sha256(checkpoint_path) == receipt["checkpoint_sha256"], "checkpoint bytes differ from fit receipt")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        require(payload["schema"] == "langbase107.owner-checkpoint.v1" and payload["step"] == 4096,
                "checkpoint schema or step differs")
        model = CausalTransformer107(TransformerConfig107(**protocol["model"]["config"])).float().eval()
        model.load_state_dict(payload["model"], strict=True)
        require(parameter_count(model) == protocol["model"]["trainable_parameters"], "checkpoint parameter count differs")
        rows.append((receipt["seed"], model, receipt))
    require(sorted(seed for seed, _, _ in rows) == protocol["seeds"], "seed cohort duplicated or incomplete")
    return sorted(rows)


@torch.inference_mode()
def score(model: CausalTransformer107, raw: bytes, *, carry: bool) -> dict:
    total_loss = correct = count = 0
    cache = None
    for start in range(0, len(raw) - 1, model.config.context):
        stop = min(start + model.config.context, len(raw) - 1)
        x = torch.tensor(list(raw[start:stop]), dtype=torch.long)[None]
        y = torch.tensor(list(raw[start + 1:stop + 1]), dtype=torch.long)[None]
        logits, cache = model(x, cache=cache if carry else None, use_cache=carry)
        total_loss += float(F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum"))
        correct += int((logits.argmax(-1) == y).sum())
        count += y.numel()
    require(count == len(raw) - 1, "held-out target accounting differs")
    nll = total_loss / count
    return {"targets": count, "nll_nats": nll, "bits_per_byte": nll / math.log(2),
            "perplexity": math.exp(nll), "accuracy": correct / count}


@torch.inference_mode()
def speed(model: CausalTransformer107) -> dict:
    torch.manual_seed(107)
    prompt = torch.randint(0, 256, (1, 128))
    repetitions = []
    cache_bytes = None
    for _ in range(3):
        cache = None
        started = time.perf_counter()
        _, cache = model(prompt, cache=cache, use_cache=True)
        prefill = time.perf_counter() - started
        token = prompt[:, -1:]
        started = time.perf_counter()
        for _ in range(128):
            logits, cache = model(token, cache=cache, use_cache=True)
            token = logits[:, -1].argmax(-1)[:, None]
        decode = time.perf_counter() - started
        cache_bytes = cache.tensor_bytes()
        repetitions.append({"prefill_bytes_per_second": 128 / prefill, "decode_bytes_per_second": 128 / decode})
    return {"prefill_bytes_per_second_median": sorted(x["prefill_bytes_per_second"] for x in repetitions)[1],
            "decode_bytes_per_second_median": sorted(x["decode_bytes_per_second"] for x in repetitions)[1],
            "carried_tensor_bytes": cache_bytes, "repetitions": repetitions}


@torch.inference_mode()
def generate(model: CausalTransformer107, seed: int) -> bytes:
    generator = torch.Generator().manual_seed(10699)
    prompt = b"ROMEO:\n"
    cache = None
    logits = None
    for value in prompt:
        logits, cache = model(torch.tensor([[value]]), cache=cache, use_cache=True)
    output = bytearray(prompt)
    for _ in range(512):
        probabilities = torch.softmax(logits[:, -1] / 0.8, dim=-1)
        token = torch.multinomial(probabilities, 1, generator=generator)
        output.append(int(token))
        logits, cache = model(token, cache=cache, use_cache=True)
    return bytes(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol_path = Path(__file__).with_name("OWNER_PROTOCOL.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(protocol["protected_test_newly_sealed"] is False and
            protocol["architecture_superiority_eligible"] is False,
            "post-hoc evaluation identity changed")
    dev = (args.corpus_dir / "evaluation/dev.bin").read_bytes()
    test = (args.corpus_dir / "evaluation/test.bin").read_bytes()
    verify_partition(dev, "dev")
    verify_partition(test, "test")
    torch.set_num_threads(protocol["fit"]["cpu_threads"])
    cohort = load_cohort(protocol, args.run_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed, model, receipt in cohort:
        sample = generate(model, seed)
        (args.output / f"sample-{seed}.bin").write_bytes(sample)
        (args.output / f"sample-{seed}.txt").write_text(sample.decode("utf-8", errors="replace"), encoding="utf-8")
        rows.append({"seed": seed, "dev_reset": score(model, dev, carry=False),
                     "dev_carry": score(model, dev, carry=True),
                     "test_reset": score(model, test, carry=False),
                     "test_carry": score(model, test, carry=True),
                     "resources": speed(model)})
    for row, (_, _, receipt) in zip(rows, cohort):
        row["checkpoint_sha256"] = receipt["checkpoint_sha256"]
    reset_nll = mean(row["test_reset"]["nll_nats"] for row in rows)
    reset_accuracy = mean(row["test_reset"]["accuracy"] for row in rows)
    carry_nll = mean(row["test_carry"]["nll_nats"] for row in rows)
    tac = json.loads((ROOT / "research/langdepth106_results/SUMMARY.json").read_text(encoding="utf-8"))["arms"]["four"]
    result = {"schema": "langbase107.owner-final-evaluation.v1",
              "status": "THREE_TRANSFORMER_FITS_POSTHOC_EVALUATED_EXPLORATORY",
              "protocol_sha256": sha256(protocol_path), "source_sha256": protocol["source"]["canonical_sha256"],
              "transformer": {"parameters": protocol["model"]["trainable_parameters"],
                  "mean_test_reset_nll_nats": reset_nll,
                  "mean_test_reset_bits_per_byte": reset_nll / math.log(2),
                  "mean_test_reset_accuracy": reset_accuracy,
                  "mean_test_carry_nll_nats": carry_nll,
                  "mean_test_carry_accuracy": mean(row["test_carry"]["accuracy"] for row in rows),
                  "mean_decode_bytes_per_second": mean(row["resources"]["decode_bytes_per_second_median"] for row in rows),
                  "carried_tensor_bytes": rows[0]["resources"]["carried_tensor_bytes"]},
              "frozen_tac_reference": tac,
              "posthoc_transformer_minus_tac_test_reset_nll": reset_nll - tac["test_reset_nll_nats"],
              "posthoc_transformer_minus_tac_test_reset_accuracy": reset_accuracy - tac["test_reset_accuracy"],
              "runs": rows,
              "limitations": ["The TAC TEST result was known before this Transformer control was registered.",
                  "The comparison is parameter- and exposure-matched but complete training FLOPs are not certified matched.",
                  "Tiny Shakespeare is a small public language diagnostic, not general intelligence evidence.",
                  "No independently signed source/reward/goal/access attestation was supplied."],
              "architecture_superiority_eligible": False, "independently_qualified": False}
    (args.output / "RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "test_reset_nll": reset_nll,
                      "test_reset_accuracy": reset_accuracy,
                      "transformer_minus_tac_nll": result["posthoc_transformer_minus_tac_test_reset_nll"]}))


if __name__ == "__main__":
    main()

