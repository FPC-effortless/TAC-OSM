"""No-fit engineering preflight for the LANGBASE107 Transformer control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import torch

from research.langbase107.model import CausalTransformer107, parameter_count


ROOT = Path(__file__).resolve().parents[2]
TAC_PARAMETERS = 998_024


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict:
    torch.set_num_threads(2)
    torch.manual_seed(107)
    model = CausalTransformer107().float().eval()
    count = parameter_count(model)
    if count != 998_816:
        raise ValueError("Transformer parameter geometry changed")
    tokens = (torch.arange(128, dtype=torch.long) * 29 % 256)[None]
    changed = tokens.clone()
    changed[:, 64:] = 255
    with torch.no_grad():
        parallel, _ = model(tokens)
        poisoned, _ = model(changed)
        cache = None
        recurrent_rows = []
        started = time.perf_counter()
        for token in tokens.unbind(1):
            logits, cache = model(token[:, None], cache=cache, use_cache=True)
            recurrent_rows.append(logits)
        decode_seconds = time.perf_counter() - started
    recurrent = torch.cat(recurrent_rows, dim=1)
    future_difference = float((parallel[:, :64] - poisoned[:, :64]).abs().max())
    recurrent_difference = float((parallel - recurrent).abs().max())
    if future_difference > 2e-6 or recurrent_difference > 2e-5:
        raise ValueError("causal or incremental numerical preflight failed")

    model.train()
    logits, _ = model(tokens)
    logits.square().mean().backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    nonfinite = [name for name, parameter in model.named_parameters()
                 if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())]
    if missing or nonfinite:
        raise ValueError("disconnected or nonfinite parameter gradients")
    return {
        "schema": "langbase107.no-fit-transformer-preflight.v1",
        "status": "PASS_ENGINEERING_ONLY_NO_TRAINING",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformer": {
            "architecture": "pre-norm causal decoder, RoPE, dense attention, GELU FFN, tied byte embedding/readout",
            "configuration": {"width": 128, "depth": 4, "heads": 8, "ffn_width": 680, "context": 128},
            "trainable_parameters": count,
            "deployed_parameters": count,
            "parameter_gap_from_tac": count - TAC_PARAMETERS,
            "relative_parameter_gap": abs(count - TAC_PARAMETERS) / max(count, TAC_PARAMETERS),
            "full_cache_tensor_bytes_fp32": cache.tensor_bytes(),
            "full_vs_incremental_max_abs": recurrent_difference,
            "future_poison_prefix_max_abs": future_difference,
            "all_parameter_gradients_connected_finite": True,
            "fixture_decode_bytes_per_second": tokens.numel() / decode_seconds,
        },
        "tac_reference": {
            "architecture": "LANGCACHE104 four-layer width-120 hybrid",
            "trainable_parameters": TAC_PARAMETERS,
            "source_sha256": sha256(ROOT / "research/langcache104/model.py"),
            "kernel_sha256": sha256(ROOT / "research/langcache104/kernels.py"),
        },
        "transformer_source_sha256": sha256(Path(__file__).with_name("model.py")),
        "optimizer_steps": 0,
        "dataset_accessed": False,
        "capability_result": None,
        "complete_training_flops": None,
        "training_blocked_by": [
            "Tiny Shakespeare is a benchmark corpus rather than the policy's required authentic real-outcome primary source",
            "complete comparable training-FLOP accounting is not yet certified",
            "independently signed prefit source/reward/goal/access evidence is absent",
            "the existing Tiny Shakespeare TEST was opened by LANGDEPTH106 and cannot be relabeled as newly sealed",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": [998_024, 998_816],
                      "relative_parameter_gap": result["transformer"]["relative_parameter_gap"]}))


if __name__ == "__main__":
    main()

