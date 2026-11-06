"""Bounded CPU NO-FIT comparison of R3 base/full/candidate emission paths.

The random-weight results measure reference implementation overhead only.
They do not measure model intelligence, trained quality, energy, GPU or
mobile inference. The 11/84 parent numerical audit failures remain OPEN.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import statistics
import time

import torch

from scripts.profile_csl125_r2_scales import configs
from tac_osm.csl125_r3 import CSL125R3, CSL125R3Config


def run():
    torch.manual_seed(53)
    torch.set_num_threads(1)
    result = {
        "schema": "woyengi.csl125-r3.cpu-no-fit-screen.v1",
        "device": "github-cpu-or-local-cpu",
        "seed": 53,
        "model_optimizer_steps": 0,
        "trained_quality": "NOT_MEASURED",
        "numerical_11_of_84": "STILL_FAILED",
        "repetitions": 7,
        "warmups": 2,
        "batch": 1,
        "sequence_length": 16,
        "precision": "fp32",
        "full_projection_includes_base_vocab": True,
        "scales": {}
    }
    for scale in ("1m", "10m"):
        cfg = CSL125R3Config(**asdict(configs()[scale]), correction_rank=16)
        model = CSL125R3(cfg).eval()
        tokens = torch.randint(0, cfg.vocab_size, (1, 16), dtype=torch.long)
        arms = (
            ("base", {}),
            ("candidate_k32", {"candidate_repair_topk": 32}),
            ("full_repair", {"emit_repair_logits": True}),
        )
        samples = {name: [] for name, _ in arms}
        with torch.inference_mode():
            for _ in range(2):
                for _, kwargs in arms:
                    model(tokens, depth=cfg.rungs[-1], **kwargs)
            for repeat in range(7):
                for name, kwargs in arms[repeat % 3:] + arms[:repeat % 3]:
                    start = time.perf_counter_ns()
                    outputs, _ = model(tokens, depth=cfg.rungs[-1], **kwargs)
                    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
                    assert cfg.rungs[-1] in outputs
                    if name == "candidate_k32":
                        assert outputs[cfg.rungs[-1]]["repair_candidate_scores"].shape[-1] == 32
                    if name == "full_repair":
                        assert "repair_logits" in outputs[cfg.rungs[-1]]
                    samples[name].append(elapsed_ms)
        result["scales"][scale] = {
            "parameter_count": model.parameter_count(),
            "vocab_size": cfg.vocab_size,
            "width": cfg.width,
            "rank": cfg.correction_rank,
            "layers": cfg.layers,
            "benchmarks": {
                name: {
                    "median_ms": round(statistics.median(times), 4),
                    "min_ms": round(min(times), 4),
                    "max_ms": round(max(times), 4),
                    "samples_ms": [round(v, 4) for v in times],
                }
                for name, times in samples.items()
            },
        }
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    run()
