#!/usr/bin/env python3
"""Synthetic fixed-gradient optimizer CPU microbenchmark, NOT model learning evidence."""
import json
import statistics
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from tac_osm.optimizer_tao import TAO

torch.set_num_threads(1)
results = []
for shape in [(32, 32), (64, 128), (128, 128)]:
    torch.manual_seed(20261030)
    original = torch.randn(shape)
    gradients = [torch.randn(shape) for _ in range(20)]
    for mode in ("adamw", "muon", "adamuon", "normuon", "pre_ns"):
        times = []
        for rep in range(3):
            p = torch.nn.Parameter(original.clone())
            group = {"params": [p], "kind": "adamw" if mode == "adamw" else "orthogonal"}
            opt = TAO([group], mode="muon" if mode == "adamw" else mode, lr=0.001)
            for gradient in gradients[:3]:
                p.grad = gradient
                opt.step()
            start = time.perf_counter()
            for gradient in gradients[3:]:
                p.grad = gradient
                opt.step()
            times.append(1000 * (time.perf_counter() - start) / 17)
        results.append({
            "shape": shape, "mode": mode,
            "median_ms_per_step": round(statistics.median(times), 4),
            "range_ms": [round(min(times), 4), round(max(times), 4)],
            "finite": bool(torch.isfinite(p).all()),
        })
print(json.dumps(results, indent=2))
