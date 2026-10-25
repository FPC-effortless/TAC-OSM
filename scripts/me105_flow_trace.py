"""No-step objective-flow diagnostic for ME104 and ME105.

Reads only the immutable FIT partition and matched ME103 parent checkpoints.
For each parent, the ME104 registered first batch is passed byte-for-byte to
both model graphs. No optimizer is constructed or stepped; no REPORT/TEST
partition is opened.

Run from the repository root with::

    python scripts/me105_flow_trace.py --output research/mathematical-engineering/me105/FLOW_TRACE.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import me104_split_panel as old_panel
from scripts import me105_protected_panel as new_panel


TERMS = ("nll", "point", "location_mse", "reconstruction_nll", "interval")
EXCLUDED_CORE_PREFIXES = ("current_head.", "width_head.", "median_head.")


def _finite_tensor(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all())


def _gradients(model, parts):
    named = list(model.named_parameters())
    parameters = tuple(parameter for _, parameter in named)
    groups = {
        "core": [i for i, (name, _) in enumerate(named)
                 if not name.startswith(EXCLUDED_CORE_PREFIXES)],
        "current_head": [i for i, (name, _) in enumerate(named)
                          if name.startswith("current_head.")],
        "width_head": [i for i, (name, _) in enumerate(named)
                       if name.startswith("width_head.")],
        "median_head": [i for i, (name, _) in enumerate(named)
                        if name.startswith("median_head.")],
    }
    component_grads = {}
    for term_name in TERMS:
        grads = torch.autograd.grad(
            parts[term_name], parameters, retain_graph=True, allow_unused=True
        )
        if any(gradient is not None and not _finite_tensor(gradient) for gradient in grads):
            raise RuntimeError(f"nonfinite gradient in {term_name}")
        component_grads[term_name] = grads
    norms = {}
    for term_name, grads in component_grads.items():
        norms[term_name] = {}
        for group, indices in groups.items():
            selected = [grads[i].detach() for i in indices if grads[i] is not None]
            norm_sq = sum((gradient.double().square().sum() for gradient in selected),
                          torch.zeros((), dtype=torch.float64))
            norm = float(norm_sq.sqrt())
            norms[term_name][group] = {
                "gradient_tensors": len(selected),
                "nonzero_tensors": sum(bool(torch.count_nonzero(gradient)) for gradient in selected),
                "norm": norm,
                "finite": math.isfinite(norm),
            }
    cosines = {}
    for left_index, left_name in enumerate(TERMS):
        for right_name in TERMS[left_index + 1:]:
            left_parts, right_parts = [], []
            for index in groups["core"]:
                left = component_grads[left_name][index]
                right = component_grads[right_name][index]
                if left is not None and right is not None:
                    left_parts.append(left.detach().reshape(-1).double())
                    right_parts.append(right.detach().reshape(-1).double())
            if not left_parts:
                cosines[f"{left_name}|{right_name}"] = {"cosine": None, "defined": False}
                continue
            left_vector = torch.cat(left_parts)
            right_vector = torch.cat(right_parts)
            denominator = left_vector.norm() * right_vector.norm()
            defined = bool(torch.isfinite(denominator) and denominator > 0)
            cosine = float(torch.dot(left_vector, right_vector) / denominator) if defined else None
            cosines[f"{left_name}|{right_name}"] = {
                "cosine": cosine,
                "defined": defined,
                "shared_parameter_count": sum(
                    parameter.numel() for i, (_, parameter) in enumerate(named)
                    if i in groups["core"]
                ),
            }
    return norms, cosines


def _trace_one(d, rawscale, index):
    old_seed = 104001 + index
    new_seed = 105001 + index
    parent_seed = 102001 + index
    parent_path = old_panel.PARENT / "strong_reconstruction" / f"seed-{parent_seed}" / "final.pt"
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)

    plans = np.load(old_panel.DOC / "FIT_PLANS.npz")
    origins = plans[str(old_seed)][0]
    rng_seed = old_seed + 10000
    batch = old_panel.tensor_batch(d, origins, old_panel.CONTEXTS[0],
                                   np.random.default_rng(rng_seed))
    if any(tensor.device.type != "cpu" for tensor in batch):
        raise RuntimeError("flow trace is expected to run on CPU")
    batch_hash = hashlib.sha256(
        b"".join(tensor.contiguous().numpy().tobytes() for tensor in batch)
    ).hexdigest()

    old_model = old_panel.initialize(d, old_seed)
    old_panel.load_parent(old_model, parent)
    new_model = new_panel.initialize(d, new_seed)
    new_panel.load_parent(new_model, parent)
    old_model.eval()
    new_model.eval()

    models = {}
    with torch.enable_grad():
        for name, model, panel in (
            ("ME104", old_model, old_panel),
            ("ME105", new_model, new_panel),
        ):
            forecast, _, trace = model(*batch[:6], return_trace=True)
            total, parts = panel.objective(model, forecast, trace, batch, rawscale,
                                           "point_guard")
            if not _finite_tensor(total) or any(not _finite_tensor(value) for value in parts.values()):
                raise RuntimeError(f"nonfinite loss component in {name}, parent {parent_seed}")
            gradients, cosines = _gradients(model, parts)
            models[name] = {
                "objective_values": {key: float(value.detach()) for key, value in parts.items()},
                "gradient_norms": gradients,
                "shared_core_pairwise_cosines": cosines,
                "finite_losses": True,
            }
    return {
        "parent_seed": parent_seed,
        "me104_seed": old_seed,
        "me105_seed": new_seed,
        "batch_sha256": batch_hash,
        "batch_size": int(batch[0].shape[0]),
        "context_length": int(batch[0].shape[1]),
        "models": models,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="output JSON path, usually research/mathematical-engineering/me105/FLOW_TRACE.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    # Deliberately load FIT only. Neither panel's score/report APIs are called.
    d = old_panel.fit_data()
    rawscale = torch.tensor(
        np.nanstd(d["raw"][..., :6].astype(np.float64), axis=(0, 1)).clip(1e-4),
        dtype=torch.float32,
    )
    rows = [_trace_one(d, rawscale, index) for index in range(5)]
    if len({row["batch_sha256"] for row in rows}) != 5:
        raise RuntimeError("expected one registered FIT batch for each matched parent")
    report = {
        "kind": "ME105_NO_STEP_OBJECTIVE_FLOW_TRACE",
        "source": "FIT partition only; matched ME103 parents; ME104 registered first batches",
        "optimizer_steps": 0,
        "uses_report_or_test_data": False,
        "component_names": list(TERMS),
        "block_definition": {
            "core": "all named model parameters except current_head, width_head, median_head",
            "pairwise_cosines": "cosines of flattened gradients over common core parameters only",
        },
        "matched_parents": rows,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"ME105_FLOW_TRACE_WRITTEN {output}")


if __name__ == "__main__":
    main()
