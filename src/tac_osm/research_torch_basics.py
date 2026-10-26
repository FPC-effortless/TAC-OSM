"""PyTorch-native but architecture-independent module and gradient inspection.

No TAC-OSM variant, memory type, head count or loss name is assumed.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
import copy
import math
from typing import Any

import torch
from torch import nn


def architecture_inventory(model: nn.Module, *,
                           invocations: Mapping[str, Callable[[nn.Module], Any]] | None = None) -> dict[str, Any]:
    """Parameter counts and observed leaf-module calls, not an inferred DAG."""
    blocks = defaultdict(lambda: {"trainable": 0, "deployed_learned": 0, "tensors": 0})
    for name, parameter in model.named_parameters(remove_duplicate=True):
        group = name.split(".")[0]
        blocks[group]["trainable"] += parameter.numel() if parameter.requires_grad else 0
        blocks[group]["deployed_learned"] += parameter.numel()
        blocks[group]["tensors"] += 1
    declared = {name: type(module).__name__ for name, module in model.named_modules()}
    phases = {}
    for phase, invoke in (invocations or {}).items():
        replica = copy.deepcopy(model).eval()
        observed = []
        handles = []
        def hook(path):
            def call(_layer, _args, output):
                tensors = []
                def walk(x):
                    if isinstance(x, torch.Tensor):
                        tensors.append({"shape": list(x.shape),
                                        "finite": bool(torch.isfinite(x).all())})
                    elif isinstance(x, (tuple, list)):
                        for child in x:
                            walk(child)
                    elif isinstance(x, Mapping):
                        for child in x.values():
                            walk(child)
                walk(output)
                observed.append({"module": path, "output_tensors": tensors})
            return call
        for path, layer in replica.named_modules():
            if path and not list(layer.children()):
                handles.append(layer.register_forward_hook(hook(path)))
        try:
            with torch.no_grad():
                invoke(replica)
        finally:
            for handle in handles:
                handle.remove()
        phases[phase] = {
            "ordered_leaf_calls": observed,
            "meaning": "runtime call order only; not a computational DAG",
            "unobserved_modules": sorted(set(declared) - {""} -
                                         {item["module"] for item in observed}),
        }
    return {
        "schema": "tac-osm-observatory/architecture-v1",
        "module_type": type(model).__name__,
        "total_trainable": sum(v["trainable"] for v in blocks.values()),
        "total_deployed_learned": sum(v["deployed_learned"] for v in blocks.values()),
        "component_parameters": dict(sorted(blocks.items())),
        "module_types": declared, "phases": phases,
        "qualification": "inspection_only_not_trained_or_proven",
    }


def gradient_conflict_atlas(model: nn.Module,
                            losses: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Measured cosine on parameter intersections; does not step optimizer."""
    if len(losses) < 2:
        raise ValueError("two or more genuine differentiable losses required")
    named = [(name, p) for name, p in model.named_parameters()
             if p.requires_grad]
    if not named:
        raise ValueError("no trainable model parameters")
    values = {}
    for label, loss in losses.items():
        if (not isinstance(loss, torch.Tensor) or loss.numel() != 1
                or not loss.requires_grad or not bool(torch.isfinite(loss).all())):
            raise ValueError(f"loss {label} must be finite differentiable scalar")
        grads = torch.autograd.grad(loss, [p for _, p in named],
                                    retain_graph=True, allow_unused=True)
        values[label] = {name: g.detach().float() if g is not None else None
                         for (name, _), g in zip(named, grads)}
    labels = list(values)
    pairs = []
    for i, left in enumerate(labels):
        for right in labels[i + 1:]:
            groups = sorted({name.split(".")[0] for name, _ in named})
            report = {}
            for group in [*groups, "<all_shared>"]:
                names = [name for name, _ in named
                         if group == "<all_shared>" or name.split(".")[0] == group]
                common = [(values[left][n], values[right][n]) for n in names
                          if values[left][n] is not None and values[right][n] is not None]
                if any(not bool(torch.isfinite(x).all()) or
                       not bool(torch.isfinite(y).all()) for x, y in common):
                    report[group] = {"cosine": None, "status": "NONFINITE_GRADIENT"}
                    continue
                if not common:
                    report[group] = {"cosine": None, "status": "NO_SHARED_GRADIENT"}
                    continue
                dot = sum(float((x * y).sum()) for x, y in common)
                l2 = math.sqrt(sum(float((x * x).sum()) for x, _ in common))
                r2 = math.sqrt(sum(float((y * y).sum()) for _, y in common))
                cosine = max(-1., min(1., dot / (l2 * r2))) if l2 and r2 else None
                report[group] = {
                    "cosine": cosine,
                    "status": "CONFLICT_AT_THIS_POINT" if cosine is not None and cosine < 0
                              else "NO_LOCAL_CONFLICT" if cosine is not None
                              else "ZERO_GRADIENT",
                    "shared_parameter_tensors": len(common),
                    "left_l2": l2, "right_l2": r2,
                }
            pairs.append({"left": left, "right": right, "components": report})
    return {"schema": "tac-osm-observatory/gradient-conflicts-v1",
            "pairs": pairs, "qualification": "local_gradients_not_training_outcome"}
