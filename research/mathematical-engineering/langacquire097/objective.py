"""Differentiable acquisition-to-query loss; no optimizer or data access.

Teacher forcing is training-only: logit t observes inputs through t and predicts
targets[t]. Query masks are owned by the loss, never fed to the model. Both
objectives compute the full cross-entropy array before selecting supervision.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class Loss097:
    loss: torch.Tensor
    logits: torch.Tensor
    state: object
    acquisition_state: object | None
    supervised_targets: int
    observed_inputs: int
    objective: str


def _mask(mask, inputs, name):
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != inputs.shape or mask.device != inputs.device:
        raise ValueError(name + " must be Bool[B,T] on the input device")
    return mask


def sequence_loss(model, inputs, targets, query_mask, *, objective="query",
                  valid_mask=None, owners=None, acquisition_end=None,
                  retain_acquisition_grad=False):
    if objective not in ("query", "full"):
        raise ValueError("registered full/query objective required")
    if (not isinstance(inputs, torch.Tensor) or inputs.dtype != torch.long
            or inputs.ndim != 2 or inputs.shape[0] < 1 or inputs.shape[1] < 1
            or not isinstance(targets, torch.Tensor) or targets.dtype != torch.long
            or targets.shape != inputs.shape or targets.device != inputs.device
            or bool((targets < 0).any()) or bool((targets > 255).any())):
        raise ValueError("aligned Long[B,T] inputs and byte targets required")
    query_mask = _mask(query_mask, inputs, "query_mask")
    if valid_mask is None:
        valid_mask = torch.ones_like(query_mask)
    valid_mask = _mask(valid_mask, inputs, "valid_mask")
    if (bool((query_mask & ~valid_mask).any())
            or not bool(query_mask.any(dim=1).all())
            or bool((valid_mask[:, 1:] & ~valid_mask[:, :-1]).any())):
        raise ValueError("nonempty query targets and contiguous right-padding required")
    if owners is None:
        owners = tuple(f"train-{i}" for i in range(inputs.shape[0]))
    acquisition = None
    if acquisition_end is None:
        if retain_acquisition_grad:
            raise ValueError("an acquisition split is needed to retain state gradients")
        logits, state = model(inputs, owners=owners)
    else:
        if type(acquisition_end) is not int or not 1 <= acquisition_end < inputs.shape[1]:
            raise ValueError("acquisition split must be inside the observed sequence")
        early, acquisition = model(inputs[:, :acquisition_end], owners=owners)
        if retain_acquisition_grad:
            for tensor in acquisition.tensors.values():
                if tensor.requires_grad:
                    tensor.retain_grad()
        later, state = model(inputs[:, acquisition_end:], state=acquisition, owners=owners)
        logits = torch.cat((early, later), dim=1)
    if logits.shape != (*inputs.shape, 256):
        raise ValueError("model byte-logit geometry differs")
    selected = query_mask if objective == "query" else valid_mask
    losses = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1),
                             reduction="none").reshape_as(targets)
    loss = (losses * selected.to(losses.dtype)).sum() / selected.sum()
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite objective")
    return Loss097(loss, logits, state, acquisition, int(selected.sum()),
                   int(valid_mask.sum()), objective)


def query_loss(model, inputs, targets, query_mask, **kwargs):
    return sequence_loss(model, inputs, targets, query_mask, objective="query", **kwargs)


def full_loss(model, inputs, targets, query_mask, **kwargs):
    return sequence_loss(model, inputs, targets, query_mask, objective="full", **kwargs)


def episode_tensors(acquisition, query, target, *, device=None):
    """Collate one genuine episode for training, with exact next-byte alignment.

    These tensors include teacher-forced Y only for training. An inference
    worker must receive A/Q and generate its own bytes instead of using this.
    """
    for name, value in (("acquisition", acquisition), ("query", query), ("target", target)):
        if (not isinstance(value, (bytes, list, tuple)) or not len(value)
                or any(type(x) is not int or not 0 <= x <= 255 for x in value)):
            raise ValueError(name + " must contain nonempty actual bytes")
    raw = list(acquisition) + list(query) + list(target)
    values = torch.tensor(raw, dtype=torch.long, device=device)[None]
    inputs, targets = values[:, :-1], values[:, 1:]
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    mask[:, len(acquisition) + len(query) - 1:] = True
    return inputs, targets, mask


def acquisition_gradient_receipt(result):
    if result.acquisition_state is None:
        return {"status": "INCONCLUSIVE", "reason": "acquisition state not retained"}
    rows = {}
    for name, tensor in result.acquisition_state.tensors.items():
        gradient = tensor.grad if tensor.requires_grad else None
        rows[name] = {"connected": gradient is not None,
                      "finite": bool(torch.isfinite(gradient).all()) if gradient is not None else None,
                      "l2": float(torch.linalg.vector_norm(gradient.detach())) if gradient is not None else None}
    useful = [v for v in rows.values() if v["connected"] and v["finite"] and v["l2"] > 0]
    nonfinite = any(v["connected"] and not v["finite"] for v in rows.values())
    return {"status": "PASS" if useful and not nonfinite else "FAIL", "state_components": rows,
            "scope": "constructed/local gradient connectivity, not learned future utility"}
