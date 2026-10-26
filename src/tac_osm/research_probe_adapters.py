"""Explicit adapter for the historical native PredictiveStateAgent contract.

All names, shapes, observation semantics and candidate-axis rules are
localized here. The generic probe module has no dependency on Belief, model
version, fixed memory size, fixed output names or input layout.
"""
from __future__ import annotations

from collections.abc import Mapping
import torch
from torch import nn

from .predictive_state_agent import Belief, PredictiveStateAgent
from .research_probe import ResearchProbeAdapter


def _validate(model: nn.Module, sample: Mapping[str, torch.Tensor]) -> None:
    if not isinstance(model, PredictiveStateAgent):
        raise ValueError("native adapter requires PredictiveStateAgent")
    if not isinstance(sample, Mapping) or set(sample) != {"history", "candidates"}:
        raise ValueError("native decision-time sample must contain ONLY history and candidates")
    h, a = sample["history"], sample["candidates"]
    if not isinstance(h, torch.Tensor) or not isinstance(a, torch.Tensor):
        raise ValueError("native source inputs must be tensors")
    if h.ndim != 3 or a.ndim != 3 or h.shape[0] != a.shape[0] or (
            h.shape[0] < 1 or h.shape[1] < 1 or a.shape[1] < 1 or
            h.shape[2] != model.context_size or a.shape[2] != model.action_size):
        raise ValueError("native source tensor feature/batch dimensions invalid")
    if not torch.is_floating_point(h) or not torch.is_floating_point(a) or (
            not torch.isfinite(h).all() or not torch.isfinite(a).all()):
        raise ValueError("nonfinite or nonfloating native source input")
    device = next(model.parameters()).device
    if h.device != device or a.device != device:
        raise ValueError("native input tensors and model must be on same device")


def _parts(state: Belief) -> dict[str, torch.Tensor]:
    return {"fast": state.fast, "slow": state.slow, "memory": state.memory}


def _patch(state: Belief, name: str, value: torch.Tensor) -> Belief:
    if name not in _parts(state):
        raise ValueError("unknown native state component: " + name)
    if value.shape != _parts(state)[name].shape:
        raise ValueError("invalid patched-state tensor dimensions")
    return Belief(**{part: value if part == name else tensor
                     for part, tensor in _parts(state).items()})


def _forecast(model: PredictiveStateAgent, state: Belief,
              sample: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    outcome, next_observation = model.predict(
        state, sample["history"][:, -1], sample["candidates"])
    return {"outcome": outcome, "next_observation": next_observation}


def _dependency(model: PredictiveStateAgent, state: Belief,
                sample: Mapping[str, torch.Tensor]):
    state_leaves = {name: t.detach().clone().requires_grad_(True)
                    for name, t in _parts(state).items()}
    context = sample["history"][:, -1].detach().clone().requires_grad_(True)
    action = sample["candidates"].detach().clone().requires_grad_(True)
    leaf = Belief(**state_leaves)
    out, nxt = model.predict(leaf, context, action)
    return ({"outcome": out, "next_observation": nxt},
            {"current_context": context, "candidate_action": action,
             "fast_state": leaf.fast, "slow_state": leaf.slow,
             "associative_memory": leaf.memory})


def _permutation(sample: Mapping[str, torch.Tensor]):
    count = sample["candidates"].shape[1]
    index = list(reversed(range(count)))
    changed = {"history": sample["history"],
               "candidates": sample["candidates"][:, index]}
    def restore(outputs):
        return {name: value[:, index] for name, value in outputs.items()}
    return changed, restore


native_predictive_adapter = ResearchProbeAdapter(
    name="native-predictive-state-v1",
    validate=_validate,
    steps=lambda sample: sample["history"].shape[1],
    start=lambda model, sample: model.initial(sample["history"].shape[0]),
    advance=lambda model, state, sample, t:
        model.observe(state, sample["history"][:, t]),
    forecast=_forecast,
    state_views=_parts,
    patch_state=_patch,
    batch_size=lambda sample: sample["history"].shape[0],
    batch_item=lambda sample, index: {
        key: value[index:index + 1] for key, value in sample.items()},
    gradient_graph=_dependency,
    permute_candidates=_permutation,
)
