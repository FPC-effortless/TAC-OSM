"""Explicit ResearchProbeAdapter for the actual S1MOTION005 native System1 state.

This is a trusted, source-cut adapter to PR #422's ARCHITECTURE-INDEPENDENT
engine. It does not import a native PredictiveStateAgent or infer dimensions
from hardcoded names. Specific observation/action semantics live HERE.

No source target, future action, outcome, reward, hazard label or authority to
act may appear in the sample. Model outputs remain advisory and uncalibrated.
"""
from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from .research_probe import ResearchProbeAdapter
from .variants.system1_velocity_anchored import VelocityAnchoredSystem1, MotionStreamState
from .variants.streaming_continuous_world import WorldStreamState


SOURCE_FIELDS = frozenset({
    "observation_history",
    "prior_logged_action_history",
    "preceding_measured_observation",
})


def validate_system1_source(model: nn.Module, sample: dict) -> None:
    if not isinstance(model, VelocityAnchoredSystem1):
        raise ValueError("System1 observatory requires genuine velocity-anchored System1")
    if type(sample) is not dict or set(sample) != SOURCE_FIELDS:
        raise ValueError("decision-time source allowlist violation: future target/action forbidden")
    obs = sample["observation_history"]
    act = sample["prior_logged_action_history"]
    prior = sample["preceding_measured_observation"]
    if not all(isinstance(v, torch.Tensor) and torch.is_floating_point(v)
               for v in (obs, act, prior)):
        raise ValueError("observations/prior actions must be floating tensors")
    if (obs.ndim != 3 or act.ndim != 3 or prior.ndim != 2
            or obs.shape[0] < 1 or obs.shape[1] < 2
            or obs.shape[-1] != model.core.core.state_dim
            or act.shape != (*obs.shape[:2], model.core.core.action_dim)
            or prior.shape != (obs.shape[0], model.core.core.state_dim)):
        raise ValueError("invalid measured history/prior action alignment")
    first = next(model.parameters())
    for value in (obs, act, prior):
        if (value.dtype != first.dtype or value.device != first.device
                or not bool(torch.isfinite(value).all())):
            raise ValueError("nonfinite/mismatched source tensors")
    if not torch.isfinite(model.core.core.delta_scale).all():
        raise ValueError("invalid model source FIT delta-scale")
    # Source provenance, actor identity and clock cannot be inferred from values:
    # the calling source reader MUST validate independently; until then probes
    # are observational, never proof of no covert data leakage.


def views(state: MotionStreamState) -> dict[str, torch.Tensor]:
    if not isinstance(state, MotionStreamState):
        raise ValueError("wrong System1 state type")
    return {
        "fast_running_sum": state.core_state.fast_sum,
        "associative_memory": state.core_state.memory,
        "previous_measured_observation": state.previous_observation,
    }


def patch_state(state: MotionStreamState, name: str,
                replacement: torch.Tensor) -> MotionStreamState:
    parts = views(state)
    if name not in parts or replacement.shape != parts[name].shape:
        raise ValueError("unrecognized or shape-mismatched System1 patch")
    if (replacement.dtype != parts[name].dtype
            or replacement.device != parts[name].device
            or not torch.isfinite(replacement).all()):
        raise ValueError("invalid diagnostic replacement")
    original = state.core_state
    if name == "fast_running_sum":
        core = replace(original, fast_sum=replacement)
    elif name == "associative_memory":
        core = replace(original, memory=replacement)
    else:
        core = original
    return MotionStreamState(core,
        replacement if name == "previous_measured_observation"
        else state.previous_observation,
        state.has_previous)


def start(model: VelocityAnchoredSystem1, sample: dict) -> MotionStreamState:
    return model.initial_state(
        sample["observation_history"].shape[0],
        previous_observation=sample["preceding_measured_observation"])


def advance(model: VelocityAnchoredSystem1, state: MotionStreamState,
            sample: dict, t: int) -> MotionStreamState:
    # Replay only the genuinely observed prefix; final t remains a forecast
    # query, so a zero-patch lesion affects the final forecast as intended.
    if not 0 <= t < sample["observation_history"].shape[1] - 1:
        raise ValueError("advance past legal source prefix")
    _, next_state = model.step(
        state,
        sample["observation_history"][:, t],
        sample["prior_logged_action_history"][:, t])
    return next_state


def forecast(model: VelocityAnchoredSystem1, state: MotionStreamState,
             sample: dict) -> dict[str, torch.Tensor]:
    # The last source-timestamp observation and *past logged action* are legal;
    # discarding returned next_state is essential for side-effect-free probes.
    estimate, _ = model.step(
        state,
        sample["observation_history"][:, -1],
        sample["prior_logged_action_history"][:, -1])
    if estimate.authorized_to_act or estimate.is_calibrated:
        raise ValueError("unqualified System1 cannot gain policy authority")
    return {
        "next_measured_state_forecast": estimate.next_observation,
        "motion_proxy_logit": estimate.adverse_event_logit,
        "motion_proxy_score": estimate.adverse_event_score,
        "uncalibrated_residual_scale": estimate.residual_scale,
    }


def gradient_graph(model: VelocityAnchoredSystem1, state: MotionStreamState,
                   sample: dict):
    parts = {name: tensor.detach().clone().requires_grad_(True)
             for name, tensor in views(state).items()}
    leaf_state = MotionStreamState(
        WorldStreamState(parts["fast_running_sum"],
                         parts["associative_memory"],
                         state.core_state.count),
        parts["previous_measured_observation"], state.has_previous)
    current = sample["observation_history"][:, -1].detach().clone().requires_grad_(True)
    logged = sample["prior_logged_action_history"][:, -1].detach().clone().requires_grad_(True)
    query = {**sample,
        "observation_history": torch.cat((
            sample["observation_history"][:, :-1].detach(),current[:,None]),dim=1),
        "prior_logged_action_history": torch.cat((
            sample["prior_logged_action_history"][:, :-1].detach(),logged[:,None]),dim=1)}
    return forecast(model,leaf_state,query), {
        "current_measured_observation":current,
        "prior_logged_action":logged,
        **parts,
    }


system1_velocity_adapter = ResearchProbeAdapter(
    name="system1-velocity-observational-v1",
    validate=validate_system1_source,
    steps=lambda sample: sample["observation_history"].shape[1] - 1,
    start=start,
    advance=advance,
    forecast=forecast,
    state_views=views,
    patch_state=patch_state,
    batch_size=lambda sample: sample["observation_history"].shape[0],
    batch_item=lambda sample,i: {
        name:value[i:i+1] for name,value in sample.items()},
    gradient_graph=gradient_graph,
    permute_candidates=None,  # no proposed actions: do NOT fake a permutation
)
