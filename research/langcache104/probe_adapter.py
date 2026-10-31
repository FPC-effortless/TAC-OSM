"""Source-specific no-fit ResearchProbeAdapter; no real-data qualification."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from tac_osm.research_probe import ResearchProbeAdapter
from research.langcache104.model import DeepCacheTAC104, LayerState104, State104


def state_views(state):
    return {f"layer.{i}.{name}": tensor for i, layer in enumerate(state.layers)
            for name, tensor in layer.tensors().items()}


def patch_state(state, name, value):
    if name not in state_views(state):
        raise ValueError("unknown source state component")
    _, index, field = name.split(".", 2)
    layers = list(state.layers)
    layers[int(index)] = replace(layers[int(index)], **{field: value})
    return replace(state, layers=tuple(layers))


def adapter_for(model):
    if not isinstance(model, DeepCacheTAC104):
        raise ValueError("LANGCACHE104 source model required")

    def validate(model, sample):
        if not isinstance(model, DeepCacheTAC104) or not isinstance(sample, dict) or set(sample) != {"history", "query", "owners"}:
            raise ValueError("exact source model and observation schema required")
        x, q, owners = sample["history"], sample["query"], sample["owners"]
        if (not isinstance(x, torch.Tensor) or x.dtype != torch.long or x.ndim != 2 or
                not isinstance(q, torch.Tensor) or q.dtype != torch.long or q.shape != (x.shape[0], 1) or
                x.shape[1] < 1 or len(owners) != x.shape[0] or len(set(owners)) != len(owners) or
                any(type(owner) is not str or not owner for owner in owners) or
                bool((x < 0).any()) or bool((x > 255).any()) or
                bool((q < 0).any()) or bool((q > 255).any())):
            raise ValueError("observed bytes and explicit owners required")

    def forecast(model, state, sample):
        logits, _ = model(sample["query"], state=state, owners=sample["owners"], mode="recurrent")
        return {"next_byte_logits": logits}

    def gradient_graph(model, state, sample):
        leaves = {name: value.detach().clone().requires_grad_()
                  for name, value in state_views(state).items() if value.numel()}
        altered = state
        for name, value in leaves.items():
            altered = patch_state(altered, name, value)
        return forecast(model, altered, sample), leaves

    return ResearchProbeAdapter(
        name="langcache104-deep-associative-source",
        validate=validate,
        steps=lambda sample: sample["history"].shape[1],
        start=lambda model, sample: model.start(sample["owners"]),
        advance=lambda model, state, sample, t: model(
            sample["history"][:, t:t+1], state=state, owners=sample["owners"], mode="recurrent")[1],
        forecast=forecast,
        state_views=state_views,
        patch_state=patch_state,
        batch_size=lambda sample: sample["history"].shape[0],
        batch_item=lambda sample, i: {"history": sample["history"][i:i+1],
            "query": sample["query"][i:i+1], "owners": (sample["owners"][i],)},
        gradient_graph=gradient_graph,
        permute_candidates=None,
    )
