"""Source-reviewed LANGCONTROL098 byte/state adapter, never a target interface.

The four wrappers expose their actual State098 tensors, including variable-size
Transformer keys and values. Integer byte IDs and clock metadata have no
continuous input derivative; local AD diagnostics explicitly concern acquired
floating state only. Candidate permutation semantics are not applicable.
"""
from dataclasses import replace

import torch

from tac_osm.research_probe import ResearchProbeAdapter


ARMS = ("tac", "gru", "ssm", "transformer")


def adapter098(arm):
    if arm not in ARMS:
        raise ValueError("an exact registered LANGCONTROL098 arm is required")

    def validate(model, sample):
        if getattr(model, "arm", None) != arm:
            raise ValueError("model does not match the source-reviewed arm")
        if not isinstance(sample, dict) or set(sample) != {"tokens", "owners"}:
            raise ValueError("only observed byte IDs and stream owners are legal")
        tokens, owners = sample["tokens"], sample["owners"]
        if (not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.long
                or tokens.ndim != 2 or tokens.shape[1] < 2
                or tokens.shape[0] != len(owners)
                or bool((tokens < 0).any()) or bool((tokens > 255).any())):
            raise ValueError("observed source tokens must be Long[B,T>=2] bytes")
        if (not isinstance(owners, tuple) or len(set(owners)) != len(owners)
                or any(type(owner) is not str or not owner for owner in owners)):
            raise ValueError("explicit unique stream-owner strings are required")
        initial = model.start(owners)
        if initial.arm != arm or initial.owners != owners or initial.clock != 0:
            raise ValueError("source initial-state identity differs")

    def forecast(model, state, sample):
        logits, _ = model(sample["tokens"][:, -1:], state=state,
                          owners=sample["owners"])
        return {"nextbyte_logits": logits[:, 0]}

    def graph(model, state, sample):
        leaves = {name: value.detach().clone().requires_grad_(True)
                  for name, value in state.tensors.items()}
        copied = replace(state, tensors=leaves)
        return forecast(model, copied, sample), leaves

    def patch(state, name, value):
        if name not in state.tensors:
            raise KeyError(name)
        return replace(state, tensors={**state.tensors, name: value})

    return ResearchProbeAdapter(
        name=f"LANGCONTROL098-{arm}-observed-byte-stream",
        validate=validate,
        steps=lambda sample: int(sample["tokens"].shape[1] - 1),
        start=lambda model, sample: model.start(sample["owners"]),
        advance=lambda model, state, sample, t: model(
            sample["tokens"][:, t:t + 1], state=state,
            owners=sample["owners"])[1],
        forecast=forecast,
        state_views=lambda state: state.tensors,
        patch_state=patch,
        batch_size=lambda sample: int(sample["tokens"].shape[0]),
        batch_item=lambda sample, i: {
            "tokens": sample["tokens"][i:i + 1],
            "owners": (sample["owners"][i],)},
        gradient_graph=graph,
        permute_candidates=None,
    )
