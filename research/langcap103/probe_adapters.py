"""Actual source adapters, with distinct native/GRU/discrete-prefix ownership.

Integer external-history memory is exposed and counted as discrete state;
it is never called stateless or differentiated as if it were a latent vector.
"""
from dataclasses import replace
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
from tac_osm.research_probe import ResearchProbeAdapter


def adapter_for(arm, source_model):
    expected_type = type(source_model)

    def validate(model, sample):
        if type(model) is not expected_type or set(sample) != {"history", "query", "owners"}:
            raise ValueError("Adapter/source model or input contract mismatch")
        x, q = sample["history"], sample["query"]
        if (x.ndim != 2 or q.shape != (x.shape[0], 1) or x.dtype != torch.long
                or q.dtype != torch.long or x.shape[1] < 1
                or len(sample["owners"]) != x.shape[0]
                or len(set(sample["owners"])) != x.shape[0]
                or bool((x < 0).any()) or bool((x > 255).any())
                or bool((q < 0).any()) or bool((q > 255).any())):
            raise ValueError("Observed byte/owner boundary invalid")

    def start(model, sample):
        b = sample["history"].shape[0]
        if arm == "tac_associative_stack":
            return model.start(sample["owners"])
        if arm == "stacked_gru":
            return {"hidden": model.embedding.weight.new_zeros(b, model.recurrent.num_layers,
                                                               model.recurrent.hidden_size)}
        return {"prefix": sample["history"][:, :0].clone()}

    def advance(model, state, sample, t):
        tokens = sample["history"][:, t:t+1]
        if arm == "tac_associative_stack":
            return model(tokens, state=state, owners=sample["owners"], parallel=False)[1]
        if arm == "stacked_gru":
            _, hidden = model.recurrent(model.embedding(tokens), state["hidden"].transpose(0, 1).contiguous())
            return {"hidden": hidden.transpose(0, 1)}
        return {"prefix": torch.cat((state["prefix"], tokens), 1)}

    def forecast(model, state, sample):
        q = sample["query"]
        if arm == "tac_associative_stack":
            logits = model(q, state=state, owners=sample["owners"], parallel=False)[0]
        elif arm == "stacked_gru":
            x, _ = model.recurrent(model.embedding(q), state["hidden"].transpose(0, 1).contiguous())
            logits = model.head(x)
        else:
            logits = model(torch.cat((state["prefix"], q), 1))[:, -1:]
        return {"next_byte_logits": logits}

    def state_views(state):
        if arm == "tac_associative_stack": return state.tensors
        if arm == "stacked_gru": return state
        return {"observed_prefix": state["prefix"]}

    def patch_state(state, name, value):
        if arm == "tac_associative_stack":
            return replace(state, tensors={**state.tensors, name: value})
        if arm == "stacked_gru": return {**state, name: value}
        if (name != "observed_prefix" or value.dtype != torch.long
                or value.shape != state["prefix"].shape
                or bool((value < 0).any()) or bool((value > 255).any())):
            raise ValueError("Invalid original-byte prefix intervention")
        return {"prefix": value}

    def gradients(model, state, sample):
        if arm == "tac_associative_stack":
            leaves = {k: v.detach().clone().requires_grad_(True) for k, v in state.tensors.items()}
            return forecast(model, replace(state, tensors=leaves), sample), leaves
        if arm == "stacked_gru":
            hidden = state["hidden"].detach().clone().requires_grad_(True)
            return forecast(model, {"hidden": hidden}, sample), {"recurrent_hidden": hidden}
        tokens = torch.cat((state["prefix"], sample["query"]), 1)
        embedded = model.embedding(tokens).detach().clone().requires_grad_(True)
        hook = model.embedding.register_forward_hook(lambda _m, _a, _o: embedded)
        try:
            output = model(tokens)[:, -1:]
        finally:
            hook.remove()
        return {"next_byte_logits": output}, {"observed_embedding_coordinates": embedded}

    def batch_item(sample, i):
        return {"history": sample["history"][i:i+1], "query": sample["query"][i:i+1],
                "owners": (sample["owners"][i],)}

    return ResearchProbeAdapter(name="langcap103-"+arm, validate=validate,
        steps=lambda sample: sample["history"].shape[1], start=start, advance=advance,
        forecast=forecast, state_views=state_views, patch_state=patch_state,
        batch_size=lambda sample: sample["history"].shape[0], batch_item=batch_item,
        gradient_graph=gradients, permute_candidates=None)
