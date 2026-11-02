"""Independent executable equations for the six source-pinned LANGCAP103 graphs.

Reads named tensors, never calls a source model's forward or recurrent/attention
module. Torch arithmetic kernels are shared: this is numerical source parity,
not a proof of PyTorch kernels or IEEE arithmetic. No optimizer, scores or fits.
"""
import math
import torch
from torch.nn import functional as F


def linear(x, p, name):
    return F.linear(x, p[name+".weight"], p.get(name+".bias"))


def norm(x, p, name):
    return F.layer_norm(x, (x.shape[-1],), p[name+".weight"], p[name+".bias"], 1e-5)


def tac(tokens, p, *, depth=8, state=None, clock=0):
    """Serial contextual/normalized-bank recurrence, independently unrolled."""
    b, t = tokens.shape; d = p["embedding.weight"].shape[1]
    e = F.embedding(tokens, p["embedding.weight"]).tanh()
    state = {} if state is None else state
    old_lexical = state.get("lexical", e.new_zeros(b, 3, d))
    history = torch.cat((old_lexical, e), 1)
    local = e
    for j in range(3):
        local = local + p["local_lag"][j].tanh() * history[:, 2-j:2-j+t]
    h = local.tanh(); final = {"lexical": history[:, -3:]}
    for layer in range(depth):
        name = f"blocks.{layer}"
        c = p[name+".context_input.bias"].numel()
        v = p[name+".compare.bias"].numel()
        raw_size = p[name+".address_value.bias"].numel()
        k = (raw_size-2*v-2)//4
        context = state.get(f"{layer}.context", e.new_zeros(b, c))
        key = state.get(f"{layer}.key", e.new_zeros(b, 2, k))
        matrix = state.get(f"{layer}.matrix", e.new_zeros(b, 2, v, k))
        mass = state.get(f"{layer}.mass", e.new_zeros(b, 2, k))
        decay = torch.sigmoid(-p[name+".raw_decay"])
        outputs = []
        for time in range(t):
            current, past = h[:, time], context
            context = decay*context + (1-decay)*linear(current, p, name+".context_input").tanh()
            addressed = linear(torch.cat((current, past), -1), p, name+".address_value")
            new_key = addressed[:, :2*k].reshape(b, 2, k).softmax(-1)
            query = addressed[:, 2*k:4*k].reshape(b, 2, k).softmax(-1)
            value = addressed[:, 4*k:4*k+2*v].reshape(b, 2, v).tanh()
            low = e.new_tensor([.97, .995]); high = e.new_tensor([.995, .99995])
            retain = low + (high-low)*addressed[:, -2:].sigmoid()
            if clock+time == 0:
                retain = torch.ones_like(retain)
            matrix = retain[:, :, None, None]*matrix + (1-retain[:, :, None, None])*value[..., None]*key[:, :, None, :]
            mass = retain[:, :, None]*mass + (1-retain[:, :, None])*key
            evidence = (mass*query).sum(-1)
            read = (matrix*query[:, :, None, :]).sum(-1)/(evidence[..., None]+1e-4)
            fast, slow = read.unbind(1)
            comparison = torch.cat((current, past, fast, slow, (fast-slow).abs(), evidence), -1)
            gate = linear(comparison, p, name+".compare").sigmoid()
            mixed = gate*fast + (1-gate)*slow
            features = torch.cat((current, past, fast, slow, mixed, evidence), -1)
            outputs.append(current + linear(norm(features, p, name+".norm"), p, name+".out").tanh())
            key = new_key
        h = torch.stack(outputs, 1)
        final.update({f"{layer}.context": context, f"{layer}.key": key,
                      f"{layer}.matrix": matrix, f"{layer}.mass": mass})
    z = norm(h, p, "head.norm")
    logits = linear(linear(z, p, "head.up")*F.silu(linear(z, p, "head.gate")), p, "head.out")
    return logits, final


def gru(tokens, p, *, depth=4, state=None):
    """PyTorch reset/update/new gate ordering; reset multiplies hidden-new affine."""
    x = F.embedding(tokens, p["embedding.weight"])
    b, t, width = x.shape
    last = []
    for layer in range(depth):
        prefix = f"recurrent."
        hidden = x.new_zeros(b, width) if state is None else state[layer]
        outputs = []
        for time in range(t):
            gi = F.linear(x[:, time], p[prefix+f"weight_ih_l{layer}"], p[prefix+f"bias_ih_l{layer}"])
            gh = F.linear(hidden, p[prefix+f"weight_hh_l{layer}"], p[prefix+f"bias_hh_l{layer}"])
            ir, iz, inn = gi.chunk(3, -1); hr, hz, hn = gh.chunk(3, -1)
            reset = (ir+hr).sigmoid(); update = (iz+hz).sigmoid()
            new = (inn+reset*hn).tanh()
            hidden = (1-update)*new + update*hidden
            outputs.append(hidden)
        x = torch.stack(outputs, 1); last.append(hidden)
    return linear(x, p, "head"), torch.stack(last)


def transformer(tokens, p, *, depth, heads=8):
    """Explicit causal QK-softmax-V attention; no source attention-module call."""
    b, t = tokens.shape; d = p["embedding.weight"].shape[1]; dh = d//heads
    position = torch.arange(t, dtype=p["embedding.weight"].dtype, device=tokens.device)[:, None]
    frequencies = torch.exp(torch.arange(0, d, 2, dtype=position.dtype, device=tokens.device)*(-math.log(10000.)/d))
    phase = position*frequencies
    h = F.embedding(tokens, p["embedding.weight"])*math.sqrt(d)
    h = h + torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)[None]
    for layer in range(depth):
        name = f"layers.{layer}"
        z = norm(h, p, name+".norm1")
        q, k, v = F.linear(z, p[name+".self_attn.in_proj_weight"], p[name+".self_attn.in_proj_bias"]).chunk(3, -1)
        def split(x):
            return x.reshape(b, t, heads, dh).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        scores = (q@k.transpose(-2, -1))/math.sqrt(dh)
        scores = scores.masked_fill(torch.ones(t, t, dtype=torch.bool, device=tokens.device).triu(1), float("-inf"))
        attended = (scores.softmax(-1)@v).transpose(1, 2).reshape(b, t, d)
        h = h + linear(attended, p, name+".self_attn.out_proj")
        h = h + linear(F.gelu(linear(norm(h, p, name+".norm2"), p, name+".linear1")), p, name+".linear2")
    return linear(norm(h, p, "norm"), p, "head"), {}


def forward(arm, tokens, parameters, *, depth, state=None, clock=0):
    if arm == "tac_associative_stack":
        return tac(tokens, parameters, depth=depth, state=state, clock=clock)
    if arm == "stacked_gru":
        return gru(tokens, parameters, depth=depth, state=state)
    if arm == "causal_transformer":
        if state is not None:
            raise ValueError("Transformer reference requires full observed prefix, not hidden-state substitution")
        return transformer(tokens, parameters, depth=depth)
    raise ValueError("Unregistered source graph")
