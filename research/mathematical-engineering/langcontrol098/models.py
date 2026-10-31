"""Source-reviewed ~100K streaming models for LANGCONTROL098.

TAC086, GRU094 and diagonal selective SSM094 are unchanged source classes.
The Transformer is a standard two-layer pre-norm causal attention control with
a differentiable KV cache. Runtime state is measured, not matched or free.
No checkpoint, data access, fitting or evaluation happens on import.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
ARMS = ("tac", "gru", "ssm", "transformer")
PARAMETERS = {"tac": 100521, "gru": 100661, "ssm": 100404, "transformer": 100484}
MAX_TRANSFORMER_CONTEXT = 4096


def _load(name, relative):
    path = ROOT / relative
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path.resolve():
            raise ValueError("source module collision: " + name)
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


SOURCE086 = _load("langcontrol098_source086", "research/mathematical-engineering/langmem086/compact_tac.py")
_prior_path = list(sys.path)
_prior_aliases = {name: sys.modules.get(name) for name in ("compact_tac", "model093")}
try:
    # The preserved historical modules use generic import names. Bind their
    # exact dependencies only during import, then restore caller-owned aliases.
    sys.modules["compact_tac"] = SOURCE086
    SOURCE093 = _load("langcontrol098_source093", "research/mathematical-engineering/langhead093/model093.py")
    sys.modules["model093"] = SOURCE093
    SOURCE094 = _load("langcontrol098_source094", "research/mathematical-engineering/langhead094/model094.py")
finally:
    sys.path[:] = _prior_path
    for _name, _module in _prior_aliases.items():
        if _module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module


@dataclass(frozen=True)
class State098:
    arm: str
    owners: tuple[str, ...]
    clock: int
    tensors: dict[str, torch.Tensor]

    def tensor_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors.values())

    def detached(self):
        """Explicit checkpoint/probe convenience; never used by training loss."""
        return State098(self.arm, self.owners, self.clock,
                        {name: value.detach() for name, value in self.tensors.items()})


def _owners(owners):
    owners = tuple(owners)
    if (not owners or len(set(owners)) != len(owners)
            or any(type(owner) is not str or not owner for owner in owners)):
        raise ValueError("explicit unique nonempty owners required")
    return owners


class Attention098(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(48, 144)
        self.projection = nn.Linear(48, 48)

    def forward(self, x, old_key, old_value):
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def heads(t):
            return t.view(batch, length, 4, 12).transpose(1, 2)
        q, k, v = heads(q), heads(k), heads(v)
        past = old_key.shape[2]
        key = torch.cat((old_key, k), dim=2)
        value = torch.cat((old_value, v), dim=2)
        # True means visible in SDPA. Each query sees exactly its causal prefix.
        mask = (torch.arange(past + length, device=x.device)[None, :]
                <= past + torch.arange(length, device=x.device)[:, None])
        output = F.scaled_dot_product_attention(q, key, value,
                                               attn_mask=mask, dropout_p=0.)
        output = output.transpose(1, 2).contiguous().view(batch, length, 48)
        return self.projection(output), key, value


class TransformerBlock098(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(48)
        self.attention = Attention098()
        self.norm2 = nn.LayerNorm(48)
        self.up = nn.Linear(48, 290)
        self.down = nn.Linear(290, 48)

    def forward(self, x, key, value):
        update, key, value = self.attention(self.norm1(x), key, value)
        x = x + update
        return x + self.down(F.gelu(self.up(self.norm2(x)))), key, value


class Transformer098(nn.Module):
    """Causal attention with actual differentiable per-layer key/value state.

    The cache retains acquired activations without replaying raw acquisition.
    There is no cache eviction, dropout, learned task ID or answer-length input.
    The declared 4,096-byte context limit bounds cache growth and is enforced.
    """
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(256, 48)
        self.blocks = nn.ModuleList([TransformerBlock098() for _ in range(2)])
        self.norm = nn.LayerNorm(48)
        self.readout = nn.Linear(48, 256)

    def start(self, owners):
        owners = _owners(owners)
        tensors = {f"{kind}_{layer}": self.embedding.weight.new_zeros((len(owners), 4, 0, 12))
                   for layer in range(2) for kind in ("key", "value")}
        return State098("transformer", owners, 0, tensors)

    def forward(self, tokens, *, state=None, owners=None):
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
            raise ValueError("observed byte tokens must be Long[B,T]")
        if owners is None:
            owners = tuple(f"train-{i}" for i in range(tokens.shape[0]))
        owners = _owners(owners)
        if state is None:
            state = self.start(owners)
        _validate_tokens_state(self, "transformer", tokens, state, owners)
        batch, length = tokens.shape
        if state.clock + length > MAX_TRANSFORMER_CONTEXT:
            raise ValueError("registered Transformer context limit exceeded")
        expected = {f"{kind}_{layer}" for layer in range(2) for kind in ("key", "value")}
        if (set(state.tensors) != expected or any(
                value.shape != (batch, 4, state.clock, 12)
                for value in state.tensors.values())):
            raise ValueError("Transformer cache shape/clock mismatch")
        x = self.embedding(tokens)
        position = torch.arange(state.clock, state.clock + length,
                                device=x.device, dtype=x.dtype)
        frequency = torch.exp(torch.arange(0, 48, 2, device=x.device, dtype=x.dtype)
                              * (-math.log(10000.) / 48))
        angles = position[:, None] * frequency[None, :]
        positional = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)
        x = x + positional[None]
        tensors = {}
        for i, block in enumerate(self.blocks):
            x, key, value = block(x, state.tensors[f"key_{i}"], state.tensors[f"value_{i}"])
            tensors[f"key_{i}"], tensors[f"value_{i}"] = key, value
        return self.readout(self.norm(x)), State098("transformer", owners,
                                                   state.clock + length, tensors)


def _validate_tokens_state(model, arm, tokens, state, owners):
    if (not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.long
            or tokens.ndim != 2 or tokens.shape[1] < 1
            or tokens.shape[0] != len(owners) or tokens.numel() == 0
            or bool((tokens < 0).any()) or bool((tokens > 255).any())):
        raise ValueError("observed byte tokens must be nonempty Long[B,T]")
    if (not isinstance(state, State098) or state.arm != arm or state.owners != owners
            or type(state.clock) is not int or state.clock < 0):
        raise ValueError("source arm/owner/clock mismatch")
    weight = next(model.parameters())
    if tokens.device != weight.device:
        raise ValueError("tokens and model devices differ")
    if any(not isinstance(t, torch.Tensor) or t.ndim < 1 or t.shape[0] != len(owners)
           or t.dtype != weight.dtype or t.device != weight.device
           or not bool(torch.isfinite(t).all()) for t in state.tensors.values()):
        raise ValueError("state must contain finite batch-leading model-dtype tensors")


class SourceModel098(nn.Module):
    """Metadata/state-layout bridge; source recurrent equations are unchanged."""
    def __init__(self, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("unregistered model family")
        self.arm = arm
        self.core = {"tac": SOURCE086.CompactDualClippedTAC086,
                     "gru": SOURCE094.GRULanguage094,
                     "ssm": SOURCE094.SelectiveSSMLanguage094,
                     "transformer": Transformer098}[arm]()

    def counts(self):
        return {"trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "deployed": sum(p.numel() for p in self.parameters()),
                "fixed_buffer_bytes": sum(b.numel() * b.element_size() for b in self.buffers())}

    def start(self, owners):
        owners = _owners(owners)
        original = self.core.start(owners)
        if self.arm in ("tac", "transformer"):
            tensors = original.tensors
        else:
            value = original.value.transpose(0, 1) if self.arm == "gru" else original.value
            tensors = {"hidden": value}
        return State098(self.arm, owners, original.clock, tensors)

    def forward(self, tokens, *, state=None, owners=None):
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
            raise ValueError("observed byte tokens must be Long[B,T]")
        if owners is None:
            owners = tuple(f"train-{i}" for i in range(tokens.shape[0]))
        owners = _owners(owners)
        if state is None:
            state = self.start(owners)
        _validate_tokens_state(self, self.arm, tokens, state, owners)
        if self.arm == "tac":
            original = SOURCE086.StreamState086(owners, state.clock, state.tensors)
        elif self.arm == "transformer":
            original = state
        else:
            expected = (len(owners), 2, 83) if self.arm == "gru" else (len(owners), 128)
            if set(state.tensors) != {"hidden"} or state.tensors["hidden"].shape != expected:
                raise ValueError("registered recurrent hidden-state shape differs")
            value = state.tensors["hidden"]
            if self.arm == "gru":
                value = value.transpose(0, 1).contiguous()
            original = SOURCE094.Recurrent094State(owners, state.clock, value)
        logits, updated = self.core(tokens, state=original, owners=owners)
        if self.arm in ("tac", "transformer"):
            tensors = updated.tensors
        else:
            value = updated.value.transpose(0, 1) if self.arm == "gru" else updated.value
            tensors = {"hidden": value}
        return logits, State098(self.arm, owners, updated.clock, tensors)

    def intervene(self, state, mode, donors=None):
        if state.arm != self.arm:
            raise ValueError("intervention arm mismatch")
        return intervene(state, mode, donors)


def make_model(arm, seed=None):
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError("integer seed required")
    if seed is None:
        model = SourceModel098(arm)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = SourceModel098(arm)
    counts = model.counts()
    if counts["trainable"] != PARAMETERS[arm] or counts["deployed"] != PARAMETERS[arm]:
        raise ValueError("live active/deployed count differs from registration")
    return model


def intervene(state, mode, donors=None):
    """Intervene on all content, retaining recipient owners and legitimate clock."""
    if not isinstance(state, State098) or mode not in ("carry", "reset", "donor", "corrupt"):
        raise ValueError("registered state intervention required")
    if mode == "donor":
        if donors is None:
            raise ValueError("explicit donor indices required")
        index = torch.as_tensor(donors, device=next(iter(state.tensors.values())).device)
        if index.is_floating_point() or index.is_complex() or index.dtype == torch.bool:
            raise ValueError("donor indices must be integers")
        index = index.to(dtype=torch.long)
        if index.shape != (len(state.owners),) or bool((index < 0).any()) or bool((index >= len(state.owners)).any()):
            raise ValueError("one in-range donor index per recipient required")
    elif donors is not None:
        raise ValueError("donors apply only to donor intervention")
    tensors = {}
    for name, tensor in state.tensors.items():
        if mode == "reset":
            changed = torch.zeros_like(tensor)
        elif mode == "donor":
            changed = tensor.index_select(0, index)
        elif mode == "corrupt":
            changed = tensor.flatten(1).roll(1, dims=1).reshape_as(tensor)
        else:
            changed = tensor.clone()
        tensors[name] = changed
    return State098(state.arm, state.owners, state.clock, tensors)


def source_digests():
    paths = [Path(__file__), Path(SOURCE086.__file__),
             Path(SOURCE093.__file__), Path(SOURCE094.__file__)]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}
