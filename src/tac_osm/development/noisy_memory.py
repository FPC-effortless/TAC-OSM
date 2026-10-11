"""Finite-capacity learned noisy-address memory, development-only.

This is a gated delta-rule research adaptation, not a faithful SwiLA or
Gated DeltaNet-2 implementation. Memory has a fixed [B,E,V,K] tensor and no
history cache. Channel-wise erase/write gates are independent. Queries read
all E finite experts, then use learned top-k routing and conditional operators.
No sublinear total-work or modern-baseline superiority is claimed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .conditional_compute import ConditionalExecutor, selected_weights


@dataclass(frozen=True)
class MemoryConfig:
    address_dim: int = 12
    key_dim: int = 24
    value_dim: int = 8
    experts: int = 4
    dim: int = 32
    read_top_k: int = 2
    operators: int = 4
    depth: int = 3
    halt_threshold: float = 0.8
    verifier_threshold: float = 0.6

    def __post_init__(self):
        if min(self.address_dim, self.key_dim, self.value_dim, self.experts, self.dim) < 1:
            raise ValueError("memory dimensions must be positive")
        if not 1 <= self.read_top_k <= self.experts:
            raise ValueError("invalid expert read budget")
        if not 0 < self.verifier_threshold < 1:
            raise ValueError("invalid verifier threshold")


@dataclass(frozen=True)
class WriteEvent:
    step: int
    batch: int
    state_elements: int
    rejected_samples: int
    mean_change: float


class DeltaMemory(nn.Module):
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        e, v, k, a = config.experts, config.value_dim, config.key_dim, config.address_dim
        self.key = nn.Linear(a, e*k)
        self.value = nn.Linear(2, v, bias=False)
        self.erase = nn.Linear(a+2, e*k)
        self.write = nn.Linear(a+2, e*v)
        self.decay = nn.Linear(a+2, e)
        self.assignment = nn.Linear(a, e)
        # Long retention at initialization; trainable rather than an imposed
        # claim that fixed capacity can preserve arbitrary-length histories.
        nn.init.zeros_(self.decay.weight)
        nn.init.constant_(self.decay.bias, 5.0)

    def initial(self, batch: int, reference: Tensor) -> Tensor:
        c = self.config
        return reference.new_zeros((batch, c.experts, c.value_dim, c.key_dim))

    def keys(self, address: Tensor) -> Tensor:
        c = self.config
        return F.normalize(self.key(address).view(-1, c.experts, c.key_dim), dim=-1)

    def update(self, state: Tensor, address: Tensor, value: Tensor,
               gate: Tensor | None = None) -> Tensor:
        c = self.config
        if address.ndim != 2 or address.shape[-1] != c.address_dim:
            raise ValueError("address must be [batch,address_dim]")
        expected = (address.shape[0], c.experts, c.value_dim, c.key_dim)
        if state.shape != expected or value.shape != (address.shape[0],):
            raise ValueError("state/value dimensions incompatible")
        encoded = F.one_hot(value.long(), 2).to(address.dtype)
        joint = torch.cat((address, encoded), dim=-1)
        k = self.keys(address)
        v = self.value(encoded)
        resp = self.assignment(address).softmax(-1)
        erase = self.erase(joint).view(-1, c.experts, c.key_dim).sigmoid()
        write = self.write(joint).view(-1, c.experts, c.value_dim).sigmoid()
        decay = self.decay(joint).sigmoid()
        # Independent key-side erase and value-side write controls.
        erased = torch.einsum('bevk,bek->bev', state, erase*k)
        proposal = decay[..., None, None]*state + resp[..., None, None]*(
            -erased[..., None]*k[:, :, None, :]
            + (write*v[:, None, :])[..., None]*k[:, :, None, :])
        if gate is None:
            return proposal
        if gate.shape != (address.shape[0],) or not torch.isfinite(gate).all():
            raise ValueError("gate must be finite [batch]")
        if ((gate < 0) | (gate > 1)).any():
            raise ValueError("gate outside [0,1]")
        return state + gate[:, None, None, None]*(proposal-state)


class NoisyMemoryModel(nn.Module):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        super().__init__()
        self.config = config
        self.memory = DeltaMemory(config)
        self.query = nn.Sequential(nn.Linear(config.address_dim+2, config.dim), nn.GELU())
        self.read_projection = nn.Linear(config.value_dim, config.dim)
        self.read_selector = nn.Sequential(nn.Linear(2*config.dim, config.dim), nn.GELU(),
                                           nn.Linear(config.dim, 1))
        self.compute = ConditionalExecutor(config.dim, config.operators, 1,
                                           config.depth, config.halt_threshold)
        self.answer = nn.Linear(2*config.dim, 2)
        # Public post-action evidence reliability, not query correctness or
        # the evaluator's target bit, drives this synthetic verifier.
        self.verifier = nn.Sequential(nn.Linear(config.address_dim+3, config.dim), nn.GELU(),
                                      nn.Linear(config.dim, 1))
        self.last_writes: tuple[WriteEvent, ...] = ()

    def encode(self, addresses: Tensor, values: Tensor, *, state: Tensor | None = None) -> Tensor:
        if addresses.ndim != 3 or values.shape != addresses.shape[:2] or addresses.shape[1] < 1:
            raise ValueError("observations must be nonempty [batch,time,dim]/[batch,time]")
        if state is None:
            state = self.memory.initial(addresses.shape[0], addresses)
        events = []
        for t in range(addresses.shape[1]):
            previous = state
            state = self.memory.update(state, addresses[:, t], values[:, t])
            events.append(WriteEvent(t, addresses.shape[0], state[0].numel(), 0,
                                    float((state-previous).detach().abs().mean())))
        self.last_writes = tuple(events)
        return state

    def predict(self, state: Tensor, address: Tensor, current_bit: Tensor) -> Tensor:
        onehot = F.one_hot(current_bit.long(), 2).to(address.dtype)
        query = self.query(torch.cat((address, onehot), dim=-1))
        reads = torch.einsum('bevk,bek->bev', state, self.memory.keys(address))
        views = self.read_projection(reads)
        scores = self.read_selector(torch.cat((views, query[:, None].expand_as(views)), -1)).squeeze(-1)
        indices, weights = selected_weights(scores, self.config.read_top_k, self.training)
        selected = views.gather(1, indices[..., None].expand(-1, -1, views.shape[-1]))
        routed = (selected * weights.gather(1, indices)[..., None]).sum(1)
        computation, _ = self.compute(routed, query)
        return self.answer(torch.cat((computation, query), -1))

    def verified_write(self, state: Tensor, address: Tensor, value: Tensor,
                       evidence: Tensor) -> tuple[Tensor, Tensor]:
        if evidence.shape != (address.shape[0],) or not torch.isfinite(evidence).all():
            raise ValueError("evidence must be finite [batch]")
        encoded = F.one_hot(value.long(), 2).to(address.dtype)
        logits = self.verifier(torch.cat((address, encoded, evidence[:, None]), -1)).squeeze(-1)
        probability = logits.sigmoid()
        gate = probability if self.training else (probability >= self.config.verifier_threshold).to(address.dtype)
        result = self.memory.update(state, address, value, gate)
        self.last_writes = (WriteEvent(0, address.shape[0], state[0].numel(),
                                      int((gate == 0).sum().detach()),
                                      float((result-state).detach().abs().mean())),)
        return result, logits


def parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(bytes(value.detach().cpu().contiguous().view(torch.uint8).flatten().tolist()))
    return digest.hexdigest()


def config_dict(model: NoisyMemoryModel) -> dict:
    return asdict(model.config)
