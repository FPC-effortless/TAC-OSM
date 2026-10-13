"""DEV-010: CED-style query-only reader over TAC-OSM's real finite PLM tensor.

* Existing learned DeltaMemory write/erase/decay and 768-scalar persistent state.
* Existing conditional structural executor and binary decision head.
* The expert-slot set encoder is permutation-equivariant (not a temporal encoder).
  Temporal causality is enforced by the inherited streaming memory writer.
* Dense and sparse modes are inference ablations on IDENTICAL trained weights.
* This is experimental, not DeepSeek's CED, a verified world model or an
  efficient/sublinear external-memory solution.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .protected_memory import ProtectionModel
from .noisy_memory import MemoryConfig


@dataclass(frozen=True)
class ReaderWork:
    batch: int
    experts: int
    encoder_score_pairs: int
    first_decoder_score_pairs: int
    second_decoder_score_pairs: int
    global_kv_scalars: int
    memory_state_scalars_per_episode: int

    @property
    def total_score_pairs(self) -> int:
        return self.encoder_score_pairs + self.first_decoder_score_pairs + self.second_decoder_score_pairs


class FiniteCEDRead(nn.Module):
    """Encoder-projected shared KV; two query-only decoder reads.

    Unlike DEV-009, no gold expert identity exists or enters the model. The
    4 experts are **unordered recurrent-memory slots**, so no positional
    embeddings or causal order over slots is assumed. A permutation-equivariant
    expert encoder is applied at the current read boundary only.
    """
    def __init__(self, dim: int = 32, heads: int = 2, experts: int = 4):
        super().__init__()
        if dim < 2 or heads < 1 or dim % heads or experts < 2:
            raise ValueError("bad reader dimensions")
        self.dim, self.heads, self.experts = dim, heads, experts
        self.expert_attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.expert_norm = nn.LayerNorm(dim)
        self.kv_projection = nn.Linear(dim, 2 * dim, bias=False)
        self.query_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(2)])
        self.query_projects = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(2)])
        self.output_projects = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(2)])
        self.ff_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(2)])
        self.feedforwards = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))
            for _ in range(2)
        ])
        self.last_work: ReaderWork | None = None
        self.last_first_scores: Tensor | None = None

    def forward(self, views: Tensor, query: Tensor, *,
                second_top_k: int | None = None) -> Tensor:
        if (views.ndim != 3 or views.shape[1:] != (self.experts, self.dim)
                or query.shape != (views.shape[0], self.dim)):
            raise ValueError("expected expert views [B,E,D], query [B,D]")
        if second_top_k is not None and not 1 <= second_top_k <= self.experts:
            raise ValueError("invalid second-stage top-k")
        if not torch.isfinite(views).all() or not torch.isfinite(query).all():
            raise ValueError("nonfinite reader input")
        batch, experts, dim = views.shape
        encoder_attended, _ = self.expert_attention(views, views, views, need_weights=False)
        context = self.expert_norm(views + encoder_attended)
        k, v = self.kv_projection(context).chunk(2, dim=-1)
        dh = dim // self.heads
        k = k.reshape(batch, experts, self.heads, dh).transpose(1, 2)
        v = v.reshape(batch, experts, self.heads, dh).transpose(1, 2)
        state = query
        first_logits = None
        for depth in range(2):
            q = self.query_projects[depth](self.query_norms[depth](state)).reshape(
                batch, self.heads, dh)
            if depth == 1 and second_top_k is not None:
                # First-stage scan is charged in full; top-k is only for
                # the *second* decoder layer. No gold item or oracle gate.
                assert first_logits is not None
                selected = first_logits.topk(second_top_k, dim=-1).indices
                index = selected[:, None, :, None].expand(batch, self.heads, second_top_k, dh)
                keys = k.gather(2, index)
                values = v.gather(2, index)
            else:
                keys, values = k, v
            scores = (q[:, :, None, :] * keys).sum(-1) / sqrt(dh)
            if depth == 0:
                first_logits = scores.mean(dim=1)
                self.last_first_scores = first_logits
            attended = (scores.softmax(-1)[:, :, :, None] * values).sum(dim=2)
            attended = attended.reshape(batch, dim)
            state = state + self.output_projects[depth](attended)
            state = state + self.feedforwards[depth](self.ff_norms[depth](state))
        self.last_work = ReaderWork(
            batch=batch, experts=experts,
            encoder_score_pairs=batch * self.heads * experts * experts,
            first_decoder_score_pairs=batch * self.heads * experts,
            second_decoder_score_pairs=batch * self.heads * (
                second_top_k if second_top_k is not None else experts),
            global_kv_scalars=2 * batch * experts * dim,
            memory_state_scalars_per_episode=experts * 8 * 24,
        )
        return state


class FiniteMemoryCEDModel(ProtectionModel):
    """Same PLM writer, 768-scalar state and conditional compute as DEV-005."""
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        super().__init__(config=config, mode="always_update")
        self.ced_read = FiniteCEDRead(config.dim, heads=2, experts=config.experts)

    def predict(self, state: Tensor, address: Tensor, current_bit: Tensor,
                *, second_top_k: int | None = None) -> Tensor:
        c = self.config
        if (state.ndim != 4 or state.shape[1:] != (c.experts,c.value_dim,c.key_dim)
                or address.shape != (state.shape[0], c.address_dim)
                or current_bit.shape != (state.shape[0],)):
            raise ValueError("memory/query dimensions incompatible")
        onehot = F.one_hot(current_bit.long(), 2).to(address.dtype)
        query = self.query(torch.cat((address, onehot), dim=-1))
        reads = torch.einsum("bevk,bek->bev",state,self.memory.keys(address))
        views = self.read_projection(reads)
        routed = self.ced_read(views, query, second_top_k=second_top_k)
        computation, _ = self.compute(routed, query)
        return self.answer(torch.cat((computation, query), dim=-1))


def copy_common_initialization(baseline: ProtectionModel,
                               candidate: FiniteMemoryCEDModel) -> None:
    """Keep writer, query, conditional executor and answer heads initially equal."""
    old = baseline.state_dict()
    new = candidate.state_dict()
    common = set(old).intersection(new)
    for name in common:
        if old[name].shape != new[name].shape:
            raise AssertionError(f"shared parameter mismatch: {name}")
        new[name] = old[name].detach().clone()
    if not any(k.startswith("memory.write.") for k in common):
        raise AssertionError("missing PLM writer in parameter overlap")
    if not any(k.startswith("compute.") for k in common):
        raise AssertionError("missing conditional executor in overlap")
    candidate.load_state_dict(new)
