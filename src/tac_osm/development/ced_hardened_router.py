"""DEV-008: experimental query-only CED router for the *existing* hardened loop.

Important exclusions: this encoder processes already-retained candidate descriptors,
NOT online observation history. The persisted temporal state, address index,
executor, verifier, repair and delayed write remain owned by HardenedLoop.
This is neither DeepSeek's model nor a proven learned TCL/PLM/CASM capability.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

import torch
from torch import Tensor, nn

from tac_osm import Candidate, Query, RoutingDecision, PersistentState


@dataclass(frozen=True)
class CEDRoutingCost:
    """Logical score pairs only; not measured FLOPs, wall time or sublinear work."""
    candidates: int
    encoder_score_pairs: int
    first_global_score_pairs: int
    subsequent_global_score_pairs: int
    final_score_pairs: int
    global_kv_scalars: int
    public_state_reads: int = 0

    @property
    def total_score_pairs(self) -> int:
        return (self.encoder_score_pairs + self.first_global_score_pairs
                + self.subsequent_global_score_pairs + self.final_score_pairs)


class _CausalCandidateBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.mix = nn.Linear(width, width, bias=False)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, x: Tensor) -> Tensor:
        n, width = x.shape
        dh = width // self.heads
        q, k, v = [t.reshape(n, self.heads, dh).transpose(0, 1)
                   for t in self.qkv(self.norm(x)).chunk(3, dim=-1)]
        logits = (q @ k.transpose(-1, -2)) / sqrt(dh)
        mask = torch.ones((n, n), dtype=torch.bool, device=x.device).tril()
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        y = (logits.softmax(-1) @ v).transpose(0, 1).reshape(n, width)
        x = x + self.mix(y)
        return x + self.ff(self.ff_norm(x))


class _QueryDecoder(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width, bias=False)
        self.mix = nn.Linear(width, width, bias=False)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, x: Tensor, kv: tuple[Tensor, Tensor],
                indices: Tensor | None = None) -> tuple[Tensor, Tensor]:
        k, v = kv  # [heads, N, head_width]
        q = self.query(self.norm(x)).reshape(self.heads, -1)
        if indices is not None:
            k, v = k[:, indices], v[:, indices]
        scores = (q[:, None, :] * k).sum(-1) / sqrt(q.shape[-1])
        attended = (scores.softmax(-1)[..., None] * v).sum(1).reshape(-1)
        x = x + self.mix(attended)
        return x + self.ff(self.ff_norm(x)), scores.mean(0)


class QueryOnlyCED(nn.Module):
    """Causal structural candidate encoder + two query-only global-read layers.

    The network sees masked candidate descriptors and a public reference
    obtained from the current query or one legal state.read(query). It never
    sees gold indices, labels, private world truth, the verifier, or outcomes.
    mode=dense constructs two global KV sets; shared constructs one; sparse
    also restricts the second read and final scores to first-read top-k.
    """
    def __init__(self, descriptor_dim: int = 8, width: int = 32,
                 heads: int = 2, mode: str = "shared", topk: int = 4):
        super().__init__()
        if descriptor_dim < 2 or width < 2 or heads < 1 or width % heads:
            raise ValueError("invalid dimensions or attention head count")
        if mode not in ("dense", "shared", "sparse") or topk < 1:
            raise ValueError("invalid attention mode or topk")
        self.descriptor_dim, self.width, self.heads = descriptor_dim, width, heads
        self.mode, self.topk = mode, topk
        self.candidate_embed = nn.Linear(2 * descriptor_dim, width)
        self.query_embed = nn.Linear(2 * descriptor_dim, width)
        self.encoder = nn.ModuleList([_CausalCandidateBlock(width, heads) for _ in range(2)])
        self.kv_producers = nn.ModuleList([
            nn.Linear(width, 2 * width, bias=False)
            for _ in range(2 if mode == "dense" else 1)
        ])
        self.decoder = nn.ModuleList([_QueryDecoder(width, heads) for _ in range(2)])
        # DEV-009: differentiable pre-topk logits for training index supervision.
        # Diagnostic only. Never fed gold or labels during model inference.
        self.last_first_index_scores: Tensor | None = None

    def forward(self, reference: Tensor, mask: Tensor, descriptors: Tensor
                ) -> tuple[Tensor, CEDRoutingCost]:
        dim = self.descriptor_dim
        if (reference.shape != (dim,) or mask.shape != (dim,)
                or descriptors.ndim != 2 or descriptors.shape[1] != dim
                or descriptors.shape[0] < 1):
            raise ValueError("reference, mask, descriptors must be [d], [d], [N,d]")
        if (not torch.isfinite(reference).all() or not torch.isfinite(mask).all()
                or not torch.isfinite(descriptors).all()):
            raise ValueError("all model-visible tensors must be finite")
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("mask must be binary")
        n = descriptors.shape[0]
        # Mask unmarked positions: do not allow irrelevant candidate bits into routing.
        shown = descriptors * mask[None, :]
        candidate_features = torch.cat((shown, mask[None, :].expand(n, -1)), dim=-1)
        x = self.candidate_embed(candidate_features)
        for layer in self.encoder:
            x = layer(x)
        kv = []
        for producer in self.kv_producers:
            k, v = producer(x).chunk(2, -1)
            k = k.reshape(n, self.heads, -1).transpose(0, 1)
            v = v.reshape(n, self.heads, -1).transpose(0, 1)
            kv.append((k, v))
        query = self.query_embed(torch.cat((reference * mask, mask), dim=-1))
        selected: Tensor | None = None
        first = 0
        later = 0
        for index, layer in enumerate(self.decoder):
            keyvalues = kv[index if self.mode == "dense" else 0]
            query, scores = layer(query, keyvalues, selected if index else None)
            if index == 0:
                self.last_first_index_scores = scores
                first += self.heads * n
                if self.mode == "sparse":
                    selected = scores.topk(min(n, self.topk)).indices
            else:
                later += self.heads * (len(selected) if selected is not None else n)
        # Final selection deliberately charges another global scoring pass.
        # It must NOT quietly make a sparse arm dense or hide this cost.
        key = kv[-1][0]
        final_q = self.decoder[-1].query(self.decoder[-1].norm(query)).reshape(self.heads, -1)
        if selected is not None:
            scores = (final_q[:, None, :] * key[:, selected]).sum(-1).mean(0)
            logits = torch.full((n,), -1e6, dtype=scores.dtype, device=scores.device)
            logits = logits.scatter(0, selected, scores)
            final = self.heads * selected.numel()
        else:
            logits = (final_q[:, None, :] * key).sum(-1).mean(0)
            final = self.heads * n
        counts = CEDRoutingCost(
            candidates=n,
            encoder_score_pairs=2 * self.heads * n * n,
            first_global_score_pairs=first,
            subsequent_global_score_pairs=later,
            final_score_pairs=final,
            global_kv_scalars=len(kv) * 2 * n * self.width,
        )
        return logits, counts


class HardenedCEDRouter:
    """Adapter preserving HardenedLoop.router(Query, state, retained) semantics."""

    def __init__(self, model: QueryOnlyCED):
        self.model = model
        self.last_cost: CEDRoutingCost | None = None
        self.last_public_source: str | None = None

    def __call__(self, query: Query, state: PersistentState,
                 candidates: Sequence[Candidate]) -> RoutingDecision:
        if not candidates:
            raise ValueError("empty retained candidate set")
        dim = self.model.descriptor_dim
        if len(query.context) != dim or any(len(c.descriptor) != dim for c in candidates):
            raise ValueError("query and candidate bit widths must match model")
        if any(bit not in (0, 1) for bit in query.context):
            raise ValueError("binary context mask required")
        # State.read() is the public TAC-OSM boundary. Never examine private
        # state dictionaries, verifier outputs or GeneratedTask.reference_bits.
        public_read = state.read(query)
        if public_read.values:
            if len(public_read.values) != 1:
                raise ValueError("ambiguous state read; fail closed")
            visible = public_read.values[0]
            source = "persistent_state"
        else:
            bits = query.text.partition("\t")[0].strip()
            if not bits:
                raise ValueError("public reference unavailable; fail closed")
            try:
                visible = tuple(int(x) for x in bits.split())
            except ValueError as exc:
                raise ValueError("malformed public query bits") from exc
            source = "query"
        if len(visible) != dim or any(bit not in (0, 1) for bit in visible):
            raise ValueError("invalid public reference bits")
        device = next(self.model.parameters()).device
        reference = torch.tensor(visible, dtype=torch.float32, device=device)
        mask = torch.tensor(query.context, dtype=torch.float32, device=device)
        descriptors = torch.tensor([c.descriptor for c in candidates],
                                   dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, cost = self.model(reference, mask, descriptors)
        from dataclasses import replace
        self.last_cost = replace(cost, public_state_reads=1)
        self.last_public_source = source
        return RoutingDecision(selected=int(logits.argmax()), scores=tuple(
            float(x) for x in logits.cpu().tolist()), provenance="dev008.ced_router")
