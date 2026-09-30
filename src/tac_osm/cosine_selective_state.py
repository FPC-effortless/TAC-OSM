"""Cosine-reranked learned selective state retrieval.

The learned state index proposes a bounded address shortlist using its compact
binary code. Continuous cosine similarity then re-ranks only that shortlist.
This keeps the indexing proposal and semantic ranking as distinct cost terms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import Query
from .learned_state_index import LearnedSemanticStateIndex, LearnedIndexLookup
from .temporal import TemporalPersistentState


@dataclass(frozen=True)
class CosineSelectiveLookup:
    address: str | None
    proposal: LearnedIndexLookup
    shortlist_scored: int
    cosine_score_macs: int
    normalization_ops: int


class CosineRerankedStateIndex:
    """Bounded learned bucket proposal followed by continuous cosine reranking."""

    def __init__(
        self,
        learned_index: LearnedSemanticStateIndex,
    ) -> None:
        self.index = learned_index
        self._state_embeddings: dict[str, tuple[float, ...]] = {}
        self._last_shortlist_size = 0
        self._built = False

    @staticmethod
    def _normalize(values: list[float]) -> tuple[float, ...]:
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 1e-8:
            raise ValueError("cannot normalize a zero embedding")
        return tuple(value / norm for value in values)

    @property
    def query_embedding_macs(self) -> int:
        return self.index.query_embedding_macs

    @property
    def build_embedding_macs(self) -> int:
        return (
            self.index.build_diagnostics.state_items
            * self.index.state_embedding_macs
        )

    @property
    def bucket_probe_count(self) -> int:
        return self.index.build_diagnostics.state_items * 0 + len(
            self.index._neighbors(0)
        )

    @property
    def max_shortlist(self) -> int:
        return self.index.config.shortlist_k

    @property
    def shortlist_size(self) -> int:
        return self._last_shortlist_size

    @property
    def state_items(self) -> int:
        return self.index.build_diagnostics.state_items

    def build(self, state: TemporalPersistentState):
        self.index.build(state)
        self._state_embeddings = {}
        for address in state.addresses():
            read = state.read(
                Query(
                    text="	" + address,
                    step=state.current_step,
                    provenance="cosine_selective_state_build",
                )
            )
            if not read.keys:
                continue
            self._state_embeddings[address] = self._normalize(
                self.index.encode_state(read.values[0])
            )
        self._built = True
        return {
            "state_items": self.state_items,
            "build_embedding_macs": self.build_embedding_macs,
            "unique_buckets": self.index.build_diagnostics.unique_buckets,
        }

    def lookup(self, query: Query) -> CosineSelectiveLookup:
        if not self._built:
            raise RuntimeError("cosine selective state index has not been built")

        proposal = self.index.lookup(query)
        self._last_shortlist_size = len(proposal.addresses)
        query_embedding = self._normalize(self.index.encode_query(query))
        scored: list[tuple[float, str]] = []
        for address in proposal.addresses:
            embedding = self._state_embeddings.get(address)
            if embedding is None:
                continue
            score = sum(a * b for a, b in zip(query_embedding, embedding))
            scored.append((score, address))
        scored.sort(key=lambda item: (-item[0], item[1]))
        address = scored[0][1] if scored else None
        return CosineSelectiveLookup(
            address=address,
            proposal=proposal,
            shortlist_scored=len(scored),
            cosine_score_macs=len(scored) * self.index.config.latent_dim,
            normalization_ops=2 * (len(scored) + 1),
        )
