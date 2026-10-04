"""Cached late-interaction reranker for G-CASM-013.

Candidate representations are encoded once for a static library. At query time
only support-row representations and pair interactions are computed. This is
the deployment form needed to test whether 012's O(M*k) candidate encoding
bottleneck is architectural rather than an optimization artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .behavioral_compatibility_router import BehavioralCompatibilityRouter


@dataclass(frozen=True)
class RerankerMacProfile:
    candidate_encoder_per_candidate: int = 290 * 64 + 64 * 32
    row_encoder_per_support_row: int = 5 * 64 + 64 * 32
    pair_head_per_candidate_row: int = 128 * 64 + 64

    def query_macs(self, support_rows: int, candidates_scored: int) -> int:
        return (
            support_rows * self.row_encoder_per_support_row
            + support_rows * candidates_scored * self.pair_head_per_candidate_row
        )

    def uncached_query_macs(self, support_rows: int, candidates_scored: int) -> int:
        return (
            candidates_scored * self.candidate_encoder_per_candidate
            + self.query_macs(support_rows, candidates_scored)
        )


class CachedBehavioralReranker:
    """012 pair scorer with candidate encoding moved to an offline cache."""

    def __init__(
        self,
        model: BehavioralCompatibilityRouter,
        candidate_features: torch.Tensor,
    ):
        if candidate_features.ndim != 2:
            raise ValueError("candidate_features must be rank-2")
        with torch.no_grad():
            self.candidate_latents = model.encode_candidates(candidate_features).detach()
        self.model = model.eval()
        self.profile = RerankerMacProfile()

    def _pair_logits(
        self,
        candidate_indices: Sequence[int],
        support_rows: torch.Tensor,
    ) -> torch.Tensor:
        if support_rows.ndim != 2:
            raise ValueError("support_rows must be rank-2")
        if not candidate_indices:
            return torch.empty((0, support_rows.shape[0]))
        c = self.candidate_latents[
            torch.tensor(candidate_indices, dtype=torch.long)
        ]
        with torch.no_grad():
            r = self.model.encode_rows(support_rows)
            c_full = c[:, None, :].expand(-1, r.shape[0], -1)
            r_full = r[None, :, :].expand(c.shape[0], -1, -1)
            pair = torch.cat(
                (c_full, r_full, c_full * r_full, torch.abs(c_full - r_full)),
                dim=-1,
            )
            return self.model.pair_head(pair).squeeze(-1)

    def score(
        self,
        candidate_indices: Sequence[int],
        support_rows: torch.Tensor,
    ) -> list[float]:
        logits = self._pair_logits(candidate_indices, support_rows)
        if logits.numel() == 0:
            return []
        tau = self.model.config.aggregate_temperature
        scores = -tau * torch.logsumexp(-logits / tau, dim=-1)
        return scores.tolist()
