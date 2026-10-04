"""Query-conditioned behavioral compatibility router for G-CASM-012.

The router learns a candidate/query compatibility relation rather than an
exact candidate identity.  Training labels are derived only from public
training-program truth tables and support rows; evaluation targets and
verifier-only rows never enter the learner.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BehavioralRouterConfig:
    candidate_dim: int = 290
    row_dim: int = 5
    hidden_dim: int = 64
    latent_dim: int = 32
    pair_hidden_dim: int = 64
    learning_rate: float = 1e-3
    temperature: float = 0.10
    aggregate_temperature: float = 0.25
    pair_loss_weight: float = 1.0
    ranking_loss_weight: float = 1.0
    steps: int = 1200
    batch_size: int = 32
    candidates_per_task: int = 8
    seed: int = 0


class BehavioralCompatibilityRouter(nn.Module):
    """Cross-encoder for candidate graph x support-row compatibility."""

    def __init__(self, config: BehavioralRouterConfig):
        super().__init__()
        self.config = config
        torch.manual_seed(config.seed)

        self.candidate_encoder = nn.Sequential(
            nn.Linear(config.candidate_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        self.row_encoder = nn.Sequential(
            nn.Linear(config.row_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        self.pair_head = nn.Sequential(
            nn.Linear(config.latent_dim * 4, config.pair_hidden_dim),
            nn.GELU(),
            nn.Linear(config.pair_hidden_dim, 1),
        )

    def encode_candidates(self, candidate_features: torch.Tensor) -> torch.Tensor:
        return self.candidate_encoder(candidate_features)

    def encode_rows(self, rows: torch.Tensor) -> torch.Tensor:
        return self.row_encoder(rows)

    def pair_logits(
        self,
        candidate_features: torch.Tensor,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        """Return [num_candidates, num_rows] compatibility logits."""
        if candidate_features.ndim != 2:
            raise ValueError("candidate_features must be rank-2")
        if rows.ndim != 2:
            raise ValueError("rows must be rank-2")
        if rows.shape[1] != self.config.row_dim:
            raise ValueError(
                f"rows width {rows.shape[1]} != configured {self.config.row_dim}"
            )

        candidates = self.encode_candidates(candidate_features)
        row_latent = self.encode_rows(rows)
        c = candidates[:, None, :]
        r = row_latent[None, :, :]
        pair = torch.cat((c.expand(-1, rows.shape[0], -1),
                          r.expand(candidates.shape[0], -1, -1),
                          c * r,
                          torch.abs(c - r)), dim=-1)
        return self.pair_head(pair).squeeze(-1)

    def compatibility_scores(
        self,
        candidate_features: torch.Tensor,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate row-level compatibility into one score per candidate."""
        logits = self.pair_logits(candidate_features, rows)
        tau = self.config.aggregate_temperature
        # Differentiable soft-min: a candidate must score well on its weakest
        # support row rather than merely accumulating a few strong matches.
        return -tau * torch.logsumexp(-logits / tau, dim=-1)

    def rank(
        self,
        candidate_features: torch.Tensor,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        return torch.argsort(
            self.compatibility_scores(candidate_features, rows),
            descending=True,
        )

    def training_loss(
        self,
        candidate_features: torch.Tensor,
        rows: torch.Tensor,
        pair_labels: torch.Tensor,
        target_index: torch.Tensor,
    ) -> torch.Tensor:
        """Pairwise compatibility supervision + target-set ranking."""
        pair_logits = self.pair_logits(candidate_features, rows)
        labels = pair_labels.to(dtype=pair_logits.dtype)
        pair_loss = nn.functional.binary_cross_entropy_with_logits(
            pair_logits, labels
        )

        scores = self.compatibility_scores(candidate_features, rows)
        rank_loss = nn.functional.cross_entropy(
            scores.unsqueeze(0) / self.config.temperature,
            target_index.reshape(1),
        )
        return (
            self.config.pair_loss_weight * pair_loss
            + self.config.ranking_loss_weight * rank_loss
        )


def row_tensor(bits: tuple[int, ...] | list[int], output: int) -> list[float]:
    return [float(x) for x in (*bits, int(output))]
