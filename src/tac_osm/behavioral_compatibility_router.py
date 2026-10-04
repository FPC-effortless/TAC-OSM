"""Query-conditioned behavioral compatibility router for G-CASM-012.

The router learns a candidate/query compatibility relation rather than an
exact candidate identity. Training labels are derived only from public
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
    aggregate_temperature: float = 0.25
    pair_loss_weight: float = 1.0
    set_loss_weight: float = 1.0
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
        """Return [M, S] for one query or [B, M, S] for a training batch."""
        if candidate_features.ndim not in (2, 3):
            raise ValueError("candidate_features must be rank-2 or rank-3")
        if rows.ndim not in (2, 3):
            raise ValueError("rows must be rank-2 or rank-3")
        if rows.shape[-1] != self.config.row_dim:
            raise ValueError(
                f"rows width {rows.shape[-1]} != configured {self.config.row_dim}"
            )
        if candidate_features.ndim != rows.ndim:
            raise ValueError("candidate and row tensors must have matching batch rank")

        candidates = self.encode_candidates(candidate_features)
        row_latent = self.encode_rows(rows)

        if candidates.ndim == 2:
            c = candidates[:, None, :]
            r = row_latent[None, :, :]
            c_full = c.expand(-1, rows.shape[0], -1)
            r_full = r.expand(candidates.shape[0], -1, -1)
        else:
            c = candidates[:, :, None, :]
            r = row_latent[:, None, :, :]
            c_full = c.expand(-1, -1, rows.shape[1], -1)
            r_full = r.expand(-1, candidates.shape[1], -1, -1)

        pair = torch.cat(
            (c_full, r_full, c_full * r_full, torch.abs(c_full - r_full)),
            dim=-1,
        )
        return self.pair_head(pair).squeeze(-1)

    def compatibility_scores(
        self,
        candidate_features: torch.Tensor,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate row compatibility into one score per candidate."""
        logits = self.pair_logits(candidate_features, rows)
        tau = self.config.aggregate_temperature
        # Soft-min aggregation makes the weakest support row influential:
        # exact behavioral compatibility requires every observed row to fit.
        return -tau * torch.logsumexp(-logits / tau, dim=-1)

    def rank(
        self,
        candidate_features: torch.Tensor,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        return torch.argsort(
            self.compatibility_scores(candidate_features, rows),
            dim=-1,
            descending=True,
        )

    def training_loss(
        self,
        candidate_features: torch.Tensor,
        rows: torch.Tensor,
        pair_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Train only on compatibility, never on target candidate identity."""
        pair_logits = self.pair_logits(candidate_features, rows)
        labels = pair_labels.to(dtype=pair_logits.dtype)
        pair_loss = nn.functional.binary_cross_entropy_with_logits(
            pair_logits, labels
        )

        # A set-level label is positive iff the candidate agrees with every
        # observed support row. This trains the actual deployment relation.
        set_labels = pair_labels.all(dim=-1).to(dtype=pair_logits.dtype)
        set_scores = self.compatibility_scores(candidate_features, rows)
        set_loss = nn.functional.binary_cross_entropy_with_logits(
            set_scores, set_labels
        )
        return (
            self.config.pair_loss_weight * pair_loss
            + self.config.set_loss_weight * set_loss
        )


def row_tensor(bits: tuple[int, ...] | list[int], output: int) -> list[float]:
    return [float(x) for x in (*bits, int(output))]
