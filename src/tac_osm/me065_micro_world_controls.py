"""ME065 compact recurrent *comparator* cells for authentic physical forecasting.

These controls intentionally reuse ME063's encoder, physical servo/persistence
experts, horizon router and factual forecast objective. They are NOT TAC variants
and the diagonal selective control is NOT a full Mamba implementation.
Neither numerical tests nor an untrained preflight establish model capability.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

from .real_robot_me063_regime import RegimeMixtureModel


class SelectiveDiagonalCell(nn.Module):
    """Input-selective stable diagonal recurrence, not the Mamba architecture."""
    def __init__(self, input_dim: int, width: int):
        super().__init__()
        self.projections = nn.Linear(input_dim, 3 * width)
        self.log_decay = nn.Parameter(torch.zeros(width))

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        candidate, delta, output_gate = self.projections(x).chunk(3, -1)
        rate = F.softplus(self.log_decay)[None, :] * F.softplus(delta)
        retention = torch.exp(-rate)
        updated = retention * state + (1. - retention) * torch.tanh(candidate)
        return torch.sigmoid(output_gate) * updated + (1. - torch.sigmoid(output_gate)) * state


class GaussianRSSMCell(nn.Module):
    """Compact GRU deterministic belief plus Gaussian stochastic prior/posterior.

    At each factual observation: p(z_t|h_{t-1}), q(z_t|h_{t-1},u_t),
    h_t=GRU([u_t,z_t],h_{t-1}). Sampling occurs in training only; posterior
    means are used during eval. KL is an explicit training loss, not a reward
    signal or proof of counterfactual planning.
    """
    def __init__(self, input_dim: int, width: int, stochastic: int = 8):
        super().__init__()
        if stochastic < 1 or stochastic >= width:
            raise ValueError("stochastic dimension must fit recurrent width")
        self.deterministic = width - stochastic
        self.stochastic = stochastic
        self.cell = nn.GRUCell(input_dim + stochastic, self.deterministic)
        self.prior = nn.Linear(self.deterministic, 2 * stochastic)
        self.posterior = nn.Linear(self.deterministic + input_dim, 2 * stochastic)
        self.kl_terms: list[torch.Tensor] = []

    def clear_kl(self) -> None:
        self.kl_terms.clear()

    def regularizer(self) -> torch.Tensor:
        if not self.kl_terms:
            return self.prior.weight.sum() * 0.
        return torch.stack(self.kl_terms).mean()

    def forward(self, u: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        h, _ = state.split((self.deterministic, self.stochastic), -1)
        pm, plog = self.prior(h).chunk(2, -1)
        qm, qlog = self.posterior(torch.cat((h, u), -1)).chunk(2, -1)
        plog = plog.clamp(-5., 3.)
        qlog = qlog.clamp(-5., 3.)
        if self.training and torch.is_grad_enabled():
            sample = qm + torch.exp(.5 * qlog) * torch.randn_like(qm)
            kl = .5 * (plog - qlog +
                        (qlog - plog).exp() +
                        (qm - pm).square() * torch.exp(-plog) - 1.)
            self.kl_terms.append(kl.sum(-1).mean())
        else:
            sample = qm
        next_h = self.cell(torch.cat((u, sample), -1), h)
        return torch.cat((next_h, sample), -1)


class MicroRegimeComparator(RegimeMixtureModel):
    """Drop-in ME063 observation-recurrence comparator; common forecast heads."""
    def __init__(self, identified, variant: str,
                 width: int = 44, decoder_width: int = 82):
        if variant not in ("compact_rssm", "selective_diagonal_ssm"):
            raise ValueError("unknown micro comparator")
        super().__init__(identified, architecture="gru_world",
                         width=width, decoder_width=decoder_width)
        self.comparator_variant = variant
        if variant == "compact_rssm":
            self.cell = GaussianRSSMCell(24, width, 8)
        else:
            self.cell = SelectiveDiagonalCell(24, width)

    def reset_auxiliary(self) -> None:
        if isinstance(self.cell, GaussianRSSMCell):
            self.cell.clear_kl()

    def auxiliary_objective(self) -> torch.Tensor:
        if isinstance(self.cell, GaussianRSSMCell):
            # Disclosed architecture-specific variational cost. The same
            # physical supervision is always used for all compared models.
            return .005 * self.cell.regularizer()
        return next(self.parameters()).sum() * 0.
