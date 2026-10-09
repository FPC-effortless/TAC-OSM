"""Fully learned multimodal controller with a generic Hamiltonian prior.

No particle count, entity IDs, operator table, retrieval rule, or task-specific
physical state is encoded. The physics arm regularizes a generic latent canonical
coordinate pair (q, p) only through Hamilton's equations and energy conservation.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LearnedPhysicsConfig:
    hidden_dim: int = 64
    vocab_size: int = 32
    action_count: int = 5
    physics_weight: float = 0.05


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden: int) -> None:
        super().__init__()
        dim = hidden // 2
        self.embedding = nn.Embedding(vocab_size, dim)
        self.rnn = nn.GRU(dim, dim, batch_first=True)

    def forward(self, tokens: Tensor) -> Tensor:
        _, h = self.rnn(self.embedding(tokens))
        return h[-1]


class ImageEncoder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2),
            nn.GELU(),
            nn.Conv2d(16, 32, 5, stride=2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(32, 32)

    def forward(self, image: Tensor) -> Tensor:
        return self.proj(self.net(image).flatten(1))


class AudioEncoder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, stride=2),
            nn.GELU(),
            nn.Conv1d(16, 32, 7, stride=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(32, 32)

    def forward(self, audio: Tensor) -> Tensor:
        return self.proj(self.net(audio).flatten(1))


class MultimodalFusion(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden // 2 + 64, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

    def forward(self, text: Tensor, image: Tensor, audio: Tensor) -> Tensor:
        return self.net(torch.cat((text, image, audio), dim=-1))


class LearnedTemporalCore(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.cell = nn.GRUCell(hidden, hidden)

    def forward(self, sequence: Tensor) -> Tensor:
        batch, steps, hidden = sequence.shape
        state = torch.zeros(batch, hidden, device=sequence.device, dtype=sequence.dtype)
        states = []
        for step in range(steps):
            state = self.cell(sequence[:, step], state)
            states.append(state)
        return torch.stack(states, dim=1)


class CanonicalLatent(nn.Module):
    """Generic latent chart used only to state the Hamiltonian prior."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.LayerNorm(hidden),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.projection(hidden_states)


class LearnedHamiltonian(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, canonical: Tensor) -> Tensor:
        return self.net(canonical).squeeze(-1)


class LearnedActionHead(nn.Module):
    def __init__(self, hidden: int, action_count: int) -> None:
        super().__init__()
        self.goal = nn.Sequential(nn.Linear(2, hidden), nn.GELU())
        self.net = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, action_count),
        )

    def forward(self, state: Tensor, canonical: Tensor, goal: Tensor) -> Tensor:
        return self.net(torch.cat((state, canonical, self.goal(goal)), dim=-1))


class LearnedPhysicsE2E(nn.Module):
    """Multimodal history -> learned persistent state -> learned action."""

    def __init__(self, config: LearnedPhysicsConfig = LearnedPhysicsConfig()) -> None:
        super().__init__()
        self.config = config
        self.text = TextEncoder(config.vocab_size, config.hidden_dim)
        self.image = ImageEncoder(config.hidden_dim)
        self.audio = AudioEncoder(config.hidden_dim)
        self.fusion = MultimodalFusion(config.hidden_dim)
        self.temporal = LearnedTemporalCore(config.hidden_dim)
        self.canonical = CanonicalLatent(config.hidden_dim)
        self.hamiltonian = LearnedHamiltonian(config.hidden_dim)
        self.action = LearnedActionHead(config.hidden_dim, config.action_count)

    def encode_step(self, text: Tensor, image: Tensor, audio: Tensor) -> Tensor:
        return self.fusion(self.text(text), self.image(image), self.audio(audio))

    def forward(
        self,
        text: Tensor,
        image: Tensor,
        audio: Tensor,
        goal: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        fused = torch.stack(
            [self.encode_step(text[:, t], image[:, t], audio[:, t]) for t in range(text.shape[1])],
            dim=1,
        )
        hidden_states = self.temporal(fused)
        canonical = self.canonical(hidden_states)
        energy = self.hamiltonian(canonical)
        logits = self.action(hidden_states[:, -1], canonical[:, -1], goal)
        return logits, canonical, energy


def hamiltonian_prior_loss(
    canonical: Tensor,
    energy: Tensor,
    dt: float,
) -> dict[str, Tensor]:
    """Hamilton's equations + energy conservation; no state/action labels."""
    half = canonical.shape[-1] // 2
    q = canonical[..., :half]
    p = canonical[..., half:]

    d_energy = torch.autograd.grad(
        energy.sum(),
        canonical,
        create_graph=True,
        retain_graph=True,
    )[0]
    dH_dq = d_energy[..., :half]
    dH_dp = d_energy[..., half:]

    qdot = (q[:, 1:] - q[:, :-1]) / dt
    pdot = (p[:, 1:] - p[:, :-1]) / dt
    hamilton_q_residual = (qdot - dH_dp[:, :-1]).pow(2).mean()
    hamilton_p_residual = (pdot + dH_dq[:, :-1]).pow(2).mean()
    energy_residual = (energy[:, 1:] - energy[:, :-1]).pow(2).mean()

    total = hamilton_q_residual + hamilton_p_residual + energy_residual
    return {
        "total": total,
        "hamilton_q": hamilton_q_residual,
        "hamilton_p": hamilton_p_residual,
        "energy": energy_residual,
    }
