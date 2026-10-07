"""Fully learned multimodal control model with an optional physics-law prior."""
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


class PhysicalStateHead(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 8),
        )

    def forward(self, state: Tensor) -> Tensor:
        raw = self.net(state)
        position = torch.sigmoid(raw[..., :4])
        velocity = 0.5 * torch.tanh(raw[..., 4:])
        return torch.cat((position, velocity), dim=-1)


class LearnedActionHead(nn.Module):
    """Action computation explicitly consumes the learned physical state."""

    def __init__(self, hidden: int, action_count: int) -> None:
        super().__init__()
        self.goal = nn.Sequential(nn.Linear(2, hidden), nn.GELU())
        self.net = nn.Sequential(
            nn.Linear(hidden * 2 + 8, hidden),
            nn.GELU(),
            nn.Linear(hidden, action_count),
        )

    def forward(
        self,
        temporal_state: Tensor,
        physical_state: Tensor,
        goal: Tensor,
    ) -> Tensor:
        return self.net(
            torch.cat((temporal_state, physical_state, self.goal(goal)), dim=-1)
        )


class LearnedPhysicsE2E(nn.Module):
    """Multimodal history -> learned state -> physics state -> learned action."""

    def __init__(self, config: LearnedPhysicsConfig = LearnedPhysicsConfig()) -> None:
        super().__init__()
        self.config = config
        self.text = TextEncoder(config.vocab_size, config.hidden_dim)
        self.image = ImageEncoder(config.hidden_dim)
        self.audio = AudioEncoder(config.hidden_dim)
        self.fusion = MultimodalFusion(config.hidden_dim)
        self.temporal = LearnedTemporalCore(config.hidden_dim)
        self.physical_state = PhysicalStateHead(config.hidden_dim)
        self.action = LearnedActionHead(config.hidden_dim, config.action_count)

    def encode_step(self, text: Tensor, image: Tensor, audio: Tensor) -> Tensor:
        return self.fusion(self.text(text), self.image(image), self.audio(audio))

    def forward(
        self,
        text: Tensor,
        image: Tensor,
        audio: Tensor,
        goal: Tensor,
    ) -> tuple[Tensor, Tensor]:
        step_representations = [
            self.encode_step(text[:, t], image[:, t], audio[:, t])
            for t in range(text.shape[1])
        ]
        fused = torch.stack(step_representations, dim=1)
        hidden_states = self.temporal(fused)
        physical_states = self.physical_state(hidden_states)
        logits = self.action(
            hidden_states[:, -1],
            physical_states[:, -1],
            goal,
        )
        return logits, physical_states


def physics_prior_loss(predicted_state: Tensor, dt: float) -> dict[str, Tensor]:
    """Only conservation/kinematic laws are privileged."""
    position = predicted_state[..., :4].reshape(-1, predicted_state.shape[1], 2, 2)
    velocity = predicted_state[..., 4:].reshape(-1, predicted_state.shape[1], 2, 2)
    momentum = velocity.sum(dim=2)
    kinetic_energy = 0.5 * velocity.pow(2).sum(dim=-1).sum(dim=-1)

    momentum_residual = (momentum[:, 1:] - momentum[:, :-1]).pow(2).mean()
    energy_residual = (kinetic_energy[:, 1:] - kinetic_energy[:, :-1]).pow(2).mean()

    average_velocity = 0.5 * (velocity[:, 1:] + velocity[:, :-1]) * dt
    displacement = position[:, 1:] - position[:, :-1]
    kinematic_residual = (displacement - average_velocity).pow(2).mean()

    total = momentum_residual + energy_residual + kinematic_residual
    return {
        "total": total,
        "momentum": momentum_residual,
        "energy": energy_residual,
        "kinematic": kinematic_residual,
    }
