from __future__ import annotations

import torch
from torch import Tensor, nn


class PersistencePLM001(nn.Module):
    """Small learned multimodal PLM whose state must carry t0 information."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text = nn.Sequential(
            nn.Embedding(32, hidden_dim),
            nn.Flatten(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.image = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2),
            nn.GELU(),
            nn.Conv2d(8, 16, 3, stride=2),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(16 * 1 * 1, hidden_dim),
            nn.GELU(),
        )
        self.audio = nn.Sequential(
            nn.Conv1d(1, 8, 5, stride=2),
            nn.GELU(),
            nn.Conv1d(8, 16, 5, stride=2),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(16 * 5, hidden_dim),
            nn.GELU(),
        )
        self.rep = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.write_delta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.write_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.read_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def encode(self, text: Tensor, image: Tensor, audio: Tensor) -> Tensor:
        return self.rep(
            torch.cat([
                self.text(text),
                self.image(image),
                self.audio(audio.unsqueeze(1) if audio.ndim == 2 else audio),
            ], dim=-1)
        )

    def initial_state(self, batch: int, device: torch.device) -> Tensor:
        return torch.zeros(batch, self.hidden_dim, device=device)

    def write(self, state: Tensor, z: Tensor) -> Tensor:
        gate = torch.sigmoid(self.write_gate(torch.cat([state, z], dim=-1)))
        delta = self.write_delta(z)
        return state + gate * delta

    def retain(self, state: Tensor) -> Tensor:
        return state

    def predict(self, state: Tensor, current_z: Tensor) -> Tensor:
        return self.read_head(torch.cat([state, current_z], dim=-1))

    def forward(self, t0, delayed, current):
        state = self.initial_state(t0[0].shape[0], t0[0].device)
        z0 = self.encode(*t0)
        state = self.write(state, z0)
        for obs in delayed:
            state = self.retain(state)
        zc = self.encode(*current)
        return self.predict(state, zc), state
