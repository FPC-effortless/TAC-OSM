"""Learned multimodal recurrent memory for PERSISTENCE-002.

The GRU receives actual encoded distractor observations at every intervening
time step. No oracle memory bit, hidden label or target is a model input.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tac_osm.persistence_002_benchmark import PairedBatch


@dataclass
class PersistenceTrace:
    logits: Tensor
    state: Tensor
    t0_state: Tensor
    current_representation: Tensor
    distractor_update_norms: tuple[Tensor, ...]
    state_write_calls: int


class PersistencePLM002(nn.Module):
    def __init__(self, hidden_dim: int = 64, embedding_dim: int = 8) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_embedding = nn.Embedding(32, embedding_dim)
        self.image_encoder = nn.Sequential(
            nn.Linear(64, 16), nn.GELU()
        )
        self.audio_encoder = nn.Sequential(
            nn.Linear(32, 16), nn.GELU()
        )
        self.rep = nn.Sequential(
            nn.Linear(4 * embedding_dim + 32, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.write_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.read_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def encode(self, observation: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        text, image, audio = observation
        return self.rep(torch.cat((
            self.text_embedding(text).flatten(1),
            self.image_encoder(image.flatten(1)),
            self.audio_encoder(audio),
        ), dim=-1))

    def predict(self, state: Tensor, current_representation: Tensor) -> Tensor:
        return self.read_head(torch.cat((state, current_representation), dim=-1))

    def forward(self, episode: PairedBatch) -> PersistenceTrace:
        n = episode.memory_bit.numel()
        state = torch.zeros(n, self.hidden_dim, dtype=torch.float32)
        state = self.write_cell(self.encode(episode.t0), state)
        t0_state = state
        norm_rows = []
        for observation in episode.distractors:
            previous = state
            state = self.write_cell(self.encode(observation), state)
            norm_rows.append(torch.linalg.vector_norm(state - previous, dim=1).mean())
        current_representation = self.encode(episode.current)
        return PersistenceTrace(
            logits=self.predict(state, current_representation),
            state=state,
            t0_state=t0_state,
            current_representation=current_representation,
            distractor_update_norms=tuple(norm_rows),
            state_write_calls=1 + len(episode.distractors),
        )
