"""Functional multimodal PLM core for E2E-005.

The model API is deliberately label-free. Training targets and environment
outcomes are supplied by the outer experiment harness only after the model has
produced an action. This makes pre-action target leakage structurally harder.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .integrated_e2e_005_benchmark import OPS


@dataclass(frozen=True)
class FunctionalConfig:
    hidden_dim: int = 40
    entity_count: int = 16
    latent_bits: int = 12


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden // 2)
        self.rnn = nn.GRU(hidden // 2, hidden // 2, batch_first=True)

    def forward(self, tokens: Tensor) -> Tensor:
        _, h = self.rnn(self.emb(tokens))
        return h[-1]


class ImageEncoder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.GELU(),
        )
        self.proj = nn.Linear(32 * 3 * 3, hidden // 2)

    def forward(self, image: Tensor) -> Tensor:
        return self.proj(self.net(image).flatten(1))


class AudioEncoder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=2),
            nn.GELU(),
            nn.Conv1d(16, 32, 5, stride=2),
            nn.GELU(),
        )
        self.proj = nn.Linear(32 * 21, hidden // 2)

    def forward(self, audio: Tensor) -> Tensor:
        return self.proj(self.net(audio).flatten(1))


class SharedRepresentation(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 3 // 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

    def forward(self, text: Tensor, image: Tensor, audio: Tensor) -> Tensor:
        return self.fuse(torch.cat([text, image, audio], dim=-1))


class ExplicitEntityState(nn.Module):
    def __init__(self, hidden: int, entity_count: int) -> None:
        super().__init__()
        self.entity_count = entity_count
        self.hidden = hidden
        self.write_value = nn.Linear(hidden, hidden)

    def initial(self, batch: int, device: torch.device) -> Tensor:
        return torch.zeros(batch, self.entity_count, self.hidden, device=device)

    def write(
        self,
        memory: Tensor,
        z: Tensor,
        entity: Tensor,
        strength: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        probs = F.one_hot(entity.long(), num_classes=self.entity_count).to(z.dtype)
        if strength is not None:
            probs = probs * strength.view(-1, 1)
        value = self.write_value(z).unsqueeze(1)
        updated = (
            (1.0 - probs.unsqueeze(-1)) * memory
            + probs.unsqueeze(-1) * value
        )
        return updated, probs

    def read(self, memory: Tensor, entity: Tensor) -> tuple[Tensor, Tensor]:
        probs = F.one_hot(
            entity.long(),
            num_classes=self.entity_count,
        ).to(memory.dtype)
        read = torch.einsum("bn,bnd->bd", probs, memory)
        return read, probs


class FixedCASM(nn.Module):
    def __init__(self, hidden: int, latent_bits: int) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_bits),
        )

    def forward(
        self,
        state_read: Tensor,
        i: Tensor,
        j: Tensor,
        op: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        bits = torch.sigmoid(self.decoder(state_read))
        a = bits.gather(1, i.view(-1, 1)).squeeze(1)
        b = bits.gather(1, j.view(-1, 1)).squeeze(1)
        values = torch.stack(
            (
                a + b - 2 * a * b,
                a * b,
                a + b - a * b,
                1 - a - b + 2 * a * b,
            ),
            dim=-1,
        )
        action = values.gather(1, op.view(-1, 1)).squeeze(1)
        logits = torch.stack((-action, action), dim=-1)
        return action, logits, bits


class FunctionalMultimodalPLM(nn.Module):
    """Multimodal input -> state -> query -> CASM -> action -> verifier."""

    def __init__(
        self,
        config: FunctionalConfig = FunctionalConfig(),
        vocab_size: int = 40,
    ) -> None:
        super().__init__()
        self.config = config
        self.text = TextEncoder(vocab_size, config.hidden_dim)
        self.image = ImageEncoder(config.hidden_dim)
        self.audio = AudioEncoder(config.hidden_dim)
        self.rep = SharedRepresentation(config.hidden_dim)
        self.state = ExplicitEntityState(config.hidden_dim, config.entity_count)
        self.casm = FixedCASM(config.hidden_dim, config.latent_bits)
        self.verifier = nn.Sequential(
            nn.Linear(config.hidden_dim + 1, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.outcome_project = nn.Linear(1, config.hidden_dim)
        self.write_gate = nn.Sequential(
            nn.Linear(config.hidden_dim * 2 + 1, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def encode(self, text: Tensor, image: Tensor, audio: Tensor) -> Tensor:
        return self.rep(
            self.text(text),
            self.image(image),
            self.audio(audio),
        )

    def query(
        self,
        memory: Tensor,
        entity: Tensor,
        i: Tensor,
        j: Tensor,
        op: Tensor,
    ) -> dict[str, Tensor]:
        state_read, attention = self.state.read(memory, entity)
        action, logits, bits = self.casm(state_read, i, j, op)
        return {
            "entity": entity,
            "read": state_read,
            "attention": attention,
            "action": action,
            "logits": logits,
            "bits": bits,
        }

    def post_action_update(
        self,
        memory: Tensor,
        query_result: dict[str, Tensor],
        outcome: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply only post-action environment feedback."""
        feedback = outcome.float().view(-1, 1)
        verifier_logit = self.verifier(
            torch.cat(
                [query_result["read"], query_result["action"].view(-1, 1)],
                dim=-1,
            )
        )
        verifier_prob = verifier_logit.sigmoid()
        outcome_embedding = self.outcome_project(feedback)
        strength = feedback * torch.sigmoid(
            self.write_gate(
                torch.cat(
                    [
                        query_result["read"],
                        outcome_embedding,
                        verifier_prob,
                    ],
                    dim=-1,
                )
            )
        )
        updated, _ = self.state.write(
            memory,
            query_result["read"] + outcome_embedding,
            query_result["entity"],
            strength=strength,
        )
        return updated, torch.cat([verifier_logit, feedback], dim=-1)
