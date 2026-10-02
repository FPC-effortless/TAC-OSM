"""Trainable multimodal persistent-computation chain.

Research status:
- synthetic benchmark substrate;
- separate text/image/audio encoders feeding a shared latent;
- differentiable persistent key-value state;
- query-conditioned state addressing;
- differentiable CASM operator selection/execution;
- post-action verifier and state update.

This module intentionally avoids pretrained external models so the first
confirmatory benchmark isolates the integrated chain itself.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F


OPS = ("xor", "and", "or", "xnor")
LATENT_BITS = 12


@dataclass(frozen=True)
class IntegratedConfig:
    hidden_dim: int = 40
    entity_count: int = 16
    memory_slots: int = 16
    latent_bits: int = LATENT_BITS


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
    def __init__(self, hidden: int, entity_count: int, latent_bits: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 3 // 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.entity_head = nn.Linear(hidden, entity_count)
        self.bit_head = nn.Linear(hidden, latent_bits)

    def forward(self, text: Tensor, image: Tensor, audio: Tensor) -> Tensor:
        return self.fuse(torch.cat([text, image, audio], dim=-1))


class PersistentKeyValueState(nn.Module):
    """Differentiable entity-addressed persistent state."""

    def __init__(self, hidden: int, entity_count: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.entity_count = entity_count
        self.slot_key = nn.Parameter(torch.randn(entity_count, hidden) * 0.05)
        self.write_value = nn.Linear(hidden, hidden)
        self.write_entity = nn.Linear(hidden, entity_count)
        self.read_query = nn.Linear(hidden, hidden)

    def initial(self, batch: int, device: torch.device) -> Tensor:
        return torch.zeros(batch, self.entity_count, self.hidden, device=device)

    def write(
        self,
        memory: Tensor,
        z: Tensor,
        strength: Tensor | None = None,
        target_entity: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if target_entity is None:
            write_probs = F.softmax(self.write_entity(z), dim=-1)
        else:
            write_probs = F.one_hot(
                target_entity.long(), num_classes=self.entity_count
            ).to(z.dtype)
        if strength is not None:
            write_probs = write_probs * strength.view(-1, 1)
        value = self.write_value(z).unsqueeze(1)
        memory = (
            (1.0 - write_probs.unsqueeze(-1)) * memory
            + write_probs.unsqueeze(-1) * value
        )
        return memory, write_probs

    def read(self, memory: Tensor, query: Tensor) -> tuple[Tensor, Tensor]:
        q = self.read_query(query)
        attention = F.softmax(
            q @ self.slot_key.t() / (self.hidden ** 0.5), dim=-1
        )
        read = torch.einsum("bn,bnd->bd", attention, memory)
        return read, attention


class CASMOperators(nn.Module):
    """Small exact differentiable Boolean operator bank."""

    def __init__(self, hidden: int, latent_bits: int) -> None:
        super().__init__()
        self.selector = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(OPS)),
        )
        self.bit_head = nn.Linear(hidden, latent_bits)

    def forward(
        self,
        state: Tensor,
        query: Tensor,
        i: Tensor,
        j: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        op_prob = F.softmax(self.selector(torch.cat([state, query], dim=-1)), dim=-1)
        bits = torch.sigmoid(self.bit_head(state))
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
        return (op_prob * values).sum(-1), op_prob, bits


class IntegratedE2EModel(nn.Module):
    """Representation -> state -> address -> CASM -> action -> verify -> write."""

    def __init__(
        self,
        config: IntegratedConfig = IntegratedConfig(),
        vocab_size: int = 40,
        operator_count: int = len(OPS),
    ) -> None:
        super().__init__()
        self.config = config
        self.text = TextEncoder(vocab_size, config.hidden_dim)
        self.image = ImageEncoder(config.hidden_dim)
        self.audio = AudioEncoder(config.hidden_dim)
        self.rep = SharedRepresentation(
            config.hidden_dim, config.entity_count, config.latent_bits
        )
        self.state = PersistentKeyValueState(config.hidden_dim, config.entity_count)
        self.entity_query = nn.Embedding(config.entity_count, config.hidden_dim)
        self.index_query = nn.Embedding(config.latent_bits, config.hidden_dim)
        self.operator_query = nn.Embedding(operator_count, config.hidden_dim)
        self.casm = CASMOperators(config.hidden_dim, config.latent_bits)
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

    def encode(
        self,
        text: Tensor,
        image: Tensor,
        audio: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        t = self.text(text)
        v = self.image(image)
        a = self.audio(audio)
        z = self.rep(t, v, a)
        return z, t, v, a

    def query_vector(
        self,
        entity: Tensor,
        i: Tensor,
        j: Tensor,
        op: Tensor,
    ) -> Tensor:
        return (
            self.entity_query(entity)
            + self.index_query(i)
            + self.index_query(j)
            + self.operator_query(op)
        )

    def observation_write(
        self, memory: Tensor, z: Tensor
    ) -> tuple[Tensor, Tensor]:
        return self.state.write(memory, z)

    def query(
        self,
        memory: Tensor,
        entity: Tensor,
        i: Tensor,
        j: Tensor,
        op: Tensor,
    ) -> dict[str, Tensor]:
        q = self.query_vector(entity, i, j, op)
        state_read, attention = self.state.read(memory, q)
        action, op_prob, bits = self.casm(state_read, q, i, j)
        logits = torch.stack((-action, action), dim=-1)
        return {
            "query": q,
            "entity": entity,
            "read": state_read,
            "attention": attention,
            "action": action,
            "logits": logits,
            "operator_prob": op_prob,
            "bits": bits,
        }

    def post_action_update(
        self,
        memory: Tensor,
        query_result: dict[str, Tensor],
        outcome: Tensor,
    ) -> tuple[Tensor, Tensor]:
        feedback = outcome.float().view(-1, 1)
        verifier_input = torch.cat(
            [query_result["read"], query_result["action"].view(-1, 1)],
            dim=-1,
        )
        verifier_logit = self.verifier(verifier_input)
        verifier_prob = verifier_logit.sigmoid()
        outcome_embedding = self.outcome_project(feedback)
        # Environment feedback is post-action evidence. Only a successful
        # outcome may create/update authoritative state; the learned verifier
        # controls the strength of that valid write.
        gate = feedback * torch.sigmoid(
            self.write_gate(
                torch.cat(
                    [query_result["read"], outcome_embedding, verifier_prob],
                    dim=-1,
                )
            )
        )
        memory, _ = self.state.write(
            memory,
            query_result["read"] + outcome_embedding,
            strength=gate,
            target_entity=query_result["entity"],
        )
        verifier_target = feedback
        return memory, torch.cat([verifier_logit, verifier_target], dim=-1)

    def forward_episode(
        self,
        observations: dict[str, Tensor],
        observation_entities: Tensor,
        payloads: Tensor,
        q1: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        q2: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ) -> dict[str, Tensor]:
        memory = self.state.initial(observations["text"].shape[1], observations["text"].device)
        entity_loss = 0.0
        bit_loss = 0.0
        for t in range(observations["text"].shape[0]):
            z, _, _, _ = self.encode(
                observations["text"][t],
                observations["image"][t],
                observations["audio"][t],
            )
            entity_loss = entity_loss + F.cross_entropy(
                self.rep.entity_head(z), observation_entities[t]
            )
            bit_loss = bit_loss + F.binary_cross_entropy_with_logits(
                self.rep.bit_head(z), payloads[t]
            )
            memory, _ = self.observation_write(memory, z)

        def run_query(
            query_tuple: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
            mem: Tensor,
        ) -> tuple[dict[str, Tensor], Tensor, Tensor]:
            entity, i, j, op, target = query_tuple
            out = self.query(mem, entity, i, j, op)
            action_loss = F.cross_entropy(out["logits"], target)
            operator_loss = F.cross_entropy(
                torch.log(out["operator_prob"] + 1e-8), op
            )
            return out, action_loss, operator_loss

        out1, action1, op1 = run_query(q1, memory)
        memory, verifier1 = self.post_action_update(memory, out1, q1[-1])
        out2, action2, op2 = run_query(q2, memory)
        _, verifier2 = self.post_action_update(memory, out2, q2[-1])

        verifier_loss = F.binary_cross_entropy_with_logits(
            verifier1[:, :1], verifier1[:, 1:]
        ) + F.binary_cross_entropy_with_logits(
            verifier2[:, :1], verifier2[:, 1:]
        )
        return {
            "action1_loss": action1,
            "action2_loss": action2,
            "operator_loss": (op1 + op2) / 2,
            "entity_loss": entity_loss / observations["text"].shape[0],
            "bit_loss": bit_loss / observations["text"].shape[0],
            "verifier_loss": verifier_loss / 2,
            "q1": out1["logits"],
            "q2": out2["logits"],
            "q1_attention": out1["attention"],
            "q2_attention": out2["attention"],
            "loss": (
                action1
                + action2
                + 0.15 * (op1 + op2)
                + 0.20 * verifier_loss
                + 0.15 * bit_loss / observations["text"].shape[0]
                + 0.05 * entity_loss / observations["text"].shape[0]
            ),
        }
