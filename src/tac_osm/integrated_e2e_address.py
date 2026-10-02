"""E2E-002 persistent-state write/read addressing diagnosis.

This reuses the frozen E2E-001 encoder/CASM stack and changes only the
persistent-state address intervention. Explicit entity addresses are metadata
available at observation/write time and query/read time; no answer labels or
evaluation outcomes are supplied before action.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .integrated_e2e import (
    OPS,
    IntegratedConfig,
    IntegratedE2EModel,
    PersistentKeyValueState,
)


@dataclass(frozen=True)
class AddressDiagnosisConfig(IntegratedConfig):
    write_mode: str = "learned"
    read_mode: str = "learned"


class AddressDiagnosisState(nn.Module):
    def __init__(self, hidden: int, entity_count: int, write_mode: str, read_mode: str):
        super().__init__()
        if write_mode not in {"learned", "explicit"}:
            raise ValueError("invalid write_mode")
        if read_mode not in {"learned", "explicit"}:
            raise ValueError("invalid read_mode")
        self.hidden = hidden
        self.entity_count = entity_count
        self.write_mode = write_mode
        self.read_mode = read_mode
        self.slot_key = nn.Parameter(torch.randn(entity_count, hidden) * 0.05)
        self.write_value = nn.Linear(hidden, hidden)
        self.write_entity = nn.Linear(hidden, entity_count)
        self.read_query = nn.Linear(hidden, hidden)

    def initial(self, batch: int, device: torch.device) -> Tensor:
        return torch.zeros(batch, self.entity_count, self.hidden, device=device)

    def write(
        self, memory: Tensor, z: Tensor, entity: Tensor | None = None,
        strength: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if self.write_mode == "explicit":
            if entity is None:
                raise ValueError("explicit write requires entity")
            probs = F.one_hot(
                entity.long(), num_classes=self.entity_count
            ).to(z.dtype)
        else:
            probs = F.softmax(self.write_entity(z), dim=-1)
        if strength is not None:
            probs = probs * strength.view(-1, 1)
        value = self.write_value(z).unsqueeze(1)
        memory = (
            (1.0 - probs.unsqueeze(-1)) * memory
            + probs.unsqueeze(-1) * value
        )
        return memory, probs

    def read(
        self, memory: Tensor, query: Tensor, entity: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if self.read_mode == "explicit":
            if entity is None:
                raise ValueError("explicit read requires entity")
            attention = F.one_hot(
                entity.long(), num_classes=self.entity_count
            ).to(query.dtype)
        else:
            q = self.read_query(query)
            attention = F.softmax(
                q @ self.slot_key.t() / (self.hidden ** 0.5), dim=-1
            )
        read = torch.einsum("bn,bnd->bd", attention, memory)
        return read, attention


class AddressDiagnosisModel(IntegratedE2EModel):
    """Same trainable multimodal/CASM chain with address intervention only."""

    def __init__(self, config: AddressDiagnosisConfig = AddressDiagnosisConfig()):
        super().__init__(config)
        self.config = config
        self.state = AddressDiagnosisState(
            config.hidden_dim,
            config.entity_count,
            config.write_mode,
            config.read_mode,
        )

    def observation_write(
        self, memory: Tensor, z: Tensor, entity: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        return self.state.write(memory, z, entity=entity)

    def query(
        self, memory: Tensor, entity: Tensor, i: Tensor, j: Tensor, op: Tensor
    ) -> dict[str, Tensor]:
        q = self.query_vector(entity, i, j, op)
        state_read, attention = self.state.read(memory, q, entity=entity)
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

    def forward_episode(
        self,
        observations: dict[str, Tensor],
        observation_entities: Tensor,
        payloads: Tensor,
        q1: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        q2: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ) -> dict[str, Tensor]:
        memory = self.state.initial(
            observations["text"].shape[1], observations["text"].device
        )
        entity_loss = 0.0
        bit_loss = 0.0
        write_correct = 0.0
        for t in range(observations["text"].shape[0]):
            z, _, _, _ = self.encode(
                observations["text"][t],
                observations["image"][t],
                observations["audio"][t],
            )
            entity_logits = self.rep.entity_head(z)
            entity_loss = entity_loss + F.cross_entropy(
                entity_logits, observation_entities[t]
            )
            bit_loss = bit_loss + F.binary_cross_entropy_with_logits(
                self.rep.bit_head(z), payloads[t]
            )
            memory, probs = self.observation_write(
                memory, z, observation_entities[t]
            )
            write_correct = write_correct + (
                probs.argmax(-1) == observation_entities[t]
            ).float().mean()

        def run_query(
            query_tuple: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
            mem: Tensor,
        ) -> tuple[dict[str, Tensor], Tensor, Tensor]:
            entity, i, j, op, target = query_tuple
            out = self.query(mem, entity, i, j, op)
            return (
                out,
                F.cross_entropy(out["logits"], target),
                F.cross_entropy(
                    torch.log(out["operator_prob"] + 1e-8), op
                ),
            )

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
            "write_target_accuracy": write_correct / observations["text"].shape[0],
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
