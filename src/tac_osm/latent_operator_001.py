"""PLM variant for latent operator induction.

The target operator ID is never supplied to this model. A permutation-invariant
set encoder infers operator weights from four demonstration rows (a, b, y),
then the existing closed-form Boolean primitives are mixed by those weights.
Entity addressing remains explicit for this experiment.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .integrated_e2e_005 import FunctionalConfig, FunctionalMultimodalPLM


class SetOperatorInducer(nn.Module):
    """Permutation-invariant inference from truth-table demonstrations."""

    def __init__(self, hidden: int, operator_count: int = 4) -> None:
        super().__init__()
        demo_hidden = max(16, hidden // 2)
        self.demo = nn.Sequential(
            nn.Linear(3, demo_hidden),
            nn.GELU(),
            nn.Linear(demo_hidden, demo_hidden),
        )
        self.head = nn.Sequential(
            nn.Linear(demo_hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, operator_count),
        )

    def forward(self, support: Tensor) -> Tensor:
        # support: [batch, demonstrations, (a,b,y)]
        return self.head(self.demo(support.float()).mean(dim=1))


class LatentOperatorPLM(FunctionalMultimodalPLM):
    """FunctionalMultimodalPLM with operator ID removed from the query API."""

    def __init__(
        self,
        config: FunctionalConfig = FunctionalConfig(
            hidden_dim=64,
            state_write_mode="residual_linear",
        ),
        vocab_size: int = 40,
    ) -> None:
        super().__init__(config=config, vocab_size=vocab_size)
        self.operator_inducer = SetOperatorInducer(config.hidden_dim, 4)

    def infer_operator(self, support: Tensor) -> Tensor:
        return self.operator_inducer(support)

    def latent_query(
        self,
        memory: Tensor,
        entity: Tensor,
        i: Tensor,
        j: Tensor,
        support: Tensor,
        operator_override: Tensor | None = None,
    ) -> dict[str, Tensor]:
        state_read, attention = self.state.read(memory, entity)
        bits = torch.sigmoid(self.casm.decoder(state_read))
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

        operator_logits = self.infer_operator(support)
        if operator_override is None:
            operator_probs = operator_logits.softmax(dim=-1)
        else:
            operator_probs = F.one_hot(
                operator_override.long(),
                num_classes=operator_logits.shape[-1],
            ).to(values.dtype)

        action = (operator_probs * values).sum(dim=-1)
        eps = torch.finfo(action.dtype).eps
        action_logit = torch.logit(action.clamp(eps, 1.0 - eps))
        logits = torch.stack((-action_logit, action_logit), dim=-1)

        return {
            "entity": entity,
            "read": state_read,
            "attention": attention,
            "action": action,
            "logits": logits,
            "bits": bits,
            "operator_logits": operator_logits,
            "operator_probs": operator_probs,
            "operator_prediction": operator_probs.argmax(dim=-1),
        }
