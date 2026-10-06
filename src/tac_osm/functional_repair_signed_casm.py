"""Development-only functional repair for the E2E multimodal learner.

This lane addresses a measured optimization pathology in the original fixed
probability-space Boolean CASM: its action gradient has a stationary point
when latent bits are uncertain. The repair keeps Boolean semantics exact at
the decision boundary but trains the CASM on signed latent logits.

This module is not a scientific result and is not used to evaluate the retired
E2E-005 holdout.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .integrated_e2e_005 import FunctionalConfig, FunctionalMultimodalPLM


def signed_boolean_margins(a: Tensor, b: Tensor) -> Tensor:
    """Return signed margins for XOR, AND, OR, XNOR."""
    return torch.stack(
        (
            -a * b,
            torch.minimum(a, b),
            torch.maximum(a, b),
            a * b,
        ),
        dim=-1,
    )


class SignedBooleanCASM(nn.Module):
    """Exact Boolean decisions with a differentiable signed-logit margin.

    Each latent bit is represented by a signed logit. At inference its Boolean
    value is sign(logit). The operator margin has the same truth table under
    that thresholding:
      XOR  = -a*b
      AND  = min(a,b)
      OR   = max(a,b)
      XNOR =  a*b
    where a,b are latent bit logits.

    The forward action is sigmoid(margin), so action >= 0.5 is exactly the
    Boolean result of thresholding the two latent bits.
    """

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
        bit_logits = self.decoder(state_read)
        a = bit_logits.gather(1, i.view(-1, 1)).squeeze(1)
        b = bit_logits.gather(1, j.view(-1, 1)).squeeze(1)
        margins = signed_boolean_margins(a, b)
        margin = margins.gather(1, op.view(-1, 1)).squeeze(1)
        action = torch.sigmoid(margin)
        logits = torch.stack((-margin, margin), dim=-1)
        bit_probs = torch.sigmoid(bit_logits)
        return action, logits, bit_probs


class FunctionalRepairPLM(FunctionalMultimodalPLM):
    """E2E-005-compatible model using the repaired signed Boolean CASM."""

    def __init__(
        self,
        config: FunctionalConfig = FunctionalConfig(),
        vocab_size: int = 40,
    ) -> None:
        super().__init__(config=config, vocab_size=vocab_size)
        self.casm = SignedBooleanCASM(config.hidden_dim, config.latent_bits)