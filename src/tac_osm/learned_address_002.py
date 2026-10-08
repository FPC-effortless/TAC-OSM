"""Learned associative address metric for PLM learned-address-002."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class LearnedAddressMetric002(nn.Module):
    """Generic learned query/key metric with a learned candidate bias.

    The metric is initialized to the identity bilinear form so the initial
    scorer is the raw dot-product reference. The candidate-key bias starts at
    zero and is learned from the structured-channel training stream. At
    inference there is no hard-coded raw-dot branch: the returned score is the
    learned metric itself.
    """

    def __init__(self, address_dim: int = 16, hidden_dim: int = 32) -> None:
        super().__init__()
        self.address_dim = address_dim
        self.query = nn.Linear(address_dim, address_dim, bias=False)
        self.key = nn.Linear(address_dim, address_dim, bias=False)
        self.candidate_bias = nn.Sequential(
            nn.Linear(address_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        with torch.no_grad():
            self.query.weight.copy_(torch.eye(address_dim))
            self.key.weight.copy_(torch.eye(address_dim))
            self.candidate_bias[-1].weight.zero_()
            self.candidate_bias[-1].bias.zero_()

    def forward(self, query: Tensor, keys: Tensor) -> Tensor:
        if query.ndim != 2:
            raise ValueError("query must be [B,D]")
        if keys.ndim != 3:
            raise ValueError("keys must be [B,M,D]")
        if keys.shape[-1] != self.address_dim or query.shape[-1] != self.address_dim:
            raise ValueError("address dimension mismatch")
        q = self.query(query)
        k = self.key(keys)
        bilinear = torch.einsum("bd,bmd->bm", q, k)
        bias = self.candidate_bias(keys).squeeze(-1)
        return bilinear + bias
