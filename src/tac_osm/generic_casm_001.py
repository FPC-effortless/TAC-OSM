from __future__ import annotations

import torch
from torch import Tensor, nn


class GenericCASM001(nn.Module):
    """Generic learned operator inference plus bilinear execution basis."""

    def __init__(
        self,
        input_dim: int = 8,
        output_dim: int = 8,
        hidden_dim: int = 64,
        basis_count: int = 8,
        rank: int = 4,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.basis_count = basis_count
        self.rank = rank

        self.support_encoder = nn.Sequential(
            nn.Linear(2 * input_dim + output_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.code_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, basis_count),
        )

        self.u_proj = nn.Parameter(torch.randn(basis_count, rank, input_dim) * 0.08)
        self.v_proj = nn.Parameter(torch.randn(basis_count, rank, input_dim) * 0.08)
        self.out_proj = nn.Parameter(torch.randn(basis_count, output_dim, rank) * 0.08)
        self.bias = nn.Parameter(torch.zeros(basis_count, output_dim))

    def infer_code(self, support_u: Tensor, support_v: Tensor, support_y: Tensor) -> Tensor:
        rows = torch.cat([support_u, support_v, support_y], dim=-1)
        return self.support_encoder(rows).mean(dim=1)

    def execute(self, u: Tensor, v: Tensor, code: Tensor) -> Tensor:
        weights = self.code_head(code).softmax(dim=-1)
        u_lat = torch.einsum("bqd,krd->bkqr", u, self.u_proj)
        v_lat = torch.einsum("bqd,krd->bkqr", v, self.v_proj)
        fused = u_lat * v_lat
        basis_out = torch.einsum("bkqr,kor->bkqo", fused, self.out_proj)
        basis_out = basis_out + self.bias.unsqueeze(0).unsqueeze(2)
        return torch.einsum("bk,bkqo->bqo", weights, basis_out)

    def forward(
        self,
        support_u: Tensor,
        support_v: Tensor,
        support_y: Tensor,
        query_u: Tensor,
        query_v: Tensor,
    ) -> Tensor:
        code = self.infer_code(support_u, support_v, support_y)
        return torch.tanh(self.execute(query_u, query_v, code))
