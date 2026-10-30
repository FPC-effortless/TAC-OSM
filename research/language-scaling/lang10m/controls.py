"""LANG10M002: untrained ~10M vocabulary-matched language controls.

All controls accept [B,T] token IDs 0..4095 and return [B,T,4096]
causal next-token logits. Each is constructed from scratch; no pretrained
weights, no optimizer steps, no TPU/GPU use, and no model-quality claim.
These are distinct architectures; the diagonal SSM is NOT Mamba/LFM3.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

VOCAB = 4096
MAX_CONTEXT = 1024
TARGET_TAC = 10_001_188
CONTROL_COUNTS = {
    "gru": 9_995_660,
    "selective_diagonal_ssm": 10_002_288,
    "causal_transformer": 10_001_940,
}


def check_tokens(tokens: torch.Tensor) -> None:
    if (tokens.ndim != 2 or tokens.dtype != torch.long or
            tokens.shape[1] < 1 or tokens.shape[1] > MAX_CONTEXT or
            tokens.shape[0] < 1):
        raise ValueError("token IDs must be long [B,1..1024]")
    # Do not inspect label/future data. Embedding performs range checks.


class GRULanguage10M(nn.Module):
    """Native PyTorch two-layer gated recurrent control (806 state dims)."""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, 128)
        self.recurrent = nn.GRU(128, 806, num_layers=2, batch_first=True)
        self.head = nn.Linear(806, VOCAB)

    def forward(self, tokens):
        check_tokens(tokens)
        y, _ = self.recurrent(self.embedding(tokens))
        return self.head(y)


class DiagonalSelectiveBlock(nn.Module):
    """Input-selective stable diagonal state, *not* a Mamba implementation.

    alpha_t = exp(-softplus(d) * sigmoid(gate(z_t)))
    h_t = alpha_t * h_{t-1} + (1-alpha_t) * tanh(input(z_t)+offer(z_t))
    The update is O(T*width) state arithmetic; Python scan is unoptimized.
    """
    def __init__(self, width=256, ffn_width=2222):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.input = nn.Linear(width, width)
        self.gate = nn.Linear(width, width)
        self.offer = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.raw_decay = nn.Parameter(torch.zeros(width))
        self.ffn_up = nn.Linear(width, ffn_width)
        self.ffn_gate = nn.Linear(width, ffn_width)
        self.ffn_down = nn.Linear(ffn_width, width)

    def forward(self, x):
        z = self.norm(x)
        decay = F.softplus(self.raw_decay).view(1, 1, -1)
        alpha = torch.exp(-decay * torch.sigmoid(self.gate(z)))
        offer = torch.tanh(self.input(z) + self.offer(z))
        h = torch.zeros_like(offer[:, 0, :])
        states = []
        for time_index in range(z.shape[1]):
            h = alpha[:, time_index] * h + (1 - alpha[:, time_index]) * offer[:, time_index]
            states.append(h)
        result = x + self.output(torch.stack(states, dim=1))
        u = self.norm(result)
        return result + self.ffn_down(
            self.ffn_up(u) * F.silu(self.ffn_gate(u))
        )


class DiagonalSelectiveSSMLanguage10M(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, 256)
        self.blocks = nn.ModuleList(
            [DiagonalSelectiveBlock() for _ in range(4)]
        )
        self.norm = nn.LayerNorm(256)
        self.head = nn.Linear(256, VOCAB)

    def forward(self, tokens):
        check_tokens(tokens)
        z = self.embedding(tokens)
        for block in self.blocks:
            z = block(z)
        return self.head(self.norm(z))


class CausalTransformerLanguage10M(nn.Module):
    """Six 256d causal-attention blocks, 4 heads, ffn=1966, no dropout."""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, 256)
        self.position = nn.Embedding(MAX_CONTEXT, 256)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=256, nhead=4, dim_feedforward=1966,
                dropout=0.0, activation="gelu",
                batch_first=True, norm_first=True,
            )
            for _ in range(6)
        ])
        self.norm = nn.LayerNorm(256)
        self.head = nn.Linear(256, VOCAB)

    def forward(self, tokens):
        check_tokens(tokens)
        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        z = self.embedding(tokens) + self.position(positions)[None, :, :]
        mask = torch.ones((length, length), dtype=torch.bool, device=tokens.device).triu(1)
        for block in self.blocks:
            z = block(z, src_mask=mask, is_causal=True)
        return self.head(self.norm(z))


def make_control(arm: str) -> nn.Module:
    builders = {
        "gru": GRULanguage10M,
        "selective_diagonal_ssm": DiagonalSelectiveSSMLanguage10M,
        "causal_transformer": CausalTransformerLanguage10M,
    }
    if arm not in builders:
        raise ValueError("unregistered arm")
    model = builders[arm]()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    deployed = sum(p.numel() for p in model.parameters())
    if count != CONTROL_COUNTS[arm] or deployed != count:
        raise ValueError(f"wrong model count for {arm}: {count}/{deployed}")
    if abs(count / TARGET_TAC - 1.0) > 0.01:
        raise ValueError("not within 1% param parity")
    return model


def status() -> dict:
    return {
        "study": "LANG10M002",
        "trained": False,
        "model_quality": "NOT_MEASURED",
        "matched_tokenizer": "LANG10M001_BPE4096_V1",
        "params": {"tac": TARGET_TAC, **CONTROL_COUNTS},
        "architecture_superiority": "NOT_CLAIMED",
        "training_compute_parity": "UNMEASURED",
        "ssm_is_mamba": False,
        "causal_window_reset": True,
    }
