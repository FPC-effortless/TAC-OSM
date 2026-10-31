"""SENSE-TOK-006 comparison candidates; NO FIT or optimizer in this module.

These are engineering candidates, not approved training registrations. The
original S004/S005 graph is reused. Exogenous random decisions require explicit
uniform draws from the runner, independent of targets and future observations.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .adaptive_byte_token import AdaptiveByteTokenGRU
from .adaptive_codebook_token import AdaptiveCodebookByteTokenGRU

ARM_NAMES = (
    "dense_byte", "forced_token", "adaptive_latent", "adaptive_codebook",
    "frozen_gate_weights", "random_gate",
)


def validate_previous_ids(ids: Tensor) -> None:
    if not isinstance(ids, Tensor) or ids.dtype != torch.long or ids.ndim != 2:
        raise ValueError("observed IDs must be int64 [batch,time]")
    if ids.shape[0] < 1 or ids.shape[1] < 1:
        raise ValueError("nonempty batch and BOS required")
    if not bool((ids[:, 0] == 256).all()):
        raise ValueError("every row must begin with BOS")
    if not bool(((ids[:, 1:] >= 0) & (ids[:, 1:] < 256)).all()):
        raise ValueError("only observed bytes may follow BOS")


class DenseByteGRU(nn.Module):
    """100,688 useful weights, with no permanently zero patch-head columns.

    e_t = E[previous_byte]; h_t = GRU(e_t,h_(t-1)); logits_t = W[h_t,e_t]+b.
    h is 118 wide; shared byte embeddings are 32 wide. No lexical rules.
    """
    def __init__(self) -> None:
        super().__init__()
        self.byte_emb = nn.Embedding(257, 32)
        self.gru_cell = nn.GRUCell(32, 118)
        self.byte_head = nn.Linear(150, 256)

    def forward(self, ids: Tensor) -> Tensor:
        validate_previous_ids(ids)
        hidden = self.byte_emb.weight.new_zeros((ids.shape[0], 118))
        outputs = []
        for item in ids.unbind(1):
            current = self.byte_emb(item)
            hidden = self.gru_cell(current, hidden)
            outputs.append(self.byte_head(torch.cat((hidden, current), -1)))
        return torch.stack(outputs, 1)


class _ExogenousGate(nn.Module):
    """No learned weights; direct invocation without reviewed draws is denied."""
    def forward(self, features: Tensor) -> Tensor:
        raise ValueError("random gate requires explicit per-example uniform draws")


class TokenComparisonArm(nn.Module):
    def __init__(self, name: str, *, seed: int, random_byte_probability: float | None = None):
        super().__init__()
        if name not in ARM_NAMES or type(seed) is not int or seed < 0:
            raise ValueError("unknown arm or invalid seed")
        if name == "random_gate" and (
            type(random_byte_probability) not in (int, float)
            or not math.isfinite(random_byte_probability)
            or not 0 < random_byte_probability < 1
        ):
            raise ValueError("random gate needs a frozen FIT-only probability in (0,1)")
        if name != "random_gate" and random_byte_probability is not None:
            raise ValueError("random probability supplied to a nonrandom arm")
        self.name, self.seed = name, seed
        self.random_byte_probability = random_byte_probability
        # Initialization is paired without perturbing the caller's RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if name == "dense_byte":
                self.backbone = DenseByteGRU()
            elif name == "adaptive_latent":
                self.backbone = AdaptiveByteTokenGRU()
            else:
                self.backbone = AdaptiveCodebookByteTokenGRU()
        if name in ("forced_token", "frozen_gate_weights"):
            self.backbone.mode_gate.requires_grad_(False)
        if name == "random_gate":
            self.backbone.mode_gate = _ExogenousGate()

    def forward(self, ids: Tensor, *, mode: str = "inference",
                random_draws: Tensor | None = None) -> Tensor:
        validate_previous_ids(ids)
        if mode not in ("inference", "surrogate"):
            raise ValueError("unknown execution mode")
        if self.name != "random_gate":
            if random_draws is not None:
                raise ValueError("random draws are only legal for the random control")
            if self.name == "dense_byte":
                return self.backbone(ids)
            return self.backbone(ids, mode=mode,
                                 force="token" if self.name == "forced_token" else None)
        if (not isinstance(random_draws, Tensor)
                or random_draws.shape != (ids.shape[0], ids.shape[1] - 1)
                or not torch.is_floating_point(random_draws)
                or not bool(torch.isfinite(random_draws).all())
                or not bool(((random_draws >= 0) & (random_draws < 1)).all())):
            raise ValueError("uniform draws must be finite [batch,time-1] in [0,1)")
        # S004/S005 execute rows then observed positions in chronological order.
        choices = iter((random_draws < self.random_byte_probability).flatten().tolist())
        gate = self.backbone.mode_gate
        calls = 0

        def exogenous(_module, features):
            nonlocal calls
            calls += 1
            return features[0].new_full((1, 1), 1000. if next(choices) else -1000.)

        # A pre-hook cannot suppress a module's forward exception. Temporarily
        # install a parameter-free callable and restore it even on failure.
        original = gate.forward
        gate.forward = lambda features: exogenous(gate, (features,))
        try:
            result = self.backbone(ids, mode=mode)
        finally:
            gate.forward = original
        if calls != random_draws.numel():
            raise ValueError("random route schedule was not consumed exactly once")
        return result

    def parameter_receipt(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        deployed = sum(p.numel() for p in self.parameters())
        return {"arm": self.name, "seed": self.seed,
                "trainable_parameters": trainable, "deployed_parameters": deployed,
                "random_byte_probability": self.random_byte_probability,
                "within_100k_band": 95000 <= trainable <= 105000 and 95000 <= deployed <= 105000,
                "training_authorized": False,
                "qualification": "instantiated_candidate_not_full_resource_parity"}
