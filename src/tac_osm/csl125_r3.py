"""CSL-125-R3 research-only risk-conditioned output revision (NO FIT).

Distinct descendant of R2; R2/R1/state-scan sources are unchanged. The shared
risk head predicts whether the base next-token argmax is wrong. A rank-limited,
bounded residual proposes revised logits without modifying carried memory.
Neither error detection nor revision has been trained, calibrated or shown to
improve accuracy. No self-certification, tool approval or state repair.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tac_osm.csl125 import CSL125State
from tac_osm.csl125_r2 import CSL125R2, CSL125R2Config


@dataclass(frozen=True)
class CSL125R3Config(CSL125R2Config):
    correction_rank: int = 32
    max_correction_gain: float = 0.25
    initial_error_bias: float = -2.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.correction_rank) is not int or not 1 <= self.correction_rank <= self.width:
            raise ValueError('correction_rank must be in [1,width]')
        if not isfinite(self.max_correction_gain) or not 0 < self.max_correction_gain <= 0.5:
            raise ValueError('max_correction_gain must be finite and in (0,0.5]')
        if not isfinite(self.initial_error_bias) or abs(self.initial_error_bias) > 6:
            raise ValueError('initial_error_bias must be finite and within [-6,6]')


class CSL125R3(CSL125R2):
    """A single shared verifier/reviser for all registered depth exits.

    The proposal path never changes R2 state. Exits are NOT dynamically
    switched across a live stream; a depth change still requires reset/replay.
    A risk logit is uncalibrated and cannot authorize actions or halting.
    """

    def __init__(self, config: CSL125R3Config = CSL125R3Config(), *, device=None):
        super().__init__(config, device=device)
        self.config = config
        with torch.device('cpu' if device is None else device):
            self.error_head = nn.Linear(config.width, 1, bias=True)
            self.revise_up = nn.Linear(config.width, config.correction_rank, bias=False)
            self.revise_down = nn.Linear(config.correction_rank, config.width, bias=True)
        if not self.error_head.weight.is_meta:
            nn.init.zeros_(self.error_head.weight)
            nn.init.constant_(self.error_head.bias, config.initial_error_bias)
            nn.init.normal_(self.revise_down.weight, mean=0., std=0.002)
            nn.init.zeros_(self.revise_down.bias)
        if self.parameter_count() > config.max_parameters:
            raise ValueError('R3 learned parameters exceed hard 125M cap')

    def _revised_features(self, z: Tensor, error_logit: Tensor) -> Tensor:
        # Bounded correction per hidden dimension, not a claim of error removal.
        delta = torch.tanh(self.revise_down(F.silu(self.revise_up(z))))
        weight = self.config.max_correction_gain * error_logit.sigmoid()
        return z + weight.to(dtype=z.dtype) * delta

    def forward(self, tokens: Tensor, *, depth: Optional[int] = None,
                state: Optional[CSL125State] = None, all_rungs: bool = False,
                stream_ids: Optional[Tuple[str, ...]] = None,
                emit_token_logits: bool = True,
                emit_repair_logits: bool = False) -> tuple[Dict[int, Dict[str, Tensor]], CSL125State]:
        # Keep the R2 state and causal layer ordering; add only an output read.
        if tokens.ndim != 2 or tokens.dtype != torch.long or tokens.shape[1] < 1:
            raise ValueError('tokens must be nonempty [B,T] integer IDs')
        d = self.config.layers if depth is None else depth
        if type(d) is not int or d not in self.config.rungs:
            raise ValueError('only preregistered exit depths supported')
        if stream_ids is not None and (len(stream_ids) != tokens.shape[0] or
            any(not isinstance(s, str) or not s for s in stream_ids) or
            len(set(stream_ids)) != len(stream_ids)):
            raise ValueError('stream_ids must uniquely name every batch row')
        h = self.embedding(tokens)
        if state is None:
            state = self.initial_state(tokens.shape[0], tokens.device, h.dtype, depth=d,
                                       stream_ids=stream_ids)
        elif stream_ids is None or state.stream_ids is None or state.stream_ids != stream_ids:
            raise ValueError('carried state requires unchanged explicit stream IDs')
        if len(state.layers) != d or len(state.attention) != d:
            raise ValueError('changing active depth requires reset/replay')

        layers, caches = list(state.layers), list(state.attention)
        outs: Dict[int, Dict[str, Tensor]] = {}
        for idx in range(d):
            h, layers[idx] = self.blocks[idx](h, layers[idx])
            layer_id = idx + 1
            if str(layer_id) in self.attention:
                if caches[idx] is None:
                    raise ValueError('missing local KV cache for active attention layer')
                h, caches[idx] = self.attention[str(layer_id)](h, caches[idx])
            if layer_id == d or (all_rungs and layer_id in self.config.rungs):
                z = self.final_norm(h)
                error_logit = self.error_head(z)
                out: Dict[str, Tensor] = {
                    'tool_logits': self.tool_head(z[:, -1]),
                    'field_logits': self.field_head(z),
                    'accept_logit': self.accept_head(z[:, -1]),
                    'error_logit': error_logit.squeeze(-1),
                }
                if emit_token_logits:
                    out['token_logits'] = F.linear(z, self.embedding.weight)
                if emit_repair_logits:
                    out['repair_logits'] = F.linear(
                        self._revised_features(z, error_logit), self.embedding.weight)
                outs[layer_id] = out
        return outs, CSL125State(tuple(layers), tuple(caches), state.stream_ids)

    def nested_repair_loss(self, tokens: Tensor, *,
                           weights: Optional[Dict[int, float]] = None,
                           loss_mask: Optional[Tensor] = None,
                           risk_weight: float = 0.1,
                           breakage_weight: float = 0.2) -> Tensor:
        """Teacher-forced research loss; labels appear ONLY after forward.

        Base/corrected NLL are supervised. Binary error labels are derived from
        detached base argmax, never exposed to the forward model. Penalize a
        correction that makes an initially correct target worse. The loss does
        NOT establish a new state-correction ability or calibrated reliability.
        """
        if tokens.ndim != 2 or tokens.dtype != torch.long or tokens.shape[1] < 2:
            raise ValueError('next-token training needs [B,T>=2] integer tokens')
        if any(not isfinite(v) or v < 0 for v in (risk_weight, breakage_weight)):
            raise ValueError('loss coefficients must be finite and nonnegative')
        if weights is None:
            weights = {r: 1. for r in self.config.rungs}
        if (set(weights) != set(self.config.rungs) or
            any(type(v) not in (int, float) or not isfinite(v) or v < 0
                for v in weights.values()) or sum(weights.values()) <= 0):
            raise ValueError('invalid per-exit loss weights')
        target = tokens[:, 1:]
        if loss_mask is None:
            mask = torch.ones_like(target, dtype=torch.bool)
        else:
            if (loss_mask.dtype != torch.bool or loss_mask.shape != target.shape or
                    not loss_mask.any()):
                raise ValueError('loss_mask must be nonempty boolean next-target mask')
            mask = loss_mask
        out, _ = self.forward(tokens, all_rungs=True, emit_repair_logits=True)
        total = tokens.new_zeros((), dtype=torch.float32)
        for rung in self.config.rungs:
            if weights[rung] == 0:
                continue
            base_logits = out[rung]['token_logits'][:, :-1]
            revised_logits = out[rung]['repair_logits'][:, :-1]
            base_nll = F.cross_entropy(base_logits.transpose(1, 2), target, reduction='none')
            revised_nll = F.cross_entropy(revised_logits.transpose(1, 2), target, reduction='none')
            incorrect = (base_logits.detach().argmax(dim=-1) != target)
            risk_loss = F.binary_cross_entropy_with_logits(
                out[rung]['error_logit'][:, :-1], incorrect.float(), reduction='none')
            breakage = F.relu(revised_nll - base_nll.detach()) * (~incorrect).float()
            item = base_nll + revised_nll + risk_weight * risk_loss + breakage_weight * breakage
            total = total + (weights[rung] / sum(weights.values())) * item[mask].mean()
        return total
