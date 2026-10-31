"""SENSE-TOK-004 — untrained, causally constrained *learnable* BYTE/TOKEN controller.

A controller chooses after observing each byte whether to commit that byte as
an individual BYTE representation, or retain it in a TOKEN buffer to be encoded
as one continuous multi-byte latent patch. A bound forces termination. Both
choices affect the recurrent state; every output predicts the NEXT byte before
its target is available. In surrogate mode a straight-through path carries the
byte-prediction gradient to the controller, while inference skips unselected
GRUCell candidates. This module contains NO optimiser, fitting, source dataset
or claim of learned behaviour. "TOKEN" means a learned latent *patch*, not a
BPE/subword vocabulary ID. The whole bytes and boundaries are retained for
mechanism reconstruction, not a learned compression rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from typing import Literal

import torch
from torch import Tensor, nn

from .causal_byte_probe import BOS_ID

Mode = Literal['inference', 'surrogate']
Force = Literal['byte', 'token']
EMBED = 32
HIDDEN = 110


@dataclass(frozen=True)
class RepresentationEvent:
    start: int
    end: int
    available_after: int
    mode: str
    payload: bytes
    forced: bool = False


@dataclass(frozen=True)
class AdaptiveAudit:
    observed_bytes: int
    decisions: tuple[str, ...]
    byte_probabilities: tuple[float, ...]
    events: tuple[RepresentationEvent, ...]
    pending: bytes
    recurrent_updates: int
    candidate_recurrent_calls: int
    byte_readouts: int
    eos_used: bool = False

    def verify(self, observed: bytes) -> None:
        if self.observed_bytes != len(observed):
            raise ValueError('observed count mismatch')
        if len(self.decisions) != len(observed) or len(self.byte_probabilities) != len(observed):
            raise ValueError('decision count mismatch')
        at = 0
        for ev in self.events:
            if ev.start != at or ev.end <= ev.start or ev.end > ev.available_after:
                raise ValueError('invalid representation span/time')
            if ev.available_after > self.observed_bytes:
                raise ValueError('future information exposed in representation')
            if ev.mode not in ('BYTE', 'TOKEN') or (ev.mode == 'BYTE') != (len(ev.payload) == 1):
                raise ValueError('byte/token type does not match span length')
            if observed[ev.start:ev.end] != ev.payload:
                raise ValueError('representation corrupt or misaligned')
            at = ev.end
        if observed[at:] != self.pending:
            raise ValueError('uncommitted data mismatch')
        if self.recurrent_updates != len(self.events):
            raise ValueError('recurrent work receipt incorrect')
        if any(not 0 <= prob <= 1 for prob in self.byte_probabilities):
            raise ValueError('invalid probability')
        if self.eos_used:
            raise ValueError('unobserved EOS must not appear during prediction')


class AdaptiveByteTokenGRU(nn.Module):
    """100,752 active weights; 0.207% above prior 100,544-weight sparse model.

    Decision for observed byte x[t-1] uses its embedding, the previous global
    state, the pending local summary and PRE-observation predictive entropy.
    It never reads x[t]. Min/max token size is a mechanical safety bound, not
    a lexical boundary or hard-coded notion of semantic importance.

    Inference: only recurrent calls for committed spans are executed.
    Surrogate: computes candidate BYTE and TOKEN branches, then inserts an
    identity-forward, nonzero-gradient straight-through switch. Surrogate is
    a training-ready approximation, not cheaper training or an unbiased
    estimator of the discrete-choice objective.
    """

    def __init__(self, *, max_token_bytes: int = 8, threshold: float = 0.5,
                 temperature: float = 1.0) -> None:
        super().__init__()
        if not 2 <= max_token_bytes <= 64:
            raise ValueError('max_token_bytes must be between 2 and 64')
        if not 0 < threshold < 1 or not isfinite(threshold):
            raise ValueError('threshold must be finite within (0,1)')
        if not 0 < temperature <= 100 or not isfinite(temperature):
            raise ValueError('temperature must be finite in (0,100]')
        self.max_token_bytes = max_token_bytes
        self.threshold = threshold
        self.temperature = temperature
        self.byte_emb = nn.Embedding(257, EMBED)
        self.gru_cell = nn.GRUCell(EMBED, HIDDEN)
        self.byte_head = nn.Linear(HIDDEN + EMBED * 2, 256)
        # 175 features -> 1 logit: 176 trainable parameters.
        self.mode_gate = nn.Linear(HIDDEN + EMBED * 2 + 1, 1)
        # Order-sensitive learned position weighting within token patches.
        # A mean pool is order-blind ("ab" == "ba" for pooled inputs).
        self.position_recency = nn.Parameter(torch.zeros(EMBED))

    def _pool(self, items: list[Tensor]) -> Tensor:
        if not items:
            raise ValueError('cannot pool empty representation')
        if len(items) == 1:
            return items[0]
        seq = torch.cat(items, dim=0)  # [n,32], chronological order
        positions = torch.arange(len(items), device=seq.device, dtype=seq.dtype)
        weights = torch.softmax(positions[:, None] * self.position_recency[None, :], dim=0)
        return (weights * seq).sum(dim=0, keepdim=True)

    def forward(self, prev_ids: Tensor, *, mode: Mode = 'inference',
                force: Force | None = None, audit: bool = False):
        if mode not in ('inference', 'surrogate'):
            raise ValueError('unknown execution mode')
        if force not in (None, 'byte', 'token'):
            raise ValueError('force is diagnostic-only: byte/token/None')
        if not isinstance(prev_ids, Tensor) or prev_ids.dtype != torch.long or prev_ids.ndim != 2:
            raise ValueError('prev_ids must be torch.long [batch,time]')
        if prev_ids.shape[1] and not bool(torch.all(prev_ids[:, 0] == BOS_ID)):
            raise ValueError('all rows must begin with BOS')
        if prev_ids.shape[1] > 1 and not bool(torch.all((prev_ids[:, 1:] >= 0) & (prev_ids[:, 1:] < 256))):
            raise ValueError('only observed 0..255 bytes may follow BOS')
        logits: list[Tensor] = []
        audits: list[AdaptiveAudit] = []
        costs: list[Tensor] = []
        for row in prev_ids:
            result, receipt, proxy = self._one(row, mode=mode, force=force)
            logits.append(result)
            audits.append(receipt)
            costs.append(proxy)
        output = (torch.stack(logits, dim=0) if logits else
                  self.byte_head.weight.new_empty((0, prev_ids.size(1), 256)))
        proxy_cost = (torch.stack(costs).mean() if costs else self.byte_head.weight.sum() * 0)
        return (output, tuple(audits), proxy_cost) if audit else output

    def _one(self, ids: Tensor, *, mode: Mode, force: Force | None):
        h = self.byte_emb.weight.new_zeros((1, HIDDEN))
        pending: list[Tensor] = []
        pending_bytes: list[int] = []
        predictions: list[Tensor] = []
        choices: list[str] = []
        probabilities: list[float] = []
        events: list[RepresentationEvent] = []
        proxy_cost = h.new_zeros(())
        actual_updates = 0
        candidates = 0
        observed = 0

        def event(payload: bytes, *, forced: bool = False) -> None:
            nonlocal actual_updates
            if not payload:
                raise AssertionError('empty representation')
            end = observed
            # In BYTE branch, a pending token is committed before current byte;
            # the current observed byte may therefore be in a separate event.
            if events:
                start = events[-1].end
            else:
                start = 0
            if start != end - len(payload):
                raise AssertionError('representation endpoints invalid')
            events.append(RepresentationEvent(
                start=start, end=end, available_after=observed,
                mode='BYTE' if len(payload) == 1 else 'TOKEN',
                payload=payload, forced=forced,
            ))
            actual_updates += 1

        for i, idx in enumerate(ids.unbind(0)):
            current = self.byte_emb(idx.view(1))
            if i:
                # x[i-1] has now been observed. x[i] remains unavailable.
                b = int(idx.item())
                observed += 1
                prev_logits = predictions[-1].detach()
                logp = torch.log_softmax(prev_logits, dim=-1)
                entropy = -(logp.exp() * logp).sum() / log(256)
                local = (self._pool(pending) if pending else torch.zeros_like(current))
                features = torch.cat((h, current, local, entropy.reshape(1, 1)), dim=-1)
                prob = torch.sigmoid(self.mode_gate(features) / self.temperature).reshape(())
                do_byte = (force == 'byte') or (force is None and bool((prob >= self.threshold).detach().item()))
                probabilities.append(float(prob.detach().item()))
                choices.append('BYTE' if do_byte else 'TOKEN')
                must_finish = (len(pending) + 1) >= self.max_token_bytes
                expected_byte_cost = 1 + int(bool(pending))
                decision_prob = (prob if force is None else
                                 (torch.ones_like(prob) if do_byte else torch.zeros_like(prob)))
                proxy_cost = proxy_cost + decision_prob * expected_byte_cost + (1-decision_prob) * int(must_finish)
                if mode == 'surrogate':
                    # Both alternatives computed for gradient attribution.
                    if pending:
                        prev_rep = self._pool(pending)
                        preh = self.gru_cell(prev_rep, h)
                        candidates += 1
                    else:
                        preh = h
                    byte_h = self.gru_cell(current, preh)
                    candidates += 1
                    if must_finish:
                        token_rep = self._pool(pending + [current])
                        token_h = self.gru_cell(token_rep, h)
                        candidates += 1
                    else:
                        token_h = h
                    forward_h = byte_h if do_byte else token_h
                    # Exactly equal to selected forward branch. Gate gradients
                    # receive downstream loss via nonzero surrogate derivative.
                    if force is None:
                        h = forward_h + (prob - prob.detach()) * (byte_h - token_h)
                    else:
                        h = forward_h
                if do_byte:
                    if pending:
                        if mode == 'inference':
                            h = self.gru_cell(self._pool(pending), h)
                            candidates += 1
                        # Pending token ends BEFORE current byte. At its
                        # reveal time one further byte has been observed.
                        old = bytes(pending_bytes)
                        old_end = observed - 1
                        last = events[-1].end if events else 0
                        if last + len(old) != old_end:
                            raise AssertionError('pending token offset mismatch')
                        events.append(RepresentationEvent(
                            start=last, end=old_end, available_after=observed,
                            mode='BYTE' if len(old) == 1 else 'TOKEN', payload=old))
                        actual_updates += 1
                        pending.clear()
                        pending_bytes.clear()
                    if mode == 'inference':
                        h = self.gru_cell(current, h)
                        candidates += 1
                    event(bytes([b]))
                else:
                    pending.append(current)
                    pending_bytes.append(b)
                    if must_finish:
                        if mode == 'inference':
                            h = self.gru_cell(self._pool(pending), h)
                            candidates += 1
                        event(bytes(pending_bytes), forced=True)
                        pending.clear()
                        pending_bytes.clear()
            local = self._pool(pending) if pending else torch.zeros_like(current)
            predictions.append(self.byte_head(torch.cat((h, current, local), dim=-1)).reshape(256))
        result = (torch.stack(predictions, 0) if predictions else
                  self.byte_head.weight.new_empty((0, 256)))
        receipt = AdaptiveAudit(
            observed_bytes=observed, decisions=tuple(choices),
            byte_probabilities=tuple(probabilities), events=tuple(events),
            pending=bytes(pending_bytes), recurrent_updates=actual_updates,
            candidate_recurrent_calls=candidates,
            byte_readouts=len(predictions), eos_used=False,
        )
        receipt.verify(bytes(ids[1:].detach().tolist()))
        return result, receipt, proxy_cost / max(1, observed)


def adaptive_parameter_receipt(model: AdaptiveByteTokenGRU) -> dict:
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    deployed = sum(p.numel() for p in model.parameters())
    return {
        'trainable_parameters': count,
        'deployed_learned_parameters': deployed,
        'reference_sparse_parameters': 100544,
        'relative_parameter_gap': abs(count - 100544) / 100544,
        'within_100k_band': 95000 <= count <= 105000 and 95000 <= deployed <= 105000,
        'within_1pct_vs_sparse': abs(count - 100544) / max(count, 100544) <= .01,
        'training_authorized': False,
        'semantic_tokenization_demonstrated': False,
    }
