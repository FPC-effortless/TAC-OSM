"""SENSE-TOK-003 no-fit patch-sparse, causally valid per-byte predictor.

A recurrent state update occurs only when a patch is *observably committed*.
The readout still predicts every byte and sees only previous observed bytes.
All segmenter choices share one architecture and 100,544 live learned weights.
This is NOT trained, does NOT implement CDL/CASM, and does not by itself prove
less end-to-end compute or memory use. Patch graph traces retain O(n) bytes.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .causal_byte_probe import BOS_ID
from .causal_patch_stream import CausalPatchStream, STREAM_VARIANTS, StreamVariant

EMBED = 32
HIDDEN = 110


@dataclass(frozen=True)
class SparseAudit:
    variant: str
    observed_bytes: int
    recurrent_updates: int
    recurrent_update_times: tuple[int, ...]
    committed_patch_ends: tuple[int, ...]
    delayed_commits: int
    byte_readouts: int
    eos_used: bool = False


class PatchSparseByteGRU(nn.Module):
    """Token-boundary-controlled recurrent updates, byte-level predictions.

    100544 weights: byte_emb=8224, GRUCell=47520, head=44800.
    Hidden recurrent updates can be much rarer than byte readout operations.
    Each new sample resets the stream and recurrent state, forbidding entity
    contamination. No EOS flush occurs before next-byte prediction.
    """

    def __init__(self) -> None:
        super().__init__()
        self.byte_emb = nn.Embedding(257, EMBED)
        self.gru_cell = nn.GRUCell(EMBED, HIDDEN)
        self.byte_head = nn.Linear(HIDDEN + EMBED * 2, 256)

    def forward(self, prev_ids: Tensor, *, variant: StreamVariant, audit: bool = False):
        if variant not in STREAM_VARIANTS:
            raise ValueError('unknown causal segmenter')
        if not isinstance(prev_ids, Tensor) or prev_ids.dtype != torch.long or prev_ids.ndim != 2:
            raise ValueError('prev_ids must be int64 [batch, time]')
        if prev_ids.size(1) and not bool(torch.all(prev_ids[:, 0] == BOS_ID)):
            raise ValueError('all sequences require BOS first')
        if prev_ids.size(1) > 1 and not bool(torch.all((prev_ids[:, 1:] >= 0) & (prev_ids[:, 1:] < 256))):
            raise ValueError('non-initial IDs must be observed raw bytes')
        all_logits: list[Tensor] = []
        receipts: list[SparseAudit] = []
        for row in prev_ids:
            logits, receipt = self._one(row, variant)
            all_logits.append(logits)
            receipts.append(receipt)
        if all_logits:
            value = torch.stack(all_logits, dim=0)
        else:
            value = self.byte_head.weight.new_zeros((0, prev_ids.size(1), 256))
        return (value, tuple(receipts)) if audit else value

    def _one(self, prev_ids: Tensor, variant: StreamVariant) -> tuple[Tensor, SparseAudit]:
        stream = CausalPatchStream(variant=variant)
        h = self.byte_emb.weight.new_zeros((1, HIDDEN))
        pending: list[Tensor] = []
        predictions: list[Tensor] = []
        update_times: list[int] = []
        ends: list[int] = []
        for i, idx in enumerate(prev_ids.unbind(dim=0)):
            current_emb = self.byte_emb(idx.view(1))  # [1,32]
            if i > 0:
                # idx is x[i-1]. Observed before the prediction for x[i].
                events = stream.observe(bytes([int(idx.item())]))
                appended = False
                for ev in events:
                    if ev.available_after != stream.observed_count:
                        raise AssertionError('future boundary unexpectedly visible')
                    # Original/fixed/surprise close INCLUDING current observed byte;
                    # T2b closes PREVIOUS patch one byte after its endpoint.
                    if ev.patch.end == stream.observed_count:
                        items = pending + [current_emb]
                        appended = True
                    elif ev.patch.end == stream.observed_count - 1:
                        items = pending
                    else:
                        raise AssertionError('unrecognized boundary delay')
                    if len(items) != len(ev.patch.payload) or not items:
                        raise AssertionError('patch tensor/byte alignment invalid')
                    pooled = torch.stack(items, dim=0).mean(dim=0)
                    h = self.gru_cell(pooled, h)
                    update_times.append(ev.available_after)
                    ends.append(ev.patch.end)
                    pending = []
                if not appended:
                    pending.append(current_emb)
            if pending:
                local = torch.stack(pending, dim=0).mean(dim=0)
            else:
                local = torch.zeros_like(current_emb)
            # All information used comes from x[:i], not x[i].
            logit = self.byte_head(torch.cat((h, current_emb, local), dim=-1))
            predictions.append(logit.squeeze(0))
        out = (torch.stack(predictions, dim=0) if predictions
               else self.byte_head.weight.new_empty((0, 256)))
        receipt = SparseAudit(
            variant=variant, observed_bytes=max(0, len(prev_ids) - 1),
            recurrent_updates=len(update_times), recurrent_update_times=tuple(update_times),
            committed_patch_ends=tuple(ends), delayed_commits=stream.delayed_commits,
            byte_readouts=len(predictions), eos_used=False,
        )
        return out, receipt


def sparse_parameter_receipt(model: PatchSparseByteGRU) -> dict:
    learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    deployed = sum(p.numel() for p in model.parameters())
    return {
        'trainable_parameters': learnable,
        'deployed_learned_parameters': deployed,
        'components': {k: p.numel() for k, p in model.named_parameters()},
        'expected_total': 100544,
        '100k_absolute_band': 95000 <= learnable <= 105000 and 95000 <= deployed <= 105000,
        'baseline_100400_relative_gap': abs(learnable - 100400) / 100400,
        'training_authorized': False,
    }
