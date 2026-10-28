"""Prospective (2026-10-28+) TAC-OSM 500K *actual learned* parameter gate.

Not a migration of legacy 95K–105K registrations. Callers MUST count unique
active learned tensor storage, exclude frozen/dead/padding weights, and record
original model/optimizer/checkpoint hashes separately. Parameter parity does not
certify equal FLOPs, objectives, inputs, data, or independent evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

TARGET = 500_000
MIN_ACTIVE = 475_000
MAX_ACTIVE = 525_000
MAX_RELATIVE_GAP = 0.01
EFFECTIVE_FROM = "2026-10-28"

@dataclass(frozen=True)
class BudgetResult:
    target: int
    lower: int
    upper: int
    architecture_counts: dict[str, int]
    max_pair_gap: float

def validate_study_counts(counts: Mapping[str, int], *,
                          study_date: str) -> BudgetResult:
    """Validate NEW models only; does not backdate/modify historic test data.

    The caller must supply *live, unique, gradient-contributing* tensor counts.
    This validator cannot infer graph participation from reported integers.
    """
    if not isinstance(study_date, str) or study_date < EFFECTIVE_FROM:
        raise ValueError("500K rule is prospective; historical receipts use original gate")
    if not counts:
        raise ValueError("at least one actual model is required")
    for model, count in counts.items():
        if not isinstance(model, str) or not model.strip():
            raise ValueError("nonblank architecture name required")
        if type(count) is not int or count < MIN_ACTIVE or count > MAX_ACTIVE:
            raise ValueError(f"{model}: active learned count {count!r} outside [{MIN_ACTIVE},{MAX_ACTIVE}]")
    ordered = sorted(counts.values())
    gap = (ordered[-1] - ordered[0]) / min(ordered)
    if len(ordered) > 1 and gap > MAX_RELATIVE_GAP:
        raise ValueError(f"active model-parameter parity failed: {gap:.5%} > 1%")
    return BudgetResult(TARGET, MIN_ACTIVE, MAX_ACTIVE, dict(counts), gap)

def live_unique_trainable_count(model) -> int:
    """PyTorch storage-identity count; not a substitute for live gradients.

    Count tied parameters only once, and refuse non-trainable model-only buffers.
    """
    seen = set()
    count = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        key = (p.device.type, p.device.index, p.untyped_storage().data_ptr(),
               p.storage_offset(), tuple(p.shape), tuple(p.stride()))
        if key not in seen:
            seen.add(key)
            count += p.numel()
    return count

def live_gradient_count(model) -> int:
    """Counts live trainable weights reached by a real scalar loss backward.

    Only use AFTER optimizer.zero_grad(set_to_none=True) and loss.backward().
    Absent gradients signal inactive parameters; zero gradients may still be
    correct for a particular example/mask and require separate tracing.
    """
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad and p.grad is not None)
