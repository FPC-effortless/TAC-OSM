"""TAC-LANG-REPAIR-018: source-authorized, parameter-neutral selective retrieval.

The trained language checkpoints optimize the historical target at lag=64.
This pure adapter NEVER changes their weights or claims longer retrieval.
For any lag other than exactly 64 it reports ABSTENTION and, when a categorical distribution is
required, returns a smoothed FIT-only frequency prior.

Routing depends ONLY on the requested lag and the frozen training horizon.
It cannot consult source targets or DEV outcomes at runtime.

The static 128-vector is additional nonlearned metadata (<=512B in FP32),
NOT free model state and NOT a 128-way learned parameter advantage.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch

TRAINED_HISTORICAL_LAG = 64
VOCAB = 128
SMOOTHING = 0.5

@dataclass(frozen=True)
class HistoricalDecision:
    log_probabilities: torch.Tensor
    abstained: bool
    used_trained_memory: bool
    lag: int
    reason: str

def fit_only_frequency_prior(documents: Sequence[np.ndarray],
                             smoothing: float = SMOOTHING) -> torch.Tensor:
    """Only training-domain source documents are authorized. Caller MUST check
    pinned SHA256 partition manifest; this function cannot validate provenance.
    """
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("strictly positive FIT-only smoothing required")
    if not documents:
        raise ValueError("at least one FIT document required")
    counts = np.full(VOCAB, smoothing, dtype=np.float64)
    for doc in documents:
        if not isinstance(doc, np.ndarray) or doc.ndim != 1:
            raise ValueError("each FIT document must be one-dimensional NumPy tokens")
        if not np.issubdtype(doc.dtype, np.integer) or not doc.size:
            raise ValueError("FIT documents require nonempty integer tokens")
        if np.any((doc < 0) | (doc >= VOCAB)):
            raise ValueError("out-of-range FIT token")
        counts += np.bincount(doc.astype(np.int64), minlength=VOCAB)
    probs = counts / counts.sum()
    if not np.isfinite(probs).all() or np.any(probs <= 0):
        raise ValueError("FIT prior failed positivity")
    return torch.from_numpy(probs.copy())

def query_selective_history(model, tokens: torch.Tensor, trace: dict, lag: int,
                            fit_prior: torch.Tensor) -> HistoricalDecision:
    if not isinstance(lag, int) or isinstance(lag, bool) or lag < 1:
        raise ValueError("positive integer lag required")
    if tokens.ndim != 2 or tokens.dtype != torch.int64 or lag >= tokens.shape[1]:
        raise ValueError("historical query requires [B,T] integer tokens and lag<T")
    if fit_prior.shape != (VOCAB,) or not torch.is_floating_point(fit_prior):
        raise ValueError("baseline requires 128 floating probabilities")
    if not bool(torch.isfinite(fit_prior).all()):
        raise ValueError("nonfinite baseline")
    if not bool(torch.all(fit_prior > 0)):
        raise ValueError("strictly positive prior mandatory")
    if abs(float(fit_prior.double().sum()) - 1.) > 1e-8:
        raise ValueError("baseline must sum to one")
    if lag == TRAINED_HISTORICAL_LAG:
        logits, _ = model.historical_logits(tokens, trace, lag)
        logp = torch.log_softmax(logits, dim=-1)
        if not bool(torch.isfinite(logp).all()):
            raise ValueError("nonfinite trained history logits")
        return HistoricalDecision(logp, False, True, lag, "exact_frozen_trained_delay")
    # No model history read, no target or DEV score, no additional training.
    dtype = model.embedding.weight.dtype
    prior = fit_prior.to(device=tokens.device, dtype=dtype)
    logp = prior.log().reshape(1, 1, VOCAB).expand(
        tokens.shape[0], tokens.shape[1] - lag, VOCAB)
    if not bool(torch.isfinite(logp).all()):
        raise ValueError("nonfinite fallback")
    return HistoricalDecision(logp, True, False, lag, "unqualified_historical_delay")
