"""GRAPH016: owner-local, target-blind Bayesian log-loss mixture of predictions.

Mathematically valid for positive normalized fixed expert predictive laws
computed using only already observed source. This is a classical online
aggregation mechanism, NOT new native TAC memory, newly trained language
intelligence, or authorization to use records beyond caller's privileges.
"""
from __future__ import annotations
import math
import numpy as np

class CausalExpertMixture:
    def __init__(self, owners, names=("native", "exact", "exact_approx")):
        if (not isinstance(owners, tuple) or not owners
            or any(type(x) is not str or not x for x in owners)
            or len(set(owners)) != len(owners)):
            raise ValueError("nonempty unique source owner IDs required")
        if (not isinstance(names, tuple) or len(names) < 2
            or len(set(names)) != len(names)):
            raise ValueError("at least two distinct named expert laws required")
        self.owners, self.names = owners, names
        self.batch, self.K = len(owners), len(names)
        self.log_w = np.full((self.batch, self.K), -math.log(self.K), dtype=np.float64)
        self.expert_loss = np.zeros((self.batch, self.K), dtype=np.float64)
        self.mixture_loss = np.zeros(self.batch, dtype=np.float64)
        self.steps = np.zeros(self.batch, dtype=np.int64)
        self._pending = None

    def predict(self, experts):
        """Produce a law without access to the next target. Shape [B,K,V]."""
        if self._pending is not None:
            raise ValueError("cannot predict twice before post-observation update")
        x = np.asarray(experts, dtype=np.float64)
        if (x.ndim != 3 or x.shape[:2] != (self.batch, self.K)
            or x.shape[-1] < 2 or not np.isfinite(x).all()
            or not (x > 0).all()
            or not np.all(np.abs(x.sum(-1) - 1) <= 2e-5)):
            raise ValueError("finite strictly-positive normalized probability laws required")
        combined = np.einsum("bk,bkv->bv", np.exp(self.log_w), x)
        if not np.allclose(combined.sum(-1), 1., atol=2e-5):
            raise ValueError("mixture not normalized")
        self._pending = (x.copy(), combined.copy())
        return combined.copy()

    def observe(self, y, owners):
        """Update owner weights *after* the outcome was disclosed to caller."""
        if self._pending is None:
            raise ValueError("must issue prediction before observing an outcome")
        if owners != self.owners:
            raise ValueError("owner mismatch / cross-source carry forbidden")
        target = np.asarray(y)
        x, combined = self._pending
        if (target.shape != (self.batch,)
            or not np.issubdtype(target.dtype, np.integer)
            or (target < 0).any() or (target >= x.shape[-1]).any()):
            raise ValueError("invalid observed label")
        rows = np.arange(self.batch)
        probs = x[rows, :, target]
        self.expert_loss -= np.log(probs)
        self.mixture_loss -= np.log(combined[rows, target])
        log_next = self.log_w + np.log(probs)
        self.log_w = log_next - np.logaddexp.reduce(log_next, axis=1, keepdims=True)
        self.steps += 1
        self._pending = None
        if (self.mixture_loss - self.expert_loss.min(axis=1) > math.log(self.K) + 1e-7).any():
            raise ArithmeticError("pathwise expert-regret bound violated")
        return self.mixture_loss.copy()

    def receipt(self):
        if self._pending is not None:
            raise ValueError("pending outcome not disclosed")
        return dict(owners=self.owners, tokens=self.steps.tolist(),
                    mixture_log_loss=self.mixture_loss.tolist(),
                    expert_log_loss=self.expert_loss.tolist(),
                    expert_names=self.names,
                    fixed_expert_regret=(self.mixture_loss - self.expert_loss.min(axis=1)).tolist(),
                    theoretical_max_regret=math.log(self.K),
                    final_weights=np.exp(self.log_w).tolist(),
                    qualification="MATH_ONLY_REQUIRES_AUTHENTIC_SOURCE_AND_CHECKPOINT_REPLAY")
