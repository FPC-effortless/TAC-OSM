"""Bounded-memory *projected-process* GP residual mathematics, no trained TAC-OSM.

This is a low-rank inducing-basis Bayesian linear filter with an exact RBF
kernel on the inducing set. It is NOT variational GP optimization, not an
end-to-end PyTorch model, and not an empirically calibrated confidence system.

For an inducing set Z, Kmm = LLᵀ, phi(x) = L⁻¹ k(Z,x).  The projected
training covariance is Qnn = Phiᵀ Phi and the prior is w ~ N(0,I).
With iid Gaussian observation noise variance nu:
  precision A = I + sum(phi phiᵀ)/nu
  information b = sum(phi*y)/nu
  mean* = phi*ᵀ A⁻¹b
  variance* = k(x*,x*) - phi*ᵀphi* + phi*ᵀA⁻¹phi*
The first two terms restore the conditional-prior variance discarded by
pure projected training. Computational state uses O(m² + m*d), not O(n²).
No learned model parameters or historical training data are stored.
"""
from __future__ import annotations
import math

from .gp_uncertainty_nofit import (Prediction, _chol, _positive, _solve,
                                   _vector, rbf_kernel)


class ProjectedResidualGP:
    """Fixed-size reference filter. Caller must supply source-legal residuals."""

    def __init__(self, inducing, *, length_scale=1.0, signal_variance=1.0,
                 noise_variance=0.05, jitter=1e-10):
        _positive("length_scale", length_scale)
        _positive("signal_variance", signal_variance)
        _positive("noise_variance", noise_variance)
        _positive("jitter", jitter)
        z = tuple(_vector(x) for x in inducing)
        if not 1 <= len(z) <= 64 or any(len(x) != len(z[0]) for x in z):
            raise ValueError("inducing basis must have 1..64 equal-size vectors")
        self.inducing = z
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        self.jitter = jitter
        self.dim = len(z[0])
        self.m = len(z)
        self.L = _chol([
            [self._kernel(x, y) + (jitter if i == j else 0.0)
             for j, y in enumerate(z)]
            for i, x in enumerate(z)
        ])
        self.precision = [[float(i == j) for j in range(self.m)] for i in range(self.m)]
        self.information = [0.0] * self.m
        self.observations = 0
        self.last_cutoff = None

    def _kernel(self, x, y):
        return rbf_kernel(x, y, length_scale=self.length_scale,
                          signal_variance=self.signal_variance)

    def _basis(self, x):
        x = _vector(x)
        if len(x) != self.dim:
            raise ValueError("feature dimension mismatch")
        k = [self._kernel(z, x) for z in self.inducing]
        phi = []
        for i in range(self.m):
            value = (k[i] - sum(self.L[i][j] * phi[j] for j in range(i))) / self.L[i][i]
            phi.append(value)
        return phi

    def observe(self, features, residual, *, feature_at, label_at, cutoff):
        """Incorporate one OBSERVED prediction residual after its legal label time.

        Reject late/early data before mutating state. Every observation at
        decision cutoff must have a previously issued prediction (caller
        supplies that verified provenance; this reference cannot attest it).
        """
        t = (feature_at, label_at, cutoff, residual)
        if any(isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(a) for a in t):
            raise ValueError("timestamps/residual must be finite numeric values")
        if not feature_at <= label_at <= cutoff:
            raise ValueError("future label or invalid feature/label order")
        if self.last_cutoff is not None and cutoff < self.last_cutoff:
            raise ValueError("cutoff must not move backwards")
        phi = self._basis(features)
        factor = 1.0 / self.noise_variance
        # Calculate all potentially failing values before mutating numerical state.
        new_a = [
            [self.precision[i][j] + phi[i] * phi[j] * factor for j in range(self.m)]
            for i in range(self.m)
        ]
        new_b = [
            self.information[i] + phi[i] * residual * factor
            for i in range(self.m)
        ]
        if any(not math.isfinite(v) for row in new_a for v in row) or any(not math.isfinite(v) for v in new_b):
            raise ValueError("nonfinite update rejected")
        _chol(new_a)
        self.precision = new_a
        self.information = new_b
        self.observations += 1
        self.last_cutoff = cutoff

    def predict(self, features):
        phi = self._basis(features)
        factor = _chol(self.precision)
        post_weights = _solve(factor, self.information)
        covariance_phi = _solve(factor, phi)
        mu = sum(a*b for a, b in zip(phi, post_weights))
        prior_residual = self.signal_variance - sum(a*a for a in phi)
        v = prior_residual + sum(a*b for a,b in zip(phi, covariance_phi))
        if v < -1e-8 * self.signal_variance or not math.isfinite(v):
            raise ValueError("predictive variance outside numerical tolerance")
        latent = max(0.0, v)
        return Prediction(mu, latent, latent + self.noise_variance)

    @property
    def retained_numeric_scalars(self):
        """Counts float state, excluding source-time metadata and Python overhead."""
        return self.m*self.m + self.m + self.m*(self.m+1)//2 + self.m*self.dim
