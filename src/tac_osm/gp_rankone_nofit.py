"""GP-UNC-002: constant-state rank-one posterior updates (constructed tests ONLY).

This implements Bayesian linear conditioning on a FIXED inducing RBF feature
basis, not a trained/neural/variational GP.  Matrix P starts as identity
and updates by a Sherman-Morrison / Kalman rank-one covariance correction:
    v = P @ phi
    d = noise_variance + phi.T @ v
    mean += v * (residual - phi.T @ mean) / d
    P -= (v @ v.T) / d
It uses O(m^2) update arithmetic, O(m^2 + m*d) retained scalars,
plus one O(m^3) factorization at initialization. It DOES NOT claim
measured realtime latency, FP correctness for arbitrary streams, calibrated
probabilities, signed-source eligibility, or any model capability.
"""
from __future__ import annotations

import math
from .gp_uncertainty_nofit import Prediction, _chol, _positive, _vector, rbf_kernel


class RankOneProjectedResidualGP:
    """Single-source-stream constructed reference, source-time guarded."""

    def __init__(self, inducing, *, length_scale=1.0, signal_variance=1.0,
                 noise_variance=0.05, jitter=1e-10):
        for key, value in (("length_scale", length_scale),
                           ("signal_variance", signal_variance),
                           ("noise_variance", noise_variance),
                           ("jitter", jitter)):
            _positive(key, value)
        z = tuple(_vector(x) for x in inducing)
        if not 1 <= len(z) <= 64 or any(len(v) != len(z[0]) for v in z):
            raise ValueError("inducing set must contain 1..64 equal-size finite vectors")
        self.inducing = z
        self.dim = len(z[0])
        self.m = len(z)
        self.length_scale = float(length_scale)
        self.signal_variance = float(signal_variance)
        self.noise_variance = float(noise_variance)
        self.jitter = float(jitter)
        self.L = _chol([
            [self._kernel(a, b) + (self.jitter if i == j else 0.0)
             for j, b in enumerate(z)]
            for i, a in enumerate(z)
        ])
        self.posterior_mean = [0.0 for _ in z]
        self.posterior_cov = [
            [float(i == j) for j in range(self.m)] for i in range(self.m)
        ]
        self.observations = 0
        self.last_cutoff = None

    def _kernel(self, x, y):
        return rbf_kernel(x, y, length_scale=self.length_scale,
                          signal_variance=self.signal_variance)

    def _basis(self, features):
        x = _vector(features)
        if len(x) != self.dim:
            raise ValueError("feature dimension mismatch")
        cross = [self._kernel(z, x) for z in self.inducing]
        phi = []
        for i in range(self.m):
            phi.append((cross[i] - sum(self.L[i][j] * phi[j]
                                     for j in range(i))) / self.L[i][i])
        return phi

    def observe(self, features, residual, *, feature_at, issued_at, label_at, cutoff):
        """Admit residual only after a claimed pre-outcome prediction.

        This checks timestamps; it CANNOT verify that supplied latent features,
        labels and predictions actually originated from independent records.
        """
        times = (feature_at, issued_at, label_at, cutoff, residual)
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
               for v in times):
            raise ValueError("nonfinite or nonnumeric time / residual")
        if not feature_at <= issued_at < label_at <= cutoff:
            raise ValueError("prediction must precede original outcome availability")
        if self.last_cutoff is not None and cutoff < self.last_cutoff:
            raise ValueError("observation cutoff moved backward")
        phi = self._basis(features)
        m = self.m
        pphi = [math.fsum(self.posterior_cov[i][j] * phi[j]
                         for j in range(m)) for i in range(m)]
        d = self.noise_variance + math.fsum(phi[i] * pphi[i] for i in range(m))
        if not math.isfinite(d) or d <= 0:
            raise ValueError("nonpositive or nonfinite innovation variance")
        error = residual - math.fsum(phi[i] * self.posterior_mean[i]
                                    for i in range(m))
        new_mean = [self.posterior_mean[i] + pphi[i] * error / d
                    for i in range(m)]
        # Symmetric correction, avoiding any O(m^3) Cholesky in the hot path.
        new_cov = [[self.posterior_cov[i][j] - pphi[i]*pphi[j]/d
                    for j in range(m)] for i in range(m)]
        if (any(not math.isfinite(v) for v in new_mean) or
                any(not math.isfinite(v) for row in new_cov for v in row) or
                any(new_cov[i][i] < -1e-10 for i in range(m))):
            raise ValueError("nonfinite or inadmissibly negative posterior")
        self.posterior_mean = new_mean
        self.posterior_cov = new_cov
        self.last_cutoff = cutoff
        self.observations += 1

    def predict(self, features):
        phi = self._basis(features)
        predicted_mean = math.fsum(phi[i]*self.posterior_mean[i]
                                   for i in range(self.m))
        prior_residual = self.signal_variance - math.fsum(x*x for x in phi)
        if prior_residual < -1e-8*self.signal_variance:
            raise ValueError("invalid inducing projection residual covariance")
        posterior_term = math.fsum(
            phi[i]*self.posterior_cov[i][j]*phi[j]
            for i in range(self.m) for j in range(self.m))
        variance = prior_residual + posterior_term
        if not math.isfinite(variance) or variance < -1e-8*self.signal_variance:
            raise ValueError("invalid or nonfinite predictive variance")
        v = max(0.0, variance)
        if not math.isfinite(predicted_mean) or not math.isfinite(v+self.noise_variance):
            raise ValueError("nonfinite prediction")
        return Prediction(predicted_mean, v, v+self.noise_variance)

    @property
    def retained_numeric_scalars(self):
        return self.m*self.m + self.m + self.m*(self.m+1)//2 + self.m*self.dim
