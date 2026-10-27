"""GP-UNC-001 bounded exact-RBF GP / calibration reference, NO model fitting.

Constructed numerical diagnostics only. This is NOT a TAC-OSM trained head,
real-data experiment, calibrated uncertainty, or replacement for PNDS Lean.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

FAMILIES = ("System 1", "Enterprise", "Temporal", "Physical", "Edge-Lite", "Counterfactual")
ARMS = ("TAC_point", "TAC_Gaussian_head", "TAC_sparse_GP_residual",
        "TAC_conformal", "matched_GRU_probabilistic")


def study_matrix() -> tuple[tuple[str, str, str], ...]:
    """Registrations, not fitted models: every cell is explicitly BLOCKED."""
    return tuple((family, arm, "NOT_FITTED_OR_QUALIFIED") for family in FAMILIES for arm in ARMS)


def _positive(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive")


def _vector(x):
    v = tuple(x)
    if not v or any(isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(a) for a in v):
        raise ValueError("nonempty finite numeric feature vector required")
    return v


def rbf_kernel(a, b, *, length_scale=1.0, signal_variance=1.0):
    _positive("length_scale", length_scale)
    _positive("signal_variance", signal_variance)
    a, b = _vector(a), _vector(b)
    if len(a) != len(b):
        raise ValueError("feature dimensions differ")
    d2 = sum((u - v) ** 2 for u, v in zip(a, b))
    return signal_variance * math.exp(-0.5 * d2 / (length_scale ** 2))


def _chol(a):
    n = len(a)
    L = [[0.] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            z = a[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if not math.isfinite(z) or z <= 0:
                    raise ValueError("covariance not numerically positive definite")
                L[i][j] = math.sqrt(z)
            else:
                L[i][j] = z / L[j][j]
    return L


def _solve(L, b):
    n = len(b)
    y = [0.] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
    x = [0.] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(L[j][i] * x[j] for j in range(i + 1, n))) / L[i][i]
    return x


@dataclass(frozen=True)
class Prediction:
    mean: float
    latent_variance: float
    observed_variance: float


def posterior(train_x, residuals, query, *, length_scale=1., signal_variance=1.,
              noise_variance=0.05, jitter=1e-10):
    """Exact dense GP reference for small diagnostic inputs; O(n^3) per call.

    Hyperparameters are fixed inputs. No optimizer, gradient or checkpoint.
    """
    _positive("length_scale", length_scale)
    _positive("signal_variance", signal_variance)
    _positive("noise_variance", noise_variance)
    _positive("jitter", jitter)
    xs = [_vector(v) for v in train_x]
    y = tuple(residuals)
    q = _vector(query)
    if len(xs) != len(y) or any(len(v) != len(q) for v in xs):
        raise ValueError("sample count or feature dimension mismatch")
    if any(isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(z) for z in y):
        raise ValueError("residuals must be finite numbers")
    if len(xs) > 128:
        raise ValueError("reference-only dense GP limited to 128 observations")
    def k(a, b):
        return rbf_kernel(a, b, length_scale=length_scale, signal_variance=signal_variance)
    if not xs:
        return Prediction(0.0, signal_variance, signal_variance + noise_variance)
    K = [[k(a, b) + (noise_variance + jitter if i == j else 0.)
          for j, b in enumerate(xs)] for i, a in enumerate(xs)]
    L = _chol(K)
    cross = [k(q, a) for a in xs]
    alpha = _solve(L, y)
    mean = sum(a * b for a, b in zip(cross, alpha))
    weight = _solve(L, cross)
    raw = signal_variance - sum(a * b for a, b in zip(cross, weight))
    if raw < -1e-8 * signal_variance:
        raise ValueError("negative predictive variance outside numerical tolerance")
    v = max(0.0, raw)
    return Prediction(mean, v, v + noise_variance)


def gaussian_nll(y, mean, variance):
    _positive("variance", variance)
    if any(not isinstance(z, (int, float)) or isinstance(z, bool) or not math.isfinite(z)
           for z in (y, mean)):
        raise ValueError("finite real target and mean required")
    return 0.5 * (math.log(2 * math.pi * variance) + (y - mean) ** 2 / variance)


def split_conformal_radius(absolute_errors, *, alpha=0.1):
    """No-fit mathematical quantile: calibration data must be supplied separately."""
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    errors = tuple(absolute_errors)
    if not errors or any(not isinstance(v, (int, float)) or isinstance(v, bool) or
                         not math.isfinite(v) or v < 0 for v in errors):
        raise ValueError("nonempty, nonnegative finite calibration residuals required")
    k = math.ceil((len(errors) + 1) * (1 - alpha))
    if k > len(errors):
        raise ValueError("insufficient calibration records for finite radius")
    return sorted(errors)[k - 1]


def validate_training_rows(rows, *, cutoff):
    """Fail-closed causal timestamp contract; no assertions of publisher origin."""
    if not isinstance(cutoff, (int, float)) or not math.isfinite(cutoff):
        raise ValueError("cutoff must be a finite numeric timestamp")
    ids = set()
    for row in rows:
        if set(row) != {"id", "feature_at", "label_at"} or row["id"] in ids:
            raise ValueError("invalid or duplicate registered row")
        ids.add(row["id"])
        for key in ("feature_at", "label_at"):
            ts = row[key]
            if not isinstance(ts, (int, float)) or isinstance(ts, bool) or not math.isfinite(ts):
                raise ValueError("invalid timestamp")
        if row["feature_at"] > row["label_at"] or row["label_at"] > cutoff:
            raise ValueError("future-dependent training label or feature")
    return len(ids)
