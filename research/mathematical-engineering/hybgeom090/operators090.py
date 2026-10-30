"""HYBGEOM090: reversible NO-FIT TAC-OSM language sidecar and analytic falsifiers.

Engineering probes only. Neither elapsed seconds nor human semantic judgments
are present in the publisher byte text: NEVER promote these fixtures as gains.
The core compact LANGMEM086 forward graph is not edited or re-trained.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
import torch
from torch.nn import functional as F


def _finite_scalar(value: float, name: str, minimum: float = 0.) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a real scalar")
    a = float(value)
    if not math.isfinite(a) or a < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return a


def linear_flow(x: torch.Tensor, dt: float, rate: float, drive: torch.Tensor | None = None) -> torch.Tensor:
    """Exact solution of xdot=-rate*x+w with bounded constant drive w.

    Rate > 0 is needed for exponential ISS; zero rate is allowed as a control.
    """
    t = _finite_scalar(dt, 'elapsed'); lam = _finite_scalar(rate, 'rate')
    if not torch.is_floating_point(x) or not torch.isfinite(x).all():
        raise ValueError('finite floating state required')
    if drive is not None and (drive.shape != x.shape or not torch.isfinite(drive).all()):
        raise ValueError('finite shape-matched causal drive required')
    if t == 0.: return x.clone()
    p = math.exp(-lam * t)
    out = p * x
    if drive is not None:
        out = out + ((-math.expm1(-lam * t)) / lam if lam else t) * drive
    return out


def lyapunov_residual(x: torch.Tensor, drive: torch.Tensor, rate: float) -> torch.Tensor:
    """dV/dt + rate*V - ||drive||²/(2*rate) <=0 for V=||x||²/2."""
    lam = _finite_scalar(rate, 'rate', 1e-12)
    if x.shape != drive.shape or not torch.isfinite(x).all() or not torch.isfinite(drive).all():
        raise ValueError('finite matching state/disturbance')
    v = .5 * x.square().sum()
    dot_v = (x * (-lam * x + drive)).sum()
    return dot_v + lam * v - .5 / lam * drive.square().sum()


@dataclass(frozen=True)
class HybridState090:
    core: object  # original StreamState086, left fully intact
    mode: int
    time: float


def start(model, owners: tuple[str, ...] = ('fixture-owner',), *, time: float = 0.) -> HybridState090:
    _finite_scalar(time, 'start time')
    if len(owners) != 1: raise ValueError('initial research adapter supports exactly one source owner')
    return HybridState090(model.start(owners), 0, float(time))


def hybrid_token(model, token: torch.Tensor, st: HybridState090, event_time: float,
                 *, rates: tuple[float, float] = (.015, .15), resets: tuple[float, float] = (1., .9),
                 min_dwell: float = 1e-6):
    """Flow under PREVIOUS mode -> observed event mode/reset -> original TAC byte step.

    Only observed current byte may choose new mode: LF -> 1, period -> 0.
    Time comes from an external registered observation-time source (not byte index).
    One source owner, one byte per call, no target access, no hidden cross-owner map.
    """
    now = _finite_scalar(event_time, 'event_time')
    dwell = _finite_scalar(min_dwell, 'min_dwell')
    if st.mode not in (0, 1) or len(rates) != 2 or len(resets) != 2:
        raise ValueError('two-mode contract')
    for r in rates: _finite_scalar(r, 'rate')
    for r in resets:
        _finite_scalar(r, 'reset')
        if r > 1: raise ValueError('expansive reset forbidden')
    if not isinstance(token, torch.Tensor) or token.shape != (1, 1) or token.dtype != torch.long:
        raise ValueError('exactly one observed integer byte required')
    byte = int(token.item())
    if not 0 <= byte <= 255: raise ValueError('byte range')
    dt = now - st.time
    if dt < 0 or (st.core.clock > 0 and dt < dwell):
        raise ValueError('time reversal or Zeno/dwell violation')
    x = st.core.tensors['upper']
    flow = linear_flow(x, dt, rates[st.mode])
    q = 1 if byte == 10 else 0 if byte == 46 else st.mode
    x_event = flow * (resets[q] if q != st.mode else 1.)
    pre = replace(st.core, tensors={**st.core.tensors, 'upper': x_event})
    logits, newcore = model(token, state=pre, owners=st.core.owners)
    return logits, HybridState090(newcore, q, now)


def unit_context(context: torch.Tensor) -> torch.Tensor:
    if context.ndim != 2 or not torch.isfinite(context).all(): raise ValueError('context [B,D]')
    return F.normalize(context, dim=-1, eps=1e-8)


def contextual_table(embeddings: torch.Tensor, context: torch.Tensor, gamma: float = .15):
    """z_a(c) = e_a + gamma <e_a,c_hat> c_hat, pseudometric on byte labels."""
    g = _finite_scalar(gamma, 'gamma')
    if embeddings.ndim != 2 or context.shape[-1] != embeddings.shape[1] or not torch.isfinite(embeddings).all():
        raise ValueError('embedding/context dimension mismatch')
    u = unit_context(context)
    a = embeddings[None, :, :]
    return a + g * (a * u[:, None, :]).sum(-1, keepdim=True) * u[:, None, :]


def metric_tensor(context: torch.Tensor, alpha: float = .5, z: torch.Tensor | None = None):
    """G(z,c)=I + alpha*sigmoid(<z,c_hat>/sqrt(D))*c_hat*c_hat^T, SPD.

    z=None selects a CONSTANT SPD metric with factor 1 rather than sigmoid.
    No learned Riemannian geodesic claim is made by the straight-line control.
    """
    a = _finite_scalar(alpha, 'alpha')
    u = unit_context(context)
    scale = (torch.sigmoid((z * u).sum(-1) / math.sqrt(u.shape[-1]))
             if z is not None else torch.ones(u.shape[0], device=u.device, dtype=u.dtype))
    return (torch.eye(u.shape[-1],device=u.device,dtype=u.dtype)[None] +
            (a * scale)[:,None,None] * (u[:,:,None] * u[:,None,:]))


def distance_constant(a: torch.Tensor, b: torch.Tensor, context: torch.Tensor, alpha: float = .5):
    d = a-b
    G = metric_tensor(context, alpha)
    return torch.einsum('bi,bij,bj->b', d,G,d).clamp_min(0).sqrt()


def straight_line_length(a: torch.Tensor, b: torch.Tensor, context: torch.Tensor, alpha: float = .5, steps: int = 64):
    """Numerical length of one path; NOT the infimum over all paths/geodesic."""
    if steps < 2: raise ValueError('steps >= 2')
    d = b-a
    lengths=[]
    for i in range(steps + 1):
        z=a+(i/steps)*d
        G=metric_tensor(context, alpha, z)
        dsq=torch.einsum('bi,bij,bj->b', d,G,d)
        lengths.append(dsq.clamp_min(0).sqrt() * (.5 if i in (0,steps) else 1.))
    return sum(lengths)/steps


def geometry_logits(logits: torch.Tensor, embeddings: torch.Tensor, observed: torch.Tensor,
                    context: torch.Tensor, *, contextual: bool, alpha: float = 0., strength: float = .01):
    """Frozen prequential logit probe; matched Euclidean/SPD geometry controls.

    Observed token and carried/current context only; candidate output byte not read.
    No parameter fit or tuning. Original logits recovered exactly at strength 0.
    """
    _finite_scalar(strength, 'strength')
    if logits.ndim != 2 or logits.shape[1] != embeddings.shape[0] or observed.shape != (logits.shape[0],):
        raise ValueError('token/logit shape')
    if not torch.isfinite(logits).all() or not torch.isfinite(embeddings).all(): raise ValueError('nonfinite logits/embeddings')
    z = contextual_table(embeddings,context, .15 if contextual else 0.)
    ref=z[torch.arange(len(observed),device=observed.device),observed]
    d=z-ref[:,None,:]
    u=unit_context(context)
    d2=d.square().sum(-1)+_finite_scalar(alpha,'alpha')*(d*u[:,None,:]).sum(-1).square()
    return logits-strength*(d2-d2.mean(-1,keepdim=True))


def macro_linear_step(s: torch.Tensor, a: float = .8, b: float = .1):
    """Permutation-equivariant all-to-all mean coupling; exact macro closure."""
    if s.ndim != 1 or len(s) < 2: raise ValueError('population 1-D size >= 2')
    return a*s+b*s.mean()


def macro_variance_sensitive_step(s: torch.Tensor, a: float = .8, b: float = .1, coupling: float = .2):
    """Constructed counterexample: same mean, different variance -> different next mean."""
    return macro_linear_step(s,a,b)+coupling*(s-s.mean()).square()


def shift_fixture(steps: int = 16):
    """Two constructed cases; demonstrates both help and harm, not adaptation skill."""
    xslow=xfast=1.
    changed_slow=changed_fast=steady_slow=steady_fast=0.
    for _ in range(steps):
        xslow *= math.exp(-.05)
        xfast *= math.exp(-.5)
        changed_slow += xslow*xslow; changed_fast += xfast*xfast
        steady_slow += (xslow-1)**2; steady_fast += (xfast-1)**2
    return dict(changed_slow=changed_slow,changed_fast=changed_fast,
                steady_slow=steady_slow,steady_fast=steady_fast)
