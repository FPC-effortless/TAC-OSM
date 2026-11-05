"""Independent NumPy reference for CSL-001 retention and causal convolution.

Only the listed operators are independently formulated; the full model's
Lean/Python/source equivalence and scientific qualification remain OPEN.
"""
from __future__ import annotations
import numpy as np


def serial_retention(z: np.ndarray, initial: np.ndarray, decay: np.ndarray) -> np.ndarray:
    """z[B,T,R,E] -> all post-token memories, exact ordered mathematical graph."""
    if z.ndim != 4 or initial.shape != (z.shape[0], *z.shape[2:]) or decay.shape != z.shape[2:]:
        raise ValueError('incompatible shapes')
    if z.shape[1] == 0 or np.any(decay <= 0) or np.any(decay >= 1):
        raise ValueError('nonempty input and 0<decay<1 required')
    prior = initial.astype(np.float64, copy=True)
    outputs = []
    for t in range(z.shape[1]):
        prior = decay * prior + (1.0 - decay) * z[:, t]
        outputs.append(prior.copy())
    return np.stack(outputs, axis=1)


def role_read(address: np.ndarray, memories: np.ndarray) -> np.ndarray:
    """Normalized address[B,T,R] and state[B,T,R,E] => read[B,T,E]."""
    if address.ndim != 3 or memories.shape[:3] != address.shape:
        raise ValueError('role shapes mismatch')
    if not np.allclose(address.sum(-1), 1., rtol=0, atol=1e-6):
        raise ValueError('addresses must normalize')
    return np.einsum('btr,btre->bte', address, memories)


def causal_depthwise(x: np.ndarray, prev: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x[B,T,D], prev[B,K-1,D], kernel[D,K]. Torch Conv1d cross-correlates."""
    if x.ndim != 3 or kernel.ndim != 2 or prev.shape != (x.shape[0], kernel.shape[1]-1, x.shape[2]):
        raise ValueError('causal convolution shape mismatch')
    joined = np.concatenate([prev, x], axis=1)
    y = np.zeros_like(x)
    for t in range(x.shape[1]):
        y[:, t, :] = (joined[:, t:t+kernel.shape[1], :].transpose(0,2,1) * kernel[None]).sum(-1)
    return y, joined[:, -prev.shape[1]:, :].copy()
