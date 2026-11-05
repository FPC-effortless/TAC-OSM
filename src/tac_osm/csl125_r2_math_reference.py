"""Independent NumPy scalar-time reference for CSL-125-R2 admission dynamics.

No neural training, source provenance claim, or real-outcome evaluation.
"""
from __future__ import annotations
import numpy as np


def selective_serial(z: np.ndarray, carry: np.ndarray, rate: np.ndarray,
                     gate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Float64 serial beta/eta recurrence, outputs [B,T,R,E,2]."""
    z = np.asarray(z, dtype=np.float64)
    carry = np.asarray(carry, dtype=np.float64).copy()
    rate = np.asarray(rate, dtype=np.float64)
    gate = np.asarray(gate, dtype=np.float64)
    if (z.ndim != 5 or z.shape[-1] != 2 or z.shape[1] < 1 or
        carry.shape != z.shape[:1]+z.shape[2:] or
        rate.shape != z.shape[2:4] or gate.shape != z.shape[:3] or
        not (np.isfinite(z).all() and np.isfinite(carry).all() and
             np.isfinite(rate).all() and (rate > 0).all() and
             np.isfinite(gate).all() and ((gate >= 0)&(gate <= 1)).all())):
        raise ValueError('invalid reference inputs')
    out = np.empty_like(z)
    for t in range(z.shape[1]):
        rg=rate[np.newaxis,:,:]*gate[:,t,:,np.newaxis]
        eta=-np.expm1(-rg)
        beta=np.exp(-rg)
        carry=beta[...,np.newaxis]*carry+eta[...,np.newaxis]*z[:,t]
        out[:,t]=carry
    return out, carry
