"""Observed absolute-error receipts for the 88-dimensional additive TAC source state.

No artificial IEEE theorem: this tests the precise chronological CPU FP32
operations and checks the per-event additive error budget of PNDS
LanguageFPContracts. Values here are constructed diagnostics, NOT real-text
language outcomes and NOT a uniform Torch/IEEE rounding-law certificate.
"""
from __future__ import annotations
import torch

def replay_fp32_additive_trace(basis64: torch.Tensor):
    if not isinstance(basis64,torch.Tensor) or basis64.ndim!=2 or basis64.shape[1]!=88:
        raise ValueError("expected chronological [T,88] basis")
    if basis64.dtype!=torch.float64 or basis64.device.type!="cpu":
        raise ValueError("certificate requires source FP64 CPU reference")
    if not torch.isfinite(basis64).all():
        raise ValueError("basis must be finite")
    observed=torch.zeros(88,dtype=torch.float32)
    ideal=torch.zeros(88,dtype=torch.float64)
    total_add_roundoff=torch.zeros_like(ideal)
    total_basis_cast=torch.zeros_like(ideal)
    peak_error=0.
    for basis in basis64:
        basis32=basis.to(dtype=torch.float32)
        cast64=basis32.to(dtype=torch.float64)
        pre=observed.to(dtype=torch.float64)
        post=(observed+basis32)
        local_error=(post.to(torch.float64)-(pre+cast64)).abs()
        total_add_roundoff+=local_error
        total_basis_cast+=(cast64-basis).abs()
        ideal+=basis
        observed=post
        residual=(observed.to(torch.float64)-ideal).abs()
        bound=total_add_roundoff+total_basis_cast
        if not torch.all(residual <= bound + 1e-11):
            raise AssertionError("finite-precision chronological trace exceeds measured local error budget")
        peak_error=max(peak_error,float(residual.max()))
    return {
        "positions":basis64.shape[0],
        "dim":88,
        "maximum_actual_prefix_abs_error":peak_error,
        "maximum_certified_trace_bound":float((total_add_roundoff+total_basis_cast).max()),
        "max_final_error":float((observed.to(torch.float64)-ideal).abs().max()),
    }
