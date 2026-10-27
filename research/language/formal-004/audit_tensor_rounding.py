"""Exact-rational local error audit for actual CPU float32 tensor dot products.

This is a FINITE, device-specific certificate for each tested operation, not
a statement that all Torch/GPU kernels obey the same summation algorithm.

Float32 operands and float32 results are converted EXACTLY to binary
rationals via Fraction.from_float. The 2n-gamma / underflow envelope
is asserted only when the kernel result satisfies it. No silent tolerance,
subnormal flushing assumptions, float64 surrogate, or train/eval scores.
"""
from fractions import Fraction
import math
import torch

U_BITS = 24
ETA_DEN = 1 << 149

def exact_rational_dot(a, b):
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("requires float32 operands")
    if a.device.type != "cpu" or b.device.type != "cpu":
        raise ValueError("only auditable CPU inputs")
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or not a.numel():
        raise ValueError("equal nonempty one-dimensional vectors required")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("nonfinite input cannot be decoded to exact rational")
    pairs=[(Fraction.from_float(float(x)),Fraction.from_float(float(y)))
           for x,y in zip(a.tolist(),b.tolist())]
    exact=sum((x*y for x,y in pairs),Fraction(0))
    magnitude=sum((abs(x*y) for x,y in pairs),Fraction(0))
    return exact,magnitude

def audit_float32_dot(a,b,operation="matmul"):
    exact,magnitude=exact_rational_dot(a,b)
    n=int(a.numel())
    if (2*n) >= (1<<U_BITS):
        raise ValueError("too many terms for gamma_{2n} condition")
    if operation=="matmul":
        out=torch.dot(a,b)
    elif operation=="einsum":
        out=torch.einsum("i,i->",a,b)
    elif operation=="rounded_product_sum":
        out=(a*b).sum()
    else:
        raise ValueError("unsupported backend operation")
    if not torch.isfinite(out):
        raise ValueError("nonfinite output not covered by finite-error theorem")
    rounded=Fraction.from_float(float(out))
    diff=abs(rounded-exact)
    # Standard, CONSERVATIVE local envelope for a 2n-operation dot
    # *if* round-to-nearest, finite range and no nonstandard FTZ.
    # The inequality is TESTED, not assumed as a universal backend law.
    gamma=Fraction(2*n,(1<<U_BITS)-2*n)
    budget=gamma*magnitude+Fraction(8*n,ETA_DEN)
    return {
      "n":n,"operation":operation,
      "within_candidate_budget":diff <= budget,
      "actual_absolute_error":float(diff),
      "candidate_absolute_budget":float(budget),
      "max_input_abs":max(float(a.abs().max()),float(b.abs().max()))
    }
