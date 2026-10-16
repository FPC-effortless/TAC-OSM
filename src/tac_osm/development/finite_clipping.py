"""Prospective CPU-FP32 clipping candidate; NOT used by frozen DEV-034 trials."""
import math
import torch


@torch.no_grad()
def clip_finite_cpu_gradients_(parameters,max_norm=1.):
    """Reject nonfinite elements before any mutation; accumulate the norm in FP64."""
    if not math.isfinite(max_norm) or max_norm<=0:raise ValueError('positive finite clipping bound required')
    gradients=[p.grad for p in parameters if p.grad is not None]
    if any(g.device.type!='cpu' or g.dtype!=torch.float32 or g.is_sparse for g in gradients):
        raise ValueError('candidate is restricted to dense CPU FP32 gradients')
    if any(not torch.isfinite(g).all() for g in gradients):raise RuntimeError('nonfinite gradient elements')
    norm=math.hypot(*(float(torch.linalg.vector_norm(g.double())) for g in gradients))
    if not math.isfinite(norm):raise RuntimeError('nonfinite FP64 total gradient norm')
    coefficient=min(1.,max_norm/(norm+1e-6))
    for gradient in gradients:gradient.mul_(coefficient)
    return norm
