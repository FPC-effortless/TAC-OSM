"""TAC-LANG-REPAIR-017: bounded-state running mean alternate update.

Research-only no model/checkpoint changes. The original source accumulates
an FP32 sum and divides by the count. The proposed replacement maintains
one 88-channel mean (same *number* of floating-state entries), satisfying
the exact-real recurrence:
  mean[n+1] = mean[n] + (basis[n]-mean[n])/(n+1).
In real arithmetic this equals sum(basis[:n+1])/(n+1), and if each
basis is tanh-bounded in [-1,1], mean remains in [-1,1].

Neither arithmetic scheme is uniformly more accurate in float32;
test adversarial scenarios and count both wins and regressions.
"""
import torch

def chronological_mean(basis:torch.Tensor,mode:str)->torch.Tensor:
    if basis.ndim!=2 or basis.dtype not in (torch.float32,torch.float64):
        raise ValueError("basis expected [events, channels] FP32/FP64")
    if not torch.isfinite(basis).all():
        raise ValueError("finite basis required")
    if mode not in ("source_sum_divide","online_mean"):
        raise ValueError("unsupported mode")
    out=basis.new_zeros(basis.shape[1])
    for i,row in enumerate(basis):
        if mode=="source_sum_divide":
            out=out+row
        else:
            out=out+(row-out)/(i+1)
    return out/basis.shape[0] if mode=="source_sum_divide" and basis.shape[0] else out
