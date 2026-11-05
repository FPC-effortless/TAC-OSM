"""NO-FIT, UNPROMOTED bounded cumprod candidate for CSL-125-R2.

This source is NOT installed in CSL125R2.forward. Replacing the stable
associative reference is not justified by short CPU tests: exact native
forward/backward, GPU numerical and end-to-end throughput gates remain open.
The form is well-conditioned only under the explicit r<=1/2, 0<=g<=1 and
chunk<=64 bounds. Then all within-chunk prefix products >= exp(-32)>1e-14
and float32 prefix division does not underflow.
"""
from __future__ import annotations
import torch
from torch import Tensor


def bounded_cumprod_scan(z: Tensor, initial: Tensor, rate: Tensor, gate: Tensor,
                         chunk_size: int) -> tuple[Tensor,Tensor]:
    if (z.ndim != 5 or z.shape[-1] != 2 or z.shape[1] < 1 or
        initial.shape != z.shape[:1]+z.shape[2:] or
        rate.shape != z.shape[2:4] or gate.shape != z.shape[:3] or
        not (1 <= chunk_size <= 64)):
        raise ValueError('invalid bounded cumprod scan shape or chunk size')
    if (not bool(torch.isfinite(rate).all()) or not bool((rate > 0).all()) or
        not bool((rate <= .5).all()) or not bool(torch.isfinite(gate).all()) or
        not bool((gate >= 0).all()) or not bool((gate <= 1).all())):
        raise ValueError('cumprod requires finite 0<rate<=1/2 and 0<=gate<=1')
    r=rate.float().view(1,1,*rate.shape)
    state=initial.float()
    outputs=[]
    for begin in range(0,z.shape[1],chunk_size):
        block=z[:,begin:begin+chunk_size].float()
        rg=r*gate[:,begin:begin+chunk_size].float().unsqueeze(-1)
        beta=torch.exp(-rg)
        eta=-torch.expm1(-rg)
        prefix=torch.cumprod(beta,dim=1)
        weighted=eta.unsqueeze(-1)*block/prefix.unsqueeze(-1)
        result=prefix.unsqueeze(-1)*(state.unsqueeze(1)+torch.cumsum(weighted,dim=1))
        outputs.append(result)
        state=result[:,-1]
    return torch.cat(outputs,dim=1),state
