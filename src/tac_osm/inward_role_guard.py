"""Prospective, no-fit inward role guard; does not replace historical optimizers.

FP32 weights/gradients/candidates only. The error envelope assumes ordinary
IEEE FP64 basic arithmetic and a dot reduction with at most n additions; full
PyTorch/IEEE refinement is OPEN. Always validate the realized candidate rather
than treating an ideal projection identity as an implementation certificate.
"""
import math
import torch


def inward_candidates(weights, gradients, raw_candidates, blocks):
    """Return independent candidates and per-block conditional certificates.

    No mutation, optimizer state, learned parameters, or gradient-through-guard.
    A conservative precision-derived inward margin avoids projecting exactly
    onto zero. Failed certificates reject protected coordinates, preserving
    zero-gradient coordinates. This cannot guarantee finite-step loss descent.
    """
    weights,gradients,raw_candidates=map(tuple,(weights,gradients,raw_candidates))
    blocks=tuple(tuple(b) for b in blocks)
    if not weights or not len(weights)==len(gradients)==len(raw_candidates):
        raise ValueError('matching nonempty lists required')
    if sorted(i for b in blocks for i in b)!=list(range(len(weights))) or any(not b for b in blocks):
        raise ValueError('blocks must partition tensors')
    for p,g,c in zip(weights,gradients,raw_candidates):
        if p.dtype!=torch.float32 or g.dtype!=p.dtype or c.dtype!=p.dtype or p.shape!=g.shape or p.shape!=c.shape or p.device!=g.device or p.device!=c.device:
            raise ValueError('matching finite FP32 tensors required')
        if not all(torch.isfinite(x).all() for x in (p,g,c)):
            raise ValueError('nonfinite input')
    result=[c.detach().clone() for c in raw_candidates];receipts=[]
    u64=torch.finfo(torch.float64).eps
    u32=torch.finfo(torch.float32).eps
    with torch.no_grad():
        for block in blocks:
            n=sum(weights[i].numel() for i in block)
            # gamma_(n+8) covers difference/product/reduction operations;
            # eps rather than half-eps is a conservative unit-roundoff choice.
            if (n+8)*u64>=.5: raise ValueError('unsupported error-envelope length')
            gamma=(n+8)*u64/(1-(n+8)*u64)
            norm=sum(gradients[i].double().square().sum() for i in block)
            dot=sum((gradients[i].double()*(raw_candidates[i].double()-weights[i].double())).sum() for i in block)
            scale=sum((gradients[i].double().abs()*(weights[i].double().abs()+raw_candidates[i].double().abs())).sum() for i in block)
            # Covers final FP32 parameter rounding as well as FP64 reduction.
            # This is precision-derived, not selected from model outcomes.
            margin=(4*u32+8*gamma)*scale
            alpha=torch.clamp((dot+margin)/norm.clamp_min(torch.finfo(torch.float64).tiny),min=0) if norm>0 else torch.zeros_like(norm)
            for i in block:
                result[i]=(raw_candidates[i].double()-alpha*gradients[i].double()).to(torch.float32)
            def certificate():
                actual=sum((gradients[i].double()*(result[i].double()-weights[i].double())).sum() for i in block)
                mass=sum((gradients[i].double().abs()*(result[i].double().abs()+weights[i].double().abs())).sum() for i in block)
                error=gamma*mass/(1-gamma)
                return actual,error
            actual,error=certificate()
            failed=bool(actual+error>0) if norm>0 else False
            if failed:
                for i in block:
                    result[i]=torch.where(gradients[i]==0,result[i],weights[i].detach())
                # All nonzero-gradient coordinates now have EXACT zero change;
                # certificate need not subtract two uncertain rounded dot sums.
                actual=torch.zeros_like(actual);error=torch.zeros_like(error)
            if not all(torch.isfinite(result[i]).all() for i in block):
                raise ValueError('nonfinite candidate')
            if actual+error>0: raise ValueError('certificate failed')
            receipts.append(dict(norm_squared=float(norm),raw_dot=float(dot),margin=float(margin),
                alpha=float(alpha),actual_dot=float(actual),dot_error_bound=float(error),
                certified_upper_dot=float(actual+error),fallback=failed,
                qualification='conditional_IEEE_error_envelope_not_full_source_proof'))
    return result,receipts
