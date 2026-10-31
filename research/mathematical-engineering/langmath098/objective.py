"""Coefficient-controlled loss only; no model/data/optimizer side effects."""
import math
import torch
from torch.nn import functional as F


def controlled_loss(logits, targets, future, valid, domains, *, prefix_weight):
    if not math.isfinite(prefix_weight) or prefix_weight<0:
        raise ValueError('finite nonnegative prefix coefficient required')
    if (logits.ndim!=3 or logits.shape[-1]!=256 or targets.dtype!=torch.long
            or logits.shape[:2]!=targets.shape or logits.device!=targets.device
            or bool((targets<0).any()) or bool((targets>255).any())):
        raise ValueError('aligned byte logits/targets required')
    for mask in (future,valid):
        if mask.shape!=targets.shape or mask.dtype!=torch.bool or mask.device!=targets.device:
            raise ValueError('aligned Boolean target masks required')
    prefix=valid & ~future
    if (bool((future & ~valid).any()) or not bool(future.any(1).all())
            or not bool(prefix.any(1).all()) or len(domains)!=targets.shape[0]
            or set(domains)!={'code','arc'}):
        raise ValueError('both domains and nonempty prefix/future in every case required')
    ce=F.cross_entropy(logits.flatten(0,1),targets.flatten(),reduction='none').reshape_as(targets)
    per_case=(ce*future).sum(1)/future.sum(1)+prefix_weight*(ce*prefix).sum(1)/prefix.sum(1)
    result=sum(per_case[torch.tensor([d==name for d in domains],device=logits.device)].mean()
               for name in ('code','arc'))/2
    if not bool(torch.isfinite(result)):
        raise ValueError('nonfinite controlled objective')
    return result
