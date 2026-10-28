"""Source-only ngram retrieval with a valid probability-law mixture.

This is an algorithmic reference available to all neural backbones, NOT a
learned TAC advantage. FIT-only calibration must choose the mixing weight.
"""
import math,torch
import torch.nn.functional as F


def suffix_source_queries(tokens, k=4):
    if tokens.ndim != 2 or tokens.dtype != torch.long or k < 1:
        raise ValueError('expected observed long[B,T] and positive k')
    b,t=tokens.shape
    ordinal=torch.full((b,t),-1,dtype=torch.long,device=tokens.device)
    seen=torch.zeros((b,t),dtype=torch.bool,device=tokens.device)
    # Source-only, no next-token target: at position t, index only j+1<t.
    for row in range(b):
        src=tokens[row].tolist();prior={}
        for now in range(t):
            j=now-2
            if j>=k-1:prior[tuple(src[j-k+1:j+1])]=j+1
            if now>=k-1:
                q=prior.get(tuple(src[now-k+1:now+1]))
                if q is not None:
                    ordinal[row,now]=q;seen[row,now]=True
    return ordinal,seen


def copy_mixture_probs(logits, observed, fit_prior, lam, k=4, eps=.05):
    if logits.ndim!=3 or logits.shape[:2]!=observed.shape or logits.size(-1)<2:
        raise ValueError('vocabulary and sequence mismatch')
    if observed.dtype!=torch.long or observed.device!=logits.device:
        raise ValueError('source token dtype/device mismatch')
    if fit_prior.shape!=(logits.size(-1),) or fit_prior.device!=logits.device:
        raise ValueError('FIT prior must match vocabulary/device')
    if not (math.isfinite(lam) and 0<=lam<=1 and math.isfinite(eps) and 0<eps<1):
        raise ValueError('finite convex mixing and nonzero smoothing required')
    if not bool(torch.isfinite(fit_prior).all()) or bool((fit_prior<=0).any()) or not torch.allclose(fit_prior.sum(),fit_prior.new_tensor(1),atol=1e-6):
        raise ValueError('FIT prior must be positive normalized probability law')
    ordinals,hits=suffix_source_queries(observed,k)
    candidate=torch.gather(observed,1,ordinals.clamp_min(0))
    prior=fit_prior.to(dtype=logits.dtype)
    copy=eps*prior.view(1,1,-1)+(1-eps)*F.one_hot(candidate,num_classes=logits.size(-1)).to(logits.dtype)
    base=F.softmax(logits,dim=-1)
    updated=(1-lam)*base+lam*copy
    distribution=torch.where(hits[...,None],updated,base)
    if not bool(torch.isfinite(distribution).all()):
        raise ValueError('nonfinite output probability')
    return distribution,ordinals,hits
