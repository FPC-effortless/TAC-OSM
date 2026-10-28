"""GRAPH009: normalized owner-bound source-count hybrid for frozen v3/v4.

Both backbones and competitors can use the same source-only cache.
The calibrated mixing constants were selected on FIT, not an external test.
"""
import torch
from torch import nn
from bayesian_cache_scan_v5 import ScannedCountCache

class SourceCalibratedTAC(nn.Module):
    def __init__(self,backbone,fit_prior,*,capacity=512,order=3,tau=.05,mixture=.6):
        super().__init__()
        if order not in (2,3,4) or tau<=0 or not 0<=mixture<=1:
            raise ValueError('invalid calibrated policy')
        if fit_prior.shape!=(128,) or not bool(torch.isfinite(fit_prior).all()) or not bool((fit_prior>0).all()) or not torch.allclose(fit_prior.sum(),fit_prior.new_tensor(1.),atol=1e-6):
            raise ValueError('FIT prior must be strictly positive and normalized')
        self.backbone=backbone
        self.register_buffer('fit_prior',fit_prior.detach().clone())
        self.cache=ScannedCountCache(capacity)
        self.order=order;self.tau=tau;self.mixture=mixture

    def forward(self,tokens,*,start=0,source_ids=None,initial_state=None,return_state=False):
        if tokens.ndim!=2 or tokens.dtype!=torch.long:raise ValueError('observed tokens must be int64[B,T]')
        if initial_state is None:
            if start!=0:raise ValueError('missing nonzero-start state')
            base_state=cache_state=None
        else:
            if initial_state.get('count')!=start or initial_state.get('source_ids')!=source_ids:
                raise ValueError('source/count mismatch')
            base_state=initial_state['backbone'];cache_state=initial_state['source_cache']
        logits,updated=self.backbone(tokens,start=start,source_ids=source_ids,
                                     initial_state=base_state,return_state=True)
        counts,new_cache=self.cache.scan_counts(tokens,start=start,source_ids=source_ids,initial_state=cache_state)
        index=self.order-2
        c=counts['counts'][:,:,index].to(logits.dtype)
        n=counts['totals'][:,:,index].to(logits.dtype)
        tau=self.tau
        prior=self.fit_prior.to(dtype=logits.dtype,device=logits.device)
        copy=(c+tau*prior[None,None,:])/(n[...,None]+tau)
        base=torch.softmax(logits,dim=-1)
        hit=n>0
        weight=torch.where(hit,torch.full_like(n,self.mixture),torch.zeros_like(n))
        probs=(1-weight[...,None])*base+weight[...,None]*copy
        if return_state:
            return probs,{'count':start+tokens.shape[1],'source_ids':source_ids,
                          'backbone':updated,'source_cache':new_cache}
        return probs
