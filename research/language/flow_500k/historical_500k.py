"""TAC-LANG-LONG-500K-003: a genuinely causal source-ordinal historical read.

Pure extension of the 499,290-parameter WaveCED500K source.  The historical
query accepts a known *past* ordinal, never the character at that position.
Historical targets are read only by the loss, never by this forward pass.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from tac_wave_ced_500k_stream import WaveCED500K, ordinal_code

LAGS=(64,192)
class HistoricalWaveCED500K(WaveCED500K):
    def __init__(self):
        super().__init__()
        self.history_head=nn.Linear(32,128)

    def forward_with_history(self,tokens,*,lags=LAGS,start=0,initial_state=None,return_state=False,key_shift=0,zero_read=False):
        # We rely on the source model's pre-write state, and never inspect any
        # token at a historical target ordinal in computing historical logits.
        logits,trace,state=self.forward(tokens,start=start,initial_state=initial_state,
                                        return_trace=True,return_state=True)
        b,t=tokens.shape
        historical={}
        for lag in lags:
            if type(lag) is not int or lag<=0 or lag>=t:
                raise ValueError('valid strictly-past historical lag required')
            indices=torch.arange(start,start+t,device=tokens.device).view(1,t).expand(b,t)
            # Pre-write prefix at t.  Fixed source-known address, not observed x_(t-lag).
            known_past=ordinal_code(indices-lag+key_shift,logits.dtype)
            hist_key=F.normalize(torch.einsum('bti,hij->bthj',known_past,self.rotation),dim=-1,eps=1e-6)
            read=torch.einsum('bthvk,bthk->bthv',trace['aligned_prewrite'],hist_key)
            if zero_read: read=torch.zeros_like(read)
            historical[lag]=self.history_head(read.flatten(-2)[:,lag:,:])
        return (logits,historical,state) if return_state else (logits,historical)


def historical_loss(model,tokens, *, history_weight=.25,lags=LAGS):
    if tokens.size(1)<max(lags)+2:raise ValueError('requires real long-context source')
    x=tokens[:,:-1];y=tokens[:,1:]
    logits,history=model.forward_with_history(x,lags=lags)
    next_loss=F.cross_entropy(logits.reshape(-1,128),y.reshape(-1))
    hist_losses={lag:F.cross_entropy(p.reshape(-1,128),x[:,:-lag].reshape(-1))
                 for lag,p in history.items()}
    # At step t, h_t must reconstruct x_(t-lag) exactly; no future token used.
    total=next_loss+history_weight*sum(hist_losses.values())/len(hist_losses)
    return total,next_loss,hist_losses
