"""TAC-LANG-500K-GRAPH-006-v2: absolute-ordinal streaming-safe historical read.

Pure extension of the 499,290-parameter WaveCED500K source.  The historical
query accepts a known *past* ordinal, never the character at that position.
Historical targets are read only by the loss, never by this forward pass.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from tac_wave_ced_500k_v2 import WaveCED500K, ordinal_code

LAGS=(64,192)
class HistoricalWaveCED500K(WaveCED500K):
    def __init__(self):
        super().__init__()
        self.history_head=nn.Linear(32,128)

    def forward_with_history(self,tokens,*,lags=LAGS,start=0,initial_state=None,
                             return_state=False,key_shift=0,zero_read=False,
                             return_positions=False,graph_scale=1.0):
        # Chronology is ABSOLUTE: after consuming 200 tokens a 1-token
        # chunk can legally ask for lag192, even though local chunk length=1.
        # Source owner and event-ID authorization remains caller's duty.
        if type(key_shift) is not int:
            raise ValueError('source-ordinal diagnostic key shift must be integer')
        if type(lags) not in (tuple,list) or len(lags)==0 or len(set(lags))!=len(lags):
            raise ValueError('lags must be a nonempty unique sequence')
        for lag in lags:
            if type(lag) is not int or lag<=0:
                raise ValueError('lag must be a positive integer')
            if key_shift>=lag:
                raise ValueError('diagnostic key shift may not query present/future')
        logits,trace,state=self.forward(
            tokens,start=start,initial_state=initial_state,
            return_trace=True,return_state=True,graph_scale=graph_scale)
        b,t=tokens.shape
        historical={}
        positions={}
        for lag in lags:
            # Each output covers the subset of this chunk with a strictly
            # past, globally valid target; no data from the target is read.
            first=max(0,lag-start,lag-key_shift-start)
            if first>=t:
                historical[lag]=logits.new_empty((b,0,self.vocab))
                positions[lag]=dict(current_ordinals=tuple(),
                                    target_ordinals=tuple(),local_start=t)
                continue
            current=torch.arange(start+first,start+t,device=tokens.device)
            query=(current-lag+key_shift).view(1,-1).expand(b,-1)
            phi=ordinal_code(query,logits.dtype)
            hist_key=F.normalize(torch.einsum('bti,hij->bthj',phi,self.rotation),dim=-1,eps=1e-6)
            read=torch.einsum('bthvk,bthk->bthv',
                              trace['aligned_prewrite'][:,first:],hist_key)
            if zero_read: read=torch.zeros_like(read)
            historical[lag]=self.history_head(read.flatten(-2))
            positions[lag]=dict(current_ordinals=tuple(current.tolist()),
                                target_ordinals=tuple((current-lag).tolist()),
                                local_start=first)
        parts=[logits,historical]
        if return_positions:parts.append(positions)
        if return_state:parts.append(state)
        return tuple(parts)


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
