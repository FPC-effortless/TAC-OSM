"""GRAPH008: source-known historical reconstruction, never label-input."""
import torch
import torch.nn.functional as F
from torch import nn
from tac_wave_ced_500k_v4 import SlotTAC500K

class HistoricalSlotTAC500K(SlotTAC500K):
    def __init__(self,capacity=512):
        super().__init__(capacity)
        self.history_head=nn.Linear(32,128)
    def forward_with_history(self,tokens,*,lags=(64,192),start=0,
                             initial_state=None,source_ids=None,
                             return_state=False,zero_read=False):
        logits,_,state,extras=self._run(tokens,start=start,initial_state=initial_state,source_ids=source_ids)
        observed,old_slot,_,_=extras
        b,t=observed.shape[:2]
        hist={}
        for lag in lags:
            if type(lag) is not int or lag<=0 or lag>self.episodic.capacity:
                raise ValueError('lag must be inside bounded memory capacity')
            # Entire current chunk is evaluated if some earlier carried
            # source is available; valid outputs start at global ordinal lag.
            first=max(0,lag-start)
            if first>=t:
                hist[lag]=logits.new_empty((b,0,128))
                continue
            tt=torch.arange(start+first,start+t,device=tokens.device,dtype=torch.long)[None,:].expand(b,-1)
            source=tt-lag
            # Use same prewrite state as primary decoder; no labels injected.
            val,_,valid=self.episodic.step_chunk(observed,start=start,
                  owner_ids=source_ids,initial_state=old_slot,
                  queries=torch.cat((torch.full((b,first),-1,dtype=torch.long,device=tokens.device),source),dim=1))
            val=val[:,first:]
            if zero_read:val=torch.zeros_like(val)
            hist[lag]=self.history_head(torch.cat((val,val),dim=-1))
        return (logits,hist,state) if return_state else (logits,hist)

def historical_loss(model,tokens,source_ids,weight=.3,lags=(64,192)):
    x=tokens[:,:-1];y=tokens[:,1:]
    logits,h=model.forward_with_history(x,source_ids=source_ids,lags=lags)
    nxt=F.cross_entropy(logits.reshape(-1,128),y.reshape(-1))
    errors={lag:F.cross_entropy(v.reshape(-1,128),x[:,:-lag].reshape(-1)) for lag,v in h.items()}
    return nxt+weight*sum(errors.values())/len(errors),nxt,errors
