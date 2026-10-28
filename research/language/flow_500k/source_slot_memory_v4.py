"""GRAPH008: bounded, source-ordinal, causal differentiable episodic slots.

128/256/512 slots hold observed *learned* latent values, not ground-truth
retrieval targets. An exact ordinal is a retrieval address; semantic query
selection is a separate problem. This is a capacity-expanding hybrid, not
free memory in 520 recurrent floats, nor a proof of LLM generalization.
"""
from __future__ import annotations
import torch

class SourceSlotMemory:
    def __init__(self, capacity=512, width=16):
        if not isinstance(capacity, int) or capacity < 1 or not isinstance(width,int) or width < 1:
            raise ValueError('positive capacity and width required')
        self.capacity=capacity;self.width=width

    def empty(self, batch, owner_ids, device, dtype):
        self._owners(owner_ids,batch)
        return {'count':0,'source_ids':owner_ids,
                'values':torch.zeros(batch,self.capacity,self.width,device=device,dtype=dtype),
                'ordinals':torch.full((batch,self.capacity),-1,dtype=torch.long,device=device)}

    @staticmethod
    def _owners(owners,b):
        if type(owners) is not tuple or len(owners)!=b or not all(type(x) is str and bool(x) for x in owners):
            raise ValueError('nonempty source owner tuple required')

    def _check(self,state,b,owners,start,values):
        self._owners(owners,b)
        if not isinstance(state,dict) or state.get('count') != start or state.get('source_ids') != owners:
            raise ValueError('owner or global counter mismatch')
        v,o=state.get('values'),state.get('ordinals')
        if (not isinstance(v,torch.Tensor) or v.shape!=(b,self.capacity,self.width) or
            v.dtype!=values.dtype or v.device!=values.device or not isinstance(o,torch.Tensor) or
            o.shape!=(b,self.capacity) or o.dtype!=torch.long or o.device!=values.device):
            raise ValueError('bad carried source slot geometry')
        if not bool(torch.isfinite(v).all()) or bool((v.abs()>1+1e-5).any()):
            raise ValueError('nonfinite or nonbounded carried slot')
        if bool(((o>=0) & ((o>=start) | (o<max(0,start-self.capacity)))).any()):
            raise ValueError('expired or future slot ordinal')
        if bool(((o>=0)&(o.remainder(self.capacity)!=torch.arange(self.capacity,device=o.device)[None,:])).any()):
            raise ValueError('wrong slot address')

    def step_chunk(self,observed, *, start, owner_ids, initial_state=None,
                   queries=None, write_mask=None):
        """Returns [B,T,W] pre-current-write reads and a carry.

        A query is -1 (no read) or a strict past global ordinal t-d with
        1<=d<=capacity, and must address an admitted source event.
        No future target or historical truth is passed in.
        """
        if observed.ndim!=3 or observed.shape[-1]!=self.width or not torch.is_floating_point(observed):
            raise ValueError('observed learned value shape mismatch')
        b,t,w=observed.shape
        if t<1 or type(start) is not int or start<0:raise ValueError('invalid sequence interval')
        if not bool(torch.isfinite(observed).all()) or bool((observed.abs()>1+1e-5).any()):
            raise ValueError('observed latent must be finite in [-1,1]')
        if initial_state is None:
            if start!=0:raise ValueError('nonzero start needs owner-bound carried state')
            initial_state=self.empty(b,owner_ids,observed.device,observed.dtype)
        self._check(initial_state,b,owner_ids,start,observed)
        if write_mask is None:
            write_mask=torch.ones((b,t),device=observed.device,dtype=torch.bool)
        if write_mask.shape!=(b,t) or write_mask.dtype!=torch.bool or write_mask.device!=observed.device:
            raise ValueError('invalid per-source admission mask')
        if queries is None:
            queries=torch.arange(start,start+t,device=observed.device,dtype=torch.long)[None,:].expand(b,-1)-1
        if queries.shape!=(b,t) or queries.dtype!=torch.long or queries.device!=observed.device:
            raise ValueError('query must be long[B,T]')
        current=torch.arange(start,start+t,device=observed.device)[None,:].expand(b,-1)
        enabled=queries>=0
        legal=(~enabled & (queries==-1)) | (enabled & (queries<current) & (current-queries<=self.capacity))
        if not bool(legal.all()):raise ValueError('not strictly past or outside retained horizon')
        # Reads are prewrite and source-causal. The observed tensors are only
        # available for the correct earlier ordinal of this same stream.
        in_chunk=enabled & (queries>=start)
        local=(queries-start).clamp(0,t-1)
        current_source=torch.gather(observed,1,local[...,None].expand(-1,-1,w))
        current_admitted=torch.gather(write_mask,1,local)
        bucket=queries.clamp_min(0).remainder(self.capacity)
        historical_source=torch.gather(initial_state['values'],1,bucket[...,None].expand(-1,-1,w))
        historical_valid=torch.gather(initial_state['ordinals'],1,bucket)==queries
        valid=(in_chunk & current_admitted) | (enabled & ~in_chunk & historical_valid)
        read=torch.where(in_chunk[...,None],current_source,historical_source)
        read=torch.where(valid[...,None],read,torch.zeros_like(read))
        # End-of-chunk carry: only the last capacity ordinals may survive;
        # all addressed slots in that trailing interval are distinct.
        tail=max(0,t-self.capacity)
        offsets=torch.arange(tail,t,device=observed.device)
        addresses=(start+offsets).remainder(self.capacity)
        slot_index=addresses[None,:].expand(b,-1)
        old_values=initial_state['values'];old_ordinals=initial_state['ordinals']
        prior_vals=torch.gather(old_values,1,slot_index[...,None].expand(-1,-1,w))
        prior_ord=torch.gather(old_ordinals,1,slot_index)
        admitted=write_mask[:,tail:]
        new_values=torch.where(admitted[...,None],observed[:,tail:],prior_vals)
        new_ord=torch.where(admitted,start+offsets[None,:],prior_ord)
        # After CAP steps, older prior slots are no longer eligible and must be
        # physically cleared to keep carry validation exact.
        out_vals=old_values.scatter(1,slot_index[...,None].expand(-1,-1,w),new_values)
        out_ord=old_ordinals.scatter(1,slot_index,new_ord)
        expiry=start+t-self.capacity
        stale=out_ord<max(0,expiry)
        out_ord=torch.where(stale,torch.full_like(out_ord,-1),out_ord)
        out_vals=torch.where(stale[...,None],torch.zeros_like(out_vals),out_vals)
        return read,{'count':start+t,'source_ids':owner_ids,'values':out_vals,'ordinals':out_ord},valid
