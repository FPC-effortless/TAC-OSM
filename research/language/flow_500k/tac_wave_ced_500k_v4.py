"""GRAPH008: unchanged 499290-weight wave/CED TAC with causal episodic slots.

Increases persistent state from 520 floats to 520 + capacity*16 floats,
plus source ordinal metadata. This is an honest hybrid memory capacity tradeoff,
not compressed 520-dimensional associative recall. No new learned parameters.
"""
import torch
from tac_wave_ced_500k_v3 import WaveCED500KSlow, phi_position
from source_slot_memory_v4 import SourceSlotMemory

class SlotTAC500K(WaveCED500KSlow):
    def __init__(self, capacity=512):
        super().__init__()
        self.episodic=SourceSlotMemory(capacity,16)
    def _run(self,tokens,*,start=0,initial_state=None,source_ids=None,
             query_ordinals=None,write_mask=None,graph_scale=1.0):
        if tokens.ndim!=2 or tokens.dtype!=torch.long:raise ValueError('invalid tokens')
        b,t=tokens.shape
        self.episodic._owners(source_ids,b)
        if initial_state is not None:
            if not isinstance(initial_state,dict) or initial_state.get('source_ids')!=source_ids:
                raise ValueError('invalid owner-bound v4 carry')
            base_state=initial_state['base'];slot_state=initial_state['slots']
        else:base_state=slot_state=None
        logits,trace,base_new=super().forward(tokens,start=start,
            initial_state=base_state,source_ids=source_ids,write_mask=write_mask,
            graph_scale=graph_scale,return_trace=True,return_state=True)
        positions=torch.arange(start,start+t,device=tokens.device,dtype=torch.long)[None,:].expand(b,-1)
        embedded=torch.tanh(self.embedding(tokens)+self.pos(phi_position(positions,self.embedding.weight.dtype)))
        gate=.75+.25*self.aligned_gate(embedded).sigmoid().view(b,t,2,3)[:,:,1,2:3]
        val=gate*torch.tanh(self.aligned_value(embedded))
        if query_ordinals is None:
            query_ordinals=positions-1
        read,slot_new,valid=self.episodic.step_chunk(val,start=start,owner_ids=source_ids,
            initial_state=slot_state,queries=query_ordinals,write_mask=write_mask)
        # Direct model prediction path: unlike GRAPH007's auxiliary history
        # head, a selected true-past read enters main decoder probabilities.
        logits=logits+self.concept_out(read)
        return logits,trace,{'count':start+t,'source_ids':source_ids,'base':base_new,'slots':slot_new},(val,slot_state,valid,read)
    def forward(self,tokens,*,start=0,initial_state=None,source_ids=None,
                query_ordinals=None,write_mask=None,graph_scale=1.0,
                return_state=False,return_trace=False):
        logits,trace,state,extra=self._run(tokens,start=start,initial_state=initial_state,
            source_ids=source_ids,query_ordinals=query_ordinals,
            write_mask=write_mask,graph_scale=graph_scale)
        if return_trace:
            trace=dict(trace)
            trace['episodic_read']=extra[-1]
            trace['episodic_valid']=extra[-2]
            return (logits,trace,state) if return_state else (logits,trace)
        return (logits,state) if return_state else logits
