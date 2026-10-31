"""Actual dense BYTE source adapter. Adaptive pending-state coverage stays OPEN."""
import torch
from .adaptive_token_controls import DenseByteGRU, validate_previous_ids
from .research_probe import ResearchProbeAdapter


def dense_byte_adapter():
    def validate(model, sample):
        if not isinstance(model, DenseByteGRU) or set(sample) != {'previous_ids'}:
            raise ValueError('exact dense BYTE source and observed-ID sample required')
        validate_previous_ids(sample['previous_ids'])
    def start(model, sample):
        batch=sample['previous_ids'].shape[0]
        return {'hidden':model.byte_emb.weight.new_zeros((batch,118)),
                'current_byte':model.byte_emb.weight.new_zeros((batch,32))}
    def advance(model,state,sample,t):
        current=model.byte_emb(sample['previous_ids'][:,t])
        return {'hidden':model.gru_cell(current,state['hidden']), 'current_byte':current}
    def forecast(model,state,sample):
        return {'next_byte_logits':model.byte_head(torch.cat((state['hidden'],state['current_byte']),-1))}
    def gradient_graph(model,state,sample):
        leaves={k:v.detach().clone().requires_grad_(True) for k,v in state.items()}
        return forecast(model,leaves,sample),leaves
    def patch(state,name,value):
        if name not in state or value.shape!=state[name].shape:
            raise ValueError('unknown state or incorrect patch shape')
        return {**state,name:value}
    return ResearchProbeAdapter(name='sense-tok-006-dense-byte-source',validate=validate,
        steps=lambda s:s['previous_ids'].shape[1],start=start,advance=advance,forecast=forecast,
        state_views=lambda state:state,patch_state=patch,
        batch_size=lambda s:s['previous_ids'].shape[0],
        batch_item=lambda s,i:{'previous_ids':s['previous_ids'][i:i+1]},
        gradient_graph=gradient_graph)
