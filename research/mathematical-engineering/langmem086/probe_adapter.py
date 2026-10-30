"""Actual086 categorical-observation/state adapter; no candidate/oracle history API."""
from dataclasses import replace
import torch
from tac_osm.research_probe import ResearchProbeAdapter

def adapter086():
    def validate(model,sample):
        if set(sample)!={'tokens','owners'}:raise ValueError('observed bytes and explicit owners ONLY')
        if sample['tokens'].dtype!=torch.long or sample['tokens'].ndim!=2 or sample['tokens'].shape[1]<2:raise ValueError('observed source prefix')
        model.validate(sample['tokens'],model.start(sample['owners']),sample['owners'])
    def forecast(model,state,sample):
        token=sample['tokens'][:,-1:]
        out,_=model(token,state=state,owners=sample['owners'])
        return {'nextbyte_logits':out[:,0]}
    def graph(model,state,sample):
        leaves={k:v.detach().clone().requires_grad_(True) for k,v in state.tensors.items()}
        copied=replace(state,tensors=leaves)
        return forecast(model,copied,sample),leaves
    return ResearchProbeAdapter(name='LANGMEM086-autonomous-source-byte-stream',validate=validate,
        steps=lambda x:x['tokens'].shape[1]-1,start=lambda m,x:m.start(x['owners']),
        advance=lambda m,s,x,t:m(x['tokens'][:,t:t+1],state=s,owners=x['owners'])[1],
        forecast=forecast,state_views=lambda s:s.tensors,
        patch_state=lambda s,k,v:replace(s,tensors={**s.tensors,k:v}),batch_size=lambda x:x['tokens'].shape[0],
        batch_item=lambda x,i:{'tokens':x['tokens'][i:i+1],'owners':(x['owners'][i],)},gradient_graph=graph)
