"""Source-reviewed043 state/observation adapter; inference AD unsupported."""
from dataclasses import replace
import torch
from tac_osm.research_probe import ResearchProbeAdapter
from incremental import IncrementalTAC043


def adapter(position_mode='native'):
    def engine(model): return IncrementalTAC043(model,position_mode)
    def validate(model,sample):
        if set(sample)!={'tokens','owners','write_mask'}: raise ValueError('source sample allowlist')
        x=sample['tokens'];mask=sample['write_mask']
        if x.ndim!=2 or x.shape[1]==0 or x.dtype!=torch.long or x.min()<0 or x.max()>255 or mask.shape!=x.shape or mask.dtype!=torch.bool:
            raise ValueError('invalid source tokens/mask')
        if len(sample['owners'])!=x.shape[0]: raise ValueError('batch owners differ')
        engine(model).start(sample['owners'])
    def advance(model,state,sample,t):
        return engine(model).step(sample['tokens'][:,t],state,owners=sample['owners'],write_mask=sample['write_mask'][:,t])[1]
    def forecast(model,state,sample):
        logits,_=engine(model).step(sample['tokens'][:,-1],state,owners=sample['owners'],write_mask=sample['write_mask'][:,-1])
        return {'next_byte_logits':logits}
    def patch(state,name,value):
        if name not in state.tensors or value.shape!=state.tensors[name].shape: raise ValueError('unknown state patch')
        return replace(state,tensors={**state.tensors,name:value.detach().clone()})
    return ResearchProbeAdapter(
        name='LANGSTREAM043_040_GRAPHFREE48_'+position_mode,validate=validate,
        steps=lambda sample:sample['tokens'].shape[1]-1,
        start=lambda model,sample:engine(model).start(sample['owners']),
        advance=advance,forecast=forecast,state_views=lambda state:state.tensors,
        patch_state=patch,batch_size=lambda sample:sample['tokens'].shape[0],
        batch_item=lambda sample,i:{'tokens':sample['tokens'][i:i+1],'write_mask':sample['write_mask'][i:i+1],'owners':sample['owners'][i:i+1]},
        gradient_graph=None,permute_candidates=None)
