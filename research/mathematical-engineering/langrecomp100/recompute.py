"""Pure execution wrapper for the unchanged LANGTRAIN099 model.

Non-reentrant checkpointing accepts integer tokens and explicit floating state.
No state is detached; owners, clock and write policy are fixed metadata. This
uses standard activation recomputation, not GRAPH011's inference replay engine.
"""
import torch
from torch.utils.checkpoint import checkpoint

POLICIES=('native','recompute')

def forward(model,tokens,*,owners,state=None,write_mask=None,policy='native',counter=None):
    if policy not in POLICIES:raise ValueError('registered native/recompute policy required')
    owners=tuple(owners)
    if state is None:state=model.start(owners)
    names=tuple(sorted(state.tensors))
    if write_mask is None:write_mask=torch.ones_like(tokens,dtype=torch.bool)
    def run(x,allowed,*tensors):
        if counter is not None:counter['calls']=counter.get('calls',0)+1
        incoming=type(state)(owners,state.clock,dict(zip(names,tensors)))
        y,final=model(x,state=incoming,owners=owners,write_mask=allowed)
        return (y,*(final.tensors[n] for n in names))
    inputs=(tokens,write_mask,*(state.tensors[n] for n in names))
    outputs=run(*inputs) if policy=='native' else checkpoint(run,*inputs,use_reentrant=False,preserve_rng_state=True,determinism_check='default')
    return outputs[0],type(state)(owners,state.clock+tokens.shape[1],dict(zip(names,outputs[1:])))
