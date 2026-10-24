"""Mathematical contracts, not real-world capability evidence."""
import pytest
import torch
from tac_osm.variants.clock_value_world import ClockValueWorld


def test_stream_chunk_causality_bounds_and_live_graph():
    torch.manual_seed(85085);torch.set_num_threads(1)
    model=ClockValueWorld().double()
    assert sum(p.numel() for p in model.parameters())==99731
    x=torch.randn(2,19,14,dtype=torch.float64)*.01
    a=torch.randn_like(x)*.01
    dt=torch.full((2,19,1),1/15,dtype=x.dtype);dt[:,7]=.9
    full,final=model(x,a,return_state=True,elapsed=dt)
    state=None;pieces=[]
    for j in range(19):
        out,state=model.step(x[:,j],a[:,j],state,elapsed=dt[:,j]);pieces.append(out)
    for k in full:torch.testing.assert_close(full[k],torch.stack([v[k] for v in pieces],1),rtol=1e-10,atol=1e-10)
    for v,w in zip(final,state):torch.testing.assert_close(v,w,rtol=1e-10,atol=1e-10)
    _,s=model(x[:,:8],a[:,:8],return_state=True,elapsed=dt[:,:8])
    tail,s=model(x[:,8:],a[:,8:],s,return_state=True,elapsed=dt[:,8:])
    for k in full:torch.testing.assert_close(full[k][:,8:],tail[k],rtol=1e-10,atol=1e-10)
    assert final.fast.abs().max()<=1 and final.slow.abs().max()<=1
    assert final.association.norm(dim=(1,2)).max()<=4+1e-10
    changed=x.clone();changed[:,8:]+=1
    suffix=model(changed,a,elapsed=dt)
    for k in full:torch.testing.assert_close(full[k][:,:8],suffix[k][:,:8],rtol=0,atol=0)
    sum(v.square().mean() for v in full.values()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters())


def test_raw_retrieval_is_constant_preserving_and_elapsed_validated():
    model=ClockValueWorld().double();x=torch.full((2,9,14),.037,dtype=torch.float64);a=torch.zeros_like(x)
    out=model(x,a)
    torch.testing.assert_close(out['current'],x[:,:,None].expand(-1,-1,3,-1),rtol=0,atol=1e-15)
    with pytest.raises(ValueError):model(x,a,elapsed=torch.zeros(2,9,1))
