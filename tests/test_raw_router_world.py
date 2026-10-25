import torch
from tac_osm.variants.raw_router_world import RawRouterWorld
from tac_osm.variants.bounded_value_world import BoundedValueWorld
from tac_osm.variants.implicit_quantile import implicit_median


def test_new_coordinate_evidence_is_active_causal_and_incrementally_equivalent():
    torch.manual_seed(89089);torch.set_num_threads(1);m=RawRouterWorld().double()
    source=BoundedValueWorld().double();loaded=m.load_state_dict(source.state_dict(),strict=False)
    assert not loaded.unexpected_keys and len(loaded.missing_keys)==5
    assert sum(p.numel() for p in m.parameters())==101964
    x=torch.randn(2,9,14,dtype=torch.float64)*.01;a=torch.randn_like(x)*.01
    dt=torch.full((2,9,1),1/15,dtype=x.dtype);dt[:,4]=.7
    full,final=m(x,a,return_state=True,elapsed=dt);state=None;parts=[]
    for t in range(9):
        out,state=m.step(x[:,t],a[:,t],state,elapsed=dt[:,t]);parts.append(out)
    for k in full:torch.testing.assert_close(full[k],torch.stack([v[k] for v in parts],1),rtol=1e-10,atol=1e-10)
    for u,v in zip(final,state):torch.testing.assert_close(u,v,rtol=1e-10,atol=1e-10)
    changed=x.clone();changed[:,5:]+=1
    alternate=m(changed,a,elapsed=dt)
    for k in full:torch.testing.assert_close(full[k][:,:5],alternate[k][:,:5],rtol=0,atol=0)
    loss=implicit_median(full).square().mean()+full['mixture_logit'].square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert all(p.grad.abs().sum()>0 for p in m.coordinate_router.parameters())
    assert m.coordinate_embedding.weight.grad.abs().sum()>0
