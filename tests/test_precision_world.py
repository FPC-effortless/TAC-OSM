import torch
from tac_osm.variants.precision_world import PrecisionWorld
from tac_osm.variants.bounded_value_world import BoundedValueWorld
from tac_osm.variants.implicit_quantile import implicit_median


def test_weight_only_initialization_new_noise_and_state_equivalence():
    torch.manual_seed(88088);torch.set_num_threads(1)
    source=BoundedValueWorld().double();model=PrecisionWorld().double();model.load_state_dict(source.state_dict(),strict=True)
    assert sum(p.numel() for p in model.parameters())==99731
    x=torch.randn(2,9,14,dtype=torch.float64)*.01;a=torch.zeros_like(x)
    dt=torch.full((2,9,1),1/15,dtype=x.dtype);dt[:,4]=.7
    full,final=model(x,a,return_state=True,elapsed=dt);state=None;parts=[]
    for t in range(9):
        out,state=model.step(x[:,t],a[:,t],state,elapsed=dt[:,t]);parts.append(out)
    for k in full:torch.testing.assert_close(full[k],torch.stack([v[k] for v in parts],1),rtol=1e-10,atol=1e-10)
    for u,v in zip(final,state):torch.testing.assert_close(u,v,rtol=1e-10,atol=1e-10)
    assert all(p.requires_grad for p in model.parameters())
    implicit_median(full).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
