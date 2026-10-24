import torch
from tac_osm.variants.bounded_value_world import BoundedValueWorld
from tac_osm.variants.implicit_quantile import implicit_median,laplace_cdf
from tac_osm.variants.reusable_execution import mixture_quantiles


def test_implicit_median_gradient_matches_independent_finite_difference():
    fields={'current':.3,'moving_delta':1.1,'quiet_scale':.2,'moving_scale':.7,'mixture_logit':-.4}
    p={k:torch.tensor([v],dtype=torch.float64,requires_grad=True) for k,v in fields.items()}
    q=implicit_median(p);actual=torch.autograd.grad(q,tuple(p.values()))
    torch.testing.assert_close(q,mixture_quantiles(p,(.5,))[0],rtol=0,atol=0)
    for name,gradient in zip(fields,actual):
        up={k:v.detach().clone() for k,v in p.items()};down={k:v.detach().clone() for k,v in p.items()};up[name]+=1e-5;down[name]-=1e-5
        finite=(mixture_quantiles(up,(.5,))[0]-mixture_quantiles(down,(.5,))[0])/2e-5
        torch.testing.assert_close(gradient,finite,rtol=2e-4,atol=2e-5)
    z=torch.tensor([0.],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.grad(laplace_cdf(z).sum(),z)[0].item()==.5


def test_quiet_innovation_is_bounded_even_after_large_observed_changes():
    torch.manual_seed(87087);torch.set_num_threads(1);m=BoundedValueWorld().double()
    assert sum(p.numel() for p in m.parameters())==99731
    x=torch.randn(2,9,14,dtype=torch.float64)*10;a=torch.randn_like(x)
    out=m(x,a);unit=m.quiet_noise*m.horizons[None,None,:,None].sqrt()
    assert ((out['current']-x[:,:,None]).abs()<=unit+1e-12).all()
    q=implicit_median(out);q.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
