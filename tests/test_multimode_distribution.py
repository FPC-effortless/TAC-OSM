import numpy as np
import torch
from tac_osm.variants.multimode_distribution import cdf,log_density,mixture_crps,mixture_quantiles,implicit_median


def example():
    return {'locations':torch.tensor([[-.4],[.1],[1.2]],dtype=torch.float64),'scales':torch.tensor([[.1],[.3],[1.]],dtype=torch.float64),'component_logits':torch.tensor([[.2],[-.3],[.1]],dtype=torch.float64)}


def test_three_laplace_crps_matches_independent_cdf_integral_and_density_normalizes():
    p=example();y=torch.tensor([.7],dtype=torch.float64);grid=torch.linspace(-35,35,300001,dtype=y.dtype)[:,None]
    left=torch.linspace(-35,float(y),150001,dtype=y.dtype)[:,None];right=torch.linspace(float(y),35,150001,dtype=y.dtype)[:,None]
    numeric=np.trapezoid(cdf(p,left).square().numpy()[:,0],left.numpy()[:,0])+np.trapezoid((1-cdf(p,right)).square().numpy()[:,0],right.numpy()[:,0])
    assert abs(float(mixture_crps(p,y))-numeric)<5e-5
    assert abs(np.trapezoid(log_density(p,grid).exp().numpy()[:,0],grid.numpy()[:,0])-1)<1e-6


def test_shared_quantiles_invert_same_distribution_and_translate():
    p=example();q=mixture_quantiles(p)
    for value,level in zip(q,[.05,.5,.95]):torch.testing.assert_close(cdf(p,value),torch.full_like(value,level),rtol=0,atol=1e-10)
    shifted={**p,'locations':p['locations']+3.}
    for a,b in zip(q,mixture_quantiles(shifted)):torch.testing.assert_close(a+3,b,rtol=0,atol=1e-9)


def test_implicit_median_gradients_match_independent_finite_differences():
    p={k:v.clone().requires_grad_() for k,v in example().items()};value=implicit_median(p).sum();value.backward()
    for field in p:
        for i in range(3):
            plus={k:v.detach().clone() for k,v in p.items()};minus={k:v.detach().clone() for k,v in p.items()}
            plus[field][i,0]+=1e-5;minus[field][i,0]-=1e-5
            numeric=(mixture_quantiles(plus,(.5,))[0]-mixture_quantiles(minus,(.5,))[0])/(2e-5)
            torch.testing.assert_close(p[field].grad[i],numeric,rtol=2e-5,atol=2e-6)
