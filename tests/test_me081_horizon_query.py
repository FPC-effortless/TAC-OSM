import torch
from tac_osm.variants.horizon_query_world import HorizonQueryWorld,mixture_crps,horizon_distribution_loss
from tac_osm.variants.reusable_execution import mixture_quantiles


def test_live_size_gradients_and_scale_order():
    torch.manual_seed(81);m=HorizonQueryWorld();x=torch.randn(2,32,14)*.01;a=torch.randn_like(x)
    p=m(x,a);y=x[:,:,None]+torch.randn(2,32,3,14)*.02
    assert sum(v.numel() for v in m.parameters())==100165
    assert (p['moving_scale']>=2*p['quiet_scale']).all()
    loss=horizon_distribution_loss(p,y,x,torch.ones(2,32,3,dtype=torch.bool),torch.ones(2),m.state_scale,.01)
    loss.backward()
    for name,v in m.named_parameters():
        assert v.grad is not None and torch.isfinite(v.grad).all(),name
        assert v.grad.abs().sum()>0,name


def test_crps_against_independent_cdf_quadrature():
    # Independent numerical integration of integral(F(z)-1[z>=y])^2 dz.
    dtype=torch.float64;z=torch.linspace(-80,80,160001,dtype=dtype)
    for mu,prob,y in ((0.,.2,.1),(2.,.7,-1.),(-3.,.8,4.)):
        b=.7;c=2.1
        p={'current':torch.tensor(0.,dtype=dtype),'moving_delta':torch.tensor(mu,dtype=dtype),
           'quiet_scale':torch.tensor(b,dtype=dtype),'moving_scale':torch.tensor(c,dtype=dtype),
           'mixture_logit':torch.logit(torch.tensor(prob,dtype=dtype))}
        analytic=mixture_crps(p,torch.tensor(y,dtype=dtype))
        F0=torch.distributions.Laplace(0.,b).cdf(z);F1=torch.distributions.Laplace(mu,c).cdf(z)
        numeric=torch.trapezoid(((1-prob)*F0+prob*F1-(z>=y).double()).square(),z)
        torch.testing.assert_close(analytic,numeric,atol=.001,rtol=.001)
        assert analytic>=0


def test_causal_prefix_and_carried_query_decoder():
    torch.manual_seed(82);m=HorizonQueryWorld().double();x=torch.randn(2,32,14,dtype=torch.double)*.01;a=torch.randn_like(x)
    p=m(x,a);xx=x.clone();xx[:,17:]+=10
    altered=m(xx,a)
    for k in p:torch.testing.assert_close(p[k][:,:17],altered[k][:,:17])
    left,state=m(x[:,:16],a[:,:16],return_state=True)
    right,final=m(x[:,16:],a[:,16:],state,True)
    for k in p:torch.testing.assert_close(p[k],torch.cat((left[k],right[k]),1))
    last=m.forecast_last(x,a)
    q=mixture_quantiles(last)
    assert (q[0]<=q[1]).all() and (q[1]<=q[2]).all()
