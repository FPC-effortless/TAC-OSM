"""Numerical mechanisms only; these are not performance evidence."""
import pytest
import torch
from tac_osm.me030_joint_readout import make_model, COUNTS


def data():
    torch.set_num_threads(1)
    torch.manual_seed(30030)
    return (torch.randint(16000,(2,128)), torch.rand(2,128,3),
            torch.randint(16000,(2,)), torch.rand(2,1))


@pytest.mark.parametrize('arm', list(COUNTS))
def test_counts_prior_script_and_causal_gradients(arm):
    args=data()
    model=make_model(arm,.5,.2,torch.full((16000,),.4))
    assert sum(p.numel() for p in model.parameters())==COUNTS[arm]
    expected=torch.logit((args[1][:,-32:,1].sum(-1,keepdim=True)+1)/34)
    assert torch.allclose(model(*args),expected)
    with torch.no_grad():
        model.head.weight.normal_(std=.1)
        model.observable_output.weight.normal_(std=.1)
    scripted=torch.jit.script(model)
    out=model.prefix_logits(*args)
    assert torch.allclose(scripted.prefix_logits(*args),out,atol=1e-6)
    for k,col in ((32,0),(64,1),(96,2)):
        items,meta,action,duration=[t.clone() for t in args]
        items[:,k+1:]=torch.randint(16000,items[:,k+1:].shape)
        meta[:,k+1:]=torch.rand_like(meta[:,k+1:])
        meta[:,k,1]=1-meta[:,k,1]
        assert torch.allclose(model.prefix_logits(items,meta,action,duration)[:,col],out[:,col],atol=1e-6)
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0
               for p in model.parameters())


def test_native_body_unchanged_and_cross_derivative():
    args=data();pop=torch.full((16000,),.4)
    torch.manual_seed(14001);a=make_model('additive_native',.5,.2,pop)
    torch.manual_seed(14001);b=make_model('joint_native',.5,.2,pop)
    assert torch.equal(a.latent_features(*args[:2]),b.latent_features(*args[:2]))
    for arm,expected_zero in [('additive_native',True),('joint_native',False)]:
        net=make_model(arm,.5,.2,pop).double()
        with torch.no_grad():net.observable_output.weight.fill_(.1)
        h=torch.randn(1,64,dtype=torch.double,requires_grad=True)
        x=torch.randn(1,26,dtype=torch.double,requires_grad=True)
        p=torch.full((1,1),.4,dtype=torch.double)
        g=torch.autograd.grad(net.predict_features(h,x,p).sum(),h,create_graph=True)[0][0,0]
        mixed=torch.autograd.grad(g,x,allow_unused=True)[0] if g.requires_grad else None
        if expected_zero:assert mixed is None or mixed.abs().max()==0
        else:
            z=net.observable_hidden(torch.cat((h,x),-1))[0]
            w=net.observable_hidden.weight
            analytic=(-2*.1*z.tanh()*(1-z.tanh().square())*w[:,0]*w[:,64]).sum()
            assert torch.allclose(mixed[0,0],analytic,atol=1e-12)
            assert mixed.abs().max()>1e-8


def test_pool_latent_permutation_invariance():
    items,meta,_,_=data()
    model=make_model('joint_pool',.5,.2,torch.full((16000,),.4))
    perm=torch.randperm(128)
    assert torch.allclose(model.latent_features(items,meta),
                          model.latent_features(items[:,perm],meta[:,perm]),atol=2e-6)
