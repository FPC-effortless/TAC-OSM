"""Centered readout, actual memory gradients and causal isolation checks."""
import pytest
import torch
from tac_osm.me032_conditional_memory import (
    CenteredReadout,CommonHead,NativeMemory,PooledMemory,ConditionalMemory,
    FrozenContext,COUNTS,make_student
)
from tac_osm.me031_crossfit_state import CrossfitNativeEncoder


def inputs():
    torch.set_num_threads(1);torch.manual_seed(32032)
    return (torch.randint(16000,(2,128)),torch.rand(2,128,3),
            torch.randint(16000,(2,)),torch.rand(2,1))


def test_centering_and_analytic_state_jacobian():
    inputs();net=CenteredReadout().double()
    with torch.no_grad():net.output.weight.normal_();net.direct.weight.normal_()
    h=torch.randn(2,64,dtype=torch.double,requires_grad=True)
    x=torch.randn(2,26,dtype=torch.double,requires_grad=True)
    assert torch.equal(net(torch.zeros_like(h),x),torch.zeros(2,dtype=torch.double))
    assert sum(p.numel() for p in net.parameters())==1260
    grad=torch.autograd.grad(net(h,x).sum(),h)[0]
    z=net.hidden(torch.cat((h,x),-1))
    manual=net.direct.weight+((1-z.tanh().square())*net.output.weight)@net.hidden.weight[:,:64]
    torch.testing.assert_close(grad,manual,rtol=0,atol=1e-12)


def test_native_equations_and_initialization_match_original():
    items,meta,_,_=inputs();pop=torch.full((16000,),.4)
    torch.manual_seed(14);a=NativeMemory()
    torch.manual_seed(14);b=CrossfitNativeEncoder(.5,.2,pop)
    assert sum(p.numel() for p in a.parameters())==134976
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    torch.testing.assert_close(a(items,meta)[:,-1],b.latent_features(items,meta),rtol=0,atol=1e-7)


@pytest.mark.parametrize('arm',list(COUNTS))
def test_active_counts_initial_base_gradient_routing_and_script(arm):
    items,meta,_,_=inputs();context=torch.randn(2,4,27)
    base=CommonHead()
    with torch.no_grad():base.output.weight.normal_(std=.1)
    torch.manual_seed(32);net=make_student(arm,base.state_dict())
    assert sum(p.numel() for p in net.parameters() if p.requires_grad)==COUNTS[arm]
    expected=context[...,26]+base(context[...,:26])
    torch.testing.assert_close(net(items,meta,context),expected)
    assert torch.allclose(net(items,meta,context),torch.jit.script(net)(items,meta,context),atol=1e-6)
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.003)
    original={k:v.clone() for k,v in net.common.state_dict().items()}
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss=net(items,meta,context).square().mean();loss.backward();opt.step()
    active=[p for p in net.parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in active)
    if arm=='joint_native':assert any(not torch.equal(v,net.common.state_dict()[k]) for k,v in original.items())
    else:
        assert all(torch.equal(v,net.common.state_dict()[k]) for k,v in original.items())
        assert all(p.grad is None for p in net.common.parameters())


def test_prefix_causality_and_pool_latent_invariance():
    items,meta,action,duration=inputs();pop=torch.full((16000,),.4)
    ctx=FrozenContext(torch.randn(16000,8),.5,.2,pop,torch.zeros(26),torch.ones(26))
    context=ctx(items,meta,action,duration)
    for pooled in [False,True]:
        net=ConditionalMemory(pooled,False)
        with torch.no_grad():net.correction.output.weight.normal_();net.correction.direct.weight.normal_()
        original=net(items,meta,context)
        for k,col in ((32,0),(64,1),(96,2)):
            i=items.clone();m=meta.clone();i[:,k+1:]=torch.randint(16000,i[:,k+1:].shape)
            m[:,k+1:]=torch.rand_like(m[:,k+1:]);m[:,k,1]=1-m[:,k,1]
            newctx=ctx(i,m,action,duration)
            torch.testing.assert_close(newctx[:,col],context[:,col],rtol=0,atol=1e-6)
            torch.testing.assert_close(net(i,m,newctx)[:,col],original[:,col],rtol=0,atol=1e-6)
    memory=PooledMemory();p=torch.randperm(128)
    torch.testing.assert_close(memory(items,meta)[:,-1],memory(items[:,p],meta[:,p])[:,-1],rtol=0,atol=2e-6)
