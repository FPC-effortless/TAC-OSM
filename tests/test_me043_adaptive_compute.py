"""Numerical execution contracts only; constructed inputs are not evidence."""
import pytest
import torch
from tac_osm.me043_adaptive_compute import ARMS, COSTS, build_model, objective


def test_me043_capacity_contract_and_finite_active_gradients():
    torch.set_num_threads(1)
    for h in (96,720):
        counts=[sum(p.numel() for p in build_model(a,h,43).parameters()) for a in ARMS]
        assert max(counts)/min(counts)<1.01
    net=build_model('adaptive_001',96,43)
    # Zero-initialized residual head legitimately blocks upstream gradients on
    # the very first update; after updates every registered coefficient learns.
    x=torch.randn(2,96,3);y=torch.randn(2,96,3)
    opt=torch.optim.AdamW(net.parameters(),lr=.001)
    for _ in range(3):
        opt.zero_grad(set_to_none=True);loss,_=objective(net,x,y);loss.backward();opt.step()
    for name,p in net.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        assert p.grad.abs().sum()>0,name


def test_me043_hard_routing_skips_native_and_matches_dense_exits():
    net=build_model('adaptive_001',96,43).eval()
    torch.nn.init.normal_(net.out.weight,std=.01)
    x=torch.randn(2,96,3)
    calls=[]
    hook=net.memory.register_forward_pre_hook(lambda _,args:calls.append(args[0].size(0)))
    with torch.no_grad():
        net.router.bias.fill_(-100)
        cheap,full,*_=net.training_exits(x);calls.clear()
        p,q=net.routed(x)
        assert calls==[] and torch.equal(p,cheap)
        net.router.bias.fill_(100)
        p,q=net.routed(x)
        assert calls==[6] and torch.equal(p,full)
    hook.remove()


def test_me043_mixed_channel_dispatch_is_correct():
    net=build_model('adaptive_free',96,43).eval()
    torch.nn.init.normal_(net.out.weight,std=.01)
    x=torch.randn(2,96,3)
    with torch.no_grad():
        net.router.weight.zero_();net.router.weight[0,14]=1
        # Force different signs through the observed mean context coordinate.
        x[0,:,0]-=10;x[0,:,1]+=10
        cheap,full,q,*_=net.training_exits(x)
        p,_=net.routed(x)
        mask=(q>.5).reshape(2,1,3)
        assert mask.any() and not mask.all()
        # Dense vs subset GEMM may differ by ordinary float32 rounding.
        torch.testing.assert_close(p,torch.where(mask,full,cheap),rtol=2e-6,atol=2e-6)


def test_me043_initial_forecasts_match_ar_and_reject_bad_contracts():
    x=torch.randn(2,96,7)
    base=build_model('context_cross',96,43)
    for arm in COSTS:
        assert torch.equal(build_model(arm,96,43)(x),base(x))
    with pytest.raises(ValueError):build_model('unknown',96,43)
    with pytest.raises(ValueError):build_model('adaptive_free',96,43)(x[:,:95])
