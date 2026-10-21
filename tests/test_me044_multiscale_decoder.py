"""Constructed tensors verify numerical contracts only, never capability."""
import pytest
import torch
from tac_osm.me044_multiscale_decoder import MATCHED,build_model,forecast_basis,objective,clip_gradients,is_expert


def test_me044_basis_orthogonality_and_matched_active_capacity():
    torch.set_num_threads(1)
    for h in (96,720):
        counts=[sum(p.numel() for p in build_model(a,h,44,24).parameters()) for a in MATCHED]
        assert max(counts)/min(counts)<1.01
        for period in (24,144):
            b=forecast_basis(h,period)
            torch.testing.assert_close(b@b.T/h,torch.eye(16),atol=1e-6,rtol=1e-6)


def test_me044_cheap_training_trajectory_is_preserved_despite_native_gradients():
    torch.set_num_threads(1)
    net=build_model('pyramid_native',96,44,24)
    old=build_model('original_context',96,44,24)
    def optimizer(n):
        return torch.optim.AdamW([{'params':list(n.ar.parameters()),'lr':.0001},
            {'params':[p for k,p in n.named_parameters() if not k.startswith('ar.')],'lr':.001}],weight_decay=1e-4)
    opts=[optimizer(n) for n in (net,old)]
    x=torch.randn(2,96,3);y=torch.randn(2,96,3)
    for step in range(1,17):
        for n,opt in zip((net,old),opts):
            opt.zero_grad(set_to_none=True);loss,_=objective(n,x,y,step);loss.backward();clip_gradients(n);opt.step()
        for name,value in old.state_dict().items():assert torch.equal(net.state_dict()[name],value),name
    for name,p in net.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        assert p.grad.abs().sum()>0,name


def test_me044_scheduled_expert_zero_initialization_and_residual_bound():
    torch.set_num_threads(1)
    x=torch.randn(2,96,3)
    for arm in MATCHED:
        net=build_model(arm,96,44,144)
        old=build_model('original_context',96,44,144)
        assert torch.equal(net(x),old(x))
        calls=[];hook=net.branch.register_forward_pre_hook(lambda *_:calls.append(1))
        net.training_outputs(x,False);assert not calls
        with torch.no_grad():net.coefficients[-1].bias.fill_(100)
        cheap,delta,residual,coef,trust,scale=net.training_outputs(x,True)
        assert calls==[1] and (residual.abs()<=.5*scale+1e-6).all()
        hook.remove()
    with pytest.raises(ValueError):forecast_basis(96,100)
