"""Numerical execution fixtures only; not synthetic capability evidence."""
import copy
import pytest
import torch
from tac_osm.me038_innovation_forecast import build_model as original
from tac_osm.me039_native_execution import build_model,compile_native_core

@pytest.mark.parametrize('h',[96,720])
def test_frozen_weights_and_active_forward_are_exact(h):
    torch.set_num_threads(1);eager=original('innovation_tac',h,1)
    with torch.no_grad():eager.out.weight.normal_(std=.05)
    scripted=compile_native_core(copy.deepcopy(eager));x=torch.randn(2,96,3)
    assert set(eager.state_dict())==set(scripted.state_dict())
    assert all(torch.equal(v,scripted.state_dict()[k]) for k,v in eager.state_dict().items())
    assert sum(p.numel() for p in eager.parameters())==sum(p.numel() for p in scripted.parameters())
    assert torch.equal(eager(x),scripted(x))


def test_active_backward_matches_registered_roundoff_contract():
    torch.set_num_threads(1);a=original('innovation_tac',96,2)
    with torch.no_grad():a.out.weight.normal_(std=.05)
    b=compile_native_core(copy.deepcopy(a));x=torch.randn(2,96,3);y=torch.randn(2,96,3)
    (a(x)-y).square().mean().backward();(b(x)-y).square().mean().backward()
    ga=torch.cat([p.grad.flatten() for p in a.parameters()]);gb=torch.cat([p.grad.flatten() for p in b.parameters()])
    u=2**-24;gamma=96*u/(1-96*u)
    assert ((ga-gb).double().norm()/ga.double().norm().clamp_min(1e-12)).item()<=4*gamma


def test_non_native_and_unknown_execution_are_rejected():
    with pytest.raises(ValueError):compile_native_core(original('context_cross',96,1))
    with pytest.raises(ValueError):build_model(96,1,backend='unknown')
