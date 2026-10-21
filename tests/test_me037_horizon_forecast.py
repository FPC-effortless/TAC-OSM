"""Numerical fixtures only; no synthetic capability or fit evidence."""
import pytest
import torch
from tac_osm.me036_patch_forecast import build_model as old_model
from tac_osm.me037_horizon_forecast import ARMS,build_model


def test_parameter_contract():
    for h in (96,192,336,720):
        n=[sum(p.numel() for p in build_model(a,h,1).parameters()) for a in ARMS[:3]]
        assert max(n)/min(n)-1<.01


@pytest.mark.parametrize('arm',ARMS)
def test_initial_forecast_is_bit_exact_ar(arm):
    torch.set_num_threads(1)
    ar=old_model('ar',192,37)
    with torch.no_grad():ar.ar.weight.normal_(std=.01);ar.ar.bias.normal_(std=.01)
    net=build_model(arm,192,38);net.initialize_ar(ar.state_dict())
    x=torch.randn(2,96,3)
    assert torch.equal(net(x),ar(x))
    assert net.anchor_loss().item()==0


def test_horizon_gate_monotonic_and_conservative_envelope():
    m=build_model('tac_horizon',720,37)
    with torch.no_grad():m.gate.weight.normal_(std=20);m.gate.bias.normal_(std=20)
    g=m.horizon_gate(torch.randn(100,16)*10)
    assert ((g>=0)&(g<=1)).all()
    assert (g[:,1:]<=g[:,:-1]).all()
    bound=1/(1+(torch.arange(384,721,dtype=torch.float32)/384)**2)
    assert (g[:,383:]<=bound+8*torch.finfo(torch.float32).eps).all()


def test_gate_cannot_be_cancelled_by_large_residual_weights():
    m=build_model('tac_horizon',720,37)
    with torch.no_grad():m.out.weight.normal_(std=100);m.out.bias.normal_(std=100)
    x=torch.randn(2,96,3)*4+10
    _,delta,alpha=m.parts(x)
    scale=torch.sqrt((x-x.mean(1,keepdim=True)).square().mean(1,keepdim=True)+1e-5)
    envelope=alpha.reshape(2,3,720).transpose(1,2)*scale
    assert (delta.abs()<=envelope).all()


@pytest.mark.parametrize('arm',ARMS[:3])
def test_every_trained_path_gets_finite_gradients_after_zero_start(arm):
    torch.set_num_threads(1)
    m=build_model(arm,96,37);opt=torch.optim.SGD(m.parameters(),lr=.01)
    x=torch.randn(2,96,3);y=torch.randn(2,96,3)
    for _ in range(2):
        opt.zero_grad();(m(x)-y).square().mean().backward();opt.step()
    for name,p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        assert p.grad.abs().sum()>0,name
