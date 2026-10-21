"""Numerical tests only; random tensors do not provide capability evidence."""
import numpy as np
import pytest
import torch
from tac_osm.me036_patch_forecast import ARMS, build_model


def test_registered_parameter_parity():
    for h in (96,192,336,720):
        counts=[sum(p.numel() for p in build_model(a,h,1).parameters()) for a in ('tac_patch','context','gru')]
        assert max(counts)/min(counts)-1 < .01


@pytest.mark.parametrize('arm', ARMS)
def test_complete_forecast_and_active_gradients(arm):
    torch.set_num_threads(1)
    model=build_model(arm,96,36001)
    x=torch.randn(2,96,3)
    p=model(x)
    assert p.shape==(2,96,3) and torch.isfinite(p).all()
    p.square().mean().backward()
    for name,param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(),name
        assert param.grad.abs().sum()>0,name


def test_channel_shared_models_are_permutation_equivariant():
    x=torch.randn(2,96,3)
    for arm in ARMS:
        m=build_model(arm,96,4).eval()
        with torch.no_grad():
            torch.testing.assert_close(m(x[:,:, [2,0,1]]),m(x)[:,:,[2,0,1]],atol=2e-6,rtol=2e-6)


def test_train_only_window_indices():
    from scripts.me036_research import batch
    z=torch.arange(1000.).reshape(-1,1)
    x,y=batch(z,[100,200],96)
    assert x[0,-1,0]==99 and y[0,0,0]==100 and y[0,-1,0]==195
    assert x[1,0,0]==104 and y[1,0,0]==200


def test_float32_dot_error_uses_prospective_bound():
    rng=np.random.default_rng(36)
    a=rng.normal(size=(20,96)).astype('float32');b=rng.normal(size=(96,12)).astype('float32')
    u=np.finfo(np.float32).eps/2;n=96;gamma=n*u/(1-n*u)
    ref=a.astype('float64')@b.astype('float64')
    bound=(gamma+2*u)*(abs(a.astype('float64'))@abs(b.astype('float64')))
    assert (abs((a@b).astype('float64')-ref)<=bound).all()


def test_streamed_metric_covers_every_window_and_full_horizon():
    from scripts.me036_research import score
    class Persistence(torch.nn.Module):
        def forward(self,x):return x[:,-1:].expand(-1,96,-1)
    t=torch.arange(400.,dtype=torch.float32)
    z=torch.stack((t,2*t),dim=-1)
    metric=score(Persistence(),z,100,300,96,stride=1)
    # For every anchor, channel errors are k and2k, k=1..96.
    expected=((96+1)*(2*96+1)/6)*2.5
    assert metric['windows']==105
    assert metric['scalar_targets']==105*96*2
    assert metric['MSE']==pytest.approx(expected,rel=4*np.finfo(np.float64).eps)
    assert metric['MAE']==72.75
