"""Exploratory kernel numerical/gradient parity; no neural fits or promotion."""
import numpy as np
import pytest
import torch
from tac_osm.csl125_r2 import selective_affine_scan
from tac_osm.csl125_r2_fast_scan import bounded_cumprod_scan
from tac_osm.csl125_r2_math_reference import selective_serial


@pytest.mark.parametrize('seed',range(12))
@pytest.mark.parametrize('chunk',[1,7,32,64])
def test_bounded_fast_reference_and_stable_gradient(seed,chunk):
    rng=np.random.default_rng(seed)
    t=137
    p=rng.dirichlet(np.ones(4),size=(2,t))
    values=rng.uniform(-1,1,(2,t,8))
    if seed%4==0:
        values[:]=np.where(np.arange(t)%2==0,1,-1)[None,:,None]
    z=np.stack([p[...,None]*values[:,:,None,:],
                np.broadcast_to(p[...,None],(2,t,4,8))],axis=-1)
    rate=rng.uniform(1/32768,.5,size=(4,8))
    gate=rng.uniform(0,1,(2,t,4))
    if seed%3==0: gate[:,1::2]=0
    carry=np.zeros((2,4,8,2))
    ref,_=selective_serial(z,carry,rate,gate)
    args=(torch.tensor(carry,dtype=torch.float32),torch.tensor(rate,dtype=torch.float32),torch.tensor(gate,dtype=torch.float32))
    x=torch.tensor(z,dtype=torch.float32).requires_grad_(True)
    base,_=selective_affine_scan(x,*args,chunk)
    fast,_=bounded_cumprod_scan(x,*args,chunk)
    serial=torch.tensor(ref,dtype=torch.float32)
    assert torch.allclose(fast,serial,atol=1e-5,rtol=1e-5)
    assert torch.allclose(fast,base,atol=1e-5,rtol=1e-5)
    gb=torch.autograd.grad(base.square().mean(),x,retain_graph=True)[0]
    gf=torch.autograd.grad(fast.square().mean(),x)[0]
    assert torch.isfinite(gf).all()
    assert torch.allclose(gb,gf,atol=5e-6,rtol=5e-5)


def test_bounded_math_prevents_small_prefix_instability():
    z=torch.zeros((1,64,1,1,2),dtype=torch.float32)
    s=torch.zeros((1,1,1,2))
    r=torch.tensor([[.5]])
    gate=torch.ones((1,64,1))
    assert float(torch.exp(torch.tensor(-32.)))>1e-14
    y,last=bounded_cumprod_scan(z,s,r,gate,64)
    assert torch.isfinite(y).all()
    with pytest.raises(ValueError,match='rate'):
        bounded_cumprod_scan(z,s,r+0.001,gate,64)
    with pytest.raises(ValueError,match='gate'):
        bounded_cumprod_scan(z,s,r,torch.full_like(gate,-1),64)
