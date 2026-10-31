"""Original-source/reference checks, not learned capability or IEEE proofs."""
import importlib.util
from pathlib import Path
import sys
import pytest
import torch

HERE=Path(__file__).resolve().parent
CANDIDATES=HERE.parent/'candidates'
if not CANDIDATES.exists():CANDIDATES=HERE.parent/'langmath101_candidates'
sys.path.insert(0,str(CANDIDATES))
import models
import reference

@pytest.mark.parametrize('arm',models.catalog())
def test_input_to_output_state_and_all_parameter_gradients(arm):
    torch.set_num_threads(1);torch.manual_seed(10101)
    m=models.make_model(arm).double();x=torch.tensor([[1,4,17,2,12],[11,21,32,7,8]])
    owners=('mathematical-a','mathematical-b');s=m.start(owners)
    p={k:v.detach().clone().requires_grad_(True) for k,v in m.named_parameters()}
    p.update({k:v.detach().clone() for k,v in m.named_buffers()})
    y,ns=m(x,state=s,owners=owners,parallel=True)
    ry,rs=reference.forward(arm,p,x,{k:v.clone() for k,v in s.tensors.items()},s.clock)
    assert torch.allclose(y,ry,atol=1e-10,rtol=1e-9),(arm,(y-ry).abs().max().item())
    assert set(ns.tensors)==set(rs)
    for key in rs:assert torch.allclose(ns.tensors[key],rs[key],atol=1e-10,rtol=1e-9),(arm,key)
    w=torch.linspace(-.3,.7,y.numel(),dtype=y.dtype).reshape_as(y)
    actual=torch.autograd.grad((y*w).mean(),list(m.parameters()),allow_unused=True)
    expected=torch.autograd.grad((ry*w).mean(),[p[k] for k,_ in m.named_parameters()],allow_unused=True)
    for (name,_),a,b in zip(m.named_parameters(),actual,expected):
        assert a is not None and b is not None,(arm,name,'disconnected')
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        assert torch.allclose(a,b,atol=1e-10,rtol=1e-8),(arm,name,(a-b).abs().max().item())
