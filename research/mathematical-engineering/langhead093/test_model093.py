"""Scoped registered mathematical/information-boundary tests, not capability claims."""
import sys
from pathlib import Path
import pytest
import torch
REG=Path(__file__).resolve().parent
sys.path.insert(0,str(REG))
from model093 import ARMS,COUNTS,TiedHeadTAC093

@pytest.mark.parametrize('arm',ARMS)
def test_live_parameter_counts(arm):
    model=TiedHeadTAC093(arm)
    assert model.counts()=={'trainable':COUNTS[arm],'deployed':COUNTS[arm]}
    assert model.core.start(('a',)).tensor_bytes()==1920

@pytest.mark.parametrize('arm',ARMS)
def test_actual_core_causality_and_incremental_parity(arm):
    torch.manual_seed(10)
    m=TiedHeadTAC093(arm).eval()
    x=torch.randint(0,256,(1,10),dtype=torch.long)
    poison=x.clone();poison[:,7:]=(poison[:,7:]+113)%256
    with torch.no_grad():
        full,_=m(x,owners=('real-owner',))
        different,_=m(poison,owners=('real-owner',))
        assert torch.equal(full[:,:7],different[:,:7])
        state=m.start(('real-owner',));ys=[]
        for i in range(x.shape[1]):
            o,state=m(x[:,i:i+1],state=state,owners=('real-owner',))
            ys.append(o)
        assert torch.allclose(full,torch.cat(ys,1),atol=3e-5,rtol=0)
        assert state.clock==10
@pytest.mark.parametrize('arm',ARMS[1:])
def test_all_extra_gradients(arm):
    torch.manual_seed(11)
    m=TiedHeadTAC093(arm)
    x=torch.randint(0,256,(2,8))
    out,_=m(x)
    loss=torch.nn.functional.cross_entropy(out.reshape(-1,256),torch.randint(0,256,(16,)))
    loss.backward()
    assert m.raw_strength.grad is not None and torch.isfinite(m.raw_strength.grad).all()
    assert float(m.raw_strength.grad.abs().max())>0
    assert m.core.embedding.weight.grad is not None
def test_same_initial_original_core():
    torch.manual_seed(1701);a=TiedHeadTAC093('original')
    torch.manual_seed(1701);b=TiedHeadTAC093('tied_dot')
    for k,v in a.core.state_dict().items():assert torch.equal(v,b.core.state_dict()[k])
def test_permutation_is_a_bijection():
    m=TiedHeadTAC093('shuffled_dot')
    assert len(set(m.shuffled.tolist()))==256
    assert not torch.equal(m.shuffled,torch.arange(256))
def test_geometry_identity_in_native_shape():
    torch.manual_seed(37)
    E=torch.randn(256,32,requires_grad=True)
    x=torch.tensor([[17,250]])
    z=E[x];v=E
    distance=(z[:,:,None,:]-v).square().sum(-1)
    expanded=2*torch.matmul(z,v.T)-v.square().sum(-1)
    assert torch.allclose((distance-distance.mean(-1,keepdim=True)),-(expanded-expanded.mean(-1,keepdim=True)),atol=1e-4,rtol=1e-4)
def test_source_owner_invalid():
    m=TiedHeadTAC093('tied_dot')
    with pytest.raises(ValueError):
        m(torch.tensor([[14]]),state=m.start(('owner-x',)),owners=('owner-y',))
