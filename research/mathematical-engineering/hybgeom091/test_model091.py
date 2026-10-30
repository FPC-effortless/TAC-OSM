"""HYBGEOM091 constructed engineering checks: no intelligence claim."""
import math
import sys
from pathlib import Path
import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research/mathematical-engineering/hybgeom091'))
from model091 import ARMS,EXPECTED,HybridGeometryTAC091

@pytest.mark.parametrize('arm',ARMS)
def test_trainable_parameter_budget_and_actual_graph(arm):
    m=HybridGeometryTAC091(arm).eval()
    assert m.counts()=={'trainable':EXPECTED[arm],'deployed':EXPECTED[arm]}
    assert 95000<=EXPECTED[arm]<=105000
    assert m.core.counts()['trainable']==100521


def test_original_exact_logits_and_state():
    torch.manual_seed(91)
    m=HybridGeometryTAC091('original').eval()
    x=torch.tensor([[97,10,98,46,99],[33,46,10,44,77]],dtype=torch.long)
    owners=('a','b')
    with torch.no_grad():
        expected,old=m.core(x,owners=owners)
        result,state=m(x,owners=owners)
    assert torch.allclose(result,expected,atol=3e-5,rtol=3e-5)
    assert all(torch.allclose(state.core.tensors[k],v,atol=3e-5,rtol=3e-5) for k,v in old.tensors.items())
    assert state.core.tensor_bytes()==2*1920

@pytest.mark.parametrize('arm',ARMS)
def test_future_poison_and_streaming_equivalence(arm):
    torch.manual_seed(10)
    m=HybridGeometryTAC091(arm).eval()
    x=torch.tensor([[97,10,99,46,20,66,77],[33,46,65,10,45,21,44]])
    altered=x.clone();altered[:,4:]=(altered[:,4:]+101)%256
    with torch.no_grad():
        logits,_=m(x,owners=('a','b'))
        future,_=m(altered,owners=('a','b'))
        st=m.start(('a','b'));parts=[]
        for j in range(x.shape[1]):
            one,st=m(x[:,j:j+1],owners=('a','b'),state=st);parts.append(one)
    assert torch.equal(logits[:,:4],future[:,:4])
    assert torch.allclose(logits,torch.cat(parts,1),rtol=0.,atol=3e-5)
    assert torch.isfinite(logits).all()

@pytest.mark.parametrize('arm',('fixed_euclidean','contextual_euclidean','contextual_spd','hybrid','hybrid_contextual_spd'))
def test_new_parameters_receive_real_loss_gradients(arm):
    torch.manual_seed(2026)
    m=HybridGeometryTAC091(arm)
    x=torch.tensor([[97,10,46,30,43,17],[46,33,10,46,91,17]])
    y=(x+7)%256
    logits,_=m(x,owners=('a','b'))
    torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten()).backward()
    for name,param in m.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    extras={k: v.grad.norm().item() for k,v in m.named_parameters() if not k.startswith('core.')}
    assert extras and any(v>0 for v in extras.values())


def test_exact_flow_positive_and_nonexpansive_reset():
    torch.manual_seed(91)
    m=HybridGeometryTAC091('hybrid')
    rates=torch.nn.functional.softplus(m.raw_rates)
    assert (rates>0).all()
    reset=torch.sigmoid(m.raw_reset)
    assert 0<reset<=1
    for mode in range(2):
        x=torch.randn(6,32)
        dt=torch.tensor([.1,.3,1.,2.,3.,5.])
        flow=x*torch.exp(-rates[mode]*dt)[:,None]
        assert torch.all(flow.norm(dim=-1)<=x.norm(dim=-1)+1e-6)
        assert torch.allclose(flow[0], x[0]*math.exp(-float(rates[mode].detach())*.1),atol=1e-6)
        assert torch.all((flow*reset).norm(dim=-1)<=x.norm(dim=-1)+1e-6)


def test_contextual_metric_strict_spd():
    torch.manual_seed(91)
    m=HybridGeometryTAC091('contextual_spd')
    alpha=torch.nn.functional.softplus(m.raw_alpha)
    c=torch.nn.functional.normalize(torch.randn(4,32),dim=-1)
    metric=torch.eye(32)[None]+alpha*c[:,:,None]*c[:,None,:]
    assert torch.linalg.eigvalsh(metric).min()>0
    assert torch.allclose(torch.linalg.eigvalsh(metric)[:,-1],(alpha+1).expand(4),atol=1e-5)


def test_mode_trigger_and_elapsed_time_validation():
    m=HybridGeometryTAC091('hybrid').eval()
    x=torch.tensor([[10,46,88],[46,10,20]])
    with torch.no_grad():
        logits,state=m(x,owners=('x','y'),elapsed=torch.tensor([[1.,2.,3.],[2.,2.,2.]]))
    assert logits.shape==(2,3,256)
    assert state.modes.tolist()==[0,1]
    with pytest.raises(ValueError):m(x,owners=('x','y'),elapsed=torch.ones(2,3)*(-1))
    with pytest.raises(ValueError):m(x,owners=('x','y'),elapsed=torch.ones(2,2))
    with pytest.raises(ValueError):m(x,owners=('x','y'),elapsed=torch.ones(2,3)*float('nan'))


def test_no_new_model_fit_on_original_archive_and_no_global_owners():
    m=HybridGeometryTAC091('hybrid')
    x=torch.tensor([[8,11],[41,42]])
    with pytest.raises(ValueError):m(x,owners=('a','a'))
    with torch.no_grad():
        a,_=m(x,owners=('a','b'))
        b,_=m(x[:1],owners=('a',))
    assert torch.allclose(a[:1],b,atol=1e-5)
