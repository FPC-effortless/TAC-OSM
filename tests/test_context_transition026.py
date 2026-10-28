"""Constructed actual-model graph controls, NOT trained improvement evidence."""
import importlib.util
from pathlib import Path
import sys
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research/mathematical-engineering/context026'))
from context_transition import ContextTransition026


def test_actual_count_and_context_gradient_path():
    torch.manual_seed(26026);m=ContextTransition026();x=torch.randint(0,256,(2,12))
    y,trace=m(x,return_trace=True)
    assert m.parameter_count()==500122 and sum(p.numel() for p in m.parameters())==500122
    y.square().mean().backward()
    assert torch.isfinite(m.context_lag_weights.grad).all()
    assert (m.context_lag_weights.grad.abs().sum(1)>0).all()
    assert m.native_key.weight.grad.abs().sum()>0
    assert m.concept_q.weight.grad.abs().sum()>0 and m.concept_k.weight.grad.abs().sum()>0


def test_shifted_source_timing_first_write_and_current_read():
    torch.manual_seed(26026);m=ContextTransition026().double();x=torch.tensor([[4,9,4,10,4]])
    with torch.no_grad():_,tr=m(x,return_trace=True)
    assert torch.equal(tr['native_write_key'][:,1:],tr['native_query_key'][:,:-1])
    assert not tr['native_admission'][0,0] and tr['native_admission'][0,1:].all()
    assert not tr['native_memory'][:,0].any()
    state=torch.zeros(1,8,8,dtype=torch.float64)
    for t in range(x.shape[1]):
        if tr['native_admission'][0,t]:
            k=tr['native_write_key'][:,t];v=tr['native_value'][:,t]
            alpha=tr['native_alpha'][:,t,None,None];beta=tr['native_beta'][:,t,None,None]
            retained=alpha*state
            state=retained+beta*(v.unsqueeze(-1)-retained@k.unsqueeze(-1))*k.unsqueeze(-2)
        assert torch.allclose(state,tr['native_memory'][:,t],atol=1e-12,rtol=1e-12)
    assert tr['native_alpha'].max()<=.995 and state.norm()<=200*(8**.5)


def test_context_changes_key_without_future_or_position_leak():
    torch.manual_seed(26026);m=ContextTransition026().eval()
    x=torch.tensor([[3,5,7,9,11,13,17,19]])
    changed=x.clone();changed[0,1]=6
    future=x.clone();future[0,5:]=torch.tensor([88,90,92])
    with torch.no_grad():
        a,ta=m(x,return_trace=True);b,tb=m(changed,return_trace=True);f,tf=m(future,return_trace=True)
    assert not torch.allclose(ta['native_query_key'][:,2],tb['native_query_key'][:,2])
    assert torch.equal(ta['native_query_key'][:,:5],tf['native_query_key'][:,:5])
    assert torch.allclose(a[:,:5],f[:,:5],atol=1e-6,rtol=0)
    repeated=torch.tensor([[1,2,3,4,1,2,3,4]])
    with torch.no_grad():_,t=m(repeated,return_trace=True)
    assert torch.equal(t['native_query_key'][:,3],t['native_query_key'][:,7])
    with pytest.raises(ValueError,match='reset source windows'):m(x,start=8)


def test_masks_and_batch_owner_isolation():
    torch.manual_seed(26026);m=ContextTransition026().eval();x=torch.randint(0,256,(2,10))
    mask=torch.ones_like(x,dtype=torch.bool);mask[:,4:7]=False
    with torch.no_grad():
        y,tr=m(x,write_mask=mask,return_trace=True)
        one=m(x[:1],write_mask=mask[:1])
    assert torch.allclose(y[:1],one,atol=1e-6,rtol=0)
    assert torch.equal(tr['native_memory'][:,3],tr['native_memory'][:,4])
    assert torch.equal(tr['native_memory'][:,4],tr['native_memory'][:,6])
