"""LANGMEM078 construction-only numerical falsifiers; not language quality."""
import sys
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F

P=Path(__file__).resolve().parents[1]/"research/mathematical-engineering"
for name in ("langmem078","langscale075","context026","langscale070","langlayer038"):
    path=str(P/name)
    if path not in sys.path:sys.path.insert(0,path)
from native_matrix import NativeMatrixTAC075
from dual_native import DualNativeTimescaleTAC078, DUAL_PARAMETERS

def test_four_parameters_original_fast_source_and_ablated_output():
    torch.manual_seed(78011)
    old=NativeMatrixTAC075("m8").eval()
    torch.manual_seed(78011)
    new=DualNativeTimescaleTAC078().eval()
    assert new.parameter_count()==499627
    d=old.state_dict();n=new.state_dict()
    assert set(d).issubset(set(n))
    for k in d:
        assert torch.equal(d[k],n[k]),k
    assert set(n)-set(d)=={"slow_log_decay","slow_erase_logit","slow_write_logit","slow_read_logit"}
    x=torch.randint(0,256,(2,18))
    with torch.no_grad():
        new.slow_read_logit.fill_(-80.)
        base=old(x);off=new(x)
    assert torch.allclose(base,off,atol=1e-6,rtol=1e-6)


def test_slow_state_bounded_mask_causal_trace():
    torch.manual_seed(78012)
    m=DualNativeTimescaleTAC078().eval()
    t=torch.randint(0,256,(2,17))
    another=t.clone();another[:,10:]+=1
    mask=torch.ones(2,17,dtype=torch.bool);mask[:,3]=False;mask[:,9]=False
    with torch.no_grad():
        a,tr=m(t,write_mask=mask,return_trace=True)
        b=m(another,write_mask=mask)
    assert a.shape==(2,17,256)
    assert torch.isfinite(a).all()
    assert torch.allclose(a[:,:10],b[:,:10],atol=1e-6,rtol=1e-6)
    assert tr["slow_memory"].shape==(2,17,8,8)
    assert tr["slow_read"].shape==(2,17,8)
    assert torch.all(tr["slow_alpha"]>0) and torch.all(tr["slow_alpha"]<1)
    assert torch.all(tr["slow_beta"]>=0) and torch.all(tr["slow_beta"]<=.05)
    assert torch.equal(tr["slow_memory"][:,3],tr["slow_memory"][:,2])
    assert torch.equal(tr["slow_memory"][:,9],tr["slow_memory"][:,8])
    assert not torch.equal(tr["slow_memory"][:,-1],torch.zeros_like(tr["slow_memory"][:,-1]))


def test_all_four_new_slow_controls_reach_next_byte_loss():
    torch.manual_seed(78013)
    m=DualNativeTimescaleTAC078()
    x=torch.randint(0,256,(2,24))
    out=m(x)
    labels=torch.randint(0,256,(2,24))
    loss=F.cross_entropy(out.reshape(-1,256),labels.reshape(-1))
    loss.backward()
    for name in ("slow_log_decay","slow_erase_logit","slow_write_logit","slow_read_logit"):
        p=getattr(m,name)
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        assert p.grad.abs().item()>0,name
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_no_slow_decay_when_source_write_mask_denied():
    torch.manual_seed(78014)
    m=DualNativeTimescaleTAC078().eval()
    x=torch.randint(0,256,(1,32))
    mask=torch.ones(1,32,dtype=torch.bool)
    mask[:,10:]=False
    with torch.no_grad():
        _,trace=m(x,write_mask=mask,return_trace=True)
    ref=trace["slow_memory"][:,9].unsqueeze(1).expand(-1,22,-1,-1)
    assert torch.equal(trace["slow_memory"][:,10:],ref)
