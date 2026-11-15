"""Untrained engineering tests; genuine FIT diagnostics have separate receipts."""
from dataclasses import replace
import hashlib
import numpy as np
import pytest
import torch
from tac_osm.byte100k_001 import ByteLanguage100K,predict_byte_chunk
from tac_osm.causal_byte_corpus import ByteDocument,causal_byte_chunks
from research.byte100k_001.checks import check_case
from research.byte100k_001.math_reference import parameter_count,mean_byte_ce_and_logit_gradient


def doc(name='fixture',delta=0):
    raw=bytes((i+delta)%256 for i in range(513))
    return ByteDocument(name,'FIT','FIT',hashlib.sha256(raw).hexdigest(),raw)

@pytest.fixture(autouse=True)
def cpu_threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_actual_all_active_count_and_no_tool_parameters():
    m=ByteLanguage100K()
    assert m.validate_live_count()=={'trainable':99050,'deployed':99050}
    assert parameter_count()==99050
    assert not any(any(h in n for h in ('tool_head','field_head','accept_head','error_head','revise')) for n,_ in m.named_parameters())
    assert all(p.requires_grad for p in m.parameters())
    assert all(type(b).__name__=='PreciseRoleStateBlock' for b in m.core.blocks)
    assert m.core.config.rungs==(4,)

@pytest.mark.parametrize('seed',[1701,1702,1703])
def test_whole_model_stream_causal_gradient_falsifiers(seed):
    torch.manual_seed(seed);m=ByteLanguage100K().eval()
    row,_=check_case(m,doc())
    assert row['status']=='PASS',row


def test_actual_document_reset_and_invalid_carry():
    m=ByteLanguage100K().eval()
    a=list(causal_byte_chunks((doc('a'),),role='FIT',chunk_length=128))[:2]
    b=next(causal_byte_chunks((doc('b',9),),role='FIT',chunk_length=128))
    with torch.no_grad():
        _,prior=predict_byte_chunk(m,a[0])
        carried,_=predict_byte_chunk(m,b,prior);fresh,_=predict_byte_chunk(m,b)
        assert torch.equal(carried,fresh)
        with pytest.raises(ValueError):predict_byte_chunk(m,a[1],replace(prior,stream_id='wrong'))
        with pytest.raises(ValueError):predict_byte_chunk(m,a[1],None)


def test_live_guard_rejects_freezing_or_padding():
    m=ByteLanguage100K();next(m.parameters()).requires_grad_(False)
    with pytest.raises(ValueError):m.validate_live_count()
    next(m.parameters()).requires_grad_(True);m.padding=torch.nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError):m.validate_live_count()

@pytest.mark.parametrize('values,targets',[
 (np.zeros((1,2,256)),np.array([[0,255]])),
 (np.arange(512).reshape(1,2,256)*4,np.array([[17,128]])),
])
def test_independent_objective_reference(values,targets):
    x=torch.tensor(values,dtype=torch.float64,requires_grad=True);y=torch.tensor(targets)
    loss=torch.nn.functional.cross_entropy(x.reshape(-1,256),y.reshape(-1));loss.backward()
    ref,gradient=mean_byte_ce_and_logit_gradient(values,targets)
    assert abs(float(loss.detach())-ref)<1e-12
    assert np.allclose(x.grad.numpy(),gradient,atol=1e-12,rtol=1e-12)
