"""Constructed engineering tests; no model capability evidence."""
import torch
import pytest
from scripts.langmem082_recover import RawRing,decode,read,wrong_writer

def test_packed_ring_wrap_and_capacity_boundary():
    r=RawRing(256)
    for t in range(500):r.append(t%256)
    assert len(r.data)==256
    assert r.get(259) is None and r.get(260)==260%256
    assert r.get(499)==499%256 and r.get(500) is None
    assert not hasattr(r,'clock')

def test_ring_history_scan_after_wrap_matches_independent_reference():
    data=(b'abcd1abcd2'*60)
    r=RawRing(256)
    for b in data:r.append(b)
    probs,n=r.predict(4)
    t=len(data)-1;start=len(data)-240;context=data[t-3:t+1]
    matches=[a for a in range(max(start+4,t-256,4),t) if data[a-4:a]==context]
    assert n==len(matches)
    for b in range(256):
        assert float(probs[b])==pytest.approx((sum(data[a]==b for a in matches)+8/256)/(n+8))

def test_actual_matrix_read_and_isolated_event_value_reconstruction():
    value=torch.tensor([1.,2.,3.,4.,5.,6.,7.,8.])
    key=torch.nn.functional.normalize(torch.arange(8,dtype=torch.float32)+1,dim=0)
    Q=value[:,None]*key[None,:]
    P=torch.eye(8)*.7
    vector=read((Q@P)[None,:,:],key[None,:])
    assert torch.allclose(vector,.7*value)
    # Row permutation changes floating-point reduction order; compare within
    # two dtype epsilons rather than requiring bitwise-identical rounding.
    torch.testing.assert_close(torch.linalg.vector_norm(Q),
                               torch.linalg.vector_norm(Q.roll(1,0)),
                               rtol=2*torch.finfo(Q.dtype).eps,atol=0)
    book=torch.randn(256,8,generator=torch.Generator().manual_seed(42))
    book=torch.nn.functional.normalize(book,dim=-1);book[137]=torch.nn.functional.normalize(value,dim=0)
    assert decode(vector,book)==137
    assert decode(torch.zeros(8),book)==0

def test_wrong_address_uses_only_previous_context_and_is_target_blind():
    prefix=bytes(range(129));keys={a:torch.ones(1,8)/8**.5 for a in range(4,129)}
    selected=wrong_writer(prefix,keys,32,128)
    assert prefix[selected-4:selected]!=prefix[28:32] and selected<128
    assert wrong_writer(prefix,keys,32,128)==selected

def test_ring_reset_owner_and_invalid_bytes():
    r=RawRing(4160)
    assert r.get(0) is None and len(r.data)==4160
    with pytest.raises(ValueError):r.append(256)
    with pytest.raises(ValueError):RawRing(255)
