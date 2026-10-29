"""LANGMEM085 constructed engineering tests: not a real-publisher capability claim."""
import sys,torch,hashlib,pytest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research/mathematical-engineering/langmem085'))
from operators import ARMS,AGES,write,prospective_wrong_keys,context_key,byte_book,decode

def test_five_256byte_no_train_value_and_key_arms():
    assert ARMS==('source','overwrite','partial025','residual_clip035','selective')
    book=byte_book()
    assert book.shape==(256,8) and len(torch.unique(book,dim=0))==256
    k=context_key(b'abcd');M=torch.zeros((8,8))
    assert k.norm().item()==pytest.approx(1.,abs=1e-6)
    for arm in ARMS:
        out,eta,res=write(M,k,book[53],arm,rho=.99,beta=.02,epsilon=.003)
        assert out.element_size()*out.numel()==256
        assert torch.isfinite(out).all() and res>=0 and 0<eta<=1
        assert decode(out,k,book)[0]==53

def test_selective_responds_to_residual_not_requested_target():
    book=byte_book();k=context_key(b'abcd')
    a,eta_a,_=write(torch.zeros(8,8),k,book[4],'selective',rho=.99,beta=.02,epsilon=.003)
    assert eta_a==.25
    prior=.8*book[4][:,None]*k[None,:]
    b,eta_b,_=write(prior,k,book[4],'selective',rho=.99,beta=.02,epsilon=.003)
    assert eta_b==1.
    c,clip,_=write(torch.zeros(8,8),k,book[4],'residual_clip035',rho=.99,beta=.02,epsilon=.003)
    assert clip==pytest.approx(.35,abs=1e-7)
    assert not torch.equal(a,b)

def test_clean_control_excludes_target_position_and_nearby_age():
    raw=(b'abcdWxyzzzafcdePjjjjjj'*20)[:320]
    keys={i:context_key(raw[i-4:i]) for i in range(4,290)}
    for a,t in ((32,128),(160,200),(256,289)):
        hard,uniform,n=prospective_wrong_keys(raw[:t+1],keys,a,t,"fixture")
        assert n>0
        for b in (hard,uniform):
            assert 4<=b<t and 5<=abs(a-b)<=16
            assert not a<b<=a+4
            assert raw[a-4:a]!=raw[b-4:b]
        assert (hard,uniform,n)==prospective_wrong_keys(raw[:t+1],keys,a,t,"fixture")
    with pytest.raises(ValueError):
        prospective_wrong_keys(raw[:130],keys,32,128,"fixture")

def test_write_does_not_consume_future_truth_or_accept_bad_geometry():
    k=context_key(b'abcd');v=byte_book()[34];M=torch.zeros(8,8)
    p=write(M,k,v,'overwrite',rho=.99,beta=.02,epsilon=.003)
    q=write(M,k,v,'overwrite',rho=.99,beta=.02,epsilon=.003)
    assert torch.equal(p[0],q[0])
    with pytest.raises(ValueError):write(torch.zeros(8,9),k,v,'overwrite',rho=.99,beta=.02,epsilon=.003)
    with pytest.raises(ValueError):write(M,k,v,'unregistered',rho=.99,beta=.02,epsilon=.003)
