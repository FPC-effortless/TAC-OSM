"""DIAGKEY002 no-fit synthetic mathematical integrity, not a model benchmark."""
import numpy as np
import pytest
import torch
from scripts.diagkey002_no_fit_mechanism import (
    oracle_update,delta_oracle_one,original_key_query,wilson,DELAYS)

def test_onehot_delta_write_and_clean_read_no_distractor():
    eye=np.eye(8)
    m=np.zeros((8,8))
    for k,v in zip([1,2,3,4],[5,6,7,0]):
        m=oracle_update(m,eye[k],eye[v])
    for k,v in zip([1,2,3,4],[5,6,7,0]):
        np.testing.assert_allclose(m@eye[k],eye[v],rtol=0,atol=1e-12)

@pytest.mark.parametrize("delay",DELAYS)
def test_correct_protected_association_reads_exactly(delay):
    rng=np.random.default_rng(75201+delay)
    for _ in range(20):
        correct,amplitude,error=delta_oracle_one(delay,"masked_write",rng)
        assert correct==1
        assert amplitude==pytest.approx(1.)
        assert error<1e-9

@pytest.mark.parametrize("delay",[0,1,2,4,8,16,64,256])
def test_ladder_has_four_real_writes_without_answer_leak(delay):
    t,k,v,y=original_key_query(75001,0,delay,batch_size=5)
    assert t.shape==(5,delay+5)
    assert torch.all(t[:,:4]==0)
    assert torch.all(t[:,4:-1]==1)
    assert torch.all(t[:,-1]==2)
    assert torch.all(v[:,-1]==0)
    for i in range(5):
        q=int(k[i,-1])
        targets=(k[i,:4]==q).nonzero().flatten()
        assert len(targets)==1
        assert not (k[i,4:-1]==q).any()
        assert int(v[i,int(targets[0])]-1)==int(y[i])
    t2,k2,v2,y2=original_key_query(75001,0,delay,batch_size=5)
    assert all(torch.equal(a,b) for a,b in zip((t,k,v,y),(t2,k2,v2,y2)))

def test_query_time_zero_value_write_erases_oracle_binding():
    eye=np.eye(8)
    mem=oracle_update(np.zeros((8,8)),eye[0],eye[3])
    assert np.argmax(mem@eye[0])==3
    mem=oracle_update(mem,eye[0],np.zeros(8))
    assert np.linalg.norm(mem@eye[0])<1e-12

def test_wilson_uses_true_case_denominator():
    a=wilson(0,512)
    b=wilson(512,512)
    assert 0<=a[0]<=a[1]<0.01
    assert .99<b[0]<=b[1]<=1.0000000001
