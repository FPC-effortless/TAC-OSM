"""No model fitting in unit tests; synthetic diagnostic integrity ONLY."""
import torch
import pytest
from scripts.diagkey001_associative_recall import ARMS,LENGTHS,NVAL,model_counts,model_for,batch_for
from torch.nn import functional as F

def test_every_architecture_has_genuinely_fitting_100k_model_counts():
    counts=model_counts()
    assert set(counts)==set(ARMS)
    assert min(counts.values())>=95000 and max(counts.values())<=105000
    assert (max(counts.values())-min(counts.values()))/max(counts.values())<.01

@pytest.mark.parametrize("gap",LENGTHS)
def test_target_answer_never_leaks_and_keys_are_disjoint(gap):
    types,keys,values,answer=batch_for(75001,1,batch_size=16,length=gap)
    assert types.shape==(16,gap+5)
    assert torch.all(types[:,:4]==0) and torch.all(types[:,4:-1]==1) and torch.all(types[:,-1]==2)
    assert torch.all(values[:,-1]==0)
    assert torch.all(keys[:,4:-1]>=17)
    for i in range(16):
        assert int(keys[i,-1]) in set(keys[i,:4].tolist())
        slot=(keys[i,:4]==keys[i,-1]).nonzero().squeeze(-1)
        assert slot.numel()==1
        assert int(values[i,int(slot.item())]-1)==int(answer[i])
        assert not (keys[i,4:-1]==keys[i,-1]).any()

def test_original_fit_batches_identical_across_architectures_and_disjoint_from_test():
    src=batch_for(75001,3,batch_size=3,split="FIT")
    repeat=batch_for(75001,3,batch_size=3,split="FIT")
    other=batch_for(75001,3,batch_size=3,split="TEST")
    assert all(torch.equal(a,b) for a,b in zip(src,repeat))
    assert any(not torch.equal(a,b) for a,b in zip(src,other))

def test_each_model_reads_same_synthetic_example_and_has_active_gradients():
    for name in ARMS:
        model=model_for(name)
        t,k,v,y=batch_for(75001,0,batch_size=2,length=16)
        logits=model(t,k,v)
        loss=F.cross_entropy(logits,y)
        loss.backward()
        assert logits.shape==(2,NVAL)
        assert torch.isfinite(logits).all()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
